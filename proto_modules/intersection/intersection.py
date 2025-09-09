# modules/intersection/intersection.py
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence

import numpy as np
import torch
import traci

from .state import build_self_raw
from .policy import SB3PolicyModule
from .action import DirectPhaseIndexActionModule
from .communication import IntersectionCommClient
from .reward import RewardModule  # Protocol with .compute(obs, action, info) -> float


@dataclass
class IntersectionConfig:
    # identity & topology
    node_id: str
    tls_id: str
    detectors_in: Sequence[str]
    detectors_out: Sequence[str] = ()
    neighbour_ids: Sequence[str] = ()

    # detector grouping and normalisation (optional)
    detector_groups: Dict[str, Sequence[str]] = field(default_factory=dict)
    e2_norm_mode: str = "none"
    e2_ref_length_m: float = 30.0

    # control / action space
    action_to_phase_index: Sequence[int] = ()
    action_to_state_str: Sequence[str] = ()            # NEW: final state strings per action
    enforce_min_green_s: float = 0.0
    amber_duration_s: float = 0.0
    all_red_after_amber: bool = False
    all_red_duration_s: float = 0.0

    # embedding and device
    embed_dim: int = 64
    device: str | torch.device = "cpu"

    # history size
    history_capacity: int = 10_000


class IntersectionAgent:
    def __init__(
        self,
        cfg: IntersectionConfig,
        policy: SB3PolicyModule,
        detector_reader: Callable[[str], Mapping[str, float]],
        *,
        bus,
        publish_timing: str = "after_predict",
        auto_subscribe: bool = True,
        reward: Optional[RewardModule] = None,
    ):
        if not cfg.action_to_phase_index and not cfg.action_to_state_str:
            raise ValueError("Provide action_to_phase_index or action_to_state_str in config")

        self.cfg = cfg

        self.detector_groups = cfg.detector_groups
        self.e2_norm_mode = cfg.e2_norm_mode
        self.e2_ref_length_m = cfg.e2_ref_length_m
        
        self.policy = policy
        self.detector_reader = detector_reader
        self.reward = reward
        self.comm = IntersectionCommClient(cfg.node_id, bus)
        self.publish_timing = publish_timing

        # Neighbour buffer (pad to at least 1 row to avoid K=0)
        self._nbr_index = {nid: i for i, nid in enumerate(cfg.neighbour_ids)}
        self._K_real = len(cfg.neighbour_ids)
        self._K_eff = max(1, self._K_real)                    # <-- NEW
        self._nbr_buf = np.zeros((self._K_eff, cfg.embed_dim), dtype=np.float32)  # <-- CHANGED

        # Action module with optional min-green and amber
        self.action = DirectPhaseIndexActionModule(
            tls_id=cfg.tls_id,
            action_to_phase_index=(cfg.action_to_phase_index or None),
            action_to_state_str=(cfg.action_to_state_str or None),   # NEW
            enforce_min_green_s=cfg.enforce_min_green_s,
            amber_duration_s=cfg.amber_duration_s,
            all_red_after_amber=cfg.all_red_after_amber,
            all_red_duration_s=cfg.all_red_duration_s,
        )

        # History
        cap = int(cfg.history_capacity)
        self._cap = cap
        self.history: Dict[str, List[float]] = {"t": [], "action": [], "phase": [], "reward": [], "loss": []}

        # Auto-subscribe
        for nid in cfg.neighbour_ids:
            self.comm.subscribe_embeddings(nid, self._on_neighbour_embedding, replay_last=True)

        # Optional context/static used by build_self_raw
        self.ctx_vec: Optional[torch.Tensor] = None
        self.static_vec: Optional[torch.Tensor] = None


    @property
    def nbr_embed_shape(self) -> tuple[int, int]:
        return self._nbr_buf.shape

    # comm hooks

    def _on_neighbour_embedding(self, sender_id: str, emb: np.ndarray) -> None:
        i = self._nbr_index.get(sender_id, None)
        if i is None:
            return
        emb = np.asarray(emb, dtype=np.float32)
        if emb.shape != (self.cfg.embed_dim,):
            raise ValueError(f"Embedding from {sender_id} has shape {emb.shape}, expected ({self.cfg.embed_dim},)")
        self._nbr_buf[i] = emb

    def publish_self_embedding(self) -> None:
        emb = self.policy.get_last_self_embedding()
        if emb is not None:
            self.comm.publish_embedding(self.cfg.node_id, emb.detach().cpu().numpy())
            

    # observation

    def _build_obs(self) -> Dict[str, np.ndarray]:
        dev = torch.device(self.cfg.device)
        self_raw = build_self_raw(self, detector_reader=self.detector_reader, device=dev)
        return {"self_raw": self_raw.detach().cpu().numpy().astype(np.float32),
                "nbr_embed": self._nbr_buf.astype(np.float32)}

    # control tick

    def step(self, deterministic: bool = True, info: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        obs = self._build_obs()
        batched = {k: v[None, ...] for k, v in obs.items()}
        action, _ = self.policy.predict(batched, deterministic=deterministic)
        action = int(action)

        if self.publish_timing == "after_predict":
            self.publish_self_embedding()

        self.action.apply(action)
        phase = traci.trafficlight.getPhase(self.cfg.tls_id)

        if self.publish_timing == "after_apply":
            self.publish_self_embedding()

        rew_val = None
        if self.reward is not None:
            info = dict(info or {})
            info["tls_id"] = self.cfg.tls_id
            # chosen action’s target green string (state-string mode)
            if self.action.map_state is not None and 0 <= action < len(self.action.map_state):
                info["action_state"] = self.action.map_state[action]
            # actual current TLS state (may be amber/red if called immediately after apply)
            info["cur_state"] = traci.trafficlight.getRedYellowGreenState(self.cfg.tls_id)
            rew_val = float(self.reward.compute(obs, action, info))

        self._log(t=traci.simulation.getTime(), action=action, phase=phase, reward=rew_val)  # type: ignore
        return {"action": action, "phase": phase, "reward": rew_val}

    # history

    def _log(self, *, t: float, action: int, phase: int, reward: Optional[float]) -> None:
        for k, v in (("t", t), ("action", float(action)), ("phase", float(phase))):
            self.history[k].append(v)
            if len(self.history[k]) > self._cap:
                self.history[k] = self.history[k][-self._cap:]
        if reward is not None:
            self.history["reward"].append(reward)
            if len(self.history["reward"]) > self._cap:
                self.history["reward"] = self.history["reward"][-self._cap:]

    def record_loss(self, loss_value: float) -> None:
        self.history["loss"].append(float(loss_value))
        if len(self.history["loss"]) > self._cap:
            self.history["loss"] = self.history["loss"][-self._cap:]

    def metrics_snapshot(self) -> Dict[str, np.ndarray]:
        return {k: np.asarray(v, dtype=np.float32) for k, v in self.history.items()}

    # attributes used by build_self_raw

    @property
    def detectors_in(self) -> Sequence[str]:
        return self.cfg.detectors_in

    @property
    def detectors_out(self) -> Sequence[str]:
        return self.cfg.detectors_out
