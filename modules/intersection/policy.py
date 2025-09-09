# modules/intersection/policy.py
"""
SB3 Policy wrapper for a single intersection agent.

- Binds a custom features extractor (NeighbourGNNFeatures) to PPO / MaskablePPO.
- Allows late env binding via set_env(env) to avoid circular construction.
- Exposes small helpers for train/infer and save/load round-trips.
"""

from __future__ import annotations

from typing import Optional, Dict, Any, Type, cast

import gymnasium as gym
import torch
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3 import PPO

from sb3_contrib import MaskablePPO



from .features_extractor import NeighbourGNNFeatures


ALGOS: Dict[str, Type[BaseAlgorithm]] = {
    "PPO": PPO,
    "MaskablePPO": cast(Type[BaseAlgorithm], MaskablePPO),
}


class SB3PolicyModule:
    """
    Thin wrapper around SB3 to couple NeighbourGNNFeatures with PPO/MaskablePPO.

    Parameters
    ----------
    env : gym.Env | None
        May be None. Bind later with set_env(env).
    algo_name : {"PPO","MaskablePPO"}
    extractor_kwargs : dict
        kwargs for NeighbourGNNFeatures:
          - self_raw_dim, embed_dim, gat_hidden, gat_heads, edge_index, device
    net_arch : dict | None
        e.g. dict(pi=[128], vf=[128]) for small MLP heads.
    algo_kwargs : dict
        Forwarded to SB3 algo (learning_rate, n_steps, etc.).
    device : str | torch.device
    verbose : int
    """

    def __init__(
        self,
        env: Optional[gym.Env] = None,
        algo_name: str = "PPO",
        extractor_kwargs: Optional[Dict[str, Any]] = None,
        net_arch: Optional[Dict[str, Any]] = None,
        algo_kwargs: Optional[Dict[str, Any]] = None,
        device: str | torch.device = "auto",
        verbose: int = 0,
    ):
        # validate algo choice
        if algo_name not in ALGOS or ALGOS[algo_name] is None:
            available = [k for k, v in ALGOS.items() if v is not None]
            raise ValueError(f"Unknown or unavailable algo '{algo_name}'. Available: {available}")

        self._algo_name = algo_name
        self._extractor_kwargs = extractor_kwargs or {}
        self._net_arch = net_arch
        self._algo_kwargs = algo_kwargs or {}
        self._device = device
        self._verbose = verbose

        self.model: Optional[BaseAlgorithm] = None
        if env is not None:
            self.set_env(env)

    # ------------------------------ wiring ----------------------------------

    def _make_policy_kwargs(self) -> Dict[str, Any]:
        pk: Dict[str, Any] = {
            "features_extractor_class": NeighbourGNNFeatures,
            "features_extractor_kwargs": self._extractor_kwargs,
        }
        if self._net_arch is not None:
            pk["net_arch"] = self._net_arch
        return pk

    def _policy_name_for(self, env: gym.Env) -> str:
        return "MultiInputPolicy" if isinstance(env.observation_space, gym.spaces.Dict) else "MlpPolicy"

    def set_env(self, env: gym.Env) -> None:
        """Bind or swap the SB3 env. Creates the underlying SB3 model on first call."""
        AlgoClass = cast(Type[BaseAlgorithm], ALGOS[self._algo_name])
        if self.model is None:
            self.model = AlgoClass(
                policy=self._policy_name_for(env),
                env=env,
                policy_kwargs=self._make_policy_kwargs(),
                device=self._device,
                verbose=self._verbose,
                **self._algo_kwargs,
            )
        else:
            self.model.set_env(env)

    # ------------------------------ API -------------------------------------

    def learn(self, total_timesteps: int, **kwargs) -> None:
        if self.model is None:
            raise RuntimeError("Policy has no env bound. Call set_env(env) first.")
        self.model.learn(total_timesteps=total_timesteps, **kwargs)

    def predict(self, obs, deterministic: bool = True):
        if self.model is None:
            raise RuntimeError("Policy has no env bound. Call set_env(env) first.")
        return self.model.predict(obs, deterministic=deterministic)

    # --------------------------- save / load ---------------------------------

    def _save(self, path: str) -> None:
        """Save SB3 zip at path (native SB3 format)."""
        if self.model is None:
            raise RuntimeError("Nothing to save; model not built. Call set_env(env) first.")
        self.model.save(path)

    @classmethod
    def _load(
        cls,
        *,
        path: str,
        env: gym.Env,
        algo_name: str = "PPO",
        extractor_kwargs: Optional[Dict[str, Any]] = None,
        net_arch: Optional[Dict[str, Any]] = None,
        algo_kwargs: Optional[Dict[str, Any]] = None,
        device: str | torch.device = "auto",
        verbose: int = 0,
    ) -> "SB3PolicyModule":
        """
        Create a new wrapper and load an SB3 .zip saved with _save().
        Note: SB3 restores the extractor and heads from the archive; the kwargs here
        are kept on the wrapper for completeness but do not affect the loaded weights.
        """
        # Construct wrapper and underlying algo bound to env
        obj = cls(
            env=None,
            algo_name=algo_name,
            extractor_kwargs=extractor_kwargs,
            net_arch=net_arch,
            algo_kwargs=algo_kwargs,
            device=device,
            verbose=verbose,
        )
        # Build an empty algo instance then load weights into it
        obj.set_env(env)
        assert obj.model is not None
        AlgoClass = cast(Type[BaseAlgorithm], ALGOS[algo_name])
        # SB3 loads into a new instance; we swap it in
        loaded = AlgoClass.load(path, env=env, device=device, print_system_info=False)
        obj.model = loaded
        return obj

    # -------------------------- convenience ----------------------------------

    def get_last_self_embedding(self) -> Optional[torch.Tensor]:
        """
        Return last self-node embedding from the extractor after a forward pass.
        Requires that either predict() or learn() ran at least once.
        """
        if self.model is None:
            return None
        fe: BaseFeaturesExtractor = self.model.policy.features_extractor  # type: ignore[attr-defined]
        getter = getattr(fe, "get_last_self_embedding", None)
        if callable(getter):
            return cast(Optional[torch.Tensor], getter())
        return None

    @property
    def algo(self) -> BaseAlgorithm:
        if self.model is None:
            raise RuntimeError("Policy has no env bound. Call set_env(env) first.")
        return self.model
