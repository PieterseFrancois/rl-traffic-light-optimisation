# environments/ingolstadt/intersections/builder.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence, Tuple
from types import SimpleNamespace

import yaml
import torch
import traci

from modules.intersection.intersection import IntersectionAgent, IntersectionConfig
from modules.intersection.policy import SB3PolicyModule
from modules.intersection.state import build_self_raw, detector_reader
from modules.intersection.env_adapter import IntersectionEnv
from utils.graph_utils import edge_index_from_edge_list, Edge


# -------- SUMO helpers to infer action mappings (indices + strings) --------

def _infer_action_mappings(tls_id: str) -> Tuple[List[int], List[str]]:
    """
    Returns (keep_indices, keep_state_strings) for phases that are not transitional (no 'y')
    and contain at least one green ('g' or 'G').
    """
    try:
        infos = traci.trafficlight.getAllProgramLogics(tls_id)
        logic = infos[0]  # active program is usually first; adjust if you version programs
        phases = getattr(logic, "phases", None) or []
        states: List[str] = []
        for ph in phases:
            st = getattr(ph, "state", None)
            if isinstance(st, str):
                states.append(st)
        keep = [i for i, s in enumerate(states) if ("y" not in s) and any(ch in ("g", "G") for ch in s)]
        return keep, [states[i] for i in keep]
    except Exception as e:
        raise ValueError(f"Cannot infer actions for '{tls_id}': {e}")


# ------------------------------ main builder -------------------------------

def build_envs_from_yaml(yaml_path: str, bus) -> Dict[str, Tuple[IntersectionEnv, IntersectionAgent, SB3PolicyModule]]:
    """
    Parse YAML and create (env, agent, policy) tuples per node.
    """
    with open(yaml_path, "r") as f:
        y = yaml.safe_load(f)

    defaults = y.get("defaults", {})
    out: Dict[str, Tuple[IntersectionEnv, IntersectionAgent, SB3PolicyModule]] = {}

    for spec in y.get("intersections", []):
        # Merge defaults with spec
        d = {**defaults, **spec}

        # Topology
        node_id = d["node_id"]
        icfg = IntersectionConfig(
            node_id=node_id,
            tls_id=d["tls_id"],
            detectors_in=d.get("detectors_in", []),
            detectors_out=d.get("detectors_out", []),
            neighbour_ids=d.get("neighbour_ids", []),
            detector_groups=d.get("detector_groups", {}),
            e2_norm_mode=d.get("e2_norm_mode", defaults.get("e2_norm_mode", "none")),
            e2_ref_length_m=float(d.get("e2_ref_length_m", defaults.get("e2_ref_length_m", 30.0))),
            enforce_min_green_s=float(d.get("tl_config", {}).get("enforce_min_green_s", 0.0)),
            amber_duration_s=float(d.get("tl_config", {}).get("amber_duration_s", 0.0)),
            all_red_after_amber=bool(d.get("tl_config", {}).get("all_red_after_amber", False)),
            all_red_duration_s=float(d.get("tl_config", {}).get("all_red_duration_s", 0.0)),
            embed_dim=int(d.get("embed_dim", defaults.get("embed_dim", 64))),
            device=str(d.get("device", defaults.get("device", "cpu"))),
        )

        # Action mappings
        if "action_to_state_str" in d and d["action_to_state_str"]:
            icfg.action_to_state_str = list(d["action_to_state_str"])
            icfg.action_to_phase_index = list(d.get("action_to_phase_index", []))
        elif d.get("infer_actions_from_sumo", False) or not d.get("action_to_phase_index"):
            keep, keep_states = _infer_action_mappings(icfg.tls_id)
            icfg.action_to_phase_index = keep
            icfg.action_to_state_str = keep_states
        else:
            icfg.action_to_phase_index = list(d.get("action_to_phase_index", []))
            icfg.action_to_state_str = list(d.get("action_to_state_str", []))

        # Local directed subgraph
        edges: Sequence[Edge] = [tuple(e) for e in d.get("edges", [])]
        edge_index = edge_index_from_edge_list(
            self_id=icfg.node_id,
            neighbour_ids=icfg.neighbour_ids,
            edges=edges,
            add_self_loops=True,
            symmetric=False,
            device=str(icfg.device),
        )

        # Infer F_RAW once
        probe = SimpleNamespace(detectors_in=icfg.detectors_in, detectors_out=icfg.detectors_out,
                                ctx_vec=None, static_vec=None)
        F_RAW = int(build_self_raw(probe, detector_reader=detector_reader,
                                   device=torch.device(icfg.device)).numel())

        # Policy (SB3) configured with this subgraph
        extractor_kwargs = dict(
            self_raw_dim=F_RAW,
            embed_dim=icfg.embed_dim,
            edge_index=edge_index,
            gat_hidden=int(d.get("gat_hidden", defaults.get("gat_hidden", 64))),
            gat_heads=int(d.get("gat_heads", defaults.get("gat_heads", 2))),
            device=icfg.device,
        )
        policy = SB3PolicyModule(
            env=None,  # set later
            algo_name=d.get("algo_name", defaults.get("algo_name", "PPO")),
            extractor_kwargs=extractor_kwargs,
            net_arch=d.get("net_arch", defaults.get("net_arch", dict(pi=[128], vf=[128]))),
            algo_kwargs=d.get("algo_kwargs", defaults.get("algo_kwargs", dict(n_steps=2048))),
            device=icfg.device,
            verbose=int(d.get("policy_verbose", defaults.get("policy_verbose", 0))),
        )

        # Agent
        agent = IntersectionAgent(
            cfg=icfg,
            policy=policy,
            detector_reader=detector_reader,
            bus=bus,
            publish_timing="after_predict",
            auto_subscribe=bool(d.get("auto_subscribe", defaults.get("auto_subscribe", True))),
        )

        # Env adapter
        env = IntersectionEnv(agent, ticks_per_decision=int(d.get("ticks_per_decision", defaults.get("ticks_per_decision", 5))))
        policy.set_env(env)

        out[node_id] = (env, agent, policy)

    return out
