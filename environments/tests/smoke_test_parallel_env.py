# ============================================================
#  DISCLAIMER:
#  This test case was written by guiding an LLM (GPT-5 Thinking) and then refining it.
#  The submodules imported and tested here were, however, developed and written individually.
# ============================================================

import argparse
import numpy as np

from environments.pz_multi_tls_env import MultiTLSParallelEnv
from modules.intersection.intersection import IntersectionConfig
from modules.intersection.preprocessor import PreprocessorConfig as FeatureConfig
from utils.sumo_helpers import SUMOConfig


def _sample_valid_action(mask: np.ndarray) -> int:
    valid = np.flatnonzero(mask.astype(np.bool_))
    return int(np.random.choice(valid)) if valid.size else 0


def quick_smoke(
    intersection_agent_configs,
    feature_config,
    sumo_config,
    episode_length=60,
    ticks_per_decision=1,
    max_steps=20,
):

    env = MultiTLSParallelEnv(
        intersection_agent_configs=intersection_agent_configs,
        feature_config=feature_config,
        sumo_config=sumo_config,
        episode_length=episode_length,
        ticks_per_decision=ticks_per_decision,
    )

    obs, infos = env.reset()
    print(f"[reset] agents: {list(obs.keys())}")
    for aid in obs:
        lf = obs[aid]["lane_features"]
        lm = obs[aid]["lane_mask"]
        am = obs[aid]["action_mask"]
        print(
            f"  - {aid}: lane_features {lf.shape}, lane_mask {lm.shape} (sum={lm.sum():.0f}), "
            f"action_mask {am.shape}, action_space.n={env.action_space(aid).n}"
        )

    for t in range(max_steps):
        actions = {
            aid: _sample_valid_action(ob["action_mask"]) for aid, ob in obs.items()
        }
        obs, rews, terms, truncs, infos = env.step(actions)

        done = all(terms.get(aid, False) for aid in env.agents)
        print(f"[step {t+1}] reward per agent: {rews}  done={done}")
        if done:
            break

    env.close()
    print("[done] closed env cleanly")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="ParallelEnv smoke test")
    ap.add_argument("--sumocfg", required=True)
    ap.add_argument(
        "--tls-ids", required=True, help="Comma-separated TLS ids, e.g. tlJ,tlK"
    )
    ap.add_argument("--episode-length", type=int, default=120)
    ap.add_argument("--ticks-per-decision", type=int, default=1)
    ap.add_argument("--max-steps", type=int, default=20)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--nogui", action="store_true")
    args = ap.parse_args()

    # Build per-intersection configs (keep it simple for the smoke test)
    tls_ids = [t.strip() for t in args.tls_ids.split(",") if t.strip()]
    intersection_agent_configs = [
        IntersectionConfig(tls_id=tls_id, warm_start=True) for tls_id in tls_ids
    ]

    # Same feature selection for all agents
    feature_config = FeatureConfig(
        include_queue=True,
        include_approach=True,
        include_total_wait=True,
        include_max_wait=True,
        include_total_speed=True,
    )

    # SUMO settings
    sumo_config = SUMOConfig(
        sumocfg_filepath=args.sumocfg,
        nogui=bool(args.nogui),
        seed=int(args.seed),
    )

    # Repro
    np.random.seed(int(args.seed))

    quick_smoke(
        intersection_agent_configs=intersection_agent_configs,
        feature_config=feature_config,
        sumo_config=sumo_config,
        episode_length=int(args.episode_length),
        ticks_per_decision=int(args.ticks_per_decision),
        max_steps=int(args.max_steps),
    )
