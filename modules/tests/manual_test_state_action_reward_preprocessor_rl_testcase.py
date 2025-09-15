# ============================================================
#  DISCLAIMER:
#  This manual test case was written by guiding an LLM (GPT-5 Thinking) and then refining it.
#  The submodules imported and tested here was, however, developed and written individually.
# ============================================================

import os
import sys
import argparse
from pathlib import Path
from typing import Optional

# ---------------- SUMO bootstrap ----------------
if "SUMO_HOME" in os.environ:
    tools = os.path.join(os.environ["SUMO_HOME"], "tools")
    sys.path.append(tools)
else:
    sys.exit('Declare environment variable "SUMO_HOME"')

from sumolib import checkBinary  # noqa: E402
import traci  # noqa: E402

# ---------------- RL / Gym ----------------------
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO

import numpy as np
import csv
import matplotlib.pyplot as plt

# ---------------- Your modules ------------------
# Adjust imports to your package layout
from modules.intersection.state import StateModule
from modules.intersection.action import ActionModule, TLSTimingStandards
from modules.intersection.preprocessor import (
    PreprocessorModule,
    PreprocessorConfig,
    PreprocessorNormalisationParameters,
)
from modules.intersection.reward import (
    RewardModule,
    RewardNormalisationParameters as RewardNormParams,
    RewardFunction,
)


# ============================================================
#                        Utilities
# ============================================================
def start_sumo(sumocfg: str, *, nogui: bool, seed: Optional[int]) -> None:
    args = [
        checkBinary("sumo" if nogui else "sumo-gui"),
        "-c",
        sumocfg,
        "--start",
        "--no-warnings",
    ]
    if seed is not None:
        args += ["--seed", str(seed)]
    traci.start(args)


def close_sumo() -> None:
    try:
        traci.close(False)
    except Exception:
        pass


def collect_kpis(state: list) -> tuple[float, float]:
    """Return (total_queue, total_wait_s) from a list[LaneMeasures]."""
    total_queue = float(sum(lm.queue for lm in state))
    total_wait = float(sum(lm.total_wait_s for lm in state))
    return total_queue, total_wait


# ============================================================
#              Minimal single-TLS Gym environment
# ============================================================
class SingleTLSEnv(gym.Env):
    """
    One env = one intersection. Each env step:
      - If ready_for_decision(): apply action -> ActionModule.set_phase()
      - Advance SUMO by ticks_per_decision (driving transitions via update_transition)
      - Build observation via Preprocessor
      - Compute reward via RewardModule
    """

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        sumocfg: str,
        tls_id: str,
        *,
        nogui: bool = False,
        seed: Optional[int] = 42,
        episode_seconds: float = 300.0,
        ticks_per_decision: int = 1,
        reward_name: str = "queue",  # "queue" or "total_wait"
        normalise_reward: bool = True,
        normalise_state: bool = True,
        max_detection_range_m: float = 50.0,
        avg_vehicle_length_m: float = 5.0,
        min_gap_m: float = 2.5,
        max_wait_time_horizon_s: float = 36.0,
        timing: Optional[TLSTimingStandards] = TLSTimingStandards(
            min_green_s=5.0, yellow_s=2.0, all_red_s=1.0, max_green_s=30.0
        ),
    ):
        super().__init__()
        self.sumocfg = sumocfg
        self.tls_id = tls_id
        self.nogui = nogui
        self.seed_val = seed
        self.episode_seconds = float(episode_seconds)
        self.ticks_per_decision = int(max(1, ticks_per_decision))
        self.reward_name = (
            RewardFunction.QUEUE
            if reward_name == "queue"
            else RewardFunction.TOTAL_WAIT
        )

        self._sumo_running = False
        self._t0 = 0.0
        self._t_end = 0.0

        # Modules (constructed after SUMO starts)
        self.state_module: Optional[StateModule] = None
        self.action_module: Optional[ActionModule] = None
        self.preproc: Optional[PreprocessorModule] = None
        self.reward_module: Optional[RewardModule] = None

        # Normalisation configs
        self._state_norm_params = (
            PreprocessorNormalisationParameters(
                max_detection_range_m=max_detection_range_m,
                avg_vehicle_occupancy_length_m=avg_vehicle_length_m + min_gap_m,
                max_wait_time_horizon_s=max_wait_time_horizon_s,
            )
            if normalise_state
            else None
        )
        self._reward_norm_params = (
            RewardNormParams(  # type: ignore
                max_detection_range_m=max_detection_range_m,
                avg_vehicle_length_m=avg_vehicle_length_m + min_gap_m,
            )
            if (normalise_reward and RewardNormParams is not None)
            else None
        )
        self._timing = timing

        # Placeholder spaces (finalized in _lazy_init())
        # Do a one-shot probe so SB3 sees the correct spaces during model init
        obs_shape, n_actions = self._probe_spaces()

        self.action_space = spaces.Discrete(n_actions)
        self.observation_space = spaces.Box(
            low=-1.0, high=1.0, shape=obs_shape, dtype=np.float32
        )

    def _probe_spaces(self) -> tuple[tuple[int, ...], int]:
        """Start SUMO briefly to discover obs shape and number of actions, then close."""
        start_sumo(self.sumocfg, nogui=True, seed=self.seed_val)
        try:
            # Build the same modules you use later
            sm = StateModule(
                traci,
                self.tls_id,
                max_detection_range_m=(
                    self._state_norm_params.max_detection_range_m
                    if self._state_norm_params
                    else 50.0
                ),
            )
            am = ActionModule(
                traci,
                self.tls_id,
                timing_standards=(
                    self._timing if self._timing else TLSTimingStandards()
                ),
            )
            pm = PreprocessorModule(
                traci,
                self.tls_id,
                config=PreprocessorConfig(
                    include_queue=True,
                    include_approach=True,
                    include_total_wait=True,
                    include_max_wait=True,
                    include_total_speed=True,
                ),
                normalisation_params=self._state_norm_params,
            )
            # deterministic warm start & tick once
            am.set_phase_immediately(0)
            traci.simulationStep()

            sample = (
                pm.get_state_tensor(sm.read_state())
                .flatten()
                .cpu()
                .numpy()
                .astype(np.float32)
            )  # noqa: E501
            obs_shape = sample.shape  # e.g., (20,)
            n_actions = am.n_actions
            return obs_shape, n_actions
        finally:
            close_sumo()

    # ---- SUMO lifecycle ----
    def _start(self):
        if not self._sumo_running:
            start_sumo(self.sumocfg, nogui=self.nogui, seed=self.seed_val)
            self._sumo_running = True

    def _stop(self):
        if self._sumo_running:
            close_sumo()
            self._sumo_running = False

    # ---- Build modules once SUMO is live ----
    def _lazy_init(self):
        self.state_module = StateModule(
            traci_connection=traci,
            tls_id=self.tls_id,
            max_detection_range_m=(
                self._state_norm_params.max_detection_range_m
                if self._state_norm_params
                else 50.0
            ),
        )
        self.action_module = ActionModule(
            traci_connection=traci,
            tls_id=self.tls_id,
            timing_standards=self._timing if self._timing else TLSTimingStandards(),
        )
        self.action_module.set_phase_immediately(0)

        self.preproc = PreprocessorModule(
            traci_connection=traci,
            tls_id=self.tls_id,
            config=PreprocessorConfig(
                include_queue=True,
                include_approach=True,
                include_total_wait=True,
                include_max_wait=True,
                include_total_speed=True,
            ),
            normalisation_params=self._state_norm_params,
        )

        self.reward_module = RewardModule(
            traci_connection=traci,
            tls_id=self.tls_id,
            normalisation_params=self._reward_norm_params,
        )
        try:
            self.reward_module.set_active_reward_function(self.reward_name)
        except Exception:
            self.reward_module.set_active_reward_function(RewardFunction.QUEUE)

    # ---- Gym API ----
    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        self._stop()
        self._start()
        self._t0 = float(traci.simulation.getTime())
        self._t_end = self._t0 + self.episode_seconds
        self._lazy_init()
        return self._get_obs(), {}

    def _get_obs(self) -> np.ndarray:
        lane_state = self.state_module.read_state()  # type: ignore
        x = self.preproc.get_state_tensor(lane_state).flatten().cpu().numpy().astype(np.float32)  # type: ignore
        return x

    def step(self, action: int):
        assert self.action_module and self.state_module and self.reward_module

        # Keep controller timers in sync before deciding
        self.action_module.update_transition()

        info = {}
        ignored = False

        if self.action_module.ready_for_decision():
            try:
                mask = (
                    self.action_module.get_action_mask()
                )  # list[bool], len = n_actions

                # Guard out-of-range
                if action < 0 or action >= len(mask):
                    info["invalid_action_index"] = int(action)
                    action = (
                        self.action_module.active_phase_memory.phase_index
                        if self.action_module.active_phase_memory
                        else 0
                    )

                if not mask[action]:
                    info["invalid_action"] = int(action)
                    # Project to a valid action; prefer switching away from current
                    valid = [i for i, ok in enumerate(mask) if ok]
                    cur = (
                        self.action_module.active_phase_memory.phase_index
                        if self.action_module.active_phase_memory
                        else None
                    )
                    if cur is not None:
                        prefer = [i for i in valid if i != cur]
                        if prefer:
                            valid = prefer
                    # Fallback if mask is (unexpectedly) all False
                    action = valid[0] if valid else (cur if cur is not None else 0)
                    info["projected_action"] = int(action)

                self.action_module.set_phase(int(action))
            except RuntimeError:
                # e.g., transition just started; skip this decision
                ignored = True
        else:
            ignored = True

        # Advance environment to the next decision boundary
        for _ in range(self.ticks_per_decision):
            self.action_module.update_transition()
            traci.simulationStep()

        # Fresh observation & reward from current state
        state = self.state_module.read_state()
        obs = self._get_obs()  # ensure this reads from state_module/preprocessor
        reward = float(self.reward_module.compute_reward(state))

        t_now = float(traci.simulation.getTime())
        terminated = bool(
            t_now >= self._t_end or traci.simulation.getMinExpectedNumber() <= 0
        )
        truncated = False

        info.update({"ignored_action": ignored, "t_sim": t_now})
        return obs, reward, terminated, truncated, info

    def close(self):
        self._stop()
        super().close()


# ============================================================
#                 Baseline & PPO evaluation
# ============================================================
def run_baseline_episode(
    sumocfg: str,
    tls_id: str,
    seconds: float,
    *,
    nogui: bool,
    seed: Optional[int],
    fixed_green_s: float = 10.0,
    timing: TLSTimingStandards = TLSTimingStandards(
        min_green_s=5, yellow_s=2, all_red_s=1, max_green_s=30
    ),
    csv_path: Path = Path("runs/ppo_test/baseline.csv"),
):
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    start_sumo(sumocfg, nogui=False, seed=seed)

    am = ActionModule(traci, tls_id, timing_standards=timing)
    sm = StateModule(traci, tls_id, max_detection_range_m=50.0)
    rm = RewardModule(traci, tls_id, normalisation_params=None)
    rm.set_active_reward_function(RewardFunction.QUEUE)

    # deterministic warm start
    am.set_phase_immediately(0)
    last_switch_t = float(traci.simulation.getTime())

    with csv_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["t", "total_queue", "total_wait_s", "reward"])

        t0 = float(traci.simulation.getTime())
        end_t = t0 + seconds
        while float(traci.simulation.getTime()) < end_t:
            # Time to switch?
            now = float(traci.simulation.getTime())
            if am.ready_for_decision() and (now - last_switch_t >= fixed_green_s):
                # simple toggler (if 2 actions), else round-robin
                next_phase = (
                    (am.active_phase_memory.phase_index + 1) % am.n_actions
                    if am.active_phase_memory
                    else 0
                )
                am.set_phase(next_phase)
                last_switch_t = now

            am.update_transition()
            traci.simulationStep()

            state = sm.read_state()
            q, wsum = collect_kpis(state)
            rew = float(rm.compute_reward(state))
            w.writerow([float(traci.simulation.getTime()), q, wsum, rew])

    close_sumo()


def run_ppo_eval_episode(
    model: PPO,
    env: SingleTLSEnv,
    seconds: float,
    csv_path: Path = Path("runs/ppo_test/ppo_eval.csv"),
):
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    obs, _ = env.reset()
    start_t = float(traci.simulation.getTime())
    end_t = start_t + seconds

    # Access modules for KPI logging (env owns them)
    assert env.state_module is not None and env.reward_module is not None

    with csv_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["t", "total_queue", "total_wait_s", "reward"])
        while True:
            action, _ = model.predict(obs, deterministic=False)
            obs, reward, done, trunc, info = env.step(int(action))

            # KPIs from latest state
            state = env.state_module.state or []
            q, wsum = collect_kpis(state)
            w.writerow([info.get("t_sim", 0.0), q, wsum, float(reward)])

            if done or trunc or float(traci.simulation.getTime()) >= end_t:
                break


def load_csv(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    t, q, w, r = [], [], [], []
    with path.open("r") as f:
        rdr = csv.DictReader(f)
        for row in rdr:
            t.append(float(row["t"]))
            q.append(float(row["total_queue"]))
            w.append(float(row["total_wait_s"]))
            r.append(float(row["reward"]))
    return np.array(t), np.array(q), np.array(w), np.array(r)


def plot_compare(baseline_csv: Path, ppo_csv: Path, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    tb, qb, wb, rb = load_csv(baseline_csv)
    tp, qp, wp, rp = load_csv(ppo_csv)

    # 1) Queue over time
    plt.figure()
    plt.plot(tb, qb, label="baseline")
    plt.plot(tp, qp, label="ppo")
    plt.xlabel("time [s]")
    plt.ylabel("total queue [veh]")
    plt.title("Total queue over time")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "queue_timeseries.png", dpi=160)

    # 2) Total wait over time
    plt.figure()
    plt.plot(tb, wb, label="baseline")
    plt.plot(tp, wp, label="ppo")
    plt.xlabel("time [s]")
    plt.ylabel("sum of waits [s]")
    plt.title("Total accumulated wait per tick")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "wait_timeseries.png", dpi=160)

    # 3) Reward over time
    plt.figure()
    plt.plot(tb, rb, label="baseline")
    plt.plot(tp, rp, label="ppo")
    plt.xlabel("time [s]")
    plt.ylabel("reward")
    plt.title("Reward over time")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "reward_timeseries.png", dpi=160)

    # 4) Means (bars)
    plt.figure()
    means_q = [qb.mean(), qp.mean()]
    means_w = [wb.mean(), wp.mean()]
    means_r = [rb.mean(), rp.mean()]
    x = np.arange(3)
    width = 0.35

    plt.bar(
        x - width / 2, [means_q[0], means_w[0], means_r[0]], width, label="baseline"
    )
    plt.bar(x + width / 2, [means_q[1], means_w[1], means_r[1]], width, label="ppo")
    plt.xticks(x, ["mean queue", "mean wait", "mean reward"])
    plt.ylabel("value")
    plt.title("Averages over episode")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "means_bar.png", dpi=160)

    print(f"[plots] saved to {out_dir.resolve()}")


# ============================================================
#                          CLI
# ============================================================
def main():
    ap = argparse.ArgumentParser(
        description="Single-intersection PPO vs baseline comparison"
    )
    ap.add_argument(
        "--sumocfg",
        default="environments/single_intersection/sumo_files/single-intersection-single-lane.sumocfg",
    )
    ap.add_argument("--tls-id", default="tlJ")
    ap.add_argument("--nogui", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--episode-seconds", type=float, default=300.0)
    ap.add_argument("--ticks-per-decision", type=int, default=1)
    ap.add_argument("--train-steps", type=int, default=5000)
    ap.add_argument("--baseline-green-s", type=float, default=10.0)
    ap.add_argument("--reward", default="queue")
    ap.add_argument("--norm-state", action="store_true")
    ap.add_argument("--norm-reward", action="store_true")
    ap.add_argument("--outdir", default="runs/ppo_test")
    ap.add_argument("--tensorboard", action="store_true")
    ap.add_argument("--logdir", default=".surrounding_modules_test_runs/ppo_test/tb")
    ap.add_argument("--tensorboard-port", type=int, default=6006)
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    baseline_csv = outdir / "baseline.csv"
    ppo_csv = outdir / "ppo_eval.csv"

    # 1) Train PPO in the env (env internally waits for decision readiness)
    env = SingleTLSEnv(
        sumocfg=args.sumocfg,
        tls_id=args.tls_id,
        nogui=args.nogui,
        seed=args.seed,
        episode_seconds=args.episode_seconds,
        ticks_per_decision=args.ticks_per_decision,
        reward_name=args.reward,
        normalise_reward=args.norm_reward,
        normalise_state=args.norm_state,
        timing=TLSTimingStandards(
            min_green_s=5.0, yellow_s=2.0, all_red_s=1.0, max_green_s=30.0
        ),
    )
    model = PPO("MlpPolicy", env, verbose=1, device="cpu")
    print("[train] PPO training…")
    model.learn(total_timesteps=args.train_steps)
    close_sumo()
    # 2) Baseline rollout (fixed-time), same seed & duration
    print("[eval] baseline rollout…")
    run_baseline_episode(
        sumocfg=args.sumocfg,
        tls_id=args.tls_id,
        seconds=args.episode_seconds,
        nogui=args.nogui,
        seed=args.seed,
        fixed_green_s=args.baseline_green_s,
        csv_path=baseline_csv,
    )

    # 3) PPO evaluation rollout (no learning)
    print("[eval] PPO rollout…")
    run_ppo_eval_episode(model, env, seconds=args.episode_seconds, csv_path=ppo_csv)

    # 4) Plots
    plot_compare(baseline_csv, ppo_csv, outdir)
    env.close()


if __name__ == "__main__":
    main()
