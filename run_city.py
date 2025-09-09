# scripts/run_city.py
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, Tuple

import yaml
import numpy as np
import traci

from utils.sumo_helpers import start_sumo, close_sumo                                      # sumo start/stop

from environments.ingolstadt.intersections.builder import build_envs_from_yaml                    # your builder
from proto_modules.communication_bus import CommunicationBus
from proto_modules.intersection.env_adapter import IntersectionEnv
from proto_modules.intersection.state import build_matrices
from proto_modules.intersection.policy import SB3PolicyModule


# --------------------- uplift utils ---------------------
def _mean_col_csv(path: Path, col: str) -> float:
    import csv, math
    xs = []
    with open(path) as f:
        r = csv.DictReader(f)
        for row in r:
            v = row.get(col)
            if v not in (None, ""):
                xs.append(float(v))
    return float(np.mean(xs)) if xs else float("nan")

def _compute_uplift(run_dir: Path, metrics=("halt", "count", "speed")) -> Dict[str, Dict[str, float]]:
    base_dir = run_dir / "baseline"
    eval_dir = run_dir / "eval"
    out: Dict[str, Dict[str, float]] = {}

    # discover node ids from csvs
    import glob, os
    base_csvs = glob.glob(str(base_dir / "baseline_*.csv"))
    eval_csvs = glob.glob(str(eval_dir / "eval_*.csv"))
    node_ids = sorted({os.path.basename(p).split("_", 1)[1].split(".")[0] for p in base_csvs + eval_csvs})

    for nid in node_ids:
        node_stats: Dict[str, float] = {}
        for m in metrics:
            b = _mean_col_csv(base_dir / f"baseline_{nid}.csv", m) if (base_dir / f"baseline_{nid}.csv").exists() else float("nan")
            e = _mean_col_csv(eval_dir / f"eval_{nid}.csv", m) if (eval_dir / f"eval_{nid}.csv").exists() else float("nan")
            if np.isfinite(b) and np.isfinite(e) and b != 0.0:
                node_stats[f"{m}_baseline"] = b
                node_stats[f"{m}_eval"] = e
                node_stats[f"{m}_uplift_pct"] = (e - b) / abs(b) * 100.0
            else:
                node_stats[f"{m}_baseline"] = b
                node_stats[f"{m}_eval"] = e
                node_stats[f"{m}_uplift_pct"] = float("nan")
        out[nid] = node_stats
    return out

def _print_uplift(uplift: Dict[str, Dict[str, float]]) -> None:
    def fmt(x): 
        return "n/a" if not np.isfinite(x) else f"{x:8.3f}"
    for nid in sorted(uplift):
        s = uplift[nid]
        line = [
            f"{nid:>8s}",
            f"halt {fmt(s['halt_baseline'])} -> {fmt(s['halt_eval'])}  ({fmt(s['halt_uplift_pct'])}%)",
            f"count {fmt(s['count_baseline'])} -> {fmt(s['count_eval'])}  ({fmt(s['count_uplift_pct'])}%)",
            f"speed {fmt(s['speed_baseline'])} -> {fmt(s['speed_eval'])}  ({fmt(s['speed_uplift_pct'])}%)",
        ]
        print(" | ".join(line))


# --------------------- plotting utils ---------------------


def _save_node_csv(ts: Dict[str, Dict[str, list]], out_dir: Path, prefix: str) -> None:
    import csv
    out_dir.mkdir(parents=True, exist_ok=True)
    for nid, d in ts.items():
        p = out_dir / f"{prefix}_{nid}.csv"
        cols = ["t", "halt", "count", "speed"]
        # include reward if present
        if "reward" in d:
            cols.append("reward")
        with open(p, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(cols)
            L = max(len(d[k]) for k in cols if k in d)
            for i in range(L):
                row = []
                for c in cols:
                    v = d.get(c, [])
                    row.append("" if i >= len(v) else float(v[i]))
                w.writerow(row)


def _plot_timeseries(ts: Dict[str, Dict[str, list]], out_dir: Path, prefix: str, show: bool) -> None:

    import matplotlib.pyplot as plt

    out_dir.mkdir(parents=True, exist_ok=True)

    # Per-node 3-panel plot
    for nid, d in ts.items():
        t = np.asarray(d["t"])
        fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
        axes[0].plot(t, d["halt"]);  axes[0].set_ylabel("halt")
        axes[1].plot(t, d["count"]); axes[1].set_ylabel("count")
        axes[2].plot(t, d["speed"]); axes[2].set_ylabel("speed"); axes[2].set_xlabel("time [s]")
        fig.suptitle(f"{prefix} — {nid}")
        fig.tight_layout()
        fig.savefig(out_dir / f"{prefix}_{nid}.png", dpi=150)
        plt.close(fig)

    # Multi-node overlays
    metrics = ["halt", "count", "speed"]
    for m in metrics:
        fig, ax = plt.subplots(figsize=(10, 4))
        for nid, d in ts.items():
            ax.plot(d["t"], d[m], label=nid)
        ax.set_title(f"{prefix} — {m}")
        ax.set_xlabel("time [s]"); ax.set_ylabel(m)
        ax.legend(ncol=max(1, len(ts)//3))
        fig.tight_layout()
        fig.savefig(out_dir / f"{prefix}_overlay_{m}.png", dpi=150)
        if show:
            plt.show()
        else:
            plt.close(fig)

    # Rewards overlays if present
    has_reward = any("reward" in d for d in ts.values())
    if has_reward:
        fig, ax = plt.subplots(figsize=(10, 4))
        for nid, d in ts.items():
            if "reward" in d and d["reward"]:
                ax.plot(d["t"][:len(d["reward"])], d["reward"], label=f"{nid} reward")
        ax.set_title(f"{prefix} — reward")
        ax.set_xlabel("time [s]"); ax.set_ylabel("reward")
        ax.legend(ncol=max(1, len(ts)//3))
        fig.tight_layout()
        fig.savefig(out_dir / f"{prefix}_overlay_reward.png", dpi=150)
        if show:
            import matplotlib.pyplot as plt
            plt.show()
        else:
            plt.close(fig)



# --------------------- optional Maskable wrapper ---------------------

def maybe_mask_env(env, agent):
    try:
        from sb3_contrib.common.wrappers.action_masker import ActionMasker
    except Exception:
        return env

    def mask_fn(e):
        mask = agent.action.compute_mask()  # None or np.ndarray(bool, shape=(A,))
        if mask is None:
            return np.ones(e.action_space.n, dtype=bool)
        mask = np.asarray(mask, dtype=bool).ravel()
        # sanity: match action space size
        if mask.size != e.action_space.n:
            # pick your policy: raise, or pad/truncate. I’d rather fail fast:
            raise ValueError(f"Action mask length {mask.size} != action space {e.action_space.n}")
        return mask

    return ActionMasker(env, mask_fn)


# --------------------- Baseline (fixed TLS) ---------------------

def run_baseline(sumocfg: str, nodes_yaml: str, nodes: Iterable[str]) -> Tuple[Dict[str, Dict[str, float]], Dict[str, Dict[str, list]]]:
    start_sumo(sumocfg, gui=True)
    bus = CommunicationBus(retain_last=True)
    print("Building baseline env from YAML")
    env_map = build_envs_from_yaml(nodes_yaml, bus)
    agents = {nid: env_map[nid][1] for nid in nodes}

    # time-series
    ts = {nid: dict(t=[], halt=[], count=[], speed=[]) for nid in nodes}

    t_now = float(traci.simulation.getTime()) # type: ignore
    print("Starting baseline loop")
    while traci.simulation.getMinExpectedNumber() > 0 and t_now < 64000: # type: ignore
        t_now = float(traci.simulation.getTime()) # type: ignore
        for nid, agent in agents.items():
            Din, _ = build_matrices(agent, agent.detector_reader)
            if Din.numel():
                ts[nid]["t"].append(t_now)
                ts[nid]["halt"].append(float(Din[:, 0].sum().item()))
                ts[nid]["count"].append(float(Din[:, 1].sum().item()))
                ts[nid]["speed"].append(float(Din[:, 2].mean().item()))
        traci.simulationStep()

    close_sumo()
    # summary
    summary = {
        nid: dict(
            mean_halt=float(np.mean(ts[nid]["halt"])) if ts[nid]["halt"] else 0.0,
            mean_count=float(np.mean(ts[nid]["count"])) if ts[nid]["count"] else 0.0,
            mean_speed=float(np.mean(ts[nid]["speed"])) if ts[nid]["speed"] else 0.0,
        )
        for nid in nodes
    }
    return summary, ts


# --------------------- Train per-agent ---------------------

def train_agents(sumocfg: str, nodes_yaml: str, nodes: Iterable[str],
                 total_timesteps: int, ticks_per_decision: int, out_dir: Path,
                 *, tensorboard: bool = False, progress: bool = False) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    tb_root = out_dir / "tb" if tensorboard else None

    for nid in nodes:
        start_sumo(sumocfg, gui=False)
        bus = CommunicationBus(retain_last=True)
        env_map = build_envs_from_yaml(nodes_yaml, bus)
        env, agent, policy = env_map[nid]

        env = IntersectionEnv(agent, ticks_per_decision=ticks_per_decision)
        env = maybe_mask_env(env, agent)
        policy.set_env(env)

        # attach SB3 logger with tensorboard if requested
        tb_name = None
        if tensorboard:
            from stable_baselines3.common.logger import configure
            logdir = (tb_root / nid) # type: ignore
            logdir.mkdir(parents=True, exist_ok=True)
            new_logger = configure(str(logdir), ["stdout", "csv", "tensorboard"])
            policy.model.set_logger(new_logger) # type: ignore
            tb_name = nid  # run name in TensorBoard

        policy.learn(total_timesteps=total_timesteps, tb_log_name=tb_name, progress_bar=progress)
        policy._save(str(out_dir / f"{nid}_sb3.zip"))

        _dump_history_from_agent(agent, out_dir / f"{nid}_train_history.csv")
        close_sumo()


def _dump_history_from_agent(agent, path: Path) -> None:
    import csv
    m = agent.metrics_snapshot()
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["t", "action", "phase", "reward", "loss"])
        L = max(len(m["t"]), len(m["action"]), len(m["phase"]), len(m["reward"]), len(m["loss"]))
        for i in range(L):
            w.writerow([
                float(m["t"][i]) if i < len(m["t"]) else "",
                float(m["action"][i]) if i < len(m["action"]) else "",
                float(m["phase"][i]) if i < len(m["phase"]) else "",
                float(m["reward"][i]) if i < len(m["reward"]) else "",
                float(m["loss"][i]) if i < len(m["loss"]) else "",
            ])


# --------------------- Evaluate trained jointly ---------------------

def evaluate_agents(sumocfg: str, nodes_yaml: str, nodes: Iterable[str],
                    ticks_per_decision: int, seconds: float, ckpt_dir: Path) -> Tuple[Dict[str, Dict[str, float]], Dict[str, Dict[str, list]]]:
    start_sumo(sumocfg, gui=True)
    bus = CommunicationBus(retain_last=True)
    env_map = build_envs_from_yaml(nodes_yaml, bus)

    # temp ignore N2 and N10
    # nodes = [nid for nid in nodes if nid not in ("N2", "N10")]

    # Load checkpoints if present, and attach fresh env adapters
    for nid in nodes:
        env, agent, policy = env_map[nid]
        ckpt = ckpt_dir / f"{nid}_sb3.zip"
        if ckpt.is_file():
            policy = SB3PolicyModule._load(
                path=str(ckpt),
                env=env,
                algo_name=policy._algo_name,
                extractor_kwargs=policy._extractor_kwargs,
                net_arch=policy._net_arch,
                algo_kwargs=policy._algo_kwargs,
                device=policy._device,
                verbose=0,
            )
        env = IntersectionEnv(agent, ticks_per_decision=ticks_per_decision)
        env = maybe_mask_env(env, agent)
        policy.set_env(env)
        env_map[nid] = (env, agent, policy) # type: ignore

    # time-series
    ts = {nid: dict(t=[], halt=[], count=[], speed=[], reward=[]) for nid in nodes}

    t0 = float(traci.simulation.getTime()) # type: ignore
    while (float(traci.simulation.getTime()) - t0) < seconds and traci.simulation.getMinExpectedNumber() > 0: # type: ignore
        # one decision for each agent
        for nid in nodes:
            out = env_map[nid][1].step(deterministic=True)  # agent.step()


        # advance remaining ticks in this decision interval
        for _ in range(max(1, ticks_per_decision)):
            traci.simulationStep()

            # drive transitional overrides for all agents
            for nid in nodes:
                env_map[nid][1].action.tick()

        # sample metrics at decision boundary
        t_now = float(traci.simulation.getTime()) # type: ignore
        for nid in nodes:
            agent = env_map[nid][1]
            Din, _ = build_matrices(agent, agent.detector_reader)
            if Din.numel():
                ts[nid]["t"].append(t_now)
                ts[nid]["halt"].append(float(Din[:, 0].sum().item()))
                ts[nid]["count"].append(float(Din[:, 1].sum().item()))
                ts[nid]["speed"].append(float(Din[:, 2].mean().item()))
                # reward trace (last added reward in agent.history)
                if agent.history["reward"]:
                    ts[nid]["reward"].append(float(agent.history["reward"][-1]))

    # summary at the end (mean over sampled points)
    summary = {
        nid: dict(
            mean_halt=float(np.mean(ts[nid]["halt"])) if ts[nid]["halt"] else 0.0,
            mean_count=float(np.mean(ts[nid]["count"])) if ts[nid]["count"] else 0.0,
            mean_speed=float(np.mean(ts[nid]["speed"])) if ts[nid]["speed"] else 0.0,
        )
        for nid in nodes
    }

    close_sumo()
    return summary, ts


# --------------------- CLI ---------------------

def main():
    ap = argparse.ArgumentParser(description="Baseline, train and evaluate with time-series and plots")
    ap.add_argument("--sumocfg", required=True, help="Path to .sumocfg")
    ap.add_argument("--yaml", required=True, help="Path to nodes.yaml")
    ap.add_argument("--nodes", nargs="*", default=None, help="Subset of node_ids (default: all in yaml)")
    ap.add_argument("--mode", choices=["baseline", "train", "eval", "all"], default="all")
    ap.add_argument("--seconds", type=float, default=1800.0, help="Eval duration in seconds")
    ap.add_argument("--ticks-per-decision", type=int, default=5, help="SUMO ticks per RL decision")
    ap.add_argument("--timesteps", type=int, default=100_000, help="SB3 total_timesteps per agent")
    ap.add_argument("--out", default="runs", help="Output directory")
    ap.add_argument("--show", action="store_true", help="Display plots at the end")
    ap.add_argument("--tensorboard", action="store_true", help="Log SB3 to OUT/tb for TensorBoard")
    ap.add_argument("--progress", action="store_true", help="Show SB3 learn() progress bar")
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)


    with open(args.yaml, "r") as f:
        y = yaml.safe_load(f)
    run_nodes = [d["node_id"] for d in y.get("intersections", [])]

    if args.mode in ("baseline", "all"):
        print("[baseline] running fixed TLS...")
        summary, ts = run_baseline(args.sumocfg, args.yaml, run_nodes)
        (out_dir / "baseline").mkdir(parents=True, exist_ok=True)
        (out_dir / "baseline" / "summary.json").write_text(json.dumps(summary, indent=2))
        _save_node_csv(ts, out_dir / "baseline", "baseline")
        _plot_timeseries(ts, out_dir / "baseline", "baseline", show=args.show)
        print("[baseline] saved to", out_dir / "baseline")

    if args.mode in ("train", "all"):
        print("[train] training agents:", ", ".join(run_nodes))
        train_agents(
            args.sumocfg, args.yaml, run_nodes, args.timesteps, args.ticks_per_decision, out_dir / "ckpts",
            tensorboard=args.tensorboard, progress=args.progress
        )


    if args.mode in ("eval", "all"):
        print("[eval] evaluating trained agents jointly...")
        summary, ts = evaluate_agents(args.sumocfg, args.yaml, run_nodes, args.ticks_per_decision, args.seconds, out_dir / "ckpts")
        (out_dir / "eval").mkdir(parents=True, exist_ok=True)
        (out_dir / "eval" / "summary.json").write_text(json.dumps(summary, indent=2))
        _save_node_csv(ts, out_dir / "eval", "eval")
        _plot_timeseries(ts, out_dir / "eval", "eval", show=args.show)
        print("[eval] saved to", out_dir / "eval")

        # quick uplift vs baseline
        uplift = _compute_uplift(out_dir)
        _print_uplift(uplift)
        (out_dir / "uplift.json").write_text(json.dumps(uplift, indent=2))


if __name__ == "__main__":
    main()
