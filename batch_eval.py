# modules/eval_only_from_bundle.py

import argparse
import os
import pickle
from pathlib import Path

import pandas as pd
import ray
from ray.rllib.algorithms.ppo import PPO

from environments.env_wrappers import make_rllib_env
from policy.independent_ppo_setup import TrainerParameters
from utils.env_config_loader import load_env_config
from utils.baseline_analysis import sumo_baseline_configured_tls
from utils.trained_analysis import evaluate_trained_scenario

from models.masked_flat_model import register_flat_model, MODEL_NAME as FLAT_MODEL_NAME
from models.masked_gnn_attention_model import (
    register_attention_gnn_model,
    MODEL_NAME as ATTENTION_GNN_MODEL_NAME,
)

from evaluation.intersection_metric_builder import MetricsReport
from modules.model_manager import ModelManager
from modules.network_state import NetworkResults

MODEL_REGISTRY = {
    FLAT_MODEL_NAME: register_flat_model,
    ATTENTION_GNN_MODEL_NAME: register_attention_gnn_model,
}


def run_eval_only(
    bundle_run_dir: str,
    new_config_file: str,
    outdir: str,
    freeflow_speed_mps: float | None = None,
    already_evaluated: bool = False,
):
    outdir = Path(outdir)
    csv_dir = outdir / "csv"
    eval_dir = outdir / "eval"
    csv_dir.mkdir(parents=True, exist_ok=True)
    eval_dir.mkdir(parents=True, exist_ok=True)

    # Fixed evaluation seeds
    TESTING_KEYS: list[int] = [
        9046837,
        5096362,
        1220450,
        139956,
        6929871,
        4640705,
        6428608,
        1144638,
        4390983,
        4556673,
    ]
    # Fresh evaluations (optional if results already cached)
    if not already_evaluated:
        # Load NEW env config (must match obs/action shapes of the trained policy)
        env_kwargs_new = load_env_config(yaml_path=new_config_file)

        # Ray init (give raylet longer to start on Windows)
        os.environ["RAY_raylet_start_wait_time_s"] = "180"
        ray.init(
            ignore_reinit_error=True,
            include_dashboard=False,
            _system_config={"raylet_start_wait_time_s": 180},
        )

        # Load bundle: env kwargs (for reference), model id, custom model cfg, checkpoint
        mm = ModelManager()
        bundle_dir = Path(bundle_run_dir)
        _, model_name, custom_model_cfg, ckpt_path = mm.load_bundle_artifacts(
            bundle_run_dir=bundle_dir
        )

        if model_name not in MODEL_REGISTRY:
            raise ValueError(f"Unknown model_name in bundle: {model_name}")
        register_fn = MODEL_REGISTRY[model_name]

        for key in TESTING_KEYS:
            print(f"\nEVALUATION SEED: {key}\n")
            env_kwargs_new["sumo_config"].seed = key
            seed_dir = csv_dir / f"seed_{key}"
            seed_dir.mkdir(parents=True, exist_ok=True)

            # Minimal trainer params (we only roll out; no gradient steps)
            tp = TrainerParameters(
                num_workers=0,
                rollout_fragment_length=env_kwargs_new["episode_length"],
                train_batch_size=2 * env_kwargs_new["episode_length"],
                minibatch_size=512,
                num_epochs=1,
                lr=3e-4,
            )

            eval_trainer: PPO = mm.build_and_restore_from_artifacts(
                register_fn=register_fn,
                model_name=model_name,
                custom_model_config=custom_model_cfg,
                ckpt_path=ckpt_path,
                env_kwargs=env_kwargs_new,
                trainer_params=tp,
            )

            env_creator = make_rllib_env(env_kwargs_new)

            # Baseline (fixed control)
            print("\n[eval] baseline rollout…")
            baseline_series = sumo_baseline_configured_tls(
                sumo_config=env_kwargs_new["sumo_config"],
                intersection_configs=env_kwargs_new["intersection_agent_configs"],
                feature_config=env_kwargs_new["feature_config"],
                simulation_duration=env_kwargs_new["episode_length"],
                ticks_per_decision=env_kwargs_new["ticks_per_decision"],
                log_directory=seed_dir,
            )

            # Trained policy rollout
            print("\n[eval] RLlib PPO rollout (reloaded)…")
            trained_series = evaluate_trained_scenario(
                eval_trainer,
                env_creator,
                max_steps=int(env_kwargs_new["episode_length"]),
                log_directory=seed_dir,
            )

            # Per-intersection report
            report = MetricsReport.build_report(baseline_series, trained_series)
            imp_df = report.improvement_summary.copy()
            imp_df.index.name = "agent"  # ensure stable name for aggregation

            # Network KPIs from XMLs written to this seed's folder
            net = NetworkResults(run_dir=seed_dir)  # <- per-seed directory
            net.load()
            kpi_df = net.kpis_comparison_df().copy()
            kpi_df.index.name = "kpi"

            # Save consolidated pickle per seed
            payload = {
                "key": key,
                "intersection_improvement": imp_df,
                "network_comparison": kpi_df,
            }
            with (outdir / f"improvement_summary_{key}.pkl").open("wb") as f:
                pickle.dump(payload, f)

            eval_trainer.stop()

    # Reload all per-seed pickles and stack with MultiIndex rows
    imp_frames: list[pd.DataFrame] = []
    net_frames: list[pd.DataFrame] = []
    for key in TESTING_KEYS:
        with (outdir / f"improvement_summary_{key}.pkl").open("rb") as f:
            data = pickle.load(f)

        imp = data["intersection_improvement"].copy()
        imp.index.name = "agent"
        imp["seed"] = key
        imp = (
            imp.set_index("seed", append=True)
            .reorder_levels(["seed", "agent"])
            .sort_index()
        )
        imp_frames.append(imp)

        net = data["network_comparison"].copy()
        net.index.name = "kpi"
        net["seed"] = key
        net = (
            net.set_index("seed", append=True)
            .reorder_levels(["seed", "kpi"])
            .sort_index()
        )
        net_frames.append(net)

    # Aggregate across seeds
    imp_all = pd.concat(imp_frames)  # index: ('seed','agent')
    imp_mean = imp_all.groupby(level="agent").mean(numeric_only=True)
    imp_std = imp_all.groupby(level="agent").std(ddof=0, numeric_only=True)
    imp_mean.to_csv(eval_dir / "intersection_improvement_mean.csv")
    imp_std.to_csv(eval_dir / "intersection_improvement_std.csv")

    net_all = pd.concat(net_frames)  # index: ('seed','kpi')
    net_mean = net_all.groupby(level="kpi").mean(numeric_only=True)
    net_std = net_all.groupby(level="kpi").std(ddof=0, numeric_only=True)
    net_mean.to_csv(eval_dir / "network_kpis_mean.csv")
    net_std.to_csv(eval_dir / "network_kpis_std.csv")

    # Build one single summary table
    # Extract AVERAGE row fromk imp_std and imp_mean
    imp_mean_avg = imp_mean.xs("AVERAGE")
    imp_std_avg = imp_std.xs("AVERAGE")
    # Convert to dataframe
    imp_mean_avg = pd.DataFrame(imp_mean_avg)
    imp_std_avg = pd.DataFrame(imp_std_avg)

    # Rename rows to indicate source
    QUEUE_NAME = "intersection_meanQueueLength"
    TOTAL_WAIT_NAME = "intersection_totalWaitingTime"
    MAX_WAIT_NAME = "intersection_maxWaitingTime"
    AVERAGE_NAME = "intersection_average"

    imp_mean_avg = imp_mean_avg.rename(
        index={
            "queue": QUEUE_NAME,
            "total_wait": TOTAL_WAIT_NAME,
            "max_wait": MAX_WAIT_NAME,
            "AVERAGE": AVERAGE_NAME,
        },
        columns={
            "AVERAGE": "mean",
        },
    )

    imp_std_avg = imp_std_avg.rename(
        index={
            "queue": QUEUE_NAME,
            "total_wait": TOTAL_WAIT_NAME,
            "max_wait": MAX_WAIT_NAME,
            "AVERAGE": AVERAGE_NAME,
        },
        columns={
            "AVERAGE": "std",
        },
    )

    # Isolate pct_imporvement column with kpi in net_mean and net_std
    net_mean = net_mean[["pct_improvement"]].rename(columns={"pct_improvement": "mean"})
    net_std = net_std[["pct_improvement"]].rename(columns={"pct_improvement": "std"})

    # Combine all mean/std tables
    summary_mean = pd.concat([imp_mean_avg, net_mean])
    summary_std = pd.concat([imp_std_avg, net_std])

    summary = pd.concat(
        [summary_mean, summary_std.rename(columns={"mean": "std"})], axis=1
    )

    # Save to csv
    summary.to_csv(eval_dir / "overall_summary.csv")

    if not already_evaluated:
        ray.shutdown()


def main():
    ap = argparse.ArgumentParser(
        description="Evaluate a saved RLlib bundle on a NEW SUMO config (baseline + trained)."
    )
    ap.add_argument(
        "--config-file",
        type=str,
        required=True,
        help="YAML for NEW evaluation SUMO config.",
    )
    ap.add_argument(
        "--outdir",
        type=str,
        required=True,
        help="Output directory for CSVs/aggregates.",
    )
    ap.add_argument(
        "--freeflow-speed-mps",
        type=float,
        default=None,
        help="Optional free-flow speed for TTI KPI.",
    )
    ap.add_argument(
        "--already-evaluated",
        action="store_true",
        default=False,
        help="Skip rollouts and only aggregate existing per-seed pickles.",
    )
    args = ap.parse_args()

    # List of bundles to evaluate; loop if you want more than one.
    BUNDLE_DIRS = [
        ".grid_search/HPC1/bundle",
    ]

    RUN_NAMES = [
        "HPC1",
    ]

    for bundle_dir in BUNDLE_DIRS:
        outdir = Path(args.outdir) / Path(RUN_NAMES[BUNDLE_DIRS.index(bundle_dir)])
        run_eval_only(
            bundle_run_dir=bundle_dir,
            new_config_file=args.config_file,
            outdir=outdir,
            freeflow_speed_mps=args.freeflow_speed_mps,
            already_evaluated=args.already_evaluated,
        )


if __name__ == "__main__":
    main()
