# modules/eval_only_from_bundle.py

import argparse
from pathlib import Path

import ray
from ray.rllib.algorithms.ppo import PPO

from environments.env_wrappers import make_rllib_env
from policy.independent_ppo_setup import build_trainer, TrainerParameters

# --- model registries (so we can pick the right register_fn from the bundle meta) ---
from models.masked_flat_model import register_flat_model, MODEL_NAME as FLAT_MODEL_NAME

from models.masked_gnn_attention_model import (
    register_attention_gnn_model,
    MODEL_NAME as ATTENTION_GNN_MODEL_NAME,
)

from utils.env_config_loader import load_env_config
from utils.baseline_analysis import sumo_baseline_configured_tls
from utils.trained_analysis import evaluate_trained_scenario
from utils.plot import plot_report, PlotSelection, PlotMask

from modules.model_manager import ModelManager
from modules.network_state import NetworkResults  # expects XMLs in output folder


MODEL_REGISTRY = {
    FLAT_MODEL_NAME: register_flat_model,
    ATTENTION_GNN_MODEL_NAME: register_attention_gnn_model,
}


def run_eval_only(
    bundle_run_dir: str,
    new_config_file: str,
    outdir: str,
    freeflow_speed_mps: float | None = None,
):
    outdir = Path(outdir)
    plots_dir = outdir / "plots"
    csv_dir = outdir / "csv"

    # Load a NEW env config (e.g., a longer whole-day SUMO cfg) — must match network/feature shapes.
    env_kwargs_new = load_env_config(yaml_path=new_config_file)

    # RLlib
    ray.init(ignore_reinit_error=True, include_dashboard=False)

    # Load bundle artifacts (meta + checkpoint path)
    mm = ModelManager()
    bundle_dir = Path(bundle_run_dir)
    saved_env_kwargs, model_name, custom_model_cfg, ckpt_path = (
        mm.load_bundle_artifacts(bundle_run_dir=bundle_dir)
    )

    # Pick the correct register function from the saved model_name
    if model_name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model_name in bundle: {model_name}")
    register_fn = MODEL_REGISTRY[model_name]

    # Build a fresh trainer (classic stack) with the NEW env kwargs, then restore weights
    training_model = {
        "custom_model": model_name,
        "custom_model_config": custom_model_cfg,
    }
    # eval-time trainer params (min requirements; no training will happen)
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
        ckpt_path=ckpt_path,  # from bundle
        env_kwargs=env_kwargs_new,  # NEW SUMO cfg here
        trainer_params=tp,
    )

    # Create env creator for rollout with the new env kwargs
    env_creator = make_rllib_env(env_kwargs_new)

    # Baseline rollout (SUMO fixed control)
    print("\n[eval] baseline rollout…")
    baseline_series = sumo_baseline_configured_tls(
        sumo_config=env_kwargs_new["sumo_config"],
        intersection_configs=env_kwargs_new["intersection_agent_configs"],
        feature_config=env_kwargs_new["feature_config"],
        simulation_duration=env_kwargs_new["episode_length"],
        ticks_per_decision=env_kwargs_new["ticks_per_decision"],
        log_directory=csv_dir,
    )

    # Trained rollout (restored policy)
    print("\n[eval] RLlib PPO rollout (reloaded)…")
    trained_series = evaluate_trained_scenario(
        eval_trainer,
        env_creator,
        max_steps=int(env_kwargs_new["episode_length"]),
        log_directory=csv_dir,
    )

    # Plots
    plot_report(
        baseline_series,
        trained_series,
        outdir=plots_dir,
        selection_mask=PlotSelection(
            timeseries=PlotMask(queue=True, total_wait=True, reward=True),
            means=PlotMask(queue=True, total_wait=True, reward=True),
        ),
    )

    # Network XML → CSV/KPIs (expects baseline_*.xml and eval_*.xml in csv_dir)
    print("\n[network] parsing summary/tripinfo XML and exporting KPIs/CSVs…")
    net = NetworkResults(run_dir=csv_dir)
    net.load()
    net_csv_dir = csv_dir / "network"
    net.export_all_csv(out_dir=csv_dir)
    net.export_kpis_csv(out_dir=net_csv_dir, freeflow_speed_mps=freeflow_speed_mps)

    print(f"\n[done] results in {outdir.resolve()}")
    print(f"[done] network CSVs & KPIs in {net_csv_dir.resolve()}")

    # Cleanup RLlib
    eval_trainer.stop()
    ray.shutdown()


def main():
    ap = argparse.ArgumentParser(
        description="Evaluate a saved RLlib bundle on a NEW SUMO config (baseline + trained)."
    )
    ap.add_argument(
        "--bundle-dir",
        type=str,
        required=True,
        help="Path to saved bundle run dir (contains meta.json).",
    )
    ap.add_argument(
        "--config-file",
        type=str,
        required=True,
        help="YAML for NEW evaluation SUMO config.",
    )
    ap.add_argument(
        "--outdir", type=str, required=True, help="Output directory for CSVs/plots."
    )
    ap.add_argument(
        "--freeflow-speed-mps",
        type=float,
        default=None,
        help="Optional free-flow speed for TTI KPI.",
    )
    args = ap.parse_args()

    run_eval_only(
        bundle_run_dir=args.bundle_dir,
        new_config_file=args.config_file,
        outdir=args.outdir,
        freeflow_speed_mps=args.freeflow_speed_mps,
    )


if __name__ == "__main__":
    main()
