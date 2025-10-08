import argparse
from pathlib import Path

import ray
from ray.rllib.algorithms.ppo import PPO

from environments.env_wrappers import make_rllib_env
from policy.independent_ppo_setup import build_trainer, TrainerParameters

from utils.training import (
    train_until_converged,
    TrainingHyperparameters,
    TrainingResult,
)

from models.masked_gnn_attention_model import (
    register_attention_gnn_model,
    MODEL_NAME as ATTENTION_GNN_MODEL_NAME,
)

from utils.env_config_loader import load_env_config
from utils.hyperparams_loader import load_hyperparameters
from utils.baseline_analysis import sumo_baseline_configured_tls
from utils.trained_analysis import evaluate_trained_scenario

from utils.plot import plot_report, PlotSelection, PlotMask
from utils.load_seed_lists import load_int_list as load_seeds

from modules.model_manager import ModelManager
from modules.network_state import NetworkResults


def run(
    config_file: str,
    hyperparams_file: str,
    outdir: str,
    freeflow_speed_mps: float = None,
):

    outdir = Path(outdir)
    plots_dir = outdir / "plots"
    csv_dir = outdir / "csv"
    bundle_dir = outdir / "bundle"

    # Build env kwargs used by both training and eval/baseline
    env_kwargs = load_env_config(yaml_path=config_file)

    # Build hyperparameters
    hyperparams = load_hyperparameters(yaml_path=hyperparams_file)

    # Load training seeds
    TRAINING_SEEDS_PATH: str = "environments/training_seeds.txt"
    training_seeds: list[int] = load_seeds(TRAINING_SEEDS_PATH)

    env_kwargs["training_seeds"] = training_seeds

    # RLlib trainer
    ray.init(ignore_reinit_error=True, include_dashboard=False)

    # small attention per node + GNN
    register_fn = register_attention_gnn_model
    training_model = {
        "custom_model": ATTENTION_GNN_MODEL_NAME,
        "custom_model_config": hyperparams.get("model_params"),
    }

    print("[train] RLlib PPO training…")
    trainer: PPO = build_trainer(
        env_kwargs=env_kwargs,
        register_fn=register_fn,
        training_model=training_model,
        trainer_params=hyperparams["trainer_params"],
    )

    model_manager = ModelManager()
    bundle_dir.mkdir(parents=True, exist_ok=True)

    training_result: TrainingResult = train_until_converged(
        trainer,
        parameters=hyperparams["training_params"],
        ckpt_dir=bundle_dir,
        training_model=training_model,
        env_kwargs=env_kwargs,
        model_manager=model_manager,
        verbose=True,
    )

    best_ckpt_path: Path = training_result.best_checkpoint_path

    # --- Save a self-contained bundle (checkpoint + meta.json) ---

    print(f"[bundle] saved to {str(best_ckpt_path)}")

    # --- Fully reload a fresh trainer from the saved bundle for evaluation ---
    trainer.stop()
    saved_env_kwargs, model_name, custom_model_cfg, ckpt_path = (
        model_manager.load_bundle_artifacts(bundle_run_dir=best_ckpt_path)
    )

    eval_trainer = model_manager.build_and_restore_from_artifacts(
        register_fn=register_fn,
        model_name=model_name,
        custom_model_config=custom_model_cfg,
        ckpt_path=ckpt_path,
        env_kwargs=saved_env_kwargs,
        trainer_params=hyperparams["trainer_params"],
    )

    print("[reload] restored trainer from bundle")

    # Create env creators for rollouts (same env_kwargs as training; modify here if you want a different SUMO day profile)
    env_creator = make_rllib_env(saved_env_kwargs)

    # Baseline rollout
    print(f"\n[eval] baseline rollout…")
    baseline_series = sumo_baseline_configured_tls(
        sumo_config=saved_env_kwargs["sumo_config"],
        intersection_configs=saved_env_kwargs["intersection_agent_configs"],
        feature_config=saved_env_kwargs["feature_config"],
        simulation_duration=saved_env_kwargs["episode_length"],
        ticks_per_decision=saved_env_kwargs["ticks_per_decision"],
        log_directory=csv_dir,
    )

    # Trained rollout (from reloaded trainer)
    print(f"\n[eval] RLlib PPO rollout (reloaded)…")
    trained_series = evaluate_trained_scenario(
        eval_trainer,
        env_creator,
        max_steps=int(saved_env_kwargs["episode_length"]),
        log_directory=csv_dir,
    )

    # Save CSV + plots
    plot_report(
        baseline_series,
        trained_series,
        outdir=plots_dir,
        selection_mask=PlotSelection(
            timeseries=PlotMask(queue=True, total_wait=True, reward=True),
            means=PlotMask(queue=True, total_wait=True, reward=True),
        ),
    )

    # -----------------------------
    # NetworkResults integration
    # -----------------------------
    # Expecting the four XML files (baseline_summary.xml, baseline_tripinfo.xml,
    # eval_summary.xml, eval_tripinfo.xml) in `outdir`.
    print("\n[network] parsing summary/tripinfo XML and exporting KPIs/CSVs…")
    net = NetworkResults(run_dir=csv_dir)
    net.load()

    # Export tidy row-level CSVs and per-second aggregates
    net_csv_dir = csv_dir / "network"
    net.export_all_csv(out_dir=csv_dir)

    # Export KPI CSVs (use --freeflow-speed-mps to enable TTI computation)
    net.export_kpis_csv(out_dir=net_csv_dir, freeflow_speed_mps=freeflow_speed_mps)

    print(f"\n[done] results in {outdir.resolve()}")
    print(f"[done] network CSVs & KPIs in {net_csv_dir.resolve()}")
