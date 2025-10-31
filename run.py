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

from event_bus import EventBus, EventNames


def run(
    config_file: str,
    hyperparams_file: str,
    outdir: str,
    freeflow_speed_mps: float = None,
    event_bus: EventBus | None = None,
):

    outdir = Path(outdir)
    plots_dir = outdir / "plots"
    csv_dir = outdir / "csv"
    bundle_dir = outdir / "bundle"

    # Build env kwargs used by both training and eval/baseline
    (
        event_bus.emit(
            EventNames.SIMULATION_INFO.value, "Loading env config for training..."
        )
        if event_bus
        else None
    )
    env_kwargs = load_env_config(yaml_path=config_file)

    # Build hyperparameters
    (
        event_bus.emit(
            EventNames.SIMULATION_INFO.value, "Loading training hyperparameters..."
        )
        if event_bus
        else None
    )
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

    (
        event_bus.emit(
            EventNames.SIMULATION_INFO.value, "Building RLlib PPO trainer..."
        )
        if event_bus
        else None
    )
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
        event_bus=event_bus,
    )

    best_ckpt_path: Path = training_result.best_checkpoint_path

    # Save episode returns to a csv
    episode_returns: list[float] = training_result.episode_returns
    with open(outdir / "episode_returns.csv", "w") as f:
        f.write("episode_return\n")
        for r in episode_returns:
            f.write(f"{r}\n")

    # --- Save a self-contained bundle (checkpoint + meta.json) ---

    print(f"[bundle] saved to {str(best_ckpt_path)}")
    (
        event_bus.emit(
            EventNames.SIMULATION_INFO.value,
            f"Saved model bundle to {str(best_ckpt_path)}.",
        )
        if event_bus
        else None
    )

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
    (
        event_bus.emit(
            EventNames.SIMULATION_INFO.value,
            "Restored trainer from saved bundle for evaluation.",
        )
        if event_bus
        else None
    )

    # Create env creators for rollouts (same env_kwargs as training; modify here if you want a different SUMO day profile)
    env_creator = make_rllib_env(saved_env_kwargs)
    sumo_seed = saved_env_kwargs["sumo_config"].seed

    # Baseline rollout
    print(f"\n[eval] baseline rollout…")
    (
        event_bus.emit(
            EventNames.BASELINE_STARTED.value,
            f"Starting baseline rollout with seed {sumo_seed}.",
        )
        if event_bus
        else None
    )
    baseline_series = sumo_baseline_configured_tls(
        sumo_config=saved_env_kwargs["sumo_config"],
        intersection_configs=saved_env_kwargs["intersection_agent_configs"],
        feature_config=saved_env_kwargs["feature_config"],
        simulation_duration=saved_env_kwargs["episode_length"],
        ticks_per_decision=saved_env_kwargs["ticks_per_decision"],
        log_directory=csv_dir,
        event_bus=event_bus,
    )
    (
        event_bus.emit(EventNames.BASELINE_ENDED.value, "Baseline rollout completed.")
        if event_bus
        else None
    )

    # Trained rollout (from reloaded trainer)
    (
        event_bus.emit(
            EventNames.EVALUATION_STARTED.value,
            f"Starting RLlib PPO rollout with seed {sumo_seed}.",
        )
        if event_bus
        else None
    )
    print(f"\n[eval] RLlib PPO rollout (reloaded)…")
    trained_series = evaluate_trained_scenario(
        eval_trainer,
        env_creator,
        max_steps=int(saved_env_kwargs["episode_length"]),
        log_directory=csv_dir,
        event_bus=event_bus,
    )
    (
        event_bus.emit(
            EventNames.EVALUATION_ENDED.value, "RLlib PPO rollout completed."
        )
        if event_bus
        else None
    )

    # Save CSV + plots
    (
        event_bus.emit(
            EventNames.SIMULATION_INFO.value, "Generating plots from results..."
        )
        if event_bus
        else None
    )
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
    (
        event_bus.emit(
            EventNames.SIMULATION_INFO.value,
            "Exporting NetworkResults CSVs and KPIs...",
        )
        if event_bus
        else None
    )
    net = NetworkResults(run_dir=csv_dir)
    net.load()

    # Export tidy row-level CSVs and per-second aggregates
    net_csv_dir = csv_dir / "network"
    net.export_all_csv(out_dir=csv_dir)

    # Export KPI CSVs (use --freeflow-speed-mps to enable TTI computation)
    net.export_kpis_csv(out_dir=net_csv_dir, freeflow_speed_mps=freeflow_speed_mps)

    print(f"\n[done] results in {outdir.resolve()}")
    print(f"[done] network CSVs & KPIs in {net_csv_dir.resolve()}")

    (
        event_bus.emit(EventNames.SIMULATION_INFO.value, "Evaluation run completed.")
        if event_bus
        else None
    )

    # Cleanup RLlib
    eval_trainer.stop()
    ray.shutdown()

    (
        event_bus.emit(EventNames.SIMULATION_DONE.value, "Evaluation simulation done.")
        if event_bus
        else None
    )
