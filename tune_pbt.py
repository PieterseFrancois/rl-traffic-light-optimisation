# tune_pbt.py
import os
from pathlib import Path
import ray
from ray import tune
from ray.air import RunConfig, CheckpointConfig
from ray.tune import Tuner, TuneConfig, FailureConfig
from ray.tune.schedulers import PopulationBasedTraining
from ray.rllib.algorithms.ppo import PPO
from ray.rllib.algorithms.ppo import PPOConfig

from utils.env_config_loader import load_env_config
from utils.hyperparams_loader import load_hyperparameters
from policy.independent_ppo_setup import build_independent_ppo_config, TrainerParameters
from modules.model_manager import ModelManager

from models.masked_gnn_attention_model import (
    register_attention_gnn_model,
    MODEL_NAME as ATTN_GNN_NAME,
)


ray.shutdown()


def build_base_config(env_kwargs: dict, trainer_params: TrainerParameters, model_cfg: dict) -> PPOConfig:
    # Keep your existing builder, but no samples here. We’ll inject Tune samplers via algo_overrides.
    return build_independent_ppo_config(
        env_kwargs=env_kwargs,
        register_fn=register_attention_gnn_model,
        training_model={"custom_model": ATTN_GNN_NAME, "custom_model_config": model_cfg},
        trainer_params=trainer_params,
    )

def main(config_file: str, hyperparams_file: str, outdir: str):
    ray.init(ignore_reinit_error=True, include_dashboard=False)

    env_kwargs = load_env_config(yaml_path=config_file)
    # TRAIN_SEEDS = [11, 23, 37, 59]
    # EVAL_SEEDS = [71, 83]
    TRAIN_SEEDS = [42]
    EVAL_SEEDS = [42]
    env_kwargs["train_seeds"] = TRAIN_SEEDS
    env_kwargs["eval_seeds"] = EVAL_SEEDS
    env_kwargs["mode"] = "train"

    # EEnsure env_kwargs["sumo_config"].sumocfg_filepath is changed to absolute path - currently relative to base dir
    sumocfg_path = Path(env_kwargs["sumo_config"].sumocfg_filepath)
    if not sumocfg_path.is_absolute():
        env_kwargs["sumo_config"].sumocfg_filepath = str((Path(sumocfg_path).resolve()))

    hps = load_hyperparameters(yaml_path=hyperparams_file)

    base_cfg = build_base_config(
        env_kwargs=env_kwargs,
        trainer_params=hps["trainer_params"],
        model_cfg=hps["model_params"],
    )

    base_cfg = (
        base_cfg
        .resources(num_cpus_for_main_process=1, num_gpus=0) 
        .env_runners(
            num_env_runners=4,            # was 0 (local only)
            num_envs_per_env_runner=2,    # vectorise inside each runner
            num_cpus_per_env_runner=1,    # 4*2 envs ~ 8 CPUs per trial
            num_gpus_per_env_runner=0,
            gym_env_vectorize_mode="SYNC",
            rollout_fragment_length='auto',  # auto-adjust to env steps
        )
        .evaluation(
        evaluation_interval=1,                    # eval every iter
        evaluation_duration=4,                    # episodes
        evaluation_duration_unit="episodes",
        evaluation_parallel_to_training=False,
        evaluation_config={
            "env_config": {
                "env_kwargs": {
                    **env_kwargs,
                    "mode": "eval",               # <- switch to eval mode
                }
            }
        }
    )
    )

    # ---- PBT scheduler (modern Tune) ----
    METRIC = "evaluation/env_runners/episode_reward_mean"
    pbt = PopulationBasedTraining(
        time_attr="training_iteration",
        perturbation_interval=3,     # iterations between exploit/explore
        metric=METRIC,
        mode="max",
        hyperparam_mutations={
            # Safe, PPO-relevant mutations (global for all independent policies)
            "lr": tune.loguniform(5e-6, 5e-4),
            "gamma": tune.choice([0.95, 0.97, 0.99]),
            "minibatch_size": tune.qrandint(64, 1024, 64),
            "train_batch_size": tune.qrandint(2000, 20000, 1000),
            # "rollout_fragment_length": tune.qrandint(100, 1000, 100),
            "num_epochs": tune.randint(2, 8),
            "clip_param": tune.uniform(0.1, 0.4),
            "entropy_coeff": tune.uniform(0.0, 0.02),
        },
        # Optional: keep some mutations consistent ratios
        synch=False,
        quantile_fraction=0.25,   
    )

    # ---- Param space: use PPOConfig().to_dict() + selective samples ----
    # Start from your base config and let Tune mutate the keys above.
    param_space = base_cfg.to_dict()
    # Put initial sample points to seed the population (PBT will kick off from here):
    param_space.update({
        "lr": tune.loguniform(1e-5, 3e-4),
        "gamma": tune.choice([0.97, 0.99]),
        "minibatch_size": tune.qrandint(128, 512, 64),
        "train_batch_size": tune.qrandint(4000, 16000, 1000),
        # "rollout_fragment_length": tune.qrandint(200, 800, 100),
        "num_epochs": tune.randint(2, 6),
        "clip_param": tune.uniform(0.15, 0.35),
        "entropy_coeff": tune.uniform(0.0, 0.01),
    })

    storage_path = Path(outdir) / "tune"
    storage_path.mkdir(parents=True, exist_ok=True)

    storage_path_uri = storage_path.resolve().as_uri()

    # Resources: reuse RLlib workers per trial; Tune will clone trials for PBT.
    trainable = PPO

    tuner = Tuner(
        trainable,
        param_space=param_space,
        run_config=RunConfig(
            name="ppo_pbt",
            storage_path=str(storage_path_uri),
            checkpoint_config=CheckpointConfig(
                checkpoint_at_end=True,
                checkpoint_frequency=1,
                num_to_keep=10,
            ),
            failure_config=FailureConfig(max_failures=5),
            stop={"training_iteration": hps["training_params"].max_iterations},
        ),
        tune_config=TuneConfig(
            num_samples=12,          # population size
            scheduler=pbt,
            reuse_actors=False,     # safer with PBT exploit/explore
            max_concurrent_trials=4,
            trial_name_creator=lambda t: f"{t.trainable_name}_{t.trial_id[:5]}",
            trial_dirname_creator=lambda t: t.trial_id,   # shortest, unique
        ),
    )

    results = tuner.fit()
    print("Best result:", results.get_best_result(METRIC, "max").metrics)

    best = results.get_best_result(METRIC, "max")

    # Restore the trained PPO from the best checkpoint
    best_ckpt = best.checkpoint
    ckpt_dir = best_ckpt.to_directory()
    algo = PPO.from_checkpoint(ckpt_dir)

    # Bundle to the same outdir you passed in
    bundle_root = Path(outdir) / "pbt_bundle"
    bundle_root.mkdir(parents=True, exist_ok=True)

    # Use the same model name/config and env kwargs you trained with
    training_model = {"custom_model": ATTN_GNN_NAME, "custom_model_config": hps["model_params"]}

    TUNED_KEYS = ['lr', 'gamma', 'minibatch_size', 'train_batch_size', 'num_epochs', 'clip_param', 'entropy_coeff']

    tuned_hps = {k: best.config.get(k, None) for k in TUNED_KEYS}

    mm = ModelManager()
    bundle_path = mm.save_bundle(
        trainer=algo,
        bundle_dir_rel=bundle_root,
        model_name=training_model["custom_model"],
        custom_model_config=training_model["custom_model_config"],
        env_kwargs=env_kwargs,   # already resolved (absolute) earlier
        notes={"metric": METRIC, "metric_value": float(best.metrics.get(METRIC, float("nan"))),
            "config": tuned_hps},   # keeps tuned hypers for traceability
    )

    print(f"[tune-pbt] bundle saved to {str(bundle_path)}")

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--hyper", required=True)
    ap.add_argument("--outdir", required=True)
    args = ap.parse_args()
    main(args.config, args.hyper, args.outdir)
