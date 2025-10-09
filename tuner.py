# tuner.py
from pathlib import Path
from collections import deque, defaultdict

import ray
from ray import tune
from ray.air import RunConfig, CheckpointConfig
from ray.tune import Tuner, TuneConfig, FailureConfig
from ray.tune.stopper import Stopper
from ray.tune.schedulers import ASHAScheduler

from ray.rllib.algorithms.ppo import PPO, PPOConfig

from utils.env_config_loader import load_env_config
from utils.hyperparams_loader import load_hyperparameters
from utils.load_seed_lists import load_int_list as load_seeds
from policy.independent_ppo_setup import build_independent_ppo_config, TrainerParameters
from modules.model_manager import ModelManager

from models.masked_gnn_attention_model import (
    register_attention_gnn_model,
    MODEL_NAME as ATTN_GNN_NAME,
)


# -----------------------------
# Custom windowed-improvement stopper
# -----------------------------
class WindowedImprovementStopper(Stopper):
    """
    Stop a trial if the moving-average of `metric` has not improved by at least
    `delta` over the best moving-average seen, for `patience` consecutive checks.

    Moving-average uses a fixed window length `window_size` in training iterations.
    """

    def __init__(self, metric: str, window_size: int, delta: float, patience: int):
        self.metric = metric
        self.window_size = int(window_size)
        self.delta = float(delta)
        self.patience = int(patience)

        # trial_id -> (deque of recent metrics, best_ma, stale_counter)
        self._state: dict[str, tuple[deque[float], float, int]] = defaultdict(
            lambda: (deque(maxlen=self.window_size), float("-inf"), 0)
        )

    def __call__(self, trial_id: str, result: dict) -> bool:
        # Pull metric safely
        if self.metric not in result:
            return False  # don't decide yet

        val = result[self.metric]
        if val is None:
            return False

        q, best_moving_average, stale = self._state[trial_id]
        q.append(float(val))

        # Not enough history yet
        if len(q) < self.window_size:
            self._state[trial_id] = (q, best_moving_average, 0)
            return False

        # Current moving average
        current_moving_average = sum(q) / len(q)

        if current_moving_average > best_moving_average + self.delta:
            best_moving_average = current_moving_average
            stale = 0
        else:
            stale += 1

        self._state[trial_id] = (q, best_moving_average, stale)
        # Stop if stale for `patience` consecutive checks
        return stale >= self.patience

    def stop_all(self) -> bool:
        return False


# ----------------------
# Build base PPO config
# ----------------------
def build_base_config(
    env_kwargs: dict, trainer_params: TrainerParameters, model_cfg: dict
) -> PPOConfig:
    return build_independent_ppo_config(
        env_kwargs=env_kwargs,
        register_fn=register_attention_gnn_model,
        training_model={
            "custom_model": ATTN_GNN_NAME,
            "custom_model_config": model_cfg,
        },
        trainer_params=trainer_params,
    )


def main(
    config_file: str,
    hyperparams_file: str,
    outdir: str,
    num_samples: int,
    max_concurrent: int,
    # Early stopping
    window_size: int,
    delta: float,
    patience: int,
    # ASHA
    grace_period: int,
):
    ray.shutdown()
    ray.init(ignore_reinit_error=True, include_dashboard=False)

    # ----- Env config and seeds -----
    env_kwargs = load_env_config(yaml_path=config_file)

    # Load training seeds
    TRAINING_SEEDS_PATH: str = "environments/training_seeds.txt"
    training_seeds: list[int] = load_seeds(TRAINING_SEEDS_PATH)

    env_kwargs["training_seeds"] = training_seeds

    # Make sumocfg absolute to avoid CWD/worker issues
    sumocfg_path = Path(env_kwargs["sumo_config"].sumocfg_filepath)
    if not sumocfg_path.is_absolute():
        env_kwargs["sumo_config"].sumocfg_filepath = str(sumocfg_path.resolve())

    # Hyperparameters from your YAML loader
    hps = load_hyperparameters(yaml_path=hyperparams_file)

    base_cfg = build_base_config(
        env_kwargs=env_kwargs,
        trainer_params=hps["trainer_params"],
        model_cfg=hps["model_params"],
    )

    # Resource plan
    base_cfg = base_cfg.resources(num_cpus_for_main_process=1, num_gpus=0).env_runners(
        sample_timeout_s=180,
        num_env_runners=4,
        num_envs_per_env_runner=2,
        num_cpus_per_env_runner=1,
        num_gpus_per_env_runner=0,
        gym_env_vectorize_mode="SYNC",
        rollout_fragment_length="auto",
    )
    # Guide by *training* episode mean return
    METRIC = "env_runners/episode_return_mean"

    # Random sweep with a fixed budget
    param_space = base_cfg.to_dict()
    param_space.update(
        {
            "lr": tune.choice([1e-5, 1e-6, 1e-7]),
            "entropy_coeff": tune.uniform(0.001, 0.012),
            "clip_param": tune.uniform(0.18, 0.34),
            "num_epochs": tune.choice([2, 4, 6]),
            "train_batch_size": tune.choice([8192, 12288, 16384]),
            "minibatch_size": tune.choice([256, 512, 1024]),
            "gamma": tune.choice([0.97, 0.99]),
            "lambda": tune.uniform(0.93, 0.97),
        }
    )
    # ASHA can kill weak trials early
    scheduler = ASHAScheduler(
        metric=METRIC,
        mode="max",
        grace_period=grace_period,
        max_t=hps["training_params"].max_iterations,
    )
    samples = int(num_samples)

    # -----------------------------
    # Output paths and short names
    # -----------------------------
    storage_path = Path(outdir) / "tune"
    storage_path.mkdir(parents=True, exist_ok=True)
    storage_path_uri = storage_path.resolve().as_uri()

    # -----------------------------
    # Early stopping (windowed moving average over training metric)
    # -----------------------------
    stopper = WindowedImprovementStopper(
        metric=METRIC,
        window_size=window_size,
        delta=delta,
        patience=patience,
    )

    # -----------------------------
    # Build tuner
    # -----------------------------
    tuner = Tuner(
        PPO,
        param_space=param_space,
        run_config=RunConfig(
            name=f"coarse_grid_search",
            storage_path=str(storage_path_uri),
            stop=stopper,
            failure_config=FailureConfig(max_failures=5),
            checkpoint_config=CheckpointConfig(
                checkpoint_at_end=True,
                checkpoint_frequency=1,
                num_to_keep=5,
            ),
        ),
        tune_config=TuneConfig(
            num_samples=samples,
            scheduler=scheduler,
            max_concurrent_trials=int(max_concurrent),
            reuse_actors=False,
            trial_name_creator=lambda t: f"{t.trainable_name}_{t.trial_id[:5]}",
            trial_dirname_creator=lambda t: t.trial_id,
        ),
    )

    # -----------------------------
    # Run, summarise, persist artefacts for analysis
    # -----------------------------
    results = tuner.fit()

    # Save a CSV of all trials for sensitivity plots
    try:
        df = results.get_dataframe()
        csv_path = Path(outdir) / "tune_results_full.csv"
        df.to_csv(csv_path, index=False)
        print(f"[tune] Wrote full results CSV to: {csv_path}")
    except Exception as e:
        print(f"[tune] Could not write results CSV: {e}")

    # Best by training metric
    best = results.get_best_result(METRIC, "max")
    print("Best trial metric summary:", {METRIC: best.metrics.get(METRIC, None)})

    # Restore the trained PPO from the best checkpoint
    best_ckpt = best.checkpoint
    ckpt_dir = best_ckpt.to_directory()
    algo = PPO.from_checkpoint(ckpt_dir)

    # Bundle the best model + metadata
    bundle_root = Path(outdir) / "random_grid_search_bundle"
    bundle_root.mkdir(parents=True, exist_ok=True)

    # Record the tuned keys present in this sweep (intersect with config)
    tuned_keys = [
        "lr",
        "gamma",
        "lambda",
        "minibatch_size",
        "train_batch_size",
        "num_epochs",
        "clip_param",
        "entropy_coeff",
    ]
    tuned_hps = {k: best.config.get(k) for k in tuned_keys if k in best.config}

    mm = ModelManager()
    bundle_path = mm.save_bundle(
        trainer=algo,
        bundle_dir_rel=bundle_root,
        model_name=ATTN_GNN_NAME,
        custom_model_config=hps["model_params"],
        env_kwargs=env_kwargs,
        notes={
            "metric": METRIC,
            "tuned_hyperparameters": tuned_hps,
        },
    )
    print(f"[tune] bundle saved to {str(bundle_path)}")


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--hyper", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--num-samples", type=int)
    ap.add_argument("--max-concurrent", type=int)

    # early stopping knobs
    ap.add_argument(
        "--window-size",
        type=int,
        help="Moving-average window (in training iterations) for the stopper",
    )
    ap.add_argument(
        "--delta",
        type=float,
        help="Minimum improvement over best moving-average to reset patience",
    )
    ap.add_argument(
        "--patience",
        type=int,
        help="Stop if no MA improvement > delta for this many consecutive checks",
    )

    ap.add_argument(
        "--grace-period",
        type=int,
        help="(ASHA) Minimum number of training iterations before a trial can be stopped",
    )

    args = ap.parse_args()
    main(
        config_file=args.config,
        hyperparams_file=args.hyper,
        outdir=args.outdir,
        num_samples=args.num_samples,
        max_concurrent=args.max_concurrent,
        window_size=args.window_size,
        delta=args.delta,
        patience=args.patience,
        grace_period=args.grace_period,
    )
