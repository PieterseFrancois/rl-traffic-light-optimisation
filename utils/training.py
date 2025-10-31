import math
from collections import deque
from pathlib import Path

from modules.model_manager import ModelManager

from dataclasses import dataclass

from datetime import datetime, timezone

from event_bus import EventBus, EventNames


@dataclass
class TrainingHyperparameters:
    """
    Hyperparameters for training.

    Attributes:
        max_iterations (int): Maximum number of training iterations.
        window (int): Window size for moving average of episode_return_mean.
        min_delta (float): Minimum improvement in moving average to reset patience.
        patience (int): Number of windows with no improvement to wait before stopping.
    """

    max_iterations: int
    window: int
    min_delta: float
    patience: int


@dataclass
class TrainingResult:
    """
    Result of training process.

    Attributes:
        best_iteration (int): The iteration with the best moving average.
        best_moving_average (float): The best moving average of episode_return_mean.
        stopped_reason (str): Reason for stopping training.
        best_checkpoint_path (Path): Path to the best checkpoint saved.
    """

    best_iteration: int
    best_moving_average: float
    stopped_reason: str
    best_checkpoint_path: Path
    episode_returns: list[float] = None


def train_until_converged(
    trainer,
    *,
    parameters: TrainingHyperparameters,
    ckpt_dir: Path,
    model_manager: ModelManager,
    env_kwargs: dict,
    training_model: dict,
    verbose: bool = True,
    event_bus: EventBus | None = None,
) -> TrainingResult:
    """
    Train the given RLlib trainer until convergence based on moving average of episode_return_mean.
    Uses early stopping based on the provided hyperparameters.

    Args:
        trainer: An initialised RLlib trainer.
        parameters (TrainingHyperparameters): Hyperparameters for training.
        ckpt_dir (Path): Directory to save checkpoints.
        model_manager (ModelManager): ModelManager instance to handle saving.
        env_kwargs (dict): Environment kwargs used for saving with the model.
        training_model (dict): Training model configuration used for saving with the model.
        verbose (bool, optional): Whether to print training progress. Defaults to True.
    """
    WARMUP_ITERS = 3
    returns = deque(maxlen=parameters.window)
    best_moving_average: float = -math.inf
    best_iteration: int = -1
    no_improve_windows: int = 0
    stopped_reason: str = f"max_iterations({parameters.max_iterations})"
    best_ckpt_path = None

    if ckpt_dir is not None:
        ckpt_dir = Path(ckpt_dir)
        ckpt_dir.mkdir(parents=True, exist_ok=True)

    (
        event_bus.emit(
            EventNames.SIMULATION_INFO.value, "Starting training until convergence..."
        )
        if event_bus
        else None
    )

    episode_returns = []

    previous_time: datetime = datetime.now(timezone.utc)
    for i in range(1, parameters.max_iterations + 1):
        result = trainer.train()
        episode_return = float(
            result.get("env_runners", {}).get("episode_return_mean", float("nan"))
        )
        if math.isnan(episode_return):
            # If env is still warming up, treat as no info.
            episode_return = -math.inf
        episode_returns.append(episode_return)

        returns.append(episode_return)
        moving_average: float = sum(returns) / max(1, len(returns))

        if verbose:
            current_time: datetime = datetime.now(timezone.utc)
            iter_str: str = (
                f"iter {i:03d} | time {current_time} | elapsed {(current_time - previous_time).total_seconds()/60:.2f} min"
            )
            iter_str += f"\n ep_return_mean={episode_return:.3f} | ma[{len(returns)}]={moving_average:.3f}\n"
            print(iter_str)
            (
                event_bus.emit(EventNames.SIMULATION_INFO.value, iter_str)
                if event_bus
                else None
            )

        improvement: float = moving_average - best_moving_average
        if improvement > parameters.min_delta and i > WARMUP_ITERS:
            best_moving_average = moving_average
            best_iteration = i
            no_improve_windows = 0
            best_ckpt_path: Path = model_manager.save_bundle(
                trainer=trainer,
                bundle_dir_rel=ckpt_dir,
                model_name=training_model["custom_model"],
                custom_model_config=training_model["custom_model_config"],
                env_kwargs=env_kwargs,
                notes=None,
            )
        else:
            # Only count against patience once a full window is collected
            if len(returns) == parameters.window:
                no_improve_windows += 1
                if no_improve_windows >= parameters.patience:
                    stopped_reason = f"plateau(window={parameters.window}, min_delta={parameters.min_delta}, patience={parameters.patience})"
                    break

    if verbose:
        verbose_str: str = (
            f"Training stopped after {i} iterations. Best moving average: {best_moving_average:.3f} at iteration {best_iteration}."
        )
        verbose_str += f"Reason: {stopped_reason}\n"

        print(verbose_str)
        (
            event_bus.emit(EventNames.SIMULATION_INFO.value, verbose_str)
            if event_bus
            else None
        )

    return TrainingResult(
        best_iteration=best_iteration,
        best_moving_average=best_moving_average,
        stopped_reason=stopped_reason,
        best_checkpoint_path=best_ckpt_path,
        episode_returns=episode_returns,
    )
