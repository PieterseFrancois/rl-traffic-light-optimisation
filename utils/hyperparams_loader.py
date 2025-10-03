from pathlib import Path
from typing import Any

import yaml

from policy.independent_ppo_setup import TrainerParameters

from utils.training import TrainingHyperparameters


def _build_training_params(param_dict: dict[str, Any]) -> TrainingHyperparameters:
    """Build the training hyperparameters from the provided dictionary."""

    REQUIRED_KEYS = [
        "max_iterations",
        "early_stopping_patience",
        "minimum_improvement",
        "moving_window_size",
    ]

    for key in REQUIRED_KEYS:
        if key not in param_dict:
            raise ValueError(f"Missing required training hyperparameter: {key}")

    return TrainingHyperparameters(
        max_iterations=int(param_dict["max_iterations"]),
        patience=int(param_dict["early_stopping_patience"]),
        min_delta=float(param_dict["minimum_improvement"]),
        window=int(param_dict["moving_window_size"]),
    )


def _build_trainer_params(param_dict: dict[str, Any]) -> TrainerParameters:
    """Build the trainer hyperparameters from the provided dictionary."""

    REQUIRED_KEYS = [
        "num_workers",
        "rollout_fragment_length",
        "train_batch_size",
        "minibatch_size",
        "num_epochs",
        "learning_rate",
        "gamma",
        "batch_mode",
    ]

    for key in REQUIRED_KEYS:
        if key not in param_dict:
            raise ValueError(f"Missing required trainer hyperparameter: {key}")

    # Ensure batch_mode is valid
    VALID_BATCH_MODES = ["truncate_episodes", "complete_episodes"]
    batch_mode = str(param_dict["batch_mode"])
    if batch_mode not in VALID_BATCH_MODES:
        raise ValueError(
            f"Invalid batch_mode: {batch_mode}. Must be one of {VALID_BATCH_MODES}"
        )

    # Create param datastructure
    return TrainerParameters(
        num_workers=int(param_dict["num_workers"]),
        rollout_fragment_length=int(param_dict["rollout_fragment_length"]),
        train_batch_size=int(param_dict["train_batch_size"]),
        minibatch_size=int(param_dict["minibatch_size"]),
        num_epochs=int(param_dict["num_epochs"]),
        lr=float(param_dict["learning_rate"]),
        gamma=float(param_dict["gamma"]),
        batch_mode=batch_mode,
    )


def _build_model_params(param_dict: dict[str, Any]) -> dict[str, Any]:
    """Build the model hyperparameters from the provided dictionary."""

    # Just return the dict as-is for now
    return param_dict


def load_hyperparameters(yaml_path: str | Path) -> dict[str, Any]:
    """Load and parse the hyperparameters from a YAML file."""

    with open(yaml_path, "r") as f:
        file = yaml.safe_load(f)

    # Start with an empty hyperparams dict
    hyperparams = {}

    # Validate and parse training parameters
    if "training" not in file or not isinstance(file["training"], dict):
        raise ValueError("Missing or invalid 'training' configuration section.")

    training_dict = file["training"]
    training_params: TrainingHyperparameters = _build_training_params(training_dict)
    hyperparams["training_params"] = training_params

    # Validate and parse trainer parameters
    if "trainer" not in file or not isinstance(file["trainer"], dict):
        raise ValueError("Missing or invalid 'trainer' configuration section.")

    trainer_dict = file["trainer"]
    trainer_params: TrainerParameters = _build_trainer_params(trainer_dict)
    hyperparams["trainer_params"] = trainer_params

    # Validate and parse model parameters
    if "model" not in file or not isinstance(file["model"], dict):
        raise ValueError("Missing or invalid 'model' configuration section.")

    model_dict = file["model"]
    model_params: dict[str, Any] = _build_model_params(model_dict)
    hyperparams["model_params"] = model_params

    return hyperparams
