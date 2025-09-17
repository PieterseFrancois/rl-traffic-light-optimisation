from pathlib import Path
from typing import Any

import yaml

from dataclasses import dataclass

from modules.intersection.action import TLSTimingStandards
from modules.intersection.reward import RewardFunction
from modules.intersection.preprocessor import PreprocessorConfig as FeatureConfig
from modules.intersection.intersection import IntersectionConfig
from utils.sumo_helpers import SUMOConfig


@dataclass
class IntersectionDefaults:
    """
    Default settings for intersections.

    Attributes:
        reward_function (RewardFunction): The reward function to use.
        normalise_rewards (bool): Whether to normalise rewards.
        timing_standards (TLSTimingStandards): Timing standards for traffic lights.
        warm_start (bool): Whether to warm start the traffic light to phase 0.
    """

    reward_function: RewardFunction
    normalise_rewards: bool
    timing_standards: TLSTimingStandards
    warm_start: bool
    max_detection_range_m: float
    green_phase_strings: list[str] | None  # None means infer from SUMO


def _convert_str_to_reward_function(function_str: str) -> RewardFunction:
    """Convert a string to a RewardFunction enum instance."""
    key = str(function_str).strip().upper()

    # Map to enum, will raise KeyError if invalid
    try:
        return getattr(RewardFunction, key)
    except Exception as e:
        raise ValueError(
            f"Error converting '{function_str}' to RewardFunction instance: {e}"
        )


def _build_timing_standards(config_dict: dict[str, Any]) -> TLSTimingStandards:
    """Build a TLSTimingStandards instance from a config dictionary."""
    return TLSTimingStandards(
        min_green_s=config_dict.get("min_green_s", 0.0),
        yellow_s=config_dict.get("yellow_s", 0.0),
        all_red_s=config_dict.get("all_red_s", 0.0),
        max_green_s=config_dict.get("max_green_s", 0.0),
    )


def _build_feature_config(config_dict: dict[str, Any]) -> FeatureConfig:
    """Build a PreprocessorConfig instance from a config dictionary."""

    KEYS = (
        "include_queue",
        "include_approach",
        "include_total_wait",
        "include_max_wait",
        "include_total_speed",
    )
    missing = [k for k in KEYS if k not in config_dict]
    if missing:
        raise ValueError(f"Missing required keys for features: {missing}")

    return FeatureConfig(
        include_queue=bool(config_dict["include_queue"]),
        include_approach=bool(config_dict["include_approach"]),
        include_total_wait=bool(config_dict["include_total_wait"]),
        include_max_wait=bool(config_dict["include_max_wait"]),
        include_total_speed=bool(config_dict["include_total_speed"]),
    )


def _build_sumo_config(config_dict: dict[str, Any]) -> SUMOConfig:
    """Build a SUMOConfig instance from a config dictionary."""
    # Validate sumcfg file
    if "sumocfg" not in config_dict or not isinstance(
        config_dict["sumocfg"], (str, Path)
    ):
        raise ValueError("Missing or invalid 'sumocfg' in sumo config.")

    return SUMOConfig(
        sumocfg_filepath=config_dict.get("sumocfg"),
        nogui=not bool(config_dict.get("gui", False)),
        seed=config_dict.get("seed", 42),
        time_to_teleport=config_dict.get("time_to_teleport_s", -1),
    )


def _get_intersection_defaults(config_dict: dict[str, Any]) -> IntersectionDefaults:
    """Extract and build the IntersectionDefaults from the config dictionary."""

    defaults = config_dict.get("defaults")
    if defaults is None or not isinstance(defaults, dict):
        raise ValueError(
            "Missing or invalid 'defaults' section in intersections config."
        )

    # Default reward
    reward_default = defaults.get("reward")
    if reward_default is None or not isinstance(reward_default, dict):
        raise ValueError(
            "Missing or invalid 'reward' section in intersections defaults."
        )

    reward_function_default_str = reward_default.get("function")
    if reward_function_default_str is None or not isinstance(
        reward_function_default_str, str
    ):
        raise ValueError(
            "Missing or invalid 'function' in intersections reward defaults."
        )
    reward_function_default: RewardFunction = _convert_str_to_reward_function(
        reward_function_default_str
    )

    normalise_default = reward_default.get("normalise")
    if normalise_default is None or not isinstance(normalise_default, bool):
        raise ValueError(
            "Missing or invalid 'normalise' in intersections reward defaults."
        )

    # Default TLS timing standards
    timing_standards_default_dict = defaults.get("timing_standards")
    if timing_standards_default_dict is None or not isinstance(
        timing_standards_default_dict, dict
    ):
        raise ValueError(
            "Missing or invalid 'timing_standards' in intersections defaults."
        )

    timing_standards_default: TLSTimingStandards = _build_timing_standards(
        timing_standards_default_dict
    )

    # Other defaults
    warm_start_default = defaults.get("warm_start")
    if warm_start_default is None or not isinstance(warm_start_default, bool):
        raise ValueError("Missing or invalid 'warm_start' in intersections defaults.")

    max_detection_range_m_default = defaults.get("max_detection_range_m")
    if max_detection_range_m_default is None or not isinstance(
        max_detection_range_m_default, (int, float)
    ):
        raise ValueError(
            "Missing or invalid 'max_detection_range_m' in intersections defaults."
        )

    green_phase_strings_default = defaults.get(
        "green_phase_strings", None
    )  # Optional, can be None

    return IntersectionDefaults(
        reward_function=reward_function_default,
        normalise_rewards=normalise_default,
        timing_standards=timing_standards_default,
        warm_start=warm_start_default,
        max_detection_range_m=max_detection_range_m_default,
        green_phase_strings=green_phase_strings_default,
    )


def _build_intersection_configs(
    intersection_config_dict: dict[str, Any], vehicle_config_dict: dict[str, Any]
) -> list[IntersectionConfig]:
    """Build a list of IntersectionConfig instances from a list of config dictionaries."""

    # Defaults to apply to each intersection
    default: IntersectionDefaults = _get_intersection_defaults(intersection_config_dict)

    # Vehicle parameters
    avg_vehicle_length_m = vehicle_config_dict.get("avg_vehicle_length_m")
    if avg_vehicle_length_m is None or not isinstance(
        avg_vehicle_length_m, (int, float)
    ):
        raise ValueError(
            "Missing or invalid 'avg_vehicle_length_m' in vehicles config."
        )
    min_gap_m = vehicle_config_dict.get("min_gap_m")
    if min_gap_m is None or not isinstance(min_gap_m, (int, float)):
        raise ValueError("Missing or invalid 'min_gap_m' in vehicles config.")

    # Get the list of intersection configs
    intersection_list = intersection_config_dict.get("list")
    if intersection_list is None or not isinstance(intersection_list, list):
        raise ValueError(
            "Missing or invalid 'list' of intersections in intersections config."
        )

    intersection_configs: list[IntersectionConfig] = []
    for idx, custom_config in enumerate(intersection_list):
        # Required TLS id
        tls_id = custom_config.get("id")
        if tls_id is None or not isinstance(tls_id, str):
            raise ValueError(
                f"Missing or invalid 'id' for intersection at index {idx}."
            )

        # Green phase strings override
        green_phase_strings = custom_config.get(
            "green_phase_strings", default.green_phase_strings
        )
        if green_phase_strings is not None and not (
            isinstance(green_phase_strings, list)
            and all(isinstance(s, str) for s in green_phase_strings)
        ):
            raise ValueError(
                f"'green_phase_strings' for '{tls_id}' must be a list[str] or null."
            )

        # Timing standards override
        timing_standards: TLSTimingStandards = (
            _build_timing_standards(custom_config["timing_standards"])
            if "timing_standards" in custom_config
            and isinstance(custom_config["timing_standards"], dict)
            else default.timing_standards
        )

        # Reward function override
        reward_config = custom_config.get("reward", {})
        reward_function: RewardFunction = (
            _convert_str_to_reward_function(reward_config["function"])
            if "function" in reward_config
            else default.reward_function
        )
        normalise: bool = (
            bool(reward_config["normalise"])
            if "normalise" in reward_config
            else default.normalise_rewards
        )

        config = IntersectionConfig(
            tls_id=tls_id,
            warm_start=custom_config.get("warm_start", default.warm_start),
            max_detection_range_m=custom_config.get(
                "max_detection_range_m", default.max_detection_range_m
            ),
            green_phase_strings=green_phase_strings,
            timing_standards=timing_standards,
            normalise_rewards=normalise,
            reward_function=reward_function,
            average_vehicle_length_m=float(avg_vehicle_length_m),
            min_gap_between_vehicles_m=float(min_gap_m),
        )
        intersection_configs.append(config)

    return intersection_configs


def load_env_config(yaml_path: str | Path) -> dict[str, Any]:
    """Load and parse the environment configuration from a YAML file."""

    with open(yaml_path, "r") as f:
        config = yaml.safe_load(f)

    # Start with an empty env_kwargs dict
    env_kwargs = {}

    # Validate and parse SUMO config
    if "sumo" not in config or not isinstance(config["sumo"], dict):
        raise ValueError("Missing or invalid 'sumo' configuration section.")

    sumo_dict = config["sumo"]
    sumo_config: SUMOConfig = _build_sumo_config(sumo_dict)
    env_kwargs["sumo_config"] = sumo_config

    # Validate and parse feature config
    if "features" not in config or not isinstance(config["features"], dict):
        raise ValueError("Missing or invalid 'features' configuration section.")

    feature_config: FeatureConfig = _build_feature_config(config["features"])
    env_kwargs["feature_config"] = feature_config

    # Validate and parse ticks_per_decision
    if "ticks_per_decision" not in sumo_dict or not isinstance(
        sumo_dict["ticks_per_decision"], int
    ):
        raise ValueError("Missing or invalid 'ticks_per_decision' configuration value.")

    ticks_per_decision = int(sumo_dict["ticks_per_decision"])
    if ticks_per_decision <= 0:
        raise ValueError("'ticks_per_decision' must be a positive integer.")
    env_kwargs["ticks_per_decision"] = ticks_per_decision

    # Validate and parse intersection
    if "intersections" not in config:
        raise ValueError("Missing 'intersections' configuration section.")
    if "vehicles" not in config:
        raise ValueError("Missing 'vehicles' configuration section.")

    intersection_configs: list[IntersectionConfig] = _build_intersection_configs(
        config["intersections"], config["vehicles"]
    )
    env_kwargs["intersection_agent_configs"] = intersection_configs

    return env_kwargs
