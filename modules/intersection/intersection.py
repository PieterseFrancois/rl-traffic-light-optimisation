from dataclasses import dataclass

import torch
import numpy as np

from .action import ActionModule, TLSTimingStandards
from .state import StateModule, LaneMeasures
from .reward import RewardModule, RewardNormalisationParameters, RewardFunction
from .preprocessor import (
    PreprocessorModule,
    PreprocessorConfig,
    PreprocessorNormalisationParameters,
)


@dataclass
class IntersectionConfig:
    """
    Configuration for an intersection module, including sub-modules.

    Attributes:
        tls_id (str): Traffic light system ID.
        warm_start (bool): Whether to warm start the controller to a known phase.
        max_detection_range_m (float): Maximum detection range for vehicles (in meters).
        green_phase_strings (list[str] | None): List of green phase strings for the action module. If None, will infer from SUMO.
        timing_standards (TLSTimingStandards | None): TLSTimingStandards for the action module. If None, no timing standards will be enforced.
        normalise_rewards (bool): Whether to normalise rewards in the reward module.
        average_vehicle_length_m (float): Average vehicle length (in meters) for normalisation in the reward module.
        reward_function (RewardFunction): Reward function to use in the reward module.
    """

    # General configuration for the intersection
    tls_id: str
    warm_start: bool

    # Configuration for the state module
    max_detection_range_m: float = 50.0

    # Configuration for the action module
    green_phase_strings: list[str] | None = None
    timing_standards: TLSTimingStandards | None = None

    # Configuration for the reward module
    normalise_rewards: bool = True
    average_vehicle_length_m: float = 5.0
    min_gap_between_vehicles_m: float = 2.5
    reward_function: RewardFunction = RewardFunction.TOTAL_WAIT


@dataclass
class BaseIntersectionKPIs:
    """
    Base KPIs for an intersection.

    Attributes:
        total_wait_time_s (float): Total wait time of all vehicles (in seconds).
        total_queue_length (int): Total queue length of all vehicles.
        max_wait_time_s (float): Maximum wait time of any vehicle (in seconds).
    """

    total_wait_time_s: float
    total_queue_length: int
    max_wait_time_s: float


class IntersectionModule:
    def __init__(
        self,
        traci_connection,
        config: IntersectionConfig,
        feature_config: PreprocessorConfig,
    ):
        # Initialise general attributes
        self.traci_connection = traci_connection
        self.tls_id = config.tls_id

        # Initialise sub-modules
        self._init_state_module(config)
        self._init_action_module(config)
        self._init_reward_module(config)
        self._init_preprocessor_module(config, feature_config)

        # Warm start controller to a known phase
        if config.warm_start:
            self.action_module.set_phase_immediately(0)

    # ---- Sub-module initialisation methods ---- #

    def _init_state_module(self, config: IntersectionConfig) -> None:
        self.state_module = StateModule(
            traci_connection=self.traci_connection,
            tls_id=self.tls_id,
            max_detection_range_m=config.max_detection_range_m,
        )

    def _init_action_module(self, config: IntersectionConfig) -> None:
        action_module_args = {
            "traci_connection": self.traci_connection,
            "tls_id": self.tls_id,
        }
        if config.green_phase_strings is not None:
            action_module_args["green_phase_strings"] = config.green_phase_strings

        if config.timing_standards is not None:
            action_module_args["timing_standards"] = config.timing_standards

        self.action_module = ActionModule(**action_module_args)

    def _init_reward_module(self, config: IntersectionConfig) -> None:
        normalisation_params = (
            RewardNormalisationParameters(
                max_detection_range_m=config.max_detection_range_m,
                avg_vehicle_length_m=config.average_vehicle_length_m
                + config.min_gap_between_vehicles_m,
            )
            if config.normalise_rewards
            else None
        )

        self.reward_module = RewardModule(
            traci_connection=self.traci_connection,
            tls_id=self.tls_id,
            normalisation_params=normalisation_params,
        )

        self.reward_module.set_active_reward_function(config.reward_function)

    def _init_preprocessor_module(
        self, config: IntersectionConfig, preprocessor_config: PreprocessorConfig
    ) -> None:

        normalisation_params = PreprocessorNormalisationParameters(
            max_detection_range_m=config.max_detection_range_m,
            avg_vehicle_occupancy_length_m=config.average_vehicle_length_m
            + config.min_gap_between_vehicles_m,
            max_wait_time_horizon_s=(
                config.timing_standards.max_green_s if config.timing_standards else 120
            ),
        )

        self.preprocessor_module = PreprocessorModule(
            traci_connection=self.traci_connection,
            tls_id=self.tls_id,
            config=preprocessor_config,
            normalisation_params=normalisation_params,
        )

    # ---- Probe utility methods ---- #
    def features_per_lane(self) -> int:
        """Get the number of features per lane in the observation space."""
        return self.preprocessor_module.num_features_per_lane()

    def num_lanes(self) -> int:
        """Get the number of approach lanes controlled by this traffic light."""
        return self.state_module.num_lanes()

    # ---- Step-time methods ---- #

    def read_state(self) -> None:
        """Read the current state from SUMO via TraCI."""
        self.state_module.read_state()

    def get_observation(self) -> np.ndarray:
        """Get the current state observation as a NumPy array [L, F], float32."""
        # Use the right StateModule API in your codebase:
        raw_state = self.state_module.get_current_state()
        x_t: torch.Tensor = self.preprocessor_module.get_state_tensor(
            raw_state
        )  # [L, F] torch
        x_np: np.ndarray = (
            x_t.detach().to(dtype=torch.float32, device="cpu").numpy().copy()
        )
        return x_np  # shape (L, F), dtype float32

    def get_observation_flattened(self) -> np.ndarray:
        """Get the current state observation flattened to [L*F], float32."""
        x_np = self.get_observation()  # (L, F) float32
        return x_np.reshape(-1).astype(np.float32, copy=False)

    def ready(self) -> bool:
        """Check if the intersection is ready for a new action."""
        return bool(self.action_module.ready_for_decision())

    def action_mask(self) -> np.ndarray:
        """Get the action mask for the current state as a NumPy array [n_actions], float32."""
        mask: list[bool] = self.action_module.get_action_mask()
        return np.asarray(mask, dtype=np.float32)

    def apply_action(self, action: int) -> None:
        """Apply the given action to the traffic light."""
        self.action_module.set_phase(action)

    def tick(self) -> None:
        """Advance the internal state of sub-modules by one simulation step."""
        self.action_module.update_transition()

    def get_reward(self) -> float:
        """Get the current reward."""
        raw_state: list[LaneMeasures] = self.state_module.get_current_state()
        return self.reward_module.compute_reward(raw_state)

    def get_kpi(self) -> BaseIntersectionKPIs:
        """Get the current KPIs for the intersection."""
        lane_states: list[LaneMeasures] = self.state_module.get_current_state()

        total_wait_time_s = sum(lane.total_wait_s for lane in lane_states)
        total_queue_length = sum(lane.queue for lane in lane_states)
        max_wait_time_s = max((lane.max_wait_s for lane in lane_states), default=0.0)

        return BaseIntersectionKPIs(
            total_wait_time_s=total_wait_time_s,
            total_queue_length=total_queue_length,
            max_wait_time_s=max_wait_time_s,
        )

    # ---- Other Properties ---- #
    @property
    def n_actions(self) -> int:
        return int(len(self.action_module.get_action_mask()))
