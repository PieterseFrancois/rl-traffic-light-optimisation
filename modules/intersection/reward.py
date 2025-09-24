from typing import Callable, Any
from enum import Enum
from dataclasses import dataclass
from numpy import clip as np_clip

from .state import LaneMeasures


@dataclass
class RewardNormalisationParameters:
    """
    Normalisation parameters for reward functions.

    Attributes:
        max_detection_range_m (float): Maximum detection range for vehicles (in meters).
        avg_vehicle_length_m (float): Average vehicle length (in meters).
    """

    max_detection_range_m: float
    avg_vehicle_length_m: float


@dataclass
class RESCOWaitNormalisationParameters:
    denominator: float = 224.0
    lower_bound: float = -4.0
    upper_bound: float = 4.0


class RewardFunction(Enum):
    """
    Available reward functions.

    - QUEUE: Negative of the total queue length across all lanes.
    - TOTAL_WAIT: Negative of the total waiting time across all lanes.
    - CUMULATIVE_WAIT_DIFF: Difference in total waiting time since the last computation.
    """

    QUEUE = "QUEUE"
    TOTAL_WAIT = "TOTAL_WAIT"
    CUMULATIVE_WAIT_DIFF = "CUMULATIVE_WAIT_DIFF"


class RewardModule:

    def __init__(
        self,
        traci_connection,
        tls_id: str,
        normalisation_params: RewardNormalisationParameters | None = None,
    ):

        self.traci_connection = traci_connection
        self.tls_id: str = tls_id

        if normalisation_params is None:
            self.normalise_rewards: bool = False
            self.normalisation_params = None
        else:
            self.normalise_rewards: bool = True
            self.normalisation_params: RewardNormalisationParameters = (
                normalisation_params
            )

        self._active_reward_function: Callable | None = None
        self._reward_fns: dict[RewardFunction, Callable] = {
            RewardFunction.QUEUE: self._queue_penalty,
            RewardFunction.TOTAL_WAIT: self._total_wait_penalty,
            RewardFunction.CUMULATIVE_WAIT_DIFF: self._difference_in_wait_reward,
        }

        self._normalisation_cache: dict[str, Any] = {
            "total_lane_length_m": None,
            "resco_normalisation_params": RESCOWaitNormalisationParameters(),
        }

        self._last_compute_time = float(self.traci_connection.simulation.getTime())
        self._last_state_cache: list[LaneMeasures] | None = None
        self._last_reward: float | None = None

    def set_normalise(self, normalisation_params: RewardNormalisationParameters | None):
        """Set whether to normalise rewards. If normalisation_params is None, disable normalisation."""
        if normalisation_params is None:
            self.normalise_rewards = False
            self.normalisation_params = None
        else:
            self.normalise_rewards = True
            self.normalisation_params = normalisation_params

    def set_active_reward_function(self, name: str):
        """Set the active reward function by name."""
        if name not in self._reward_fns:
            raise ValueError(f"Unknown reward function '{name}'")
        self._active_reward_function = self._reward_fns[name]

    def compute_reward(self, state: list[LaneMeasures]) -> float:
        """Compute the reward for the given state using the active reward function.

        Args:
            state: Current state read from SUMO (list of LaneMeasures).
            previous_state: Previous state read from SUMO (list of LaneMeasures).
        Returns:
            A float representing the reward for the current state.
        """

        if self._active_reward_function is None:
            raise RuntimeError("No active reward function set")
        
        # If no time has passed since the last computation, return the last reward
        current_time = float(self.traci_connection.simulation.getTime())
        if current_time == self._last_compute_time and self._last_reward is not None:
            return self._last_reward

        reward = self._active_reward_function(state)
        self._last_compute_time = current_time
        self._last_reward = reward
        # Normalisation handled in the reward functions themselves
        return reward

    def _calculate_total_lane_length(self, lane_ids: list[str]) -> float:
        """Calculate the total length of the given lanes.

        Args:
            lane_ids: List of lane IDs to calculate the total length for.

        Returns:
            Total length of the lanes in meters.
        """
        total_length = 0.0
        for lane_id in lane_ids:
            lane_length = self.traci_connection.lane.getLength(lane_id)
            if self.normalisation_params is not None:
                usable_length: float = min(
                    lane_length, self.normalisation_params.max_detection_range_m
                )
            else:
                usable_length: float = lane_length
            total_length += usable_length

        if total_length <= 0.0:
            raise ValueError(
                f"Total lane length for TLS {self.tls_id} is zero or negative"
            )

        return total_length

    def _queue_penalty(self, state: list[LaneMeasures]) -> float:
        """Calculate a penalty based on the total queue length across all lanes.

        Args:
            state: Current state read from SUMO (list of LaneMeasures).

        Returns:
            A negative float representing the penalty for the current state.
        """
        total_queue: int = sum(lane.queue for lane in state)

        if not self.normalise_rewards:
            return -total_queue

        # Normalise by maximum possible queue length
        total_lane_length_m: float = 0.0
        if self._normalisation_cache["total_lane_length_m"] is None:
            lane_ids = [lane.lane_id for lane in state]
            total_lane_length_m = self._calculate_total_lane_length(lane_ids)
            self._normalisation_cache["total_lane_length_m"] = total_lane_length_m
        else:
            total_lane_length_m = self._normalisation_cache["total_lane_length_m"]

        if self.normalisation_params is None:
            raise RuntimeError(
                "Normalisation parameters are required for normalised rewards."
            )

        estimated_max_vehicles: float = (
            total_lane_length_m / self.normalisation_params.avg_vehicle_length_m
        )

        if estimated_max_vehicles == 0:
            raise ValueError(
                "Estimated maximum vehicles is zero, cannot normalise queue penalty."
            )

        return -(total_queue / estimated_max_vehicles)

    def _total_wait_penalty(self, state: list[LaneMeasures]) -> float:

        total_wait: float = sum(lane.total_wait_s for lane in state)

        if not self.normalise_rewards:
            return -total_wait

        # Normalise using RESCO parameters and clipping
        RESCO_PARAMS = self._normalisation_cache["resco_normalisation_params"]
        normalised_wait = float(-total_wait / RESCO_PARAMS.denominator)
        clipped_wait = np_clip(
            normalised_wait, RESCO_PARAMS.lower_bound, RESCO_PARAMS.upper_bound
        )

        return clipped_wait

    def _difference_in_wait_reward(self, state: list[LaneMeasures]) -> float:
        """Calculate the difference in total wait time since the last computation.

        Args:
            state: Current state read from SUMO (list of LaneMeasures).

        Returns:
            A float representing the difference in total wait time.
        """
        # If this is the first call, we cannot compute a difference
        if self._last_state_cache is None:
            self._last_state_cache = state
            return 0.0

        # Extract previous state from cache and update cache
        previous_state = self._last_state_cache
        self._last_state_cache = state

        current_total_wait = sum(lane.total_wait_s for lane in state)
        previous_total_wait = sum(lane.total_wait_s for lane in previous_state)

        if not self.normalise_rewards:
            return float(previous_total_wait - current_total_wait)

        # Normalise using RESCO parameters and clipping
        RESCO_PARAMS = self._normalisation_cache["resco_normalisation_params"]
        normalised_diff = float(
            (previous_total_wait - current_total_wait) / RESCO_PARAMS.denominator
        )
        clipped_diff = np_clip(
            normalised_diff, RESCO_PARAMS.lower_bound, RESCO_PARAMS.upper_bound
        )

        return clipped_diff
