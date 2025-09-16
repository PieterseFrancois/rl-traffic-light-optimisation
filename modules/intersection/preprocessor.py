from dataclasses import dataclass
from torch import Tensor, tensor, float32
from typing import Any

from math import ceil
from numpy import clip as np_clip

from .state import LaneMeasures


@dataclass
class PreprocessorConfig:
    """
    Configuration for the preprocessor module.

    Attributes:
        include_queue (bool): Whether to include the queue length feature.
        include_approach (bool): Whether to include the approach count feature.
        include_total_wait (bool): Whether to include the total wait time feature.
        include_max_wait (bool): Whether to include the maximum wait time feature.
        include_total_speed (bool): Whether to include the total speed feature.
    """

    include_queue: bool
    include_approach: bool
    include_total_wait: bool
    include_max_wait: bool
    include_total_speed: bool


@dataclass
class NormalisedLaneMeasures:
    """
    Normalised features for a lane.

    Attributes:
        queue_occupancy (float): Normalised queue length (0 to 1).
        approach_occupancy (float): Normalised approach count (0 to 1).
        avg_wait_time_fraction (float): Normalised total wait time as average wait time as a fraction of max wait time.
        max_wait_time_fraction (float): Normalised max wait time using time horizon and clipping.
        avg_speed_fraction (float): Normalised total speed as average speed as a fraction of speed limit.
    """

    queue_occupancy: float
    approach_occupancy: float
    avg_wait_time_fraction: float
    max_wait_time_fraction: float
    avg_speed_fraction: float


@dataclass
class PreprocessorNormalisationParameters:
    """
    Parameters for normalising features in the preprocessor module.

    Attributes:
        max_detection_range_m (float): Maximum detection range for vehicles (in meters).
        avg_vehicle_occupancy_length_m (float): Average vehicle occupancy length (in meters).
        max_wait_time_horizon_s (float): Time horizon for normalising wait times (in seconds). Recommended to be equal to the max phase duration.
    """

    max_detection_range_m: float
    avg_vehicle_occupancy_length_m: float
    max_wait_time_horizon_s: float


class PreprocessorModule:
    def __init__(
        self,
        traci_connection,
        tls_id: str,
        config: PreprocessorConfig,
        normalisation_params: PreprocessorNormalisationParameters | None,
    ):
        self.traci_connection = traci_connection
        self.tls_id: str = tls_id
        self.config: PreprocessorConfig = config

        self.lane_order: list[str] = self._build_lane_order()

        self.normalisation_params: PreprocessorNormalisationParameters | None = (
            normalisation_params
        )

        if self.normalisation_params is not None:
            self._normalisation_cache: dict[str, dict[str, Any]] = (
                self._build_normalisation_cache()
            )
        else:
            self._normalisation_cache: None = None

    def _build_lane_order(self) -> list[str]:
        """Uses TraCI controlled lanes to determine a consistent lane order aligned with traffic light phase strings."""
        lanes = self.traci_connection.trafficlight.getControlledLanes(self.tls_id)

        # Remove duplicates while preserving order
        seen = set()
        unique_lanes: list[str] = []
        for lane in lanes:
            if lane not in seen:
                seen.add(lane)
                unique_lanes.append(str(lane))

        return unique_lanes

    def _build_normalisation_cache(self) -> dict[str, dict[str, Any]]:
        """Precompute normalisation values for each lane."""

        lane_cache: dict[str, dict[str, Any]] = {}
        for lane_id in self.lane_order:
            # Lane length
            actual_lane_length_m = float(self.traci_connection.lane.getLength(lane_id))
            observed_length_m: float = min(
                actual_lane_length_m, self.normalisation_params.max_detection_range_m
            )
            observed_length_m = max(1, observed_length_m)  # Avoid division by zero

            # Estimated max vehicles in lane
            estimated_max_vehicles: int = ceil(
                observed_length_m
                / self.normalisation_params.avg_vehicle_occupancy_length_m
            )
            estimated_max_vehicles = max(1, estimated_max_vehicles)  # Ensure at least 1

            # Lane speed limit
            lane_speed_limit_m_s = float(
                self.traci_connection.lane.getMaxSpeed(lane_id)
            )
            lane_speed_limit_m_s = max(
                0.1, lane_speed_limit_m_s
            )  # Avoid division by zero

            # Store in cache
            lane_cache[lane_id] = {
                "observed_length_m": observed_length_m,
                "estimated_max_vehicles": estimated_max_vehicles,
                "lane_speed_limit_m_s": lane_speed_limit_m_s,
            }

        return lane_cache

    def _normalise_lane_measures(
        self, lane_measures: LaneMeasures
    ) -> NormalisedLaneMeasures:
        """Normalise the features of a single lane based on the configuration and normalisation parameters."""

        if self.normalisation_params is None:
            raise RuntimeError(
                "Normalisation parameters are required for normalised features."
            )

        if self._normalisation_cache is None:
            self._normalisation_cache = self._build_normalisation_cache()

        # Normalise queue length
        lane_capacity = self._normalisation_cache[lane_measures.lane_id][
            "estimated_max_vehicles"
        ]
        queue_occupancy = lane_measures.queue / lane_capacity

        # Normalise approach count
        approach_occupancy = lane_measures.approach / lane_capacity

        # Normalise total wait time
        number_of_vehicles = len(lane_measures.vehicles)
        divisor = max(
            1, number_of_vehicles * lane_measures.max_wait_s
        )  # Avoid division by zero
        avg_wait_time_fraction = (
            lane_measures.total_wait_s / divisor
        )  # Effectively average wait time per vehicle as a fraction of max wait time

        # Normalise max wait time
        max_wait_time_fraction = (
            lane_measures.max_wait_s / self.normalisation_params.max_wait_time_horizon_s
        )
        max_wait_time_fraction = np_clip(max_wait_time_fraction, 0, 1)

        # Normalise total speed - effectively average speed as a fraction of speed limit
        lane_speed_limit = self._normalisation_cache[lane_measures.lane_id][
            "lane_speed_limit_m_s"
        ]
        if number_of_vehicles == 0:
            avg_speed_fraction = 1  # No vehicles means free flow, so set to 1 as the lane is basically operating at full speed - if 0 it would imply a jam
        else:
            avg_speed_fraction = min(
                1.0, lane_measures.total_speed / (lane_speed_limit * number_of_vehicles)
            )  # Cap at 1.0 to avoid issues if vehicles exceed speed limit

        return NormalisedLaneMeasures(
            queue_occupancy=queue_occupancy,
            approach_occupancy=approach_occupancy,
            avg_wait_time_fraction=avg_wait_time_fraction,
            max_wait_time_fraction=max_wait_time_fraction,
            avg_speed_fraction=avg_speed_fraction,
        )

    def get_state_tensor(self, state: list[LaneMeasures]) -> Tensor:
        """Convert the current state into a tensor based on the configuration."""

        lane_map: dict[str, LaneMeasures] = {lm.lane_id: lm for lm in state}

        unique_keys = lane_map.keys()
        if set(self.lane_order) != unique_keys:
            raise ValueError(
                f"Mismatch between expected lanes and state lanes. "
                f"Expected: {self.lane_order}, got: {list(unique_keys)}"
            )

        USE_NORMALISATION = bool(self.normalisation_params is not None)

        feature_matrix = []
        for lane_id in self.lane_order:
            lane_state: LaneMeasures = lane_map[lane_id]

            # Normalise features if required
            if USE_NORMALISATION:
                normalised_lane_state: NormalisedLaneMeasures = (
                    self._normalise_lane_measures(lane_state)
                )

            # Construct a row of features based on configuration where the features form the columns
            features = []
            if self.config.include_queue:
                value = float(
                    lane_state.queue
                    if not USE_NORMALISATION
                    else normalised_lane_state.queue_occupancy
                )
                features.append(value)

            if self.config.include_approach:
                value = float(
                    lane_state.approach
                    if not USE_NORMALISATION
                    else normalised_lane_state.approach_occupancy
                )
                features.append(value)

            if self.config.include_total_wait:
                value = float(
                    lane_state.total_wait_s
                    if not USE_NORMALISATION
                    else normalised_lane_state.avg_wait_time_fraction
                )
                features.append(value)

            if self.config.include_max_wait:
                value = float(
                    lane_state.max_wait_s
                    if not USE_NORMALISATION
                    else normalised_lane_state.max_wait_time_fraction
                )
                features.append(value)

            if self.config.include_total_speed:
                value = float(
                    lane_state.total_speed
                    if not USE_NORMALISATION
                    else (1 - normalised_lane_state.avg_speed_fraction)
                )  # Invert speed fraction to represent congestion
                features.append(value)

            # Append the row to the feature matrix
            feature_matrix.append(features)

        # Convert feature matrix to tensor
        return tensor(feature_matrix, dtype=float32)

    def num_features_per_lane(self) -> int:
        """Get the number of features per lane based on the configuration."""
        return sum(
            [
                self.config.include_queue,
                self.config.include_approach,
                self.config.include_total_wait,
                self.config.include_max_wait,
                self.config.include_total_speed,
            ]
        )
