from dataclasses import dataclass

from action import ActionModule, TLSTimingStandards
from state import StateModule
from reward import RewardModule, RewardNormalisationParameters, RewardFunction

@dataclass
class IntersectionConfig:
    """
    Configuration for an intersection module, including sub-modules.

    Attributes:
        tls_id (str): Traffic light system ID.
        max_detection_range_m (float): Maximum detection range for vehicles (in meters).
        green_phase_strings (list[str] | None): List of green phase strings for the action module. If None, will infer from SUMO.
        timing_standards (TLSTimingStandards | None): TLSTimingStandards for the action module. If None, no timing standards will be enforced.
        normalise_rewards (bool): Whether to normalise rewards in the reward module.
        average_vehicle_length_m (float): Average vehicle length (in meters) for normalisation in the reward module.
        reward_function (RewardFunction): Reward function to use in the reward module.
    """

    # General configuration for the intersection
    tls_id: str

    # Configuration for the state module
    max_detection_range_m: float = 50.0 # Maximum detection range for vehicles (in meters) from the traffic light (realism constraint)

    # Configuration for the action module
    green_phase_strings: list[str] | None = None # If None, will infer green phases from SUMO
    timing_standards: TLSTimingStandards | None = None # If None, no timing standards will be enforced

    # Configuration for the reward module
    normalise_rewards: bool = True # Whether to normalise rewards
    average_vehicle_length_m: float = 5 # Average vehicle length (in meters) for normalisation
    min_gap_between_vehicles_m: float = 2.5 # Minimum gap between vehicles (in meters) for normalisation
    reward_function: RewardFunction = RewardFunction.QUEUE # Default reward function


class IntersectionModule:
    def __init__(
        self,
        traci_connection,
        config: IntersectionConfig,
    ):
        # Initialise general attributes
        self.traci_connection = traci_connection
        self.tls_id = config.tls_id

        # Initialise sub-modules
        self._init_state_module(config)
        self._init_action_module(config)
        self._init_reward_module(config)


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
        normalisation_params = RewardNormalisationParameters(
            max_detection_range_m=config.max_detection_range_m,
            avg_vehicle_length_m=config.average_vehicle_length_m + config.min_gap_between_vehicles_m,
        ) if config.normalise_rewards else None

        self.reward_module = RewardModule(
            traci_connection=self.traci_connection,
            tls_id=self.tls_id,
            normalisation_params=normalisation_params,
        )
