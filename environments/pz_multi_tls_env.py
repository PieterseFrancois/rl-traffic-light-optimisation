import traci
import numpy as np

from gymnasium import spaces
from pettingzoo.utils import ParallelEnv

from dataclasses import dataclass

from modules.intersection.intersection import (
    IntersectionConfig,
    BaseIntersectionKPIs,
    IntersectionModule,
)
from modules.intersection.preprocessor import PreprocessorConfig as FeatureConfig


from .sumo_helpers import start_sumo, close_sumo, SUMOConfig


@dataclass
class IntersectionKPIs(BaseIntersectionKPIs):
    """
    KPIs for a traffic light intersection, extending BaseIntersectionKPIs.

    Attributes:
        timestamp (float): Simulation time when the KPIs were recorded (in seconds).

        total_wait_time_s (float): Total wait time of all vehicles (in seconds).
        total_queue_length (int): Total queue length of all vehicles.
        max_wait_time_s (float): Maximum wait time of any vehicle (in seconds).
    """

    timestamp: float


class MultiTLSParallelEnv(ParallelEnv):

    def __init__(
        self,
        intersection_agent_configs: list[IntersectionConfig],
        feature_config: FeatureConfig,
        sumo_config: SUMOConfig,
        episode_length: int,
        ticks_per_decision: int = 1,
    ):
        super().__init__()

        if episode_length <= 0 or ticks_per_decision <= 0:
            raise ValueError(
                "ParallelEnv episode_length and ticks_per_decision must be positive integers"
            )

        self.sumo_config: SUMOConfig = sumo_config
        self.episode_length: int = episode_length
        self.ticks_per_decision: int = ticks_per_decision
        self.intersection_agent_configs: list[IntersectionConfig] = (
            intersection_agent_configs
        )
        self.feature_config: FeatureConfig = feature_config  # Same for all agents

        # Runtime initialisation
        self.possible_agents: list[str] = [
            cfg.tls_id for cfg in intersection_agent_configs
        ]
        self.agents: list[str] = []
        self._tls_agents: dict[str, IntersectionModule] = {}
        self._observation_spaces: dict[str, spaces.Box] = {}
        self._action_spaces: dict[str, spaces.Discrete] = {}

        # Padding sizes decided at reset
        self._L_max: int = 0
        self._F: int = 0

        # Episode timing
        self._t0: float = 0.0
        self._t_end: float = 0.0

    # ---- PettingZoo API spaces ---- #
    def observation_space(self, agent: str):
        return self._observation_spaces[agent]

    def action_space(self, agent: str):
        return self._action_spaces[agent]

    # ---- Helpers ---- #

    def _build_tls_agents(self) -> None:
        """Build IntersectionModules for each traffic light system (TLS) in the environment."""

        # Reset runtime initialisation
        self.agents = []
        self._tls_agents: dict[str, IntersectionModule] = {}

        # Build agents
        for config in self.intersection_agent_configs:
            if config.tls_id in self.agents:
                raise ValueError(f"Duplicate TLS ID found: {config.tls_id}")

            self.agents.append(config.tls_id)

            self._tls_agents[config.tls_id] = IntersectionModule(
                traci_connection=traci,
                config=config,
                feature_config=self.feature_config,
            )

    def _decide_padding_sizes(self) -> None:
        """Decide L_max and F for this episode based on current agents."""
        self._L_max = max(agent.num_lanes() for agent in self._tls_agents.values())
        self._F = self._tls_agents[next(iter(self._tls_agents))].features_per_lane()

    def _build_spaces_from_snapshot(self) -> None:
        """Build observation and action spaces for each agent based on a snapshot observation."""
        self._observation_spaces = {}
        self._action_spaces = {}

        for tls_id, agent in self._tls_agents.items():
            self._observation_spaces[tls_id] = spaces.Dict(
                {
                    # Keep bounds broad; values are typically in [0,1] after preprocessing
                    "lane_features": spaces.Box(
                        low=0.0,
                        high=1.0,
                        shape=(self._L_max, self._F),
                        dtype=np.float32,
                    ),
                    "lane_mask": spaces.Box(
                        low=0.0, high=1.0, shape=(self._L_max,), dtype=np.float32
                    ),
                    "action_mask": spaces.Box(
                        low=0.0, high=1.0, shape=(agent.n_actions,), dtype=np.float32
                    ),
                }
            )
            self._action_spaces[tls_id] = spaces.Discrete(agent.n_actions)

    def _project_action(self, agent: IntersectionModule, action: int) -> int:
        """Map an invalid action to the first valid one under the current mask."""
        mask = agent.action_mask().astype(np.bool_)
        if action < 0 or action >= mask.size or not mask[action]:
            valid = np.flatnonzero(mask)
            return int(valid[0]) if valid.size else 0
        return action

    def _build_observation(self, tls_id: str) -> dict[str, np.ndarray]:
        agent = self._tls_agents[tls_id]

        lane_features = agent.get_observation()  # (L, F)
        L_i = int(lane_features.shape[0])
        F = int(lane_features.shape[1])

        if F != self._F:
            raise RuntimeError(
                f"Feature width mismatch for agent {tls_id}: got {F}, expected {self._F}"
            )

        # Create a full zero array and copy in the real features
        padded_features = np.zeros(
            (self._L_max, self._F), dtype=np.float32
        )  # (L_max, F)
        padded_features[:L_i, :] = lane_features

        lane_mask = np.zeros((self._L_max,), dtype=np.float32)  # (L_max,)
        lane_mask[:L_i] = 1.0

        return {
            "lane_features": padded_features,
            "lane_mask": lane_mask,
            "action_mask": agent.action_mask(),
        }

    def _build_observations(self) -> dict[str, dict[str, np.ndarray]]:
        return {tls_id: self._build_observation(tls_id) for tls_id in self.agents}

    # ---- PettingZoo API: reset/step/close ---- #
    def reset(self, seed: int | None = None, options: dict | None = None):
        close_sumo()
        start_sumo(self.sumo_config)

        # Build agents and mark them active
        self._build_tls_agents()

        # Episode timing
        self._t0 = float(traci.simulation.getTime())
        self._t_end = self._t0 + self.episode_length

        # Sync controllers, tick once, and snapshot state
        for agent in self._tls_agents.values():
            agent.tick()
        traci.simulationStep()
        for agent in self._tls_agents.values():
            agent.read_state()

        # Decide padding sizes and build spaces
        self._decide_padding_sizes()
        self._build_spaces_from_snapshot()

        # First observation
        observations: dict[str, dict[str, np.ndarray]] = self._build_observations()
        infos = {tls_id: {} for tls_id in self.agents}
        return observations, infos

    def step(self, actions: dict[str, int]):
        # Ensure all agents tick to check for readiness
        for agent in self._tls_agents.values():
            agent.tick()

        # Apply actions for ready agents, projecting invalid actions
        for tls_id, action in actions.items():
            if tls_id not in self.agents:
                # raise ValueError(f"Action provided for inactive agent: {tls_id}")
                continue

            agent = self._tls_agents[tls_id]
            if agent.ready():
                projected_action = self._project_action(agent, action)
                agent.apply_action(projected_action)

        # Advance the simulation to next decision boundary
        for _ in range(self.ticks_per_decision):
            traci.simulationStep()
            for agent in self._tls_agents.values():
                agent.tick()

        # Read new state
        for agent in self._tls_agents.values():
            agent.read_state()

        # Build observations, rewards, terminations, truncations, metrics
        observations: dict[str, dict[str, np.ndarray]] = self._build_observations()
        rewards: dict[str, float] = {
            tls_id: agent.get_reward() for tls_id, agent in self._tls_agents.items()
        }

        t_now = float(traci.simulation.getTime())
        episode_done = bool(
            t_now >= self._t_end or traci.simulation.getMinExpectedNumber() <= 0
        )

        terminations: dict[str, bool] = {tls_id: episode_done for tls_id in self.agents}
        truncations: dict[str, bool] = {tls_id: False for tls_id in self.agents}

        tracked_metrics = {}
        infos = {}
        for tls_id, agent in self._tls_agents.items():
            kpis: BaseIntersectionKPIs = agent.get_kpi()
            tracked_metrics[tls_id] = IntersectionKPIs(
                timestamp=t_now,
                total_wait_time_s=kpis.total_wait_time_s,
                total_queue_length=kpis.total_queue_length,
                max_wait_time_s=kpis.max_wait_time_s,
            )

            # CConvert intersection kpis to plain dict for info
            infos[tls_id] = tracked_metrics[tls_id].__dict__

        # Remove done agents
        if episode_done:
            self.agents = []  # required by PettingZoo API

        return observations, rewards, terminations, truncations, infos

    def close(self):
        close_sumo()
