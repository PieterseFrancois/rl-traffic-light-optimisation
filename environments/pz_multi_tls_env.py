import traci
import numpy as np
from torch import Tensor

from gymnasium import spaces
from pettingzoo.utils import ParallelEnv

from modules.intersection.intersection import (
    IntersectionConfig,
    IntersectionModule,
)
from modules.intersection.preprocessor import PreprocessorConfig as FeatureConfig
from modules.intersection.memory import LogEntry

from modules.traffic_graph import TrafficGraph, TrafficGraphConfig, NeighbourInfo
from modules.communication_bus import CommunicationBus

from dataclasses import asdict


from utils.sumo_helpers import start_sumo, close_sumo, SUMOConfig


class MultiTLSParallelEnv(ParallelEnv):

    def __init__(
        self,
        intersection_agent_configs: list[IntersectionConfig],
        feature_config: FeatureConfig,
        sumo_config: SUMOConfig,
        traffic_graph_config: TrafficGraphConfig,
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
        self._K_max: int = 0

        # Episode timing
        self._t0: float = 0.0
        self._t_end: float = 0.0
        self._agent_logs: dict[str, list[LogEntry]] = {}

        # Traffic graph
        self.traffic_graph = TrafficGraph(config=traffic_graph_config)
        self.traffic_graph.get_neighbour_table()  # Precompute neighbour table to build internal cache

        # Comm Bus
        NEIGHBOUR_LAG_DEFAULT = 0
        self.neighbour_lag_steps: int = NEIGHBOUR_LAG_DEFAULT
        self.comm_bus = CommunicationBus()

        self._sumo_step_size_s: float = 0.0

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

            self.comm_bus.register(
                tls_id=config.tls_id,
                memory_module=self._tls_agents[config.tls_id].memory_module,
            )

    def _decide_padding_sizes(self) -> None:
        """Decide L_max and F for this episode based on current agents."""
        self._L_max = max(agent.num_lanes() for agent in self._tls_agents.values())
        self._F = self._tls_agents[next(iter(self._tls_agents))].features_per_lane()

        # Neighbour info
        tab = self.traffic_graph.get_neighbour_table()
        self._K_max = max(1, max((len(v) for v in tab.values()), default=0))

    def _build_spaces_from_snapshot(self) -> None:
        """Build observation and action spaces for each agent based on a snapshot observation."""
        self._observation_spaces = {}
        self._action_spaces = {}

        for tls_id, agent in self._tls_agents.items():
            self._observation_spaces[tls_id] = spaces.Dict(
                {
                    # Base observation
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
                    # Neighbour info
                    "nbr_features": spaces.Box(
                        0.0,
                        1.0,
                        shape=(self._K_max, self._L_max, self._F),
                        dtype=np.float32,
                    ),
                    "nbr_lane_mask": spaces.Box(
                        0.0, 1.0, shape=(self._K_max, self._L_max), dtype=np.float32
                    ),
                    "nbr_mask": spaces.Box(
                        0.0, 1.0, shape=(self._K_max,), dtype=np.float32
                    ),
                    "nbr_discount": spaces.Box(
                        0.0, 1.0, shape=(self._K_max,), dtype=np.float32
                    ),
                }
            )
            self._action_spaces[tls_id] = spaces.Discrete(agent.n_actions)

    def _project_action(self, agent: IntersectionModule, action: int) -> int:
        """Map an invalid action to the first valid one under the current mask."""
        mask = agent.action_mask().astype(np.bool_)
        if action < 0 or action >= mask.size or not mask[action]:
            valid = np.flatnonzero(mask)
            print(
                f"[warn] Invalid action {action} for TLS {agent.tls_id}, valid: {valid}"
            )
            return int(valid[0]) if valid.size else 0
        return action

    def _build_base_observations(
        self, agent: IntersectionModule
    ) -> dict[str, np.ndarray]:
        lane_features = agent.get_observation()  # (L, F)
        L_i = int(lane_features.shape[0])
        F = int(lane_features.shape[1])

        if F != self._F:
            raise RuntimeError(
                f"Feature width mismatch for agent {agent.tls_id}: got {F}, expected {self._F}"
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

    def _query_time_for_bus(self) -> float:
        """Time to read from the bus based on neighbour_lag."""
        t_now = float(traci.simulation.getTime())
        if self.neighbour_lag_steps == 0:
            return t_now

        # Get sumo step size if not known
        if self._sumo_step_size_s <= 0:
            self._sumo_step_size_s = traci.simulation.getDeltaT()

        return max(self._t0, t_now - self.neighbour_lag_steps * self._sumo_step_size_s)

    def _build_neighbour_observations(
        self, agent: IntersectionModule
    ) -> dict[str, np.ndarray]:

        neighbour_table: dict[str, list[NeighbourInfo]] = (
            self.traffic_graph.get_neighbour_table()
        )  # Should be cached
        neighbours: list[NeighbourInfo] = neighbour_table.get(agent.tls_id, [])

        t_query: float = self._query_time_for_bus()

        # Init matrices for neighbour info
        nbr_features = np.zeros(
            (self._K_max, self._L_max, self._F), dtype=np.float32
        )  # (K_max, L_max, F)
        nbr_lane_mask = np.zeros(
            (self._K_max, self._L_max), dtype=np.float32
        )  # (K_max, L_max)
        nbr_mask = np.zeros((self._K_max,), dtype=np.float32)  # (K_max,)
        nbr_discount = np.zeros((self._K_max,), dtype=np.float32)  # (K_max,)

        # If no neighbours, return zero matrices
        if not neighbours:
            return {
                "nbr_features": nbr_features,
                "nbr_lane_mask": nbr_lane_mask,
                "nbr_mask": nbr_mask,
                "nbr_discount": nbr_discount,
            }

        # else, fill in neighbour info
        for k, nbr in enumerate(neighbours):
            if k >= self._K_max:
                break

            # Get neighbour entry from comm bus
            nbr_entry_dict: dict[str, LogEntry | None] = self.comm_bus.at(
                tls_ids=[nbr.node_id], t=t_query
            )
            nbr_entry: LogEntry | None = nbr_entry_dict.get(nbr.node_id, None)

            if nbr_entry is None:
                continue

            # Get neighbour agent for preprocessor use
            if nbr.node_id not in self._tls_agents:
                continue
            nbr_agent = self._tls_agents[nbr.node_id]
            state_tensor: Tensor = nbr_agent.preprocessor_module.get_state_tensor(
                state=nbr_entry.lane_measures
            )
            x_np: np.ndarray = nbr_agent.convert_tensor_to_numpy(state_tensor)  # (L, F)
            L_k = int(x_np.shape[0])

            nbr_features[k, :L_k, :] = x_np
            nbr_lane_mask[k, :L_k] = 1.0
            nbr_mask[k] = 1.0
            nbr_discount[k] = nbr.discount

        return {
            "nbr_features": nbr_features,
            "nbr_lane_mask": nbr_lane_mask,
            "nbr_mask": nbr_mask,
            "nbr_discount": nbr_discount,
        }

    def _build_observations(self) -> dict[str, dict[str, np.ndarray]]:

        # Build observations for all agents
        observations_dict = {}
        for tls_id in self.agents:
            agent = self._tls_agents[tls_id]
            observations_dict[tls_id] = self._build_base_observations(agent)

        # Add neighbour info
        for tls_id in self.agents:
            agent = self._tls_agents[tls_id]
            observations_dict[tls_id].update(self._build_neighbour_observations(agent))

        return observations_dict

    # ---- PettingZoo API: reset/step/close ---- #
    def reset(self, seed: int | None = None, options: dict | None = None):

        # Ensure comm_bus is clear
        self.comm_bus = CommunicationBus()

        # (Re)start SUMO
        close_sumo()
        start_sumo(self.sumo_config)

        # Clear logs
        self._agent_logs = {}

        # Build agents and mark them active
        self._build_tls_agents()

        # Episode timing
        self._t0 = float(traci.simulation.getTime())
        self._t_end = self._t0 + self.episode_length
        self._sumo_step_size_s = traci.simulation.getDeltaT()

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

        for tls_id, agent in self._tls_agents.items():
            # agent.log_to_memory(t=t_now) Now included in read_state()
            # Update logs with reward from this step
            agent.memory_module.set_latest_reward(rewards[tls_id])
            self._agent_logs[tls_id] = self._agent_logs.get(tls_id, []) + [
                agent.memory_module.get_latest()
            ]

        # Remove done agents
        if episode_done:
            self.agents = []  # required by PettingZoo API

        infos = {
            tls_id: (
                asdict(latest)
                if (latest := agent.memory_module.get_latest()) is not None
                else {}
            )
            for tls_id, agent in self._tls_agents.items()
        }

        return observations, rewards, terminations, truncations, infos

    def get_agent_logs(self) -> dict[str, list[LogEntry]]:
        return self._agent_logs

    def close(self):
        close_sumo()
