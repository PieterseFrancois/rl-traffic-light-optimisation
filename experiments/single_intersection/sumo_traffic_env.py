import gymnasium
from gymnasium import spaces
import numpy as np
import traci
import random
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from modules import lane_metrics
from modules.metrics_structs import SingleIntersectionMetrics


class SumoTrafficEnv(gymnasium.Env):
    def __init__(self):
        super(SumoTrafficEnv, self).__init__()

        self.lanes = {"North": "NJ_0", "East": "EJ_0", "South": "SJ_0", "West": "WJ_0"}
        self.lane_ids = list(self.lanes.values())
        self.phases = [
            {"id": 0, "green_lanes": ["NJ_0", "SJ_0"]},  # NS green
            {"id": 1, "green_lanes": ["EJ_0", "WJ_0"]},  # EW green
        ]
        self.current_phase = 0

        # Action = which phase to apply (0 or 1)
        self.action_space = spaces.Discrete(len(self.phases))

        # Observation = queue lengths on all 4 lanes
        self.observation_space = spaces.Box(
            low=0, high=100, shape=(4,), dtype=np.float32
        )

        self.metrics = SingleIntersectionMetrics()
        self.sim_step = 0
        self.steps_in_current_phase = 0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        if traci.isLoaded():
            traci.close()

        sumo_binary = "sumo-gui"  # or "sumo-gui" if you want visuals
        sumo_config = "experiments/single_intersection/sumo_files/single-intersection-single-lane.sumocfg"  # Replace with the actual path
        traci.start([sumo_binary, "-c", sumo_config])

        self._setup_traffic()
        self.sim_step = 0
        self.current_phase = 0
        self.steps_in_current_phase = 0

        traci.trafficlight.setPhase("tlJ", self.current_phase)

        return self._get_observation(), {}

    def step(self, action):
        min_phase_duration = (
            10  # minimum time (in simulation steps) a phase must be active
        )

        # Check if we are allowed to switch the phase
        if action != self.current_phase:
            if self.steps_in_current_phase >= min_phase_duration:
                self.current_phase = action
                traci.trafficlight.setPhase("tlJ", self.current_phase)
                self.steps_in_current_phase = 0  # reset phase timer
            else:
                action = self.current_phase  # override agent's action if too soon

        # Advance simulation by 1 step (you can batch steps outside this method if needed)
        traci.simulationStep()
        self.sim_step += 1
        self.steps_in_current_phase += 1

        obs = self._get_observation()
        reward = float(-np.sum(obs))
        done = traci.simulation.getMinExpectedNumber() == 0
        truncated = False
        info = {}

        return obs, reward, done, truncated, info

    def _get_observation(self):
        return np.array(
            [traci.lane.getLastStepVehicleNumber(lane_id) for lane_id in self.lane_ids],
            dtype=np.float32,
        )

    def render(self, mode="human"):
        pass  # optional

    def close(self):
        traci.close()

    def _setup_traffic(self):
        """Spawn vehicles from all four directions with mixed movements."""

        # Define all routes
        routes = {
            "N_straight": ["NJ", "JS"],
            "N_left": ["NJ", "JE"],
            "N_right": ["NJ", "JW"],
            "S_straight": ["SJ", "JN"],
            "S_left": ["SJ", "JE"],
            "S_right": ["SJ", "JW"],
            "E_straight": ["EJ", "JW"],
            "E_left": ["EJ", "JN"],
            "E_right": ["EJ", "JS"],
            "W_straight": ["WJ", "JE"],
            "W_left": ["WJ", "JS"],
            "W_right": ["WJ", "JN"],
        }

        # Register routes in SUMO
        for route_id, edges in routes.items():
            traci.route.add(route_id, edges)

        # Vehicle distribution (tunable per direction)
        num_vehicles_per_direction = 30
        movement_probs = {"straight": 0.6, "left": 0.2, "right": 0.2}

        # Northbound vehicles
        for i in range(num_vehicles_per_direction):
            move = random.choices(
                ["N_straight", "N_left", "N_right"], weights=movement_probs.values()
            )[0]
            traci.vehicle.add(f"veh_N_{i}", move, depart=i * 1.0)

        # Southbound vehicles
        for i in range(num_vehicles_per_direction):
            move = random.choices(
                ["S_straight", "S_left", "S_right"], weights=movement_probs.values()
            )[0]
            traci.vehicle.add(f"veh_S_{i}", move, depart=i * 1.0)

        # Eastbound vehicles
        for i in range(num_vehicles_per_direction):
            move = random.choices(
                ["E_straight", "E_left", "E_right"], weights=movement_probs.values()
            )[0]
            traci.vehicle.add(f"veh_E_{i}", move, depart=i * 1.0)

        # Westbound vehicles
        for i in range(num_vehicles_per_direction):
            move = random.choices(
                ["W_straight", "W_left", "W_right"], weights=movement_probs.values()
            )[0]
            traci.vehicle.add(f"veh_W_{i}", move, depart=i * 1.0)
