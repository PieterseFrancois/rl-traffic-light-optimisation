import traci
import random
import numpy as np

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from modules import lane_metrics, plotting
from modules.metrics_structs import SingleIntersectionMetrics


def _setup_traffic():
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
            "W_right": ["WJ", "JN"]
        }

        # Register routes in SUMO
        for route_id, edges in routes.items():
            traci.route.add(route_id, edges)

        # Vehicle distribution (tunable per direction)
        num_vehicles_per_direction = 30
        movement_probs = {
            "straight": 0.6,
            "left": 0.2,
            "right": 0.2
        }

        # Northbound vehicles
        for i in range(num_vehicles_per_direction):
            move = random.choices(["N_straight", "N_left", "N_right"],
                                weights=movement_probs.values())[0]
            traci.vehicle.add(f"veh_N_{i}", move, depart=i * 1.0)

        # Southbound vehicles
        for i in range(num_vehicles_per_direction):
            move = random.choices(["S_straight", "S_left", "S_right"],
                                weights=movement_probs.values())[0]
            traci.vehicle.add(f"veh_S_{i}", move, depart=i * 1.0)

        # Eastbound vehicles
        for i in range(num_vehicles_per_direction):
            move = random.choices(["E_straight", "E_left", "E_right"],
                                weights=movement_probs.values())[0]
            traci.vehicle.add(f"veh_E_{i}", move, depart=i * 1.0)

        # Westbound vehicles
        for i in range(num_vehicles_per_direction):
            move = random.choices(["W_straight", "W_left", "W_right"],
                                weights=movement_probs.values())[0]
            traci.vehicle.add(f"veh_W_{i}", move, depart=i * 1.0)

def run():
    _setup_traffic()

    lanes = {
        "North": "NJ_0",
        "East": "EJ_0",
        "South": "SJ_0",
        "West": "WJ_0"
    }

    # Create custom logic
    phases = [
        {"id": 0, "green_lanes": ["NJ_0", "SJ_0"], "duration": 30},
        {"id": 1, "green_lanes": ["EJ_0", "WJ_0"], "duration": 30},
    ]

    current_phase = 0
    phase_timer = 0
    green_extension = 10  # seconds
    queue_threshold = 5
    phase_changes = 0

    data = SingleIntersectionMetrics()

    while traci.simulation.getMinExpectedNumber() > 0:
        traci.simulationStep()
        step = len(data.steps) + 1
        data.steps.append(step)

        # Store current phase (manually managed)
        data.phases.append(phases[current_phase]["id"])

        # Gather metrics
        waits = lane_metrics.get_lane_average_waiting_times(lanes.values())
        speeds = lane_metrics.get_lane_speeds(lanes.values())
        queues = lane_metrics.get_lane_queue_lengths(lanes.values())

        for dir, lane_id in lanes.items():
            data.wait[dir].append(waits[lane_id])
            data.speed[dir].append(speeds[lane_id])
            data.queue[dir].append(queues[lane_id])

        # Phase control logic
        phase_timer += 1
        active_lanes = phases[current_phase]["green_lanes"]
        active_queues = sum(queues[l] for l in active_lanes)

        if phase_timer >= phases[current_phase]["duration"]:
            if active_queues >= queue_threshold and phase_timer < (phases[current_phase]["duration"] + green_extension):
                continue  # extend green
            # switch phase
            current_phase = (current_phase + 1) % len(phases)
            phase_timer = 0
            phase_changes += 1
            traci.trafficlight.setPhase("tlJ", phases[current_phase]["id"])

    print(f"Total Phase Changes: {phase_changes}")

    # Calculate average metrics
    avg_wait = np.mean([np.mean(data.wait[dir]) for dir in lanes])
    avg_speed = np.mean([np.mean(data.speed[dir]) for dir in lanes])
    avg_queue = np.mean([np.mean(data.queue[dir]) for dir in lanes])

    print(f"Average Wait Time: {avg_wait:.2f} seconds")
    print(f"Average Speed: {avg_speed:.2f} m/s")
    print(f"Average Queue Length: {avg_queue:.2f} vehicles")

    # Plot
    plotting.init_plot_layout(num_subplots=3)
    plotting.plot_lane_metrics(data.steps, data.wait, data.phases, "Dynamic Light: Wait Time", "Seconds", [0], [1])
    plotting.plot_lane_metrics(data.steps, data.speed, data.phases, "Dynamic Light: Speed", "m/s", [0], [1])
    plotting.plot_lane_metrics(data.steps, data.queue, data.phases, "Dynamic Light: Queue", "Vehicles", [0], [1])
    plotting.show_plots()
