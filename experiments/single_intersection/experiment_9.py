import traci
import random

import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from modules import lane_metrics, plotting
from modules.metrics_structs import SingleIntersectionMetrics


def _setup_turning_and_straight_flow_NS(num_vehicles=50):
    """Vehicles from North with mixed movements: straight, left, right."""
    traci.route.add("route_straight", ["NJ", "JS"])
    traci.route.add("route_left", ["NJ", "JE"])
    traci.route.add("route_right", ["NJ", "JW"])

    for i in range(num_vehicles):
        move = random.choices(
            ["route_straight", "route_left", "route_right"],
            weights=[0.6, 0.2, 0.2],  # 60% straight, 20% left, 20% right
            k=1,
        )[0]
        traci.vehicle.add(f"veh1{i}", move, depart=i * 1.0)


def _setup_turning_and_straight_flow_SN(num_vehicles=50):
    """Vehicles from South with mixed movements: straight, left, right."""
    traci.route.add("route_straight2", ["SJ", "JN"])
    traci.route.add("route_left2", ["SJ", "JE"])
    traci.route.add("route_right2", ["SJ", "JW"])

    for i in range(num_vehicles):
        move = random.choices(
            ["route_straight2", "route_left2", "route_right2"],
            weights=[0.6, 0.2, 0.2],  # 60% straight, 20% left, 20% right
            k=1,
        )[0]
        traci.vehicle.add(f"veh2{i}", move, depart=i * 1.0)


def run():
    _setup_turning_and_straight_flow_NS(num_vehicles=50)
    _setup_turning_and_straight_flow_SN(num_vehicles=50)

    lanes = {"North": "NJ_0", "East": "EJ_0", "South": "SJ_0", "West": "WJ_0"}

    data = SingleIntersectionMetrics()

    while traci.simulation.getMinExpectedNumber() > 0:
        traci.simulationStep()
        data.steps.append(len(data.steps) + 1)
        data.phases.append(traci.trafficlight.getPhase("tlJ"))

        waits = lane_metrics.get_lane_average_waiting_times(lanes.values())
        speeds = lane_metrics.get_lane_speeds(lanes.values())
        queues = lane_metrics.get_lane_queue_lengths(lanes.values())

        for dir, lane_id in lanes.items():
            data.wait[dir].append(waits[lane_id])
            data.speed[dir].append(speeds[lane_id])
            data.queue[dir].append(queues[lane_id])

    # Visualize
    plotting.init_plot_layout(num_subplots=3)
    plotting.plot_lane_metrics(
        data.steps,
        data.wait,
        data.phases,
        "Turn vs. Straight: Wait Time",
        "Seconds",
        [0],
        [2],
    )
    plotting.plot_lane_metrics(
        data.steps, data.speed, data.phases, "Turn vs. Straight: Speed", "m/s", [0], [2]
    )
    plotting.plot_lane_metrics(
        data.steps,
        data.queue,
        data.phases,
        "Turn vs. Straight: Queue",
        "Vehicles",
        [0],
        [2],
    )
    plotting.show_plots()
