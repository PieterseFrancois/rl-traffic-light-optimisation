import traci
import random

import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from modules import lane_metrics, plotting
from modules.metrics_structs import SingleIntersectionMetrics


def _setup_random_arrivals(num_vehicles=60):
    """Vehicles depart randomly from all directions with jittered timing."""
    traci.route.add("route_NS", ["NJ", "JS"])
    traci.route.add("route_SN", ["SJ", "JN"])
    traci.route.add("route_EW", ["EJ", "JW"])
    traci.route.add("route_WE", ["WJ", "JE"])

    for i in range(num_vehicles):
        route = random.choice(["route_NS", "route_SN", "route_EW", "route_WE"])
        depart_time = random.uniform(0, 90)  # random between 0 and 90s
        traci.vehicle.add(f"veh{i}", route, depart=depart_time)


def run():
    _setup_random_arrivals(num_vehicles=60)

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

    plotting.init_plot_layout(num_subplots=3)
    plotting.plot_lane_metrics(
        data.steps,
        data.wait,
        data.phases,
        "Random Arrival: Avg Wait Time",
        "Seconds",
        [0],
        [2],
    )
    plotting.plot_lane_metrics(
        data.steps, data.speed, data.phases, "Random Arrival: Speed", "m/s", [0], [2]
    )
    plotting.plot_lane_metrics(
        data.steps,
        data.queue,
        data.phases,
        "Random Arrival: Queue",
        "Vehicles",
        [0],
        [2],
    )
    plotting.show_plots()
