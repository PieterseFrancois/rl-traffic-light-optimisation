import traci

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from modules import lane_metrics, plotting
from modules.metrics_structs import SingleIntersectionMetrics



def _setup_single_direction_flow(num_vehicles=30):
    """Spawn vehicles moving North to South only."""
    traci.route.add("route_EW", ["EJ", "JW"])
    for i in range(num_vehicles):
        traci.vehicle.add(f"veh{i}", "route_EW", depart=i * 1.0)


def run():
    _setup_single_direction_flow(num_vehicles=30)

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

    # Plotting
    plotting.init_plot_layout(num_subplots=3)
    plotting.plot_lane_metrics(
        data.steps,
        data.wait,
        data.phases,
        "Average Waiting Time per Lane",
        "Seconds",
        [0],
        [2],
    )
    plotting.plot_lane_metrics(
        data.steps, data.speed, data.phases, "Lane Speed", "m/s", [0], [2]
    )
    plotting.plot_lane_metrics(
        data.steps,
        data.queue,
        data.phases,
        "Queue Length per Lane",
        "Vehicles",
        [0],
        [2],
    )
    plotting.show_plots()
