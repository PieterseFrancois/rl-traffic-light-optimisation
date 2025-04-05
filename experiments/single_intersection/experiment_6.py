import traci

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from modules import lane_metrics, plotting
from modules.metrics_structs import SingleIntersectionMetrics


def _setup_asymmetric_flow(num_vehicles=30):
    """Spawn vehicles North→South (green direction) and East→West (starved direction)."""
    traci.route.add("route_NS", ["NJ", "JS"])
    traci.route.add("route_EW", ["EJ", "JW"])

    for i in range(num_vehicles):
        traci.vehicle.add(f"vehN{i}", "route_NS", depart=i * 1.0)
        traci.vehicle.add(f"vehE{i}", "route_EW", depart=i * 1.0)


def _set_asymmetric_traffic_light():

    # Always green for North-South (G = go), always red for East-West (r = stop)
    asymmetric_program = traci.trafficlight.Logic(
        programID="asymmetric-green-ns",
        type=0,  # static
        currentPhaseIndex=0,
        phases=[
            traci.trafficlight.Phase(
                duration=9999, state="GGgrrrGGgrrr"
            )  # Only NS directions green
        ],
    )
    traci.trafficlight.setProgramLogic("tlJ", asymmetric_program)


def run():
    _setup_asymmetric_flow(num_vehicles=30)
    _set_asymmetric_traffic_light()

    lanes = {"North": "NJ_0", "East": "EJ_0", "South": "SJ_0", "West": "WJ_0"}

    data = SingleIntersectionMetrics()

    while (
        traci.simulation.getMinExpectedNumber() > 0 and traci.simulation.getTime() < 500
    ):
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
        "Avg Wait Time: Asymmetric Signal",
        "Seconds",
        [0],
        [2],
    )
    plotting.plot_lane_metrics(
        data.steps, data.speed, data.phases, "Speed: Asymmetric Signal", "m/s", [0], [2]
    )
    plotting.plot_lane_metrics(
        data.steps,
        data.queue,
        data.phases,
        "Queue: Asymmetric Signal",
        "Vehicles",
        [0],
        [2],
    )
    plotting.show_plots()
