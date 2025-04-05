import traci
import random

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from modules import lane_metrics, plotting
from modules.metrics_structs import SingleIntersectionMetrics

def _setup_flow_with_random_bursts(duration=100, base_interval=2.0, burst_chance=0.3):
    """
    Create a steady flow of vehicles with a chance of burst injection every 10s.
    - duration: total simulation time (in seconds)
    - base_interval: how often base vehicles are released
    - burst_chance: probability of burst every 10s
    """
    traci.route.add("route_NS", ["NJ", "JS"])

    vehicle_id = 0
    t = 0
    while t < duration:
        # Base flow vehicle
        traci.vehicle.add(f"veh{vehicle_id}", "route_NS", depart=t)
        vehicle_id += 1
        t += base_interval

        # Burst injection check every ~10s
        if int(t) % 10 == 0 and random.random() < burst_chance:
            burst_size = random.randint(10, 15)
            for b in range(burst_size):
                traci.vehicle.add(f"veh{vehicle_id}", "route_NS", depart=t + b * 0.3)
                vehicle_id += 1
            t += burst_size * 0.3  # slight pause after burst


def run():
    _setup_flow_with_random_bursts(duration=120, base_interval=5.0)

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
        "Burst Flow: Avg Wait Time",
        "Seconds",
        [0],
        [2],
    )
    plotting.plot_lane_metrics(
        data.steps, data.speed, data.phases, "Burst Flow: Speed", "m/s", [0], [2]
    )
    plotting.plot_lane_metrics(
        data.steps, data.queue, data.phases, "Burst Flow: Queue", "Vehicles", [0], [2]
    )
    plotting.show_plots()
