import traci
import modules.lane_metrics as lane_metrics
import modules.plotting as plotting
from modules.metrics_structs import SingleIntersectionMetrics


def _setup_single_vehicle():
    traci.route.add("route_EW", ["EJ", "JW"])
    traci.vehicle.add("veh0", "route_EW", depart=0.0)


def run():
    _setup_single_vehicle()

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
