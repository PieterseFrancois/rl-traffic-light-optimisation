from dataclasses import dataclass

from enum import Enum

from event_bus import EventNames
from modules.intersection.state import LaneMeasures


class RunMode(Enum):
    BASELINE = "Baseline"
    EVALUATION = "Evaluation"


@dataclass
class BaseIntersectionKPIs:
    """
    Base KPIs for an intersection.

    Attributes:
        total_wait_time_s (float): Total wait time of all vehicles (in seconds).
        total_queue_length (int): Total queue length of all vehicles.
        max_wait_time_s (float): Maximum wait time of any vehicle (in seconds).
    """

    total_wait_time_s: float
    total_queue_length: int
    max_wait_time_s: float


def get_intersection_kpi(lane_states: list[LaneMeasures]) -> BaseIntersectionKPIs:
    """Compute and return the current KPIs for the intersection."""
    total_wait_time_s = sum(lane.total_wait_s for lane in lane_states)
    total_queue_length = sum(lane.queue for lane in lane_states)
    max_wait_time_s = max((lane.max_wait_s for lane in lane_states), default=0.0)

    return BaseIntersectionKPIs(
        total_wait_time_s=total_wait_time_s,
        total_queue_length=total_queue_length,
        max_wait_time_s=max_wait_time_s,
    )


def collect_and_emit_kpis(event_bus, traci_conn, run_mode: RunMode):
    """
    Read KPIs from TraCI for the current timestep and emit a dict via the event bus.

    Args:
        event_bus: your shared bus instance
        traci_conn: a TraCI connection (e.g. from traci.getConnection() or the global traci module)

    Emits:
        EventNames.SIMULATION_KPIS with a payload dict
    """
    # Simulation time
    sim_time_s = traci_conn.simulation.getTime()

    # Vehicle set
    veh_ids = traci_conn.vehicle.getIDList()

    n = len(veh_ids)

    # Per-vehicle aggregates
    total_wait = 0.0
    total_time_loss = 0.0
    total_speed = 0.0
    total_acc_wait = 0.0

    if n:
        for vid in veh_ids:
            total_wait += float(traci_conn.vehicle.getWaitingTime(vid))

            tl = float(traci_conn.vehicle.getTimeLoss(vid))
            if tl < 0:
                tl = 0.0
            total_time_loss += tl

            total_speed += float(traci_conn.vehicle.getSpeed(vid))

            total_acc_wait += float(traci_conn.vehicle.getAccumulatedWaitingTime(vid))

    avg_wait = (total_wait / n) if n else 0.0
    avg_time_loss = (total_time_loss / n) if n else 0.0
    mean_speed = (total_speed / n) if n else 0.0
    avg_acc_wait = (total_acc_wait / n) if n else 0.0

    # Throughput and departures since last step
    arrived = int(traci_conn.simulation.getArrivedNumber())
    departed = int(traci_conn.simulation.getDepartedNumber())

    payload = {
        "sim_time": sim_time_s,
        "n_vehicles": n,
        "avg_wait": avg_wait,
        "avg_time_loss": avg_time_loss,
        "avg_acc_wait": avg_acc_wait,
        "mean_speed_vehicle": mean_speed,
        "throughput_arrived": arrived,
        "throughput_departed": departed,
    }

    # Emit on the bus
    try:
        if run_mode == RunMode.EVALUATION:
            event_bus.emit(EventNames.SIMULATION_KPIS_EVAL.value, payload)
        elif run_mode == RunMode.BASELINE:
            event_bus.emit(EventNames.SIMULATION_KPIS_BASELINE.value, payload)
    except Exception:
        pass

    return payload
