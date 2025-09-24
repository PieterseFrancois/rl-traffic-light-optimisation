from dataclasses import dataclass

from modules.intersection.state import LaneMeasures


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
