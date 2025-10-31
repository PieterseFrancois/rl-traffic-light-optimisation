from collections import defaultdict

from enum import Enum


class EventNames(Enum):
    SIMULATION_INFO = "simulation_info"
    BASELINE_STARTED = "baseline_started"
    BASELINE_ENDED = "baseline_ended"
    EVALUATION_STARTED = "evaluation_started"
    EVALUATION_ENDED = "evaluation_ended"
    SIMULATION_DONE = "simulation_done"
    SIMULATION_FAILED = "simulation_failed"
    SIMULATION_KPIS_EVAL = "simulation_kpis_eval"
    SIMULATION_KPIS_BASELINE = "simulation_kpis_baseline"


class EventBus:
    """
    Components can subscribe to events by name,
    and publishers can emit events with arbitrary data.
    """

    def __init__(self):
        self._subscribers = defaultdict(list)

    def subscribe(self, event_name: str, callback):
        """Register a callback to an event."""
        self._subscribers[event_name].append(callback)

    def unsubscribe(self, event_name: str, callback):
        """Remove a callback."""
        if callback in self._subscribers[event_name]:
            self._subscribers[event_name].remove(callback)

    def emit(self, event_name: str, data=None):
        """Notify all subscribers of the event."""
        for callback in self._subscribers[event_name]:
            try:
                callback(data)
            except Exception as e:
                print(f"Error in event subscriber for {event_name}: {e}")
