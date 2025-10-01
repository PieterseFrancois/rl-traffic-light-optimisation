from pathlib import Path

from modules.intersection.memory import MemoryModule, LogEntry


class CommunicationBus:
    """
    A registry and access point for multiple MemoryModules, allowing
    cross-agent communication and data retrieval.
    """

    def __init__(self):
        self._modules: dict[str, MemoryModule] = {}

    # ---- Registration methods ---- #

    def register(self, tls_id: str, memory_module: MemoryModule) -> None:
        """Register a MemoryModule under a given traffic light ID."""
        self._modules[tls_id] = memory_module

    def unregister(self, tls_id: str) -> None:
        """Unregister a MemoryModule from a given traffic light ID."""
        self._modules.pop(tls_id, None)

    def ids(self) -> list[str]:
        """Get a list of all registered traffic light IDs."""
        return list(self._modules.keys())

    def get(self, tls_id: str) -> MemoryModule:
        """Get the MemoryModule registered under a given traffic light ID."""
        return self._modules[tls_id]

    # ---- Cross-agent reads ---- #

    def at(
        self, t: float, tls_ids: list[str] | None = None
    ) -> dict[str, LogEntry | None]:
        """Get the log entry at a specific time for all or specified traffic lights."""
        ids = self.ids() if tls_ids is None else tls_ids
        return {i: self._modules[i].get_log_at(t) for i in ids if i in self._modules}

    def latest(self, tls_ids: list[str] | None = None) -> dict[str, LogEntry | None]:
        """Get the latest log entry for all or specified traffic lights."""
        ids = self.ids() if tls_ids is None else tls_ids
        return {i: self._modules[i].get_latest() for i in ids if i in self._modules}

    def before(
        self, t: float, tls_ids: list[str] | None = None
    ) -> dict[str, list[LogEntry]]:
        """Get all log entries before a specific time for all or specified traffic lights."""
        ids = self.ids() if tls_ids is None else tls_ids
        return {
            i: self._modules[i].get_logs_before(t) for i in ids if i in self._modules
        }

    def after(
        self, t: float, tls_ids: list[str] | None = None
    ) -> dict[str, list[LogEntry]]:
        """Get all log entries after a specific time for all or specified traffic lights."""
        ids = self.ids() if tls_ids is None else tls_ids
        return {
            i: self._modules[i].get_logs_after(t) for i in ids if i in self._modules
        }

    def between(
        self, t_start: float, t_end: float, tls_ids: list[str] | None = None
    ) -> dict[str, list[LogEntry]]:
        """Get all log entries between two specific times for all or specified traffic lights."""
        ids = self.ids() if tls_ids is None else tls_ids
        return {
            i: self._modules[i].get_logs_between(t_start, t_end)
            for i in ids
            if i in self._modules
        }
