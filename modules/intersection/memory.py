import os
import csv
import json

import bisect
from dataclasses import dataclass, asdict

from .state import LaneMeasures


@dataclass
class LogEntry:
    """
    Immutable time-stamped log entry for one intersection at simulation time t.

    Attributes:
        t (float): TraCI simulation time.
        reward (float): scalar reward at time t.
        total_wait_s (float): Total waiting time across all lanes at time t.
        total_queue_length (int): Total queue length across all lanes at time t.
        max_wait_s (float): Maximum waiting time among all lanes at time t.
        lane_measures (list[LaneMeasures]): List of LaneMeasures for each approach lane at time t.
    """

    t: float
    reward: float
    total_wait_s: float
    total_queue_length: int
    max_wait_s: float
    lane_measures: list[LaneMeasures]


class MemoryModule:
    """
    Per-intersection logger + time-indexed retrieval (O(1) append, O(log N) lookup).
    - Rolling window via `max_records` (None = unlimited).
    - CSV export (arrays serialized as JSON strings).
    """

    def __init__(
        self,
        tls_id: str,
        max_records: int | None = None,
    ):
        self.tls_id = tls_id
        self.max_records = max_records
        self._logs: list[LogEntry] = []
        self._timestamps: list[float] = []

    def log(self, entry: LogEntry):
        """Append a new log entry."""
        # Ensure strictly increasing timestamps.
        if self._timestamps and entry.t <= self._timestamps[-1]:
            raise ValueError("Log entries must have strictly increasing timestamps.")

        self._logs.append(entry)
        self._timestamps.append(entry.t)

        # Enforce max_records if set.
        if self.max_records is not None and len(self._logs) > self.max_records:
            self._logs.pop(0)
            self._timestamps.pop(0)

    def get_timestamps(self) -> list[float]:
        """Get all recorded timestamps."""
        return self._timestamps.copy()

    def get_all_logs(self) -> list[LogEntry]:
        """Get all recorded log entries."""
        return self._logs.copy()

    def get_log_at(self, t: float) -> LogEntry | None:
        """Get the log entry at or before time t, or None if no such entry exists."""
        idx = bisect.bisect_right(self._timestamps, t) - 1
        return self._logs[idx] if idx >= 0 else None

    def get_logs_before(self, t: float) -> list[LogEntry]:
        """Get all log entries before and including time t."""
        idx = bisect.bisect_right(self._timestamps, t)
        return self._logs[:idx]

    def get_logs_after(self, t: float) -> list[LogEntry]:
        """Get all log entries after and including time t."""
        idx = bisect.bisect_left(self._timestamps, t)
        return self._logs[idx:]

    def get_logs_between(self, t_start: float, t_end: float) -> list[LogEntry]:
        """Get all log entries between and including t_start and t_end."""
        idx_start = bisect.bisect_left(self._timestamps, t_start)
        idx_end = bisect.bisect_right(self._timestamps, t_end)
        if idx_start < idx_end:
            return self._logs[idx_start:idx_end]
        return []

    def get_latest(self) -> LogEntry | None:
        return self._logs[-1] if self._logs else None

    def set_latest_reward(self, reward: float) -> None:
        """Set the reward of the latest log entry."""
        if not self._logs:
            raise ValueError("No log entries to update.")
        self._logs[-1] = LogEntry(
            t=self._logs[-1].t,
            reward=reward,
            total_wait_s=self._logs[-1].total_wait_s,
            total_queue_length=self._logs[-1].total_queue_length,
            max_wait_s=self._logs[-1].max_wait_s,
            lane_measures=self._logs[-1].lane_measures,
        )

    def export_csv(self, filepath: str) -> None:
        """Export all recorded logs to a CSV file."""
        dirpath = os.path.dirname(filepath)
        if dirpath:
            os.makedirs(dirpath, exist_ok=True)

        with open(filepath, mode="w", newline="") as csvfile:
            writer = csv.DictWriter(
                csvfile,
                fieldnames=[
                    "t",
                    "reward",
                    "total_wait_s",
                    "total_queue_length",
                    "max_wait_s",
                    "lane_measures",
                ],
            )
            writer.writeheader()
            for e in self._logs:
                writer.writerow(
                    {
                        "t": f"{e.t:.2f}",
                        "reward": f"{e.reward:.6f}",
                        "total_wait_s": f"{e.total_wait_s:.2f}",
                        "total_queue_length": str(e.total_queue_length),
                        "max_wait_s": f"{e.max_wait_s:.2f}",
                        "lane_measures": json.dumps(
                            [asdict(lm) for lm in e.lane_measures]
                        ),
                    }
                )

    def clear_before(self, t: float) -> None:
        """Clear all recorded logs before time t (exclusive)."""
        idx = bisect.bisect_left(self._timestamps, t)
        if idx > 0:
            del self._logs[:idx]
            del self._timestamps[:idx]

    def clear(self):
        """Clear all recorded logs."""
        self._logs.clear()
        self._timestamps.clear()
