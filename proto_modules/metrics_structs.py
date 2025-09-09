from dataclasses import dataclass, field


@dataclass
class SingleIntersectionMetrics:
    """
    Holds time series simulation data for a single 4-way intersection.
    Tracks per-lane metrics like waiting time, speed, and queue length over time.
    """

    steps: list = field(default_factory=list)
    phases: list = field(default_factory=list)

    # Each metric is a dictionary of lists, keyed by lane direction (e.g., "North")
    wait: dict = field(
        default_factory=lambda: {"North": [], "East": [], "South": [], "West": []}
    )

    speed: dict = field(
        default_factory=lambda: {"North": [], "East": [], "South": [], "West": []}
    )

    queue: dict = field(
        default_factory=lambda: {"North": [], "East": [], "South": [], "West": []}
    )
