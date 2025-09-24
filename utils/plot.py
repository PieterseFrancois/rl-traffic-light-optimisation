import numpy as np
from matplotlib import pyplot as plt

from pathlib import Path
from dataclasses import dataclass

from modules.intersection.memory import LogEntry


# -------------------------
# Metric configuration
# -------------------------

# Map short metric names to LogEntry attribute names
METRIC_ATTR: dict[str, str] = {
    "queue": "total_queue_length",
    "total_wait": "total_wait_s",
    "max_wait": "max_wait_s",
    "reward": "reward",
}

# Labels for axes/titles
YLABELS: dict[str, str] = {
    "queue": "total queue [veh]",
    "total_wait": "sum of waits [s]",
    "max_wait": "max wait [s]",
    "reward": "reward",
}


# -------------------------
# Plot selection masks
# -------------------------


@dataclass(frozen=True)
class PlotMask:
    """Toggle which metrics to plot."""

    queue: bool = False
    total_wait: bool = False
    max_wait: bool = False
    reward: bool = False


@dataclass(frozen=True)
class PlotSelection:
    """Separate masks for time-series and mean bar plots."""

    timeseries: PlotMask = PlotMask()
    means: PlotMask = PlotMask()


def _selected_metrics(mask: PlotMask) -> list[str]:
    return [k for k, v in vars(mask).items() if v]


# -------------------------
# Core helpers
# -------------------------


def ensure_dir(path: Path | str) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def agents_union(
    baseline: dict[str, list[LogEntry]],
    trained: dict[str, list[LogEntry]],
) -> list[str]:
    return sorted(set(baseline.keys()) | set(trained.keys()))


def _extract_metric_array(logs: list[LogEntry], metric: str) -> np.ndarray:
    """Extract a 1D array for a metric from a list of LogEntry."""
    if not logs:
        return np.empty((0,), dtype=float)
    attr = METRIC_ATTR[metric]
    return np.fromiter((getattr(e, attr) for e in logs), dtype=float)


def series_from_logs(logs: list[LogEntry], metrics: list[str]) -> dict[str, np.ndarray]:
    """Build arrays for time and the requested metrics."""
    if not logs:
        out = {"t": np.empty((0,), dtype=float)}
        out.update({m: np.empty((0,), dtype=float) for m in metrics})
        return out
    out = {"t": np.fromiter((e.t for e in logs), dtype=float)}
    for m in metrics:
        out[m] = _extract_metric_array(logs, m)
    return out


def compute_means(S: dict[str, list[LogEntry]], metric: str) -> dict[str, float]:
    """Mean of a metric per agent from LogEntry lists."""
    out: dict[str, float] = {}
    for aid, logs in S.items():
        arr = _extract_metric_array(logs, metric)
        out[aid] = float(np.mean(arr)) if arr.size else np.nan
    return out


# -------------------------
# Plotting primitives
# -------------------------


def plot_timeseries_for_agent(
    aid: str,
    baseline_logs: list[LogEntry],
    trained_logs: list[LogEntry],
    outdir: Path | str,
    metric: str,
    dpi: int,
    title_suffix: str | None = None,
) -> Path:
    """Line plot of baseline vs trained for one agent and one metric."""
    outdir = ensure_dir(outdir)
    Sb = series_from_logs(baseline_logs, [metric])
    St = series_from_logs(trained_logs, [metric])

    plt.figure()
    if Sb["t"].size and Sb[metric].size:
        plt.plot(Sb["t"], Sb[metric], label="baseline")
    if St["t"].size and St[metric].size:
        plt.plot(St["t"], St[metric], label="ppo")

    plt.xlabel("time [s]")
    plt.ylabel(YLABELS.get(metric, metric))
    suffix = f" {title_suffix}" if title_suffix else ""
    plt.title(f"{metric} over time ({aid}){suffix}")
    plt.legend()
    plt.tight_layout()
    out_path = Path(outdir) / f"{aid}_{metric}_timeseries.png"
    plt.savefig(out_path, dpi=dpi)
    plt.close()
    return out_path


def plot_timeseries_all_agents(
    baseline: dict[str, list[LogEntry]],
    trained: dict[str, list[LogEntry]],
    outdir: Path | str,
    metrics: list[str],
    dpi: int,
) -> list[Path]:
    """Timeseries plots for all agents and selected metrics."""
    out_paths: list[Path] = []
    for aid in agents_union(baseline, trained):
        b_logs = baseline.get(aid, [])
        t_logs = trained.get(aid, [])
        for m in metrics:
            out_paths.append(
                plot_timeseries_for_agent(aid, b_logs, t_logs, outdir, m, dpi=dpi)
            )
    return out_paths


def plot_means_bar(
    baseline: dict[str, list[LogEntry]],
    trained: dict[str, list[LogEntry]],
    outdir: Path | str,
    metric: str,
    dpi: int,
    title: str | None = None,
) -> Path:
    """Grouped bar chart of mean(metric) per agent for baseline vs trained."""
    outdir = ensure_dir(outdir)
    mb = compute_means(baseline, metric)
    mt = compute_means(trained, metric)
    agents = sorted(set(mb.keys()) | set(mt.keys()))
    x = np.arange(len(agents))
    width = 0.35

    plt.figure()
    plt.bar(x - width / 2, [mb.get(a, np.nan) for a in agents], width, label="baseline")
    plt.bar(x + width / 2, [mt.get(a, np.nan) for a in agents], width, label="ppo")
    plt.xticks(x, agents, rotation=30)
    plt.ylabel(f"mean {YLABELS.get(metric, metric)}")
    plt.title(title or f"Mean {metric} per agent")
    plt.legend()
    plt.tight_layout()
    out_path = Path(outdir) / f"means_{metric}.png"
    plt.savefig(out_path, dpi=dpi)
    plt.close()
    return out_path


# -------------------------
# High-level driver
# -------------------------


def plot_report(
    baseline: dict[str, list[LogEntry]],
    trained: dict[str, list[LogEntry]],
    outdir: Path | str,
    selection_mask: PlotSelection,
    dpi: int = 160,
) -> dict[str, list[Path]]:
    """
    Choose plots via boolean masks or explicit lists.
    If mask is provided, it overrides timeseries_metrics and mean_metrics.
    Returns paths of generated figures.
    """

    timeseries_metrics = _selected_metrics(selection_mask.timeseries)
    mean_metrics = _selected_metrics(selection_mask.means)

    out: dict[str, list[Path]] = {"timeseries": [], "means": []}
    out["timeseries"] = plot_timeseries_all_agents(
        baseline, trained, outdir, timeseries_metrics, dpi=dpi
    )
    for m in mean_metrics:
        out["means"].append(
            plot_means_bar(baseline, trained, outdir, metric=m, dpi=dpi)
        )
    return out
