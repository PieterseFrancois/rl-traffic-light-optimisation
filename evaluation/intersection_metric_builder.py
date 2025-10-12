import numpy as np
import pandas as pd

from modules.intersection.memory import LogEntry

from dataclasses import dataclass


@dataclass
class MetricsReport:
    """
    Container for per-agent time series and per-metric means, plus labels.
    Use `MetricsReport.build_report(...)` to construct from baseline/eval logs.
    """

    # ---- payload ----
    agent_ids: list[str]
    # timeseries[agent_id][metric] = ((t_baseline, y_baseline), (t_eval, y_eval))
    timeseries: dict[
        str,
        dict[str, tuple[tuple[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]]],
    ]
    # means[metric] = {"agents": list[str], "baseline": np.ndarray, "eval": np.ndarray}
    means: dict[str, dict[str, object]]
    improvement_summary: pd.DataFrame
    # axis labels for metrics
    ylabels: dict[str, str]

    # ---- config (class-level) ----
    METRIC_ATTR: dict[str, str] = None
    YLABELS: dict[str, str] = None
    HIGHER_IS_BETTER: dict[str, bool] = None

    # ------------- public API -------------
    @classmethod
    def build_report(
        cls,
        baseline_logs_by_agent: dict[str, list[LogEntry]],
        eval_logs_by_agent: dict[str, list[LogEntry]],
        timeseries_metrics: list[str] | None = None,
        mean_metrics: list[str] | None = None,
    ) -> "MetricsReport":
        """
        Build a MetricsReport from per-agent LogEntry sequences.

        Args:
            baseline_logs_by_agent: {agent_id: [LogEntry, ...]}
            eval_logs_by_agent:     {agent_id: [LogEntry, ...]}  (your tuned/evaluation run)
            timeseries_metrics:     which metrics to include as time series
            mean_metrics:           which metrics to include as per-agent means

        Returns:
            MetricsReport
        """
        if timeseries_metrics is None:
            timeseries_metrics = list(cls.METRIC_ATTR.keys())
        if mean_metrics is None:
            mean_metrics = list(cls.METRIC_ATTR.keys())

        agent_ids = cls._agents_union(baseline_logs_by_agent, eval_logs_by_agent)

        # Per-agent time series
        ts: dict[
            str,
            dict[
                str, tuple[tuple[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]]
            ],
        ] = {}
        for aid in agent_ids:
            ts[aid] = {}
            for m in timeseries_metrics:
                tb, yb = cls._extract_metric_array(
                    baseline_logs_by_agent.get(aid, []), m
                )
                te, ye = cls._extract_metric_array(eval_logs_by_agent.get(aid, []), m)
                ts[aid][m] = ((tb, yb), (te, ye))

        # Per-metric means
        means: dict[str, dict[str, object]] = {}
        for m in mean_metrics:
            mb, me, aids = cls._compute_means(
                baseline_logs_by_agent, eval_logs_by_agent, m
            )

            hib = bool(cls.HIGHER_IS_BETTER.get(m, True))
            # signed delta so that positive means "improved"
            delta = (me - mb) if hib else (mb - me)
            # percentage improvement w.r.t baseline magnitude (avoid div-by-zero)
            denom = np.maximum(np.abs(mb), 1e-8)
            pct = (delta / denom) * 100.0

            means[m] = {
                "agents": aids,
                "baseline": mb,
                "eval": me,
                "delta": delta,  
                "pct_improvement": pct, 
                "higher_is_better": hib, 
            }

        # --- flat improvement summary: rows=agents, cols=metrics, values=% improvement ---
        rows: dict[str, dict[str, float]] = {}
        for metric, pack in means.items():
            aids = pack["agents"]
            pct = np.asarray(pack["pct_improvement"], dtype=float)
            for i, aid in enumerate(aids):
                rows.setdefault(aid, {})[metric] = float(pct[i]) if np.isfinite(pct[i]) else np.nan

        improvement_df = pd.DataFrame.from_dict(rows, orient="index")
        improvement_df = improvement_df.reindex(index=agent_ids)
        improvement_df = improvement_df[[m for m in mean_metrics if m in improvement_df.columns]]

        # Average row at the bottom
        avg_row = improvement_df.mean(axis=0, skipna=True)
        avg_row.name = "AVERAGE"
        improvement_df = pd.concat([improvement_df, avg_row.to_frame().T], axis=0)

        # Average column at the right
        improvement_df["AVERAGE"] = improvement_df.mean(axis=1, skipna=True)
        improvement_df = improvement_df.round(2)
        improvement_df.index.name = "agent_id"
        improvement_df.columns.name = "% improvement"

        return cls(
            agent_ids=agent_ids,
            timeseries=ts,
            means=means,
            ylabels=cls.YLABELS,
            improvement_summary=improvement_df,
        )

    # ------------- helpers (private) -------------
    @staticmethod
    def _agents_union(
        baseline: dict[str, list[LogEntry]],
        evald: dict[str, list[LogEntry]],
    ) -> list[str]:
        return sorted(set(baseline.keys()) | set(evald.keys()))

    @classmethod
    def _extract_metric_array(
        cls, logs: list[LogEntry], metric: str
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return (t, y) as float arrays for a single metric from a list of LogEntry."""
        if not logs:
            return np.array([]), np.array([])
        t = np.array([le.t for le in logs], dtype=float)
        attr = cls.METRIC_ATTR.get(metric, metric)
        y = np.array([getattr(le, attr) for le in logs], dtype=float)
        return t, y

    @classmethod
    def _compute_means(
        cls,
        baseline: dict[str, list[LogEntry]],
        evald: dict[str, list[LogEntry]],
        metric: str,
    ) -> tuple[np.ndarray, np.ndarray, list[str]]:
        """Per-agent means for baseline vs eval for one metric."""
        agent_ids = cls._agents_union(baseline, evald)
        mb, me = [], []
        for aid in agent_ids:
            _, yb = cls._extract_metric_array(baseline.get(aid, []), metric)
            _, ye = cls._extract_metric_array(evald.get(aid, []), metric)
            mb.append(float(np.nanmean(yb)) if yb.size else np.nan)
            me.append(float(np.nanmean(ye)) if ye.size else np.nan)
        return np.asarray(mb, dtype=float), np.asarray(me, dtype=float), agent_ids


MetricsReport.METRIC_ATTR = {
    "queue": "total_queue_length",
    # "approach": "approach",
    "total_wait": "total_wait_s",
    "max_wait": "max_wait_s",
    # "total_speed": "total_speed",
    # "reward": "reward",
}
MetricsReport.YLABELS = {
    "queue": "queue length [veh]",
    "approach": "approach count [veh]",
    "total_wait": "total wait [s]",
    "max_wait": "max wait [s]",
    "total_speed": "total speed [m/s]",
    "reward": "reward",
}
MetricsReport.HIGHER_IS_BETTER = {
    "queue": False,
    "approach": False,
    "total_wait": False,
    "max_wait": False,
    "total_speed": True,
    "reward": True,
}
