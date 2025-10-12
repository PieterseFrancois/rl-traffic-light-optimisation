import math
import csv
import xml.etree.ElementTree as ET
from dataclasses import dataclass, asdict
from pathlib import Path
from collections import defaultdict
import pandas as pd


# -----------------------------
# Data rows (typed / stable)
# -----------------------------


@dataclass(frozen=True)
class SummaryStepRow:
    time: float
    loaded: int
    inserted: int
    running: int
    waiting: int
    ended: int
    arrived: int
    collisions: int
    teleports: int
    halting: int
    stopped: int
    meanWaitingTime: float
    meanTravelTime: float
    meanSpeed: float
    meanSpeedRelative: float
    duration: int  # SUMO's "duration" field inside <step>


@dataclass(frozen=True)
class TripInfoRow:
    id: str
    depart: float
    arrival: float
    duration: float
    routeLength: float
    waitingTime: float
    waitingCount: int
    stopTime: float
    timeLoss: float
    departDelay: float
    speedFactor: float
    vType: str


@dataclass(frozen=True)
class TripInfoAggRow:
    """Per-second aggregate derived from tripinfo (binned by arrival time)."""

    t: int
    nArrived: int
    meanWaitingTime: float
    meanTimeLoss: float
    meanTravelTime: float


# -----------------------------
# Parsing helpers
# -----------------------------


def _f(x: str | None, default: float = 0.0) -> float:
    try:
        if x is None:
            return default
        return float(x)
    except Exception:
        return default


def _i(x: str | None, default: int = 0) -> int:
    try:
        if x is None:
            return default
        return int(float(x))  # SUMO sometimes stores ints as "0.00"
    except Exception:
        return default


def parse_summary_xml(path: Path) -> list[SummaryStepRow]:
    rows: list[SummaryStepRow] = []
    if not path.exists():
        return rows

    root = ET.parse(path).getroot()
    for step in root.iter("step"):
        a = step.attrib
        rows.append(
            SummaryStepRow(
                time=_f(a.get("time"), 0.0),
                loaded=_i(a.get("loaded"), 0),
                inserted=_i(a.get("inserted"), 0),
                running=_i(a.get("running"), 0),
                waiting=_i(a.get("waiting"), 0),
                ended=_i(a.get("ended"), 0),
                arrived=_i(a.get("arrived"), 0),
                collisions=_i(a.get("collisions"), 0),
                teleports=_i(a.get("teleports"), 0),
                halting=_i(a.get("halting"), 0),
                stopped=_i(a.get("stopped"), 0),
                meanWaitingTime=_f(a.get("meanWaitingTime"), math.nan),
                meanTravelTime=_f(a.get("meanTravelTime"), math.nan),
                meanSpeed=_f(a.get("meanSpeed"), math.nan),
                meanSpeedRelative=_f(a.get("meanSpeedRelative"), math.nan),
                duration=_i(a.get("duration"), 0),
            )
        )
    return rows


def parse_tripinfo_xml(path: Path) -> list[TripInfoRow]:
    rows: list[TripInfoRow] = []
    if not path.exists():
        return rows

    root = ET.parse(path).getroot()
    for ti in root.iter("tripinfo"):
        a = ti.attrib
        rows.append(
            TripInfoRow(
                id=a.get("id", ""),
                depart=_f(a.get("depart"), math.nan),
                arrival=_f(a.get("arrival"), math.nan),
                duration=_f(a.get("duration"), math.nan),
                routeLength=_f(a.get("routeLength"), math.nan),
                waitingTime=_f(a.get("waitingTime"), 0.0),
                waitingCount=_i(a.get("waitingCount"), 0),
                stopTime=_f(a.get("stopTime"), 0.0),
                timeLoss=_f(a.get("timeLoss"), 0.0),
                departDelay=_f(a.get("departDelay"), 0.0),
                speedFactor=_f(a.get("speedFactor"), math.nan),
                vType=a.get("vType", ""),
            )
        )
    return rows


def aggregate_tripinfo_per_second(trips: list[TripInfoRow]) -> list[TripInfoAggRow]:
    if not trips:
        return []
    bins: dict[int, dict[str, float]] = defaultdict(
        lambda: {"n": 0, "wait": 0.0, "loss": 0.0, "dur": 0.0}
    )
    for r in trips:
        if math.isnan(r.arrival):
            continue
        tbin = int(math.floor(r.arrival))
        b = bins[tbin]
        b["n"] += 1
        b["wait"] += r.waitingTime
        b["loss"] += r.timeLoss
        b["dur"] += r.duration

    out: list[TripInfoAggRow] = []
    for t in sorted(bins.keys()):
        b = bins[t]
        n = int(b["n"])
        out.append(
            TripInfoAggRow(
                t=t,
                nArrived=n,
                meanWaitingTime=(b["wait"] / n) if n else math.nan,
                meanTimeLoss=(b["loss"] / n) if n else math.nan,
                meanTravelTime=(b["dur"] / n) if n else math.nan,
            )
        )
    return out


# -----------------------------
# CSV writers
# -----------------------------


def write_csv_rows(path: Path, rows: list[dataclass]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        if not rows:
            f.write("")  # create empty file
            return
        fieldnames = list(asdict(rows[0]).keys())
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(asdict(r))


# -----------------------------
# KPI helpers (no NumPy)
# -----------------------------


def _mean(vals: list[float]) -> float:
    vals = [v for v in vals if not math.isnan(v)]
    return sum(vals) / len(vals) if vals else math.nan


def _percentile(vals: list[float], p: float) -> float:
    """
    Linear interpolation percentile in [0,100].
    Ignores NaNs. Returns NaN if empty.
    """
    xs = sorted(v for v in vals if not math.isnan(v))
    if not xs:
        return math.nan
    if p <= 0:
        return xs[0]
    if p >= 100:
        return xs[-1]
    k = (len(xs) - 1) * (p / 100.0)
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return xs[int(k)]
    return xs[f] + (xs[c] - xs[f]) * (k - f)


def _safe_div(a: float, b: float) -> float:
    return (
        a / b
        if b not in (0.0, 0) and not math.isnan(a) and not math.isnan(b)
        else math.nan
    )


# -----------------------------
# High-level manager
# -----------------------------


class NetworkResults:
    """
    Offline loader for:
      baseline_summary.xml / baseline_tripinfo.xml
      eval_summary.xml     / eval_tripinfo.xml
    found inside a given run directory.

    Provides:
      - parsed rows
      - per-second aggregates from tripinfo
      - CSV exporters
      - overall KPIs from tripinfo and summary (means, percentiles, TTI, etc.)
    """

    def __init__(self, run_dir: Path):
        self.run_dir = Path(run_dir)

        # Expected file names
        self.paths = {
            "baseline_summary": self.run_dir / "baseline_summary.xml",
            "baseline_tripinfo": self.run_dir / "baseline_tripinfo.xml",
            "eval_summary": self.run_dir / "eval_summary.xml",
            "eval_tripinfo": self.run_dir / "eval_tripinfo.xml",
        }

        # Loaded data
        self.baseline_summary: list[SummaryStepRow] = []
        self.baseline_tripinfo: list[TripInfoRow] = []
        self.eval_summary: list[SummaryStepRow] = []
        self.eval_tripinfo: list[TripInfoRow] = []

        # Aggregates (computed on demand)
        self._baseline_tripinfo_agg: list[TripInfoAggRow] | None = None
        self._eval_tripinfo_agg: list[TripInfoAggRow] | None = None

        # For KPI "better" direction. Keys are *unprefixed* metric names.
        self._BASE_HIGHER_IS_BETTER: dict[str, bool] = {
            # tripinfo KPIs
            "throughput": True,
            "meanTravelTime": False,
            "meanTimeLoss": False,
            "meanWaitingTime": False,
            "stopRate": False,          # avg waitingCount per trip
            "meanDepartDelay": False,
            "TTI_mean": False,
            "TTI_median": False,
            # summary KPIs
            "horizon_s": True,        
            "n_steps": True,     
            "meanSpeed": True,
            "meanSpeed_running_weighted": True,
            "meanHalting": False,
            "maxHalting": False,
            "totalArrived": True,
            "totalCollisions": False,
            "totalTeleports": False,
        }

    # ---- Helpers ----


    # Include percentile patterns: p50_TravelTime, p90_TimeLoss, p95_WaitingTime -> lower is better
    def _higher_is_better_for_key(self, metric_key: str) -> bool:
        # metric_key is like "trip_meanTravelTime" / "summary_meanSpeed" / "trip_p90_TimeLoss"
        name = metric_key.split("_", 1)[1] if "_" in metric_key else metric_key
        if name.startswith("p") and ("TravelTime" in name or "TimeLoss" in name or "WaitingTime" in name):
            return False
        return self._BASE_HIGHER_IS_BETTER.get(name, True)


    # ---- Loaders ----

    def load(self) -> None:
        """Load all four files (silently skips missing ones)."""
        self.baseline_summary = parse_summary_xml(self.paths["baseline_summary"])
        self.baseline_tripinfo = parse_tripinfo_xml(self.paths["baseline_tripinfo"])
        self.eval_summary = parse_summary_xml(self.paths["eval_summary"])
        self.eval_tripinfo = parse_tripinfo_xml(self.paths["eval_tripinfo"])
        # Invalidate caches
        self._baseline_tripinfo_agg = None
        self._eval_tripinfo_agg = None

    # ---- Aggregates ----

    def baseline_tripinfo_per_second(self) -> list[TripInfoAggRow]:
        if self._baseline_tripinfo_agg is None:
            self._baseline_tripinfo_agg = aggregate_tripinfo_per_second(
                self.baseline_tripinfo
            )
        return self._baseline_tripinfo_agg

    def eval_tripinfo_per_second(self) -> list[TripInfoAggRow]:
        if self._eval_tripinfo_agg is None:
            self._eval_tripinfo_agg = aggregate_tripinfo_per_second(self.eval_tripinfo)
        return self._eval_tripinfo_agg

    # ---- KPI computation (tripinfo) ----

    @staticmethod
    def _tripinfo_kpis(
        trips: list[TripInfoRow],
        *,
        freeflow_speed_mps: float | None = None,
        percentiles: tuple[float, ...] = (50, 90, 95),
    ) -> dict[str, float]:
        """
        KPIs based on per-vehicle data:
          - throughput (n trips)
          - meanTravelTime, meanTimeLoss, meanWaitingTime
          - stopRate (mean waitingCount)
          - departDelay mean
          - percentiles for duration/timeLoss/waitingTime
          - TTI (needs freeflow_speed_mps to compute per-trip free-flow time as routeLength / speed)
        """
        n = len(trips)
        if n == 0:
            out = {
                "throughput": 0,
                "meanTravelTime": math.nan,
                "meanTimeLoss": math.nan,
                "meanWaitingTime": math.nan,
                "stopRate": math.nan,
                "meanDepartDelay": math.nan,
            }
            # Add empty percentile keys
            for p in percentiles:
                out[f"p{int(p)}_TravelTime"] = math.nan
                out[f"p{int(p)}_TimeLoss"] = math.nan
                out[f"p{int(p)}_WaitingTime"] = math.nan
            # TTI fields
            out["TTI_mean"] = math.nan
            out["TTI_median"] = math.nan
            return out

        durations = [t.duration for t in trips]
        time_losses = [t.timeLoss for t in trips]
        waiting_times = [t.waitingTime for t in trips]
        waiting_counts = [float(t.waitingCount) for t in trips]
        depart_delays = [t.departDelay for t in trips]

        out: dict[str, float] = {
            "throughput": n,
            "meanTravelTime": _mean(durations),
            "meanTimeLoss": _mean(time_losses),
            "meanWaitingTime": _mean(waiting_times),
            "stopRate": _mean(waiting_counts),
            "meanDepartDelay": _mean(depart_delays),
        }

        # Percentiles
        for p in percentiles:
            out[f"p{int(p)}_TravelTime"] = _percentile(durations, p)
            out[f"p{int(p)}_TimeLoss"] = _percentile(time_losses, p)
            out[f"p{int(p)}_WaitingTime"] = _percentile(waiting_times, p)

        # TTI (if we can derive free-flow time from a constant free-flow speed)
        tti_vals: list[float] = []
        if freeflow_speed_mps and freeflow_speed_mps > 0.0:
            for t in trips:
                if not math.isnan(t.routeLength) and not math.isnan(t.duration):
                    ff_time = _safe_div(t.routeLength, freeflow_speed_mps)
                    if not math.isnan(ff_time) and ff_time > 0.0:
                        tti_vals.append(t.duration / ff_time)
        out["TTI_mean"] = _mean(tti_vals) if tti_vals else math.nan
        out["TTI_median"] = _percentile(tti_vals, 50) if tti_vals else math.nan

        return out

    # ---- KPI computation (summary) ----

    @staticmethod
    def _summary_kpis(steps: list[SummaryStepRow]) -> dict[str, float]:
        """
        KPIs from the per-step network summary:
          - horizon_s, n_steps
          - meanSpeed (simple), meanSpeed_running_weighted
          - meanHalting, maxHalting
          - totalArrived (from last step), totalCollisions, totalTeleports (sum across steps)
        """
        if not steps:
            return {
                "horizon_s": 0.0,
                "n_steps": 0,
                "meanSpeed": math.nan,
                "meanSpeed_running_weighted": math.nan,
                "meanHalting": math.nan,
                "maxHalting": math.nan,
                "totalArrived": 0,
                "totalCollisions": 0,
                "totalTeleports": 0,
            }

        # Horizon & count
        horizon_s = (
            (steps[-1].time - steps[0].time) if steps[-1].time >= steps[0].time else 0.0
        )
        n_steps = len(steps)

        # Speeds
        mean_speed = _mean([s.meanSpeed for s in steps])

        # Running-weighted mean speed (per-step weighting by running vehicles)
        sum_w = 0
        sum_ws = 0.0
        for s in steps:
            if not math.isnan(s.meanSpeed) and s.running > 0:
                sum_w += s.running
                sum_ws += s.meanSpeed * s.running
        mean_speed_weighted = _safe_div(sum_ws, float(sum_w))

        # Halting
        halts = [float(s.halting) for s in steps]
        mean_halting = _mean(halts)
        max_halting = max(halts) if halts else math.nan

        # Totals
        total_arrived = steps[-1].arrived  # cumulative
        total_collisions = sum(s.collisions for s in steps)
        total_teleports = sum(s.teleports for s in steps)

        return {
            "horizon_s": float(horizon_s),
            "n_steps": n_steps,
            "meanSpeed": mean_speed,
            "meanSpeed_running_weighted": mean_speed_weighted,
            "meanHalting": mean_halting,
            "maxHalting": max_halting,
            "totalArrived": int(total_arrived),
            "totalCollisions": int(total_collisions),
            "totalTeleports": int(total_teleports),
        }

    # ---- CSV exports ----

    def export_all_csv(self, out_dir: Path) -> None:
        """
        Writes:
          baseline_summary.csv
          baseline_tripinfo.csv
          baseline_tripinfo_agg.csv
          eval_summary.csv
          eval_tripinfo.csv
          eval_tripinfo_agg.csv
        """
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        write_csv_rows(out_dir / "baseline_summary.csv", self.baseline_summary)
        write_csv_rows(out_dir / "baseline_tripinfo.csv", self.baseline_tripinfo)
        write_csv_rows(
            out_dir / "baseline_tripinfo_agg.csv", self.baseline_tripinfo_per_second()
        )

        write_csv_rows(out_dir / "eval_summary.csv", self.eval_summary)
        write_csv_rows(out_dir / "eval_tripinfo.csv", self.eval_tripinfo)
        write_csv_rows(
            out_dir / "eval_tripinfo_agg.csv", self.eval_tripinfo_per_second()
        )

    def export_kpis_csv(
        self,
        out_dir: Path,
        *,
        freeflow_speed_mps: float | None = None,
        percentiles: tuple[float, ...] = (50, 90, 95),
    ) -> None:
        """
        Writes two small CSV files with overall KPIs:
          baseline_kpis.csv
          eval_kpis.csv
        """
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        base_kpis = self.baseline_kpis(
            freeflow_speed_mps=freeflow_speed_mps, percentiles=percentiles
        )
        eval_kpis = self.eval_kpis(
            freeflow_speed_mps=freeflow_speed_mps, percentiles=percentiles
        )

        # Dict -> one-row CSV
        def _write_one(path: Path, d: dict[str, float]) -> None:
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=list(d.keys()))
                w.writeheader()
                w.writerow(d)

        _write_one(out_dir / "baseline_kpis.csv", base_kpis)
        _write_one(out_dir / "eval_kpis.csv", eval_kpis)

    # ---- Public KPI getters ----

    def baseline_kpis(
        self,
        *,
        freeflow_speed_mps: float | None = None,
        percentiles: tuple[float, ...] = (50, 90, 95),
    ) -> dict[str, float]:
        """
        Merge tripinfo KPIs + summary KPIs for baseline.
        """
        trip_kpis = self._tripinfo_kpis(
            self.baseline_tripinfo,
            freeflow_speed_mps=freeflow_speed_mps,
            percentiles=percentiles,
        )
        sum_kpis = self._summary_kpis(self.baseline_summary)
        return {
            **{f"trip_{k}": v for k, v in trip_kpis.items()},
            **{f"summary_{k}": v for k, v in sum_kpis.items()},
        }

    def eval_kpis(
        self,
        *,
        freeflow_speed_mps: float | None = None,
        percentiles: tuple[float, ...] = (50, 90, 95),
    ) -> dict[str, float]:
        """
        Merge tripinfo KPIs + summary KPIs for eval.
        """
        trip_kpis = self._tripinfo_kpis(
            self.eval_tripinfo,
            freeflow_speed_mps=freeflow_speed_mps,
            percentiles=percentiles,
        )
        sum_kpis = self._summary_kpis(self.eval_summary)
        return {
            **{f"trip_{k}": v for k, v in trip_kpis.items()},
            **{f"summary_{k}": v for k, v in sum_kpis.items()},
        }
    
    def kpis_comparison_df(
        self,
        *,
        freeflow_speed_mps: float | None = None,
        percentiles: tuple[float, ...] = (50, 90, 95),
    ) -> pd.DataFrame:
        """
        Return a flat DataFrame with columns:
        baseline, eval, delta, pct_improvement, higher_is_better
        Index = KPI keys, e.g. 'trip_meanTravelTime', 'summary_meanSpeed', ...
        delta is signed so that positive means "improved" according to the chosen direction.
        """
        base = self.baseline_kpis(
            freeflow_speed_mps=freeflow_speed_mps, percentiles=percentiles
        )
        eva = self.eval_kpis(
            freeflow_speed_mps=freeflow_speed_mps, percentiles=percentiles
        )

        keys = sorted(set(base.keys()) | set(eva.keys()))
        rows = []
        for k in keys:
            b = base.get(k, math.nan)
            e = eva.get(k, math.nan)
            hib = self._higher_is_better_for_key(k)

            # Signed delta: positive = improvement
            delta = (e - b) if hib else (b - e)

            # Percentage improvement relative to baseline magnitude
            denom = abs(b) if (isinstance(b, (int, float)) and not math.isnan(b) and b != 0.0) else 1e-8
            pct = (delta / denom) * 100.0 if isinstance(delta, (int, float)) and not math.isnan(delta) else math.nan

            rows.append({
                "kpi": k,
                "baseline": b,
                "eval": e,
                "delta": delta,
                "pct_improvement": pct,
                "higher_is_better": hib,
            })

        df = pd.DataFrame(rows).set_index("kpi")
        return df

