# ============================================================
#  DISCLAIMER:
#  This manual test case was written by guiding an LLM (GPT-5 Thinking) and then refining it.
#  The submodules imported and tested here were, however, developed and written individually.
# ============================================================

import os
import sys
import optparse
from pathlib import Path
import csv
import numpy as np


# ---------- SUMO tooling bootstrap ----------
def _ensure_sumo_tools_in_path():
    if "SUMO_HOME" in os.environ:
        tools = os.path.join(os.environ["SUMO_HOME"], "tools")
        if tools not in sys.path:
            sys.path.append(tools)
    else:
        sys.exit(
            'Please set environment variable "SUMO_HOME" to your SUMO installation path.'
        )


_ensure_sumo_tools_in_path()
from sumolib import checkBinary  # noqa: E402
import traci  # noqa: E402
from modules.intersection.action import ActionModule, TLSTimingStandards
from modules.intersection.state import StateModule


# ---------- CLI options ----------
def get_options():
    opt_parser = optparse.OptionParser()
    opt_parser.add_option(
        "--nogui", action="store_true", default=False, help="Run headless SUMO (no GUI)"
    )
    opt_parser.add_option(
        "--sumocfg",
        type="string",
        default="environments/single_intersection/sumo_files/single-intersection-single-lane.sumocfg",
        help="Path to .sumocfg",
    )
    opt_parser.add_option(
        "--tls-id", type="string", default="tlJ", help="Traffic light ID"
    )
    opt_parser.add_option(
        "--seconds", type="float", default=600.0, help="Sim length per run [s]"
    )
    opt_parser.add_option(
        "--fixed-green-s",
        type="float",
        default=10.0,
        help="Fixed green per phase for the baseline [s]",
    )
    opt_parser.add_option(
        "--out-dir",
        type="string",
        default=".state_action_adaptive_control_logs",
        help="Output dir for CSV logs",
    )
    opt_parser.add_option("--seed", type="int", default=None, help="SUMO random seed")
    opt_parser.add_option(
        "--plot", action="store_true", default=False, help="Show comparison plots"
    )

    # Timing standards for ActionModule
    opt_parser.add_option(
        "--min-green", type="float", default=10.0, help="Min green [s]"
    )
    opt_parser.add_option("--yellow", type="float", default=3.0, help="Yellow [s]")
    opt_parser.add_option("--all-red", type="float", default=2.0, help="All-red [s]")
    opt_parser.add_option(
        "--max-green", type="float", default=30.0, help="Max green [s] (0 disables)"
    )

    # StateModule sensing
    opt_parser.add_option(
        "--max-det-range",
        type="float",
        default=50.0,
        help="Max detection range to TLS [m]",
    )

    options, _ = opt_parser.parse_args()
    return options


# ---------- SUMO lifecycle ----------
def start_sumo(sumocfg: str, gui: bool, seed: int | None):
    bin_name = "sumo-gui" if gui else "sumo"
    sumo_binary = checkBinary(bin_name)
    cmd = [
        sumo_binary,
        "-c",
        sumocfg,
        "--start",
    ]
    if seed is not None:
        cmd += ["--seed", str(seed)]
    traci.start(cmd)


def close_sumo():
    try:
        traci.close(False)
    except Exception:
        pass


# ---------- Helpers for rule-based scoring ----------
_GREEN_CHARS = {"G", "g"}


def phase_to_inbound_lanes(am: ActionModule) -> dict[int, list[str]]:
    """
    Build phase_index -> inbound lane_ids mapping using SUMO's controlled links.
    The i-th char in the TLS state corresponds to the i-th link group returned by getControlledLinks.
    """
    links = traci.trafficlight.getControlledLinks(
        am.tls_id
    )  # list[list[(inLane, outLane, via)]]
    mapping: dict[int, list[str]] = {}
    for idx, pstr in am.green_phase_map.items():
        open_inbounds = []
        for sig_idx, link_group in enumerate(links):
            if not link_group:
                continue
            if pstr[sig_idx] in _GREEN_CHARS:
                in_lane = link_group[0][0]  # inbound lane for this signal index
                open_inbounds.append(in_lane)
        # de-dup while preserving order
        seen = set()
        ordered = []
        for ln in open_inbounds:
            if ln not in seen:
                ordered.append(ln)
                seen.add(ln)
        mapping[idx] = ordered
    return mapping


def score_phase_by_queue(
    phase_to_lanes: dict[int, list[str]], lane_measures
) -> dict[int, int]:
    q_by_lane = {lm.lane_id: lm.queue for lm in lane_measures}
    return {
        idx: int(sum(q_by_lane.get(lid, 0) for lid in lanes))
        for idx, lanes in phase_to_lanes.items()
    }


def argmax_tiebreak(scores: dict[int, int], prefer_keep: int | None):
    """
    Pick phase with the largest score.
    If tied and prefer_keep is provided, prefer switching away (i.e., pick the smallest index != prefer_keep).
    Deterministic result.
    """
    maxv = max(scores.values())
    cands = [k for k, v in scores.items() if v == maxv]
    if prefer_keep is not None and prefer_keep in cands and len(cands) > 1:
        cands.remove(prefer_keep)  # prefer switching when tie
    return min(cands)


# ---------- Controllers ----------
def adaptive_step(am: ActionModule, sm: StateModule):
    """
    1) Read state
    2) If ready, choose phase with max summed inbound queue
    3) Tick transition engine
    Returns (total_queue, total_wait_s)
    """
    lanes = sm.read_state()
    total_queue = sum(lm.queue for lm in lanes)
    total_wait = sum(lm.total_wait_s for lm in lanes)

    if am.ready_for_decision():
        pmap = phase_to_inbound_lanes(am)
        scores = score_phase_by_queue(pmap, lanes)
        prefer = am.active_phase_memory.phase_index if am.active_phase_memory else None
        target = argmax_tiebreak(scores, prefer_keep=prefer)
        am.set_phase(target)

    am.update_transition()
    return total_queue, total_wait


def make_baseline(fixed_green_s: float = 10.0):
    last_switch_time = None

    def step(am: ActionModule, sm: StateModule):
        nonlocal last_switch_time
        lanes = sm.read_state()
        total_queue = sum(lm.queue for lm in lanes)
        total_wait = sum(lm.total_wait_s for lm in lanes)

        now = float(sm.traci.simulation.getTime())
        if am.ready_for_decision():
            if last_switch_time is None:
                am.set_phase(0)
                last_switch_time = now
            elif now - last_switch_time >= fixed_green_s:
                if am.active_phase_memory is None:
                    am.set_phase(0)
                else:
                    cur = am.active_phase_memory.phase_index
                    nxt = (cur + 1) % am.n_actions if am.n_actions > 0 else 0
                    am.set_phase(nxt)
                last_switch_time = now

        am.update_transition()
        return total_queue, total_wait

    return step


# ---------- Runner and plotting ----------
def run_controller(
    tls_id: str,
    timing: TLSTimingStandards,
    max_det_range_m: float,
    controller_step,
    seconds: float,
    log_csv_path: Path,
):
    am = ActionModule(traci_connection=traci, tls_id=tls_id, timing_standards=timing)
    sm = StateModule(
        traci_connection=traci, tls_id=tls_id, max_detection_range_m=max_det_range_m
    )

    # Deterministic warm start
    am.set_phase(0)

    with open(log_csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["t", "total_queue", "total_wait_s"])
        t0 = float(traci.simulation.getTime())
        end_t = t0 + seconds
        while (
            float(traci.simulation.getTime()) < end_t
            and traci.simulation.getMinExpectedNumber() >= 0
        ):
            q, wsum = controller_step(am, sm)
            t_now = float(traci.simulation.getTime())
            writer.writerow([t_now, q, wsum])
            traci.simulationStep()  # one tick


def load_csv(path: Path):
    t, q, w = [], [], []
    with open(path, "r") as f:
        rdr = csv.DictReader(f)
        for row in rdr:
            t.append(float(row["t"]))
            q.append(float(row["total_queue"]))
            w.append(float(row["total_wait_s"]))
    return np.array(t), np.array(q), np.array(w)


def plot_compare(baseline_csv: Path, adaptive_csv: Path):
    import matplotlib.pyplot as plt

    tb, qb, wb = load_csv(baseline_csv)
    ta, qa, wa = load_csv(adaptive_csv)

    # queues over time
    plt.figure()
    plt.plot(tb, qb, label="baseline")
    plt.plot(ta, qa, label="adaptive")
    plt.xlabel("time [s]")
    plt.ylabel("total queue [veh]")
    plt.title("Total queue over time")
    plt.legend()
    plt.tight_layout()

    # waits over time
    plt.figure()
    plt.plot(tb, wb, label="baseline")
    plt.plot(ta, wa, label="adaptive")
    plt.xlabel("time [s]")
    plt.ylabel("sum of waits [s]")
    plt.title("Total accumulated wait per tick")
    plt.legend()
    plt.tight_layout()

    # mean bars
    plt.figure()
    means_q = [qb.mean() if qb.size else 0.0, qa.mean() if qa.size else 0.0]
    means_w = [wb.mean() if wb.size else 0.0, wa.mean() if wa.size else 0.0]
    x = np.arange(2)
    width = 0.4
    plt.bar(x - width / 2, means_q, width, label="mean queue")
    plt.bar(x + width / 2, means_w, width, label="mean wait")
    plt.xticks(x, ["baseline", "adaptive"])
    plt.ylabel("value")
    plt.title("Mean KPIs")
    plt.legend()
    plt.tight_layout()

    plt.show()


def run_two(
    sumocfg: str,
    tls_id: str,
    seconds: float,
    fixed_green_s: float,
    out_dir: Path,
    seed: int | None,
    gui: bool,
    timing: TLSTimingStandards,
    max_det_range_m: float,
):
    out_dir.mkdir(parents=True, exist_ok=True)
    baseline_csv = out_dir / "baseline.csv"
    adaptive_csv = out_dir / "adaptive.csv"

    # Baseline run
    start_sumo(sumocfg, gui=gui, seed=seed)
    try:
        baseline_step = make_baseline(fixed_green_s=fixed_green_s)
        run_controller(
            tls_id=tls_id,
            timing=timing,
            max_det_range_m=max_det_range_m,
            controller_step=baseline_step,
            seconds=seconds,
            log_csv_path=baseline_csv,
        )
    finally:
        close_sumo()

    # Adaptive run (restart SUMO with same seed for comparable traffic)
    start_sumo(sumocfg, gui=gui, seed=seed)
    try:
        run_controller(
            tls_id=tls_id,
            timing=timing,
            max_det_range_m=max_det_range_m,
            controller_step=adaptive_step,
            seconds=seconds,
            log_csv_path=adaptive_csv,
        )
    finally:
        close_sumo()

    return baseline_csv, adaptive_csv


# ---------- Main ----------
if __name__ == "__main__":
    opts = get_options()

    timing = TLSTimingStandards(
        min_green_s=opts.min_green,
        yellow_s=opts.yellow,
        all_red_s=opts.all_red,
        max_green_s=opts.max_green,
    )

    out_dir = Path(opts.out_dir)

    baseline_csv, adaptive_csv = run_two(
        sumocfg=opts.sumocfg,
        tls_id=opts.tls_id,
        seconds=opts.seconds,
        fixed_green_s=opts.fixed_green_s,
        out_dir=out_dir,
        seed=opts.seed,
        gui=not opts.nogui,
        timing=timing,
        max_det_range_m=opts.max_det_range,
    )

    print(f"Wrote: {baseline_csv}")
    print(f"Wrote: {adaptive_csv}")

    if opts.plot:
        plot_compare(baseline_csv, adaptive_csv)
