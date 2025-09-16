# modules/intersection/reward.py
from __future__ import annotations
from typing import Callable, Sequence, Mapping, Dict, Any, Optional
import numpy as np

# Signature for reward functions
# det_in, det_out, reader, obs, action, info, **kwargs -> float
RewardFn = Callable[
    [
        Sequence[str],
        Sequence[str],
        Callable[[str], Mapping[str, float]],
        Dict[str, np.ndarray],
        int,
        Dict[str, Any],
    ],
    float,
]


class RewardModule:  # light interface
    def compute(
        self, obs: Dict[str, np.ndarray], action: int, info: Dict[str, Any]
    ) -> float: ...


class SimpleReward(RewardModule):
    """
    Generic reward: bind a function once, call compute() each step.

    Example:
        R = SimpleReward(det_in, reader, fn_name="queue_penalty", fn_kwargs=dict(w_count=1.0, w_halt=0.5))
        r = R.compute(obs, action, info)
    """

    def __init__(
        self,
        detectors_in: Sequence[str],
        detector_reader: Callable[[str], Mapping[str, float]],
        detectors_out: Sequence[str] = (),
        *,
        fn: Optional[RewardFn] = None,
        fn_name: Optional[str] = None,
        fn_kwargs: Optional[Dict[str, Any]] = None,
    ):
        if fn is None and fn_name is None:
            raise ValueError("Provide either fn or fn_name")
        self.det_in = list(detectors_in)
        self.det_out = list(detectors_out)
        self.reader = detector_reader
        self._kwargs = fn_kwargs or {}

        if fn is None:
            if fn_name not in _REGISTRY:
                raise ValueError(f"Unknown reward function '{fn_name}'")
            fn = _REGISTRY[fn_name]

        # Bind parameters so compute(obs, action, info) is clean
        def _bound(
            obs: Dict[str, np.ndarray], action: int, info: Dict[str, Any]
        ) -> float:
            return float(
                fn(
                    self.det_in,
                    self.det_out,
                    self.reader,
                    obs,
                    action,
                    info,
                    **self._kwargs,
                )
            )

        self._compute = _bound

    def compute(
        self, obs: Dict[str, np.ndarray], action: int, info: Dict[str, Any]
    ) -> float:
        return self._compute(obs, action, info)


# ---------------- Built-in reward functions (optional) ----------------


def queue_penalty(
    det_in: Sequence[str],
    det_out: Sequence[str],
    reader,
    obs,
    action,
    info,
    *,
    w_count: float = 1.0,
    w_halt: float = 0.5,
) -> float:
    """reward = -(w_count * sum(count_in) + w_halt * sum(halt_in))"""
    total = 0.0
    for d in det_in:
        m = reader(d) or {}
        total += w_count * float(m.get("count", 0.0))
        total += w_halt * float(m.get("halt", 0.0))
    return -total


def throughput_out(
    det_in: Sequence[str],
    det_out: Sequence[str],
    reader,
    obs,
    action,
    info,
    *,
    w: float = 1.0,
) -> float:
    """reward = +w * sum(count_out)"""
    total = 0.0
    for d in det_out:
        m = reader(d) or {}
        total += float(m.get("count", 0.0))
    return w * total


def avg_speed_in(
    det_in: Sequence[str],
    det_out: Sequence[str],
    reader,
    obs,
    action,
    info,
    *,
    w: float = 1.0,
) -> float:
    """reward = +w * mean(speed_in)"""
    if not det_in:
        return 0.0
    s = 0.0
    for d in det_in:
        m = reader(d) or {}
        s += float(m.get("speed", 0.0))
    return w * (s / float(len(det_in)))


def linear_combo(
    det_in: Sequence[str],
    det_out: Sequence[str],
    reader,
    obs,
    action,
    info,
    *,
    terms: Sequence[tuple[str, float]] = (),
) -> float:
    """
    Combine other named functions: terms=[("queue_penalty", 1.0), ("throughput_out", 0.2)]
    """
    val = 0.0
    for name, w in terms:
        fn = _REGISTRY.get(name)
        if fn is None:
            raise ValueError(f"Unknown reward function '{name}' in linear_combo")
        val += float(w) * float(fn(det_in, det_out, reader, obs, action, info))
    return val


def max_pressure_lite(
    det_in, det_out, reader, obs, action, info, *, beta: float = 1.0
) -> float:
    import traci

    tls_id: str = info["tls_id"]

    # Prefer the current TLS state if it is non-transitional; else fall back to the chosen action’s state
    cur_state: str = info.get("cur_state", "") or ""
    act_state: str = info.get("action_state", "") or ""

    def is_greenful(s: str) -> bool:
        return any(c in ("g", "G") for c in s)

    def is_transitional(s: str) -> bool:
        return "y" in s  # simple but effective

    state = (
        cur_state
        if (cur_state and is_greenful(cur_state) and not is_transitional(cur_state))
        else act_state
    )
    if not state or not is_greenful(state):
        # Nothing useable; evaluate to 0 rather than injecting noise mid-amber
        return 0.0

    links = traci.trafficlight.getControlledLinks(tls_id)

    active_in, active_out = set(), set()
    for i, group in enumerate(links):
        if i >= len(state):
            break
        if state[i] in ("g", "G"):
            for in_lane, _, out_lane in group:
                if in_lane:
                    active_in.add(in_lane)
                if out_lane:
                    active_out.add(out_lane)

    def lane_of(det_id: str):
        try:
            return traci.lanearea.getLaneID(det_id)
        except Exception:
            return None

    q_in = 0.0
    for d in det_in:
        L = lane_of(d)
        if L and L in active_in:
            m = reader(d) or {}
            q_in += float(m.get("halt", 0.0)) + float(m.get("count", 0.0))

    q_out = 0.0
    for d in det_out:
        L = lane_of(d)
        if L and L in active_out:
            m = reader(d) or {}
            q_out += float(m.get("halt", 0.0)) + float(m.get("count", 0.0))

    return float(q_in - beta * q_out)


# Name → function registry so you can reference by string
_REGISTRY: Dict[str, RewardFn] = {
    "queue_penalty": queue_penalty,
    "throughput_out": throughput_out,
    "avg_speed_in": avg_speed_in,
    "linear_combo": linear_combo,
    "max_pressure_lite": max_pressure_lite,
}
