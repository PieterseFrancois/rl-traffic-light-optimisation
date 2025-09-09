# modules/intersection/action.py
from __future__ import annotations

from dataclasses import dataclass
from collections import deque
from typing import Deque, Iterable, List, Optional
import numpy as np
import traci


@dataclass
class ActionRecord:
    sim_time: float
    event: str                     # "apply", "skip_min_green",
                                   # "amber_start", "amber_done",
                                   # "all_red_start", "all_red_done",
                                   # "warn_clamp_phase", "warn_reset_program"
    action: Optional[int] = None   # RL action index (if applicable)
    phase: Optional[int] = None    # TLS phase index after the call (if set)
    note: str = ""


class DirectPhaseIndexActionModule:
    """
    Direct mapping from RL action -> SUMO control, with transitional safety.

    Supports either:
      - action_to_phase_index: list[int]    (uses setPhase for the final apply), or
      - action_to_state_str : list[str]     (uses setRedYellowGreenState for the final apply)

    Features
      - Optional minimum green enforcement.
      - Amber from current (greens -> yellow, others -> red).
      - Optional all-red clearance after amber.
      - Ring buffer of recent control decisions.
      - Transitional timing driven in Python (tick() each SUMO step).
    """

    def __init__(
        self,
        tls_id: str,
        action_to_phase_index: Iterable[int] | None = None,
        *,
        action_to_state_str: Iterable[str] | None = None,
        enforce_min_green_s: float = 0.0,
        amber_duration_s: float = 0.0,
        all_red_after_amber: bool = False,
        all_red_duration_s: float = 0.0,
        buffer_capacity: int = 256,
    ):
        self.tls_id = tls_id
        self.map: List[int] | None = list(action_to_phase_index) if action_to_phase_index is not None else None
        self.map_state: List[str] | None = list(action_to_state_str) if action_to_state_str is not None else None
        if (self.map is None) and (self.map_state is None):
            raise ValueError("Provide action_to_phase_index or action_to_state_str")

        self.enforce_min_green_s = float(enforce_min_green_s)
        self.amber_duration_s = float(amber_duration_s)
        self.all_red_after_amber = bool(all_red_after_amber)
        self.all_red_duration_s = float(all_red_duration_s)

        self.buffer: Deque[ActionRecord] = deque(maxlen=int(buffer_capacity))

        # Internal state
        try:
            self._phase_enter_time: float = float(traci.simulation.getTime())  # type: ignore
        except Exception:
            self._phase_enter_time = 0.0
        try:
            self._last_phase: int = int(traci.trafficlight.getPhase(self.tls_id)) # type: ignore
        except Exception:
            self._last_phase = 0

        # Handover tracking
        self._pending_target_action: Optional[int] = None  # remember action index
        self._handover_stage: Optional[str] = None         # None | "amber" | "all_red"
        self._handover_until: Optional[float] = None
        self._amber_cached: Optional[str] = None

        # Cache signal heads
        try:
            cur = traci.trafficlight.getRedYellowGreenState(self.tls_id)
            self._n_signals = len(cur)
        except Exception:
            self._n_signals = 0

        # Remember a base program for index-based fallback
        self._base_program: Optional[str] = None
        try:
            self._base_program = traci.trafficlight.getProgram(self.tls_id) # type: ignore
        except Exception:
            pass

    # ----------------------------- helpers ---------------------------------

    def _now(self) -> float:
        try:
            return float(traci.simulation.getTime())  # type: ignore
        except Exception:
            return 0.0

    def _record(self, event: str, *, action: Optional[int] = None,
                phase: Optional[int] = None, note: str = "") -> None:
        self.buffer.append(ActionRecord(self._now(), event, action, phase, note))

    def _amber_from_current(self) -> str:
        """Only current greens become yellow; everything else red."""
        cur = traci.trafficlight.getRedYellowGreenState(self.tls_id)
        self._n_signals = len(cur)
        return "".join(("y" if c in ("g", "G") else "r") for c in cur)

    def _all_red_state(self) -> str:
        if self._n_signals <= 0:
            try:
                self._n_signals = len(traci.trafficlight.getRedYellowGreenState(self.tls_id))
            except Exception:
                self._n_signals = 0
        return "r" * max(0, self._n_signals)

    # ---- index-mode robustness (only used if map is provided) ----

    def _ensure_program(self) -> None:
        """Re-activate the base TLS program if available."""
        if not self._base_program:
            try:
                self._base_program = traci.trafficlight.getProgram(self.tls_id) # type: ignore
            except Exception:
                return
        try:
            cur = traci.trafficlight.getProgram(self.tls_id)
            if cur != self._base_program:
                traci.trafficlight.setProgram(self.tls_id, self._base_program)
        except Exception:
            pass

    def _current_phase_count(self) -> int:
        """Return number of phases in the currently active program (>=1)."""
        try:
            pid = traci.trafficlight.getProgram(self.tls_id)
            infos = traci.trafficlight.getAllProgramLogics(self.tls_id)
            for L in infos:
                prog_id = getattr(L, "programID", None) or getattr(L, "programId", None)
                if prog_id == pid:
                    phases = getattr(L, "phases", None) or []
                    return max(1, len(phases))
            phases = getattr(infos[0], "phases", None) or []
            return max(1, len(phases))
        except Exception:
            return 1

    def _safe_set_phase(self, target: int) -> int:
        """Restore program, clamp target to valid range, then setPhase. Return final index."""
        self._ensure_program()
        n = self._current_phase_count()
        if target < 0 or target >= n:
            self._record("warn_clamp_phase", phase=target, note=f"clamped to [0,{n-1}]")
            target = min(max(target, 0), n - 1)
        try:
            traci.trafficlight.setPhase(self.tls_id, target)
        except Exception:
            # last-ditch fallback
            self._ensure_program()
            traci.trafficlight.setPhase(self.tls_id, 0)
            self._record("warn_reset_program", note="fallback to phase 0")
            target = 0
        return target

    # ----------------------------- API -------------------------------------

    def compute_mask(self) -> Optional[np.ndarray]:
        """
        Optional action mask for MaskablePPO.
        While current phase age < min-green, only the action that keeps current state is allowed.
        """
        if self.enforce_min_green_s <= 0:
            return None

        time_in_phase = self._now() - float(self._phase_enter_time)
        n_actions = len(self.map or self.map_state or [])
        mask = np.ones(n_actions, dtype=bool)

        if time_in_phase < self.enforce_min_green_s:
            if self.map is not None:
                current = int(traci.trafficlight.getPhase(self.tls_id)) # type: ignore
                allowed = [i for i, p in enumerate(self.map) if p == current]
            else:
                cur = traci.trafficlight.getRedYellowGreenState(self.tls_id)
                allowed = [i for i, s in enumerate(self.map_state or []) if s == cur]
            mask[:] = False
            if allowed:
                mask[allowed] = True
            else:
                mask[0] = True
        return mask

    def tick(self) -> None:
        """
        Drive transitional overrides each SUMO tick.
        Re-assert amber/all-red states and advance to the next stage at deadline.
        """
        if self._pending_target_action is None or self._handover_stage is None:
            return

        now = self._now()
        if self._handover_until is None or now < self._handover_until:
            if self._handover_stage == "amber":
                st = self._amber_cached or self._amber_from_current()
                traci.trafficlight.setRedYellowGreenState(self.tls_id, st)
            elif self._handover_stage == "all_red":
                traci.trafficlight.setRedYellowGreenState(self.tls_id, self._all_red_state())
            return

        # Deadline reached -> advance
        a = int(self._pending_target_action)
        if self._handover_stage == "amber" and self.all_red_after_amber and self.all_red_duration_s > 0.0:
            traci.trafficlight.setRedYellowGreenState(self.tls_id, self._all_red_state())
            self._handover_stage = "all_red"
            self._handover_until = now + self.all_red_duration_s
            self._phase_enter_time = now
            self._record("all_red_start", note=f"dur={self.all_red_duration_s}s")
            return

        # Finalise to target
        if self.map_state is not None:
            traci.trafficlight.setRedYellowGreenState(self.tls_id, self.map_state[a])
        else:
            target_phase = int(self.map[a])  # type: ignore[index]
            target_phase = self._safe_set_phase(target_phase)

        self._last_phase = int(traci.trafficlight.getPhase(self.tls_id)) # type: ignore
        self._phase_enter_time = now
        evt = "all_red_done" if self._handover_stage == "all_red" else "amber_done"
        self._pending_target_action = None
        self._handover_stage = None
        self._handover_until = None
        self._amber_cached = None
        self._record(evt, phase=self._last_phase)

    def apply(self, action: int) -> None:
        """
        Apply the chosen action. If a handover is ongoing, only log and return.
        Otherwise:
          - enforce min-green,
          - start amber with an internal deadline,
          - optionally follow with all-red,
          - or switch immediately to the target (state string preferred if provided).
        """
        n_actions = len(self.map or self.map_state or [])
        if not (0 <= action < n_actions):
            raise ValueError(f"Action {action} out of range [0, {n_actions - 1}]")

        if self._pending_target_action is not None:
            self._record("apply", action=action, note=f"waiting_{self._handover_stage or 'handover'}")
            return

        if self.map is not None:
            current = int(traci.trafficlight.getPhase(self.tls_id)) # type: ignore
            target_same = (int(self.map[action]) == current)
        else:
            cur = traci.trafficlight.getRedYellowGreenState(self.tls_id)
            target_same = ((self.map_state or [])[action] == cur)

        if target_same:
            self._record("apply", action=action, phase=int(traci.trafficlight.getPhase(self.tls_id)), # type: ignore
                         note="noop_same_phase_or_state")
            return

        if self.enforce_min_green_s > 0.0:
            time_in_phase = self._now() - float(self._phase_enter_time)
            if time_in_phase < self.enforce_min_green_s:
                self._record(
                    "skip_min_green",
                    action=action,
                    phase=int(traci.trafficlight.getPhase(self.tls_id)), # type: ignore
                    note=f"time_in_phase={time_in_phase:.2f}s < {self.enforce_min_green_s:.2f}s",
                )
                return

        # Start amber window if configured
        if self.amber_duration_s > 0.0:
            self._amber_cached = self._amber_from_current()
            traci.trafficlight.setRedYellowGreenState(self.tls_id, self._amber_cached)
            self._pending_target_action = int(action)
            self._handover_stage = "amber"
            now = self._now()
            self._handover_until = now + self.amber_duration_s
            self._phase_enter_time = now
            self._record("amber_start", action=action, note=f"dur={self.amber_duration_s}s")
            return

        # All-red without amber
        if self.all_red_after_amber and self.all_red_duration_s > 0.0:
            traci.trafficlight.setRedYellowGreenState(self.tls_id, self._all_red_state())
            self._pending_target_action = int(action)
            self._handover_stage = "all_red"
            now = self._now()
            self._handover_until = now + self.all_red_duration_s
            self._phase_enter_time = now
            self._record("all_red_start", action=action, note=f"dur={self.all_red_duration_s}s")
            return

        # Immediate switch to final
        if self.map_state is not None:
            traci.trafficlight.setRedYellowGreenState(self.tls_id, self.map_state[action])
        else:
            self._safe_set_phase(int(self.map[action]))  # type: ignore[index]
        self._last_phase = int(traci.trafficlight.getPhase(self.tls_id)) # type: ignore
        self._phase_enter_time = self._now()
        self._record("apply", action=action, phase=self._last_phase, note="immediate_switch")

    # ----------------------------- utilities --------------------------------

    def last_n(self, n: int = 10) -> List[ActionRecord]:
        n = max(0, min(n, len(self.buffer)))
        return list(list(self.buffer)[-n:])
