# tests/test_action.py
"""
Action Module Test Suite (Direct Phase Index Control, TraCI-faked)

This suite validates the DirectPhaseIndexActionModule without requiring a live
SUMO/TraCI connection. We inject a small FakeTraCI that exposes only the calls
the module uses.

Covers
------
1) No-op when selecting the current phase
2) Minimum green enforcement and action masking
3) Amber-from-current + optional all-red handover
4) Ring buffer capacity behaviour
5) Invalid action range raises

Notes
-----
- The fake models `simulation.getTime()` as a scalar clock you can advance.
- Transitional phases are modelled via an internal "remaining duration".
"""

import types
import numpy as np
import pytest


# ------------------------------- TraCI Fake ---------------------------------


class _FakeSimulation:
    def __init__(self):
        self._t = 0.0

    def getTime(self):
        return self._t

    def advance(self, dt):
        self._t += float(dt)


class _FakeTLS:
    def __init__(self):
        self._phase = 0
        self._phase_duration = (
            0.0  # remaining duration for current special state (amber/all-red)
        )
        self._ryg_state = "ggrr"  # default current state (2 green, 2 red) for amber-from-current tests
        # Shape only; some code inspects length of phases
        self._program_logics = [{"phases": [0, 1, 2, 3]}]

    # --- API used by the action module ---
    def getPhase(self, tls_id):
        return self._phase

    def setPhase(self, tls_id, phase):
        self._phase = int(phase)
        # Setting a concrete phase clears any special remaining duration
        self._phase_duration = 0.0

    def getPhaseDuration(self, tls_id):
        return self._phase_duration

    def setPhaseDuration(self, tls_id, dur):
        self._phase_duration = float(dur)

    def setRedYellowGreenState(self, tls_id, state):
        self._ryg_state = state

    def getRedYellowGreenState(self, tls_id):
        return self._ryg_state

    def getAllProgramLogics(self, tls_id):
        return self._program_logics


class FakeTraci(types.SimpleNamespace):
    def __init__(self):
        super().__init__()
        self.simulation = _FakeSimulation()
        self.trafficlight = _FakeTLS()


@pytest.fixture()
def bind_traci(monkeypatch):
    """
    Bind a fresh FakeTraCI into the action module under test.
    """
    import importlib

    fake = FakeTraci()
    # Be tolerant to different repo layouts
    try:
        am = importlib.import_module("modules.intersection.action")
    except ModuleNotFoundError:
        try:
            am = importlib.import_module("intersection.action")
        except ModuleNotFoundError:
            am = importlib.import_module("action")
    # Replace the 'traci' symbol in the module with our fake
    monkeypatch.setattr(am, "traci", fake, raising=True)
    return fake, am


# --------------------------------- Tests ------------------------------------


def test_noop_same_phase(bind_traci):
    """
    1) No-op when selecting the current phase
       Selecting an action that maps to the already active phase must not change
       the SUMO phase and should record a 'noop' apply event in the buffer.
    """
    fake, am = bind_traci
    fake.trafficlight.setPhase("tls", 1)

    module = am.DirectPhaseIndexActionModule(
        tls_id="tls",
        action_to_phase_index=[0, 1, 2, 3],
        enforce_min_green_s=0.0,
        amber_duration_s=0.0,
        buffer_capacity=16,
    )

    module.apply(action=1)
    assert fake.trafficlight.getPhase("tls") == 1
    rec = module.last_n(1)[0]
    assert rec.event == "apply" and "noop" in rec.note


def test_min_green_blocks_switch_and_masks(bind_traci):
    """
    2) Minimum green enforcement and action masking
       While time-in-phase is below the configured threshold, switching must be
       blocked and the mask should only allow actions that keep the current phase.
    """
    fake, am = bind_traci
    fake.trafficlight.setPhase("tls", 0)

    module = am.DirectPhaseIndexActionModule(
        tls_id="tls",
        action_to_phase_index=[0, 1, 2, 3],
        enforce_min_green_s=5.0,
        amber_duration_s=0.0,
    )

    module.apply(action=2)
    assert fake.trafficlight.getPhase("tls") == 0
    assert module.last_n(1)[0].event == "skip_min_green"

    mask = module.compute_mask()
    assert mask.dtype == bool and mask.shape == (4,)
    assert np.array_equal(np.where(mask)[0], np.array([0]))

    fake.simulation.advance(5.1)
    module.apply(action=2)
    assert fake.trafficlight.getPhase("tls") == 2
    assert module.last_n(1)[0].event == "apply"


def test_amber_from_current_and_all_red_sequence(bind_traci):
    """
    3) Amber-from-current + optional all-red handover
       When a change is requested and amber is configured, the module must:
         - derive amber from current RYG (greens -> yellow, others -> red),
         - hold during amber, then if enabled start an all-red clearance,
         - finally switch to the queued target phase.
    """
    fake, am = bind_traci
    fake.trafficlight.setPhase("tls", 0)
    fake.trafficlight.setRedYellowGreenState("tls", "ggrr")  # two greens, two reds

    module = am.DirectPhaseIndexActionModule(
        tls_id="tls",
        action_to_phase_index=[0, 1, 2, 3],
        enforce_min_green_s=0.0,
        amber_duration_s=3.0,
        all_red_after_amber=True,
        all_red_duration_s=1.5,
    )

    # Request 0 -> 3: should start amber built from current state "ggrr" -> "yyrr"
    module.apply(action=3)
    assert fake.trafficlight._ryg_state == "yyrr"
    assert fake.trafficlight.getPhaseDuration("tls") == 3.0
    assert module.last_n(1)[0].event == "amber_start"
    assert fake.trafficlight.getPhase("tls") == 0  # still old phase

    # While amber remains (> 0), module waits
    fake.simulation.advance(1.0)
    fake.trafficlight.setPhaseDuration("tls", 2.0)
    module.apply(action=0)
    assert "waiting_amber" in module.last_n(1)[0].note
    assert fake.trafficlight.getPhase("tls") == 0

    # Amber finishes -> all-red starts
    fake.simulation.advance(2.1)
    fake.trafficlight.setPhaseDuration("tls", 0.0)
    module.apply(action=0)
    assert module.last_n(1)[0].event == "all_red_start"
    assert fake.trafficlight._ryg_state == "rrrr"

    # All-red finishes -> switch to target phase
    fake.simulation.advance(1.6)
    fake.trafficlight.setPhaseDuration("tls", 0.0)
    module.apply(action=0)
    assert fake.trafficlight.getPhase("tls") == 3
    assert module.last_n(1)[0].event == "all_red_done"


def test_buffer_capacity(bind_traci):
    """
    4) Ring buffer capacity behaviour
       When more events are recorded than the configured capacity, only the
       most recent events up to capacity should be retained.
    """
    fake, am = bind_traci
    fake.trafficlight.setPhase("tls", 0)

    module = am.DirectPhaseIndexActionModule(
        tls_id="tls",
        action_to_phase_index=[0, 1],
        enforce_min_green_s=0.0,
        amber_duration_s=0.0,
        buffer_capacity=3,
    )

    module.apply(0)
    module.apply(1)
    module.apply(0)
    module.apply(1)
    assert len(module.last_n(10)) == 3


def test_invalid_action_raises(bind_traci):
    """
    5) Invalid action range raises
       The module must raise ValueError if the selected action is outside the
       defined action space mapping.
    """
    fake, am = bind_traci
    fake.trafficlight.setPhase("tls", 0)

    module = am.DirectPhaseIndexActionModule(
        tls_id="tls",
        action_to_phase_index=[0, 1],
        enforce_min_green_s=0.0,
        amber_duration_s=0.0,
    )

    with pytest.raises(ValueError):
        module.apply(-1)
    with pytest.raises(ValueError):
        module.apply(2)
