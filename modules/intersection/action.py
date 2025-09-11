from dataclasses import dataclass
from typing import Literal


@dataclass
class TLSTimingStandards:
    """Traffic Light System timing standards as defined by relevant traffic authority. Set to zero if not used."""

    min_green_s: float = 0.0
    yellow_s: float = 0.0
    all_red_s: float = 0.0
    max_green_s: float = 0.0


@dataclass
class TransitionMemory:
    """Memory of an ongoing traffic light phase transition."""

    to_phase: int
    yellow_end_time: float
    red_end_time: float
    current_transition: Literal["yellow", "all_red"]


@dataclass
class ActivePhaseMemory:
    """Memory of the currently active green phase."""

    phase_index: int
    start_time: float


class ActionModule:
    _ALLOWED_GREEN_PHASE_CHARACTERS = {"G", "g", "r"}
    _GREEN_CHARACTERS = {"G", "g"}

    def __init__(
        self,
        traci_connection,
        tls_id: str,
        green_phase_strings: list[str] | None = None,
        timing_standards: TLSTimingStandards = TLSTimingStandards(),
    ):
        self.traci = traci_connection
        self.tls_id: str = tls_id
        self.timing_standards: TLSTimingStandards = timing_standards
        self._validate_timing_standards()

        if green_phase_strings is None:
            green_phase_strings = self._infer_green_phases_from_sumo()

        self._validate_green_phases(green_phase_strings)
        self.green_phase_map: dict[int, str] = {
            i: phase for i, phase in enumerate(green_phase_strings)
        }
        self.n_actions: int = len(self.green_phase_map)

        self.active_phase_memory: ActivePhaseMemory | None = None

        self.in_transition: bool = False
        self.transition_memory: TransitionMemory | None = None

    def get_action_space(self) -> list[int]:
        """Get the list of valid action indices (green phases) for the TLS."""
        return list(self.green_phase_map.keys())

    def set_phase_immediately(self, phase_index: int) -> None:
        """Immediately set the traffic light to the specified green phase index ignoring any transitional standards."""
        if phase_index not in self.green_phase_map:
            raise ValueError(
                f"Phase index {phase_index} is not valid for TLS '{self.tls_id}'."
            )

        phase_str = self.green_phase_map[phase_index]
        self.traci.trafficlight.setRedYellowGreenState(self.tls_id, phase_str)

        # Update active phase memory
        self.active_phase_memory = ActivePhaseMemory(
            phase_index=phase_index, start_time=self.traci.simulation.getTime()
        )

    def set_phase(self, phase_index: int) -> None:
        """Set the traffic light to the specified green phase index, following any transitional timing standards."""
        if phase_index not in self.green_phase_map:
            raise ValueError(
                f"Phase index {phase_index} is not valid for TLS '{self.tls_id}'."
            )

        if self.in_transition:
            raise RuntimeError(
                f"TLS '{self.tls_id}' is already in a transition. Cannot set a new phase."
            )

        # Check if phase has been set at least once
        if not self.active_phase_memory:
            self.set_phase_immediately(phase_index)
            return

        current_time = float(self.traci.simulation.getTime())
        elapsed_green_time: float = current_time - self.active_phase_memory.start_time

        # Check if the requested phase is already active
        if self.active_phase_memory.phase_index == phase_index:
            # Check if max green time is enforced and exceeded
            if (
                self.timing_standards.max_green_s > 0.0
                and elapsed_green_time >= self.timing_standards.max_green_s
            ):
                # TODO: Max green time exceeded for the current phase - for now print a warning
                print(
                    f"Warning: Maximum green time of {self.timing_standards.max_green_s}s exceeded for TLS '{self.tls_id}' on phase index {phase_index}. Current green time: {elapsed_green_time:.2f}s."
                )

            return  # No action needed, already in the desired phase

        # Check if minimum green time is enforced and not yet met
        if (
            self.timing_standards.min_green_s > 0.0
            and elapsed_green_time < self.timing_standards.min_green_s
        ):
            # Minimum green time not met, cannot switch phase
            # Debug log
            print(
                f"Minimum green time of {self.timing_standards.min_green_s}s not met for TLS '{self.tls_id}'. Current green time: {elapsed_green_time:.2f}s. Phase change to index {phase_index} ignored."
            )
            return

        # Check if a transition period is needed
        transition_needed: bool = (
            self.timing_standards.yellow_s > 0.0
            or self.timing_standards.all_red_s > 0.0
        )
        if transition_needed:
            transition_colour: str = ""
            if self.timing_standards.yellow_s > 0.0:
                transition_colour = "yellow"
                self._set_yellow_phase()
            else:
                transition_colour = "all_red"
                self._set_all_red_phase()

            yellow_end_time = (
                current_time + self.timing_standards.yellow_s
            )  # If no yellow, this is just current_time
            red_end_time = (
                yellow_end_time + self.timing_standards.all_red_s
            )  # If no all-red, this is just yellow_end_time

            self.transition_memory = TransitionMemory(
                to_phase=phase_index,
                yellow_end_time=yellow_end_time,
                red_end_time=red_end_time,
                current_transition=transition_colour,
            )
            self.in_transition = True
        else:
            self.set_phase_immediately(phase_index)

    def update_transition(self) -> None:
        """Update the transition state of the traffic light."""
        if not self.in_transition or not self.transition_memory:
            return

        current_time = float(self.traci.simulation.getTime())

        # Check if yellow phase has ended
        if (
            current_time >= self.transition_memory.yellow_end_time
            and current_time < self.transition_memory.red_end_time
        ):
            # Yellow phase ended, set to all red if all_red_s > 0
            if (
                self.timing_standards.all_red_s > 0.0
                and self.transition_memory.current_transition != "all_red"
            ):  # Only set all-red if not already in all-red
                self._set_all_red_phase()
                self.transition_memory.current_transition = "all_red"
        elif current_time >= self.transition_memory.red_end_time:
            # Red phase ended, set to the new green phase
            new_phase_index = self.transition_memory.to_phase
            self.set_phase_immediately(new_phase_index)
            self.in_transition = False
            self.transition_memory = None

    def _set_yellow_phase(self) -> None:
        """Set the traffic light to the yellow phase based on the current green phase."""
        current_state = self.traci.trafficlight.getRedYellowGreenState(self.tls_id)
        yellow_phase = "".join(
            "y" if char in self._GREEN_CHARACTERS else char for char in current_state
        )
        self.traci.trafficlight.setRedYellowGreenState(self.tls_id, yellow_phase)

    def _set_all_red_phase(self) -> None:
        """Set the traffic light to the all-red phase."""
        length_of_phase_string: int = len(
            self.green_phase_map[0]
        )  # All phase strings should be the same length
        red_phase_string = "r" * length_of_phase_string
        self.traci.trafficlight.setRedYellowGreenState(self.tls_id, red_phase_string)

    def _validate_green_phases(self, green_phase_strings: list[str]) -> None:
        """Validate that the provided green phase strings are valid for the TLS."""

        # Ensure no duplicate phase strings are provided
        if len(green_phase_strings) != len(set(green_phase_strings)):
            raise ValueError(
                f"Duplicate green phase strings found for TLS '{self.tls_id}'."
            )

        # Ensure only G, g, or r characters are used
        for phase_str in green_phase_strings:
            if any(
                character not in self._ALLOWED_GREEN_PHASE_CHARACTERS
                for character in phase_str
            ):
                raise ValueError(
                    f"Invalid character in green phase string '{phase_str}' for TLS '{self.tls_id}'. Allowed characters are {self._ALLOWED_GREEN_PHASE_CHARACTERS}."
                )

        # Ensure each phase string is correct length
        tls_lanes = self.traci.trafficlight.getControlledLanes(self.tls_id)
        expected_length = len(tls_lanes)
        for phase_str in green_phase_strings:
            if len(phase_str) != expected_length:
                raise ValueError(
                    f"Green phase string '{phase_str}' length does not match number of controlled lanes ({expected_length}) for TLS '{self.tls_id}'."
                )

    def _infer_green_phases_from_sumo(self) -> list[str]:
        """Infer green phase strings from the TLS in SUMO by examining each phase."""
        definitions = self.traci.trafficlight.getAllProgramLogics(self.tls_id)

        # Assume only one definition exists
        logical_definition = definitions[0] if definitions else None

        phases = logical_definition.phases if logical_definition else None

        if not phases:
            raise ValueError(f"TLS '{self.tls_id}' has no phases defined in SUMO.")

        green_phases: list[str] = []
        for phase in phases:
            phase_str = phase.state
            if all(
                char in self._ALLOWED_GREEN_PHASE_CHARACTERS for char in phase_str
            ) and any(char in self._GREEN_CHARACTERS for char in phase_str):
                green_phases.append(phase_str)

        if not green_phases:
            raise ValueError(f"No green phases found for TLS '{self.tls_id}' in SUMO.")

        return green_phases

    def _validate_timing_standards(self) -> None:
        """Validate the timing standards provided."""
        if self.timing_standards.min_green_s < 0.0:
            raise ValueError("Minimum green time cannot be negative.")
        if self.timing_standards.yellow_s < 0.0:
            raise ValueError("Yellow time cannot be negative.")
        if self.timing_standards.all_red_s < 0.0:
            raise ValueError("All-red time cannot be negative.")
        if self.timing_standards.max_green_s < 0.0:
            raise ValueError("Maximum green time cannot be negative.")
        if (
            self.timing_standards.min_green_s > 0.0
            and self.timing_standards.max_green_s > 0.0
        ):
            if self.timing_standards.min_green_s > self.timing_standards.max_green_s:
                raise ValueError(
                    "Minimum green time cannot be greater than maximum green time."
                )

    def ready_for_decision(self) -> bool:
        """Check if the TLS is ready for a new phase decision."""

        # Transition in progress
        if self.in_transition:
            return False

        # Minimum green time enforcement
        enforce_min_green: bool = (
            self.timing_standards.min_green_s > 0.0
            and self.active_phase_memory is not None
        )
        if enforce_min_green:
            current_time = float(self.traci.simulation.getTime())
            elapsed_green_time: float = (
                current_time - self.active_phase_memory.start_time
            )
            if elapsed_green_time < self.timing_standards.min_green_s:
                return False

        return True

    def get_action_mask(self, ready_for_decision: bool | None = None) -> list[bool]:
        """Get a mask of valid actions (green phases) for the TLS."""

        # Fallback for not ready for decision - ideally wait until ready and pass from outside to get action mask
        if ready_for_decision is None:
            ready_for_decision = self.ready_for_decision()

        if not ready_for_decision:
            return [False] * self.n_actions

        # Initialise all actions as valid
        mask: list[bool] = [True] * self.n_actions

        if not self.active_phase_memory:
            return mask  # All actions valid if no active phase

        current_time = float(self.traci.simulation.getTime())
        elapsed_green_time: float = current_time - self.active_phase_memory.start_time

        # Minimum green time enforcement - handled using the decision boundary
        # enforce_min_green: bool = (self.timing_standards.min_green_s > 0.0 and self.active_phase_memory is not None)
        # if enforce_min_green and elapsed_green_time < self.timing_standards.min_green_s:
        #     # Only the currently active phase is valid
        #     for i in range(self.n_actions):
        #         if i != self.active_phase_memory.phase_index:
        #             mask[i] = False

        # Maximum green time enforcement
        enforce_max_green: bool = (
            self.timing_standards.max_green_s > 0.0
            and self.active_phase_memory is not None
        )
        if (
            enforce_max_green
            and elapsed_green_time >= self.timing_standards.max_green_s
        ):
            # Only phases other than the currently active phase are valid
            if self.active_phase_memory:
                mask[self.active_phase_memory.phase_index] = False

        return mask
