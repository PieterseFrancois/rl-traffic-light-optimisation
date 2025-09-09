import copy
from dataclasses import dataclass, field


@dataclass
class VehicleRecord:
    vehicle_id: str
    type_id: str
    wait_time_s: float = 0.0
    speed: float = 0.0


@dataclass
class LaneMeasures:
    lane_id: str  # lane identifier
    queue: int = 0  # vehicles with wait > 0
    approach: int = 0  # vehicles within detection distyance and wait == 0
    total_wait_s: float = 0.0  # sum of waits on this lane
    max_wait_s: float = 0.0  # max wait on this lane
    total_speed: float = 0.0  # sum of speeds on this lane
    vehicles: list[VehicleRecord] = field(default_factory=list)  # per-vehicle records
    outbound_lanes: list[str] | None = None  # outbound lanes connected to this lane


class StateModule:
    """StateModule for a traffic light intersection in SUMO via TraCI.

    Attributes:
        tls_id: Traffic light system ID.
        max_detection_range_m: Maximum detection range for vehicles (in meters).
        approach_lanes: List of LaneMeasures for each approach lane.
        previous_state: Previous state read from SUMO (list of LaneMeasures).
        state: Current state read from SUMO (list of LaneMeasures).

    Methods:
        read_state: Reads the current state from SUMO via TraCI.
    """

    def __init__(
        self,
        traci_connection,
        tls_id: str,
        max_detection_range_m: float = 50.0,
    ):
        self.traci = traci_connection
        self.tls_id: str = tls_id
        self.max_detection_range_m: float = max_detection_range_m

        self.approach_lanes: list[LaneMeasures] = (
            self._build_approach_lanes_and_outbounds()
        )

        self.previous_state: list[LaneMeasures] | None = None
        self.state: list[LaneMeasures] | None = None
        self.last_read_time: float | None = None

    def _build_approach_lanes_and_outbounds(self) -> list[LaneMeasures]:
        """
        Derive inbound lanes in SUMO signal-index order and attach their outbound lanes.
        Uses traci.trafficlight.getControlledLinks(tls_id) so phase-index alignment is correct.
        """
        links = self.traci.trafficlight.getControlledLinks(self.tls_id)
        lane_order: list[str] = []
        out_map: dict[str, set[str]] = {}

        # Establish lane order and collect all (deduped) out-lanes per in-lane
        for group in links or []:
            if not group:
                continue
            for in_lane, out_lane, _ in group:
                if in_lane not in lane_order:
                    lane_order.append(in_lane)
                if in_lane not in out_map:
                    out_map[in_lane] = set()
                if out_lane:
                    out_map[in_lane].add(out_lane)

        # Fallback: include any controlled lanes that didn’t appear in links (rare)
        for lane in self.traci.trafficlight.getControlledLanes(self.tls_id):
            if lane not in lane_order:
                lane_order.append(lane)
                out_map.setdefault(lane, set())

        if not lane_order:
            raise ValueError(f"TLS '{self.tls_id}' does not control any lanes")

        # Build LaneMeasures with outbound_lanes already filled (sorted for determinism)
        return [
            LaneMeasures(lane_id=lane, outbound_lanes=sorted(out_map.get(lane, ())))
            for lane in lane_order
        ]

    def _get_vehicles(self, lane_id: str) -> list[str]:
        """Get all vehicles on a specific lane within detection range."""
        vehicles = self.traci.lane.getLastStepVehicleIDs(lane_id)

        in_range_vehicles = []
        for vehicle_id in vehicles:
            vehicle_path = self.traci.vehicle.getNextTLS(vehicle_id)

            # Check id path is not empty
            if not vehicle_path:
                continue

            next_tls_id, _, distance_to_tls, _ = vehicle_path[0]

            # Check if next TLS is this TLS
            if next_tls_id != self.tls_id:
                continue

            # Check if within detection range
            if distance_to_tls <= self.max_detection_range_m:
                in_range_vehicles.append(vehicle_id)

        return in_range_vehicles

    def read_state(self, simulation_timestep_s: float = 1.0) -> list[LaneMeasures]:
        """Read current state from SUMO via TraCI.

        Args:
            simulation_timestep_s: Simulation timestep in seconds (default: 1.0s). Used as fallback if no previous timestamp exists.

        Returns:
            List of LaneMeasures for each approach lane.
        """

        # Store previous state
        self.previous_state = (
            copy.deepcopy(self.state) if self.state is not None else None
        )

        # Build a per lane previous vehicle dict for reference - reduces O(n^2) lookups to O(n)
        previous_vehicles_per_lane: dict[str, dict[str, VehicleRecord]] = (
            {
                lane.lane_id: {v.vehicle_id: v for v in lane.vehicles}
                for lane in self.previous_state
            }
            if self.previous_state
            else {}
        )

        current_simulation_time = float(self.traci.simulation.getTime())
        delta_time: float = (
            current_simulation_time - self.last_read_time
            if self.last_read_time is not None
            else simulation_timestep_s
        )

        temp_state_variable: list[LaneMeasures] = []
        for lane in self.approach_lanes:
            # Get vehicles from previous state
            previous_vehicles: dict[str, VehicleRecord] = (
                previous_vehicles_per_lane.get(lane.lane_id, {})
            )

            # Get vehicles currently on this lane within detection range
            current_vehicle_ids = self._get_vehicles(lane.lane_id)

            new_lane_measures = LaneMeasures(
                lane_id=lane.lane_id,
                outbound_lanes=lane.outbound_lanes,
            )

            for vehicle_id in current_vehicle_ids:
                # Check if vehicle was present in previous state
                previous_record: VehicleRecord | None = previous_vehicles.get(
                    vehicle_id, None
                )

                # Vehicle stats
                speed = self.traci.vehicle.getSpeed(vehicle_id)
                wait_time_s = self.traci.vehicle.getWaitingTime(vehicle_id)
                record: VehicleRecord
                if not previous_record:
                    # New vehicle, create new record
                    type = str(self.traci.vehicle.getTypeID(vehicle_id))

                    record = VehicleRecord(
                        vehicle_id=vehicle_id,
                        wait_time_s=wait_time_s,
                        speed=speed,
                        type_id=type,
                    )
                else:
                    # Existing vehicle, update record
                    if (
                        previous_record.wait_time_s > 0.0
                    ):  # Already waiting - if slowly creeping forward whilst waiting (e.g. at red light), keep counting wait time
                        wait_time_s = previous_record.wait_time_s + delta_time
                    else:  # Not waiting previously
                        wait_time_s = (
                            wait_time_s  # Just use current wait time from SUMO
                        )

                    record = VehicleRecord(
                        vehicle_id=vehicle_id,
                        wait_time_s=wait_time_s,
                        speed=speed,
                        type_id=previous_record.type_id,
                    )

                # Update lane measures
                new_lane_measures.total_wait_s += wait_time_s
                new_lane_measures.max_wait_s = max(
                    new_lane_measures.max_wait_s, wait_time_s
                )
                new_lane_measures.total_speed += speed

                if wait_time_s > 0:
                    new_lane_measures.queue += 1  # Stationary vehicles
                else:
                    new_lane_measures.approach += 1  # Moving vehicles

                new_lane_measures.vehicles.append(record)

            temp_state_variable.append(new_lane_measures)

        # Update current state and timestamp
        self.last_read_time = current_simulation_time
        self.state = temp_state_variable

        return self.state
