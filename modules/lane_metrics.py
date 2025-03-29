import traci

# lane_metrics.py
# Reusable functions for collecting SUMO lane-level metrics via TraCI


def get_lane_queue_lengths(lane_ids):
    """Returns the number of halted vehicles per lane."""
    return {lane: traci.lane.getLastStepHaltingNumber(lane) for lane in lane_ids}


def get_lane_speeds(lane_ids):
    """Returns the mean speed per lane."""
    return {lane: traci.lane.getLastStepMeanSpeed(lane) for lane in lane_ids}


def get_lane_occupancies(lane_ids):
    """Returns the occupancy percentage per lane (0.0 to 100.0)."""
    return {lane: traci.lane.getLastStepOccupancy(lane) for lane in lane_ids}


def get_lane_vehicle_counts(lane_ids):
    """Returns the number of vehicles currently on each lane."""
    return {lane: traci.lane.getLastStepVehicleNumber(lane) for lane in lane_ids}


def get_lane_emissions(lane_ids):
    """Returns the CO2 emissions per lane (if emission model is active)."""
    return {lane: traci.lane.getCO2Emission(lane) for lane in lane_ids}


def get_lane_waiting_times(lane_ids):
    """
    Returns the total waiting time of all vehicles on each lane.
    Waiting time is the time a vehicle has spent at speed ≤ 0.1 m/s.
    """
    lane_waits = {lane: 0.0 for lane in lane_ids}

    for vid in traci.vehicle.getIDList():
        lane_id = traci.vehicle.getLaneID(vid)
        if lane_id in lane_ids:
            wait = traci.vehicle.getWaitingTime(vid)
            lane_waits[lane_id] += wait

    return lane_waits


def get_lane_average_waiting_times(lane_ids):
    """
    Returns the average waiting time per lane.
    This is the mean of all waiting times of vehicles on each lane.
    """
    lane_waits = {lane: 0.0 for lane in lane_ids}
    lane_counts = {lane: 0 for lane in lane_ids}

    for vid in traci.vehicle.getIDList():
        lane_id = traci.vehicle.getLaneID(vid)
        if lane_id in lane_ids:
            lane_waits[lane_id] += traci.vehicle.getWaitingTime(vid)
            lane_counts[lane_id] += 1

    avg_waits = {}
    for lane in lane_ids:
        if lane_counts[lane] > 0:
            avg_waits[lane] = lane_waits[lane] / lane_counts[lane]
        else:
            avg_waits[lane] = 0.0

    return avg_waits
