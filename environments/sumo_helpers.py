import os
import sys

from sumolib import checkBinary
import traci


def start_sumo(
    sumocfg: str, *, nogui: bool, seed: int | None, time_to_teleport: int = -1
) -> None:
    """
    Start the SUMO simulation.

    Arguments:
        sumocfg (str): Path to the SUMO configuration file.
        nogui (bool): If True, start SUMO without GUI.
        seed (int | None): Random seed for the simulation. If None, no seed is set.
        time_to_teleport (int): Time in seconds before a vehicle is teleported. Default is -1 (no teleportation).
    """

    if "SUMO_HOME" in os.environ:
        tools = os.path.join(os.environ["SUMO_HOME"], "tools")
        if tools not in sys.path:
            sys.path.append(tools)
    else:
        raise RuntimeError('Declare environment variable "SUMO_HOME"')

    args = [
        checkBinary("sumo" if nogui else "sumo-gui"),
        "-c",
        sumocfg,
        "--start",
        "--no-warnings",
        "--time-to-teleport",
        str(time_to_teleport),
    ]
    if seed is not None:
        args += ["--seed", str(seed)]
    traci.start(args)


def close_sumo() -> None:
    """Close the SUMO simulation."""
    try:
        traci.close(False)
    except Exception:
        pass
