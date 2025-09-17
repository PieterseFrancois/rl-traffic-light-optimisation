import os
import sys

from sumolib import checkBinary
import traci

from dataclasses import dataclass


@dataclass
class SUMOConfig:
    """
    Configuration for the SUMO simulation.

    Attributes:
        sumocfg_filepath (str): Path to the SUMO configuration file.
        nogui (bool): If True, start SUMO without GUI.
        seed (int | None): Random seed for the simulation. If None, no seed is set.
        time_to_teleport (int): Time in seconds before a vehicle is teleported. Default is -1 (no teleportation).
    """
    sumocfg_filepath: str
    nogui: bool
    seed: int | None
    time_to_teleport: int = -1


def start_sumo(
    config: SUMOConfig,
) -> None:
    """Start the SUMO simulation."""

    if "SUMO_HOME" in os.environ:
        tools = os.path.join(os.environ["SUMO_HOME"], "tools")
        if tools not in sys.path:
            sys.path.append(tools)
    else:
        raise RuntimeError('Declare environment variable "SUMO_HOME"')

    args = [
        checkBinary("sumo" if config.nogui else "sumo-gui"),
        "-c",
        config.sumocfg_filepath,
        "--start",
        "--no-warnings",
        "--time-to-teleport",
        str(config.time_to_teleport),
    ]
    
    if config.seed is not None:
        args += ["--seed", str(config.seed)]

    traci.start(args)


def close_sumo() -> None:
    """Close the SUMO simulation."""
    try:
        traci.close(False)
    except Exception:
        pass
