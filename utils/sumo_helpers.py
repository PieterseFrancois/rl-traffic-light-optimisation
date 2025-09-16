import traci


def start_sumo(sumocfg: str, gui: bool = False) -> None:
    binname = "sumo-gui" if gui else "sumo"
    traci.start(
        [
            binname,
            "-c",
            sumocfg,
            "--start",
            "--no-step-log",
            "true",
            "--time-to-teleport",
            "-1",
        ]
    )


def close_sumo() -> None:
    try:
        traci.close(False)
    except Exception as e:
        print(f"Error closing SUMO: {e}")
        pass
