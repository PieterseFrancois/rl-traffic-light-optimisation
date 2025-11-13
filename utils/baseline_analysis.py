import traci
import yaml

from pathlib import Path

from modules.intersection.intersection import (
    IntersectionModule,
    IntersectionConfig,
)
from modules.intersection.memory import LogEntry
from modules.intersection.preprocessor import PreprocessorConfig as FeatureConfig

from utils.sumo_helpers import start_sumo, close_sumo, SUMOConfig, NetworkStateLogging

from event_bus import EventBus
from utils.kpi import collect_and_emit_kpis, RunMode

from disruptions.fault_manager import FaultManager
from disruptions.all_red_fault import AllRedFault


def sumo_baseline_configured_tls(
    sumo_config: SUMOConfig,
    intersection_configs: list[IntersectionConfig],
    feature_config: FeatureConfig,
    simulation_duration: float,
    log_directory: Path | str,
    ticks_per_decision: int = 1,
    event_bus: EventBus | None = None,
) -> dict[str, list[LogEntry]]:
    """
    Runs a SUMO simulation with configured traffic light systems (TLS) to
    establish a baseline for comparison with RL-based approaches.

    The TLS follow their pre-defined programs without any intervention,
    therefore the TLS must already be configured before the simulation starts.

    Logs KPIs and rewards at each decision step for each intersection.

    Args:
        sumo_config (SUMOConfig): Configuration for SUMO simulation.
        intersection_configs (list[IntersectionConfig]): List of intersection configurations.
        feature_config (FeatureConfig): Configuration for feature extraction.
        simulation_duration (float): Duration of the simulation in seconds.
        log_directory (Path | str): Directory to save the log CSV files.
        ticks_per_decision (int, optional): Number of simulation ticks between each decision step. Defaults to 1.
        event_bus (EventBus | None, optional): Event bus for emitting events. Defaults to None.

    Returns:
        dict[tls_id -> list[LogEntry]]
    """
    # Configure faults
    CONFIG_FILE_PATH: str = "environments/ingolstadt/config.yaml"
    with open(CONFIG_FILE_PATH, "r") as f:
        config_yaml = yaml.safe_load(f)

    all_red_fault: bool = config_yaml.get("all_red_fault", {}).get("enabled", False)
    all_red_fault_tls_id: str | None = config_yaml.get("all_red_fault", {}).get(
        "tls_id", None
    )
    all_red_fault_duration_s: int | None = config_yaml.get("all_red_fault", {}).get(
        "duration_s", None
    )
    all_red_fault_start_time_s: int | None = config_yaml.get("all_red_fault", {}).get(
        "start_time_s", None
    )

    if all_red_fault:
        if (
            all_red_fault_tls_id is None
            or all_red_fault_duration_s is None
            or all_red_fault_start_time_s is None
        ):
            raise ValueError("All-Red fault configuration is incomplete in config.yaml")

    faults = []
    if all_red_fault:
        faults.append(
            AllRedFault(
                tls_id=all_red_fault_tls_id,
                duration_steps=all_red_fault_duration_s,
                start_step=all_red_fault_start_time_s,
            )
        )

    fault_manager = FaultManager(faults) if faults else None

    # (Re)start SUMO
    close_sumo()
    start_sumo(
        sumo_config,
        network_logging=NetworkStateLogging(
            log_directory=str(log_directory),
            run_label="baseline",
        ),
        verbose=True,
    )

    # Ensure all configs have warm_start=False (ensures a pure program-following baseline)
    for config in intersection_configs:
        config.warm_start = False

    # Build agents
    agents: dict[str, IntersectionModule] = {
        config.tls_id: IntersectionModule(
            traci_connection=traci, config=config, feature_config=feature_config
        )
        for config in intersection_configs
    }

    # Initial sync: tick once and read state
    for agent in agents.values():
        agent.tick()
    traci.simulationStep()
    for agent in agents.values():
        agent.read_state()

    t0 = float(traci.simulation.getTime())
    t_end = t0 + float(simulation_duration)

    cache_active_tls_program = traci.trafficlight.getProgram(all_red_fault_tls_id)
    restored = False

    t_now = t0
    while t_now < t_end and traci.simulation.getMinExpectedNumber() > 0:
        t_now = float(traci.simulation.getTime())

        # Advance simulation without issuing any actions (SUMO follows its own TLS program)
        for _ in range(max(1, int(ticks_per_decision))):
            traci.simulationStep()
            if (
                fault_manager is not None
                and t_now <= all_red_fault_start_time_s + all_red_fault_duration_s
            ):
                fault_manager.step(t_now)
            elif fault_manager is not None and not restored:
                # Restore cached programs if no fault is active
                traci.trafficlight.setProgram(
                    all_red_fault_tls_id, cache_active_tls_program
                )
                restored = True

            for agent in agents.values():
                agent.tick()

        # Read and log KPIs + reward
        for tls_id, agent in agents.items():
            agent.read_state(hold_reward=False)
            # agent.log_to_memory(t=t_now) # Now included in read_state()

            log_entry: LogEntry = agent.memory_module.get_latest()
            if log_entry is None:
                print("Warning: No log entry found for TLS", tls_id)
                continue  # should not happen

        # If event bus is used, collect and emit kpis
        if event_bus:
            collect_and_emit_kpis(event_bus, traci, run_mode=RunMode.BASELINE)

    agent_logs: dict[str, list[LogEntry]] = {
        tls_id: agent.memory_module.get_all_logs() for tls_id, agent in agents.items()
    }

    for tls_id, agent in agents.items():
        filepath = Path(log_directory) / f"baseline_{tls_id}.csv"
        filepath.parent.mkdir(parents=True, exist_ok=True)
        agent.memory_module.export_csv(filepath)

    close_sumo()
    return agent_logs
