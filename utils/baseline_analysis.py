import traci

from pathlib import Path

from modules.intersection.intersection import (
    IntersectionModule,
    IntersectionConfig,
)
from modules.intersection.memory import LogEntry
from modules.intersection.preprocessor import PreprocessorConfig as FeatureConfig

from utils.sumo_helpers import start_sumo, close_sumo, SUMOConfig, NetworkStateLogging


def sumo_baseline_configured_tls(
    sumo_config: SUMOConfig,
    intersection_configs: list[IntersectionConfig],
    feature_config: FeatureConfig,
    simulation_duration: float,
    log_directory: Path | str,
    ticks_per_decision: int = 1,
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

    Returns:
        dict[tls_id -> list[LogEntry]]
    """
    # (Re)start SUMO
    close_sumo()
    start_sumo(
        sumo_config,
        network_logging=NetworkStateLogging(
            log_directory=str(log_directory),
            run_label="baseline",
        ),
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

    t_now = t0
    while t_now < t_end and traci.simulation.getMinExpectedNumber() > 0:
        t_now = float(traci.simulation.getTime())

        # Advance simulation without issuing any actions (SUMO follows its own TLS program)
        for _ in range(max(1, int(ticks_per_decision))):
            traci.simulationStep()
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

    agent_logs: dict[str, list[LogEntry]] = {
        tls_id: agent.memory_module.get_all_logs() for tls_id, agent in agents.items()
    }

    for tls_id, agent in agents.items():
        filepath = Path(log_directory) / f"baseline_{tls_id}.csv"
        filepath.parent.mkdir(parents=True, exist_ok=True)
        agent.memory_module.export_csv(filepath)

    close_sumo()
    return agent_logs
