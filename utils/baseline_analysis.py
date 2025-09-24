import traci

from modules.intersection.intersection import (
    IntersectionModule,
    IntersectionConfig,
    BaseIntersectionKPIs,
)
from modules.intersection.preprocessor import PreprocessorConfig as FeatureConfig
from utils.sumo_helpers import start_sumo, close_sumo, SUMOConfig


def sumo_baseline_configured_tls(
    sumo_config: SUMOConfig,
    intersection_configs: list[IntersectionConfig],
    feature_config: FeatureConfig,
    simulation_duration: float,
    ticks_per_decision: int = 1,
) -> dict[str, dict[str, list]]:
    """
    Runs a SUMO simulation with configured traffic light systems (TLS) to
    establish a baseline for comparison with RL-based approaches.

    The TLS follow their pre-defined programs without any intervention,
    therefore the TLS must already be configured before the simulation starts.

    Logs {t, queue, total_wait, reward} per agent each tick.

    Args:
        sumo_config (SUMOConfig): Configuration for SUMO simulation.
        intersection_configs (list[IntersectionConfig]): List of intersection configurations.
        feature_config (FeatureConfig): Configuration for feature extraction.
        simulation_duration (float): Duration of the simulation in seconds.
        ticks_per_decision (int, optional): Number of simulation ticks between each decision step. Defaults to 1.

    Returns:
        dict[tls_id -> dict[str, list]] with keys: "t", "queue", "total_wait", "reward".
    """
    # (Re)start SUMO
    close_sumo()
    start_sumo(sumo_config)

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

    series: dict[str, dict[str, list]] = {
        tls_id: {"t": [], "queue": [], "total_wait": [], "reward": []}
        for tls_id in agents.keys()
    }

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
            agent.read_state()
            kpi: BaseIntersectionKPIs = agent.get_kpi()
            reward = float(agent.get_reward())
            series[tls_id]["t"].append(t_now)
            series[tls_id]["queue"].append(float(kpi.total_queue_length))
            series[tls_id]["total_wait"].append(float(kpi.total_wait_time_s))
            series[tls_id]["reward"].append(reward)

    close_sumo()
    return series
