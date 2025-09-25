from ray.rllib.algorithms.ppo import PPO

from modules.intersection.memory import LogEntry


def evaluate_trained_scenario(
    trainer: PPO,
    env_creator,
    *,
    max_steps: int,
) -> dict[str, list[LogEntry]]:
    """
    Use RLlib trainer to act. One independent policy per agent (policy id == agent id).
    """
    env = env_creator()
    obs, _infos = env.reset()

    steps = 0
    while env.agents and steps < max_steps:
        actions = {}
        for tls_id, ob in obs.items():
            act = trainer.compute_single_action(
                observation=ob, policy_id=tls_id, explore=True
            )
            actions[tls_id] = int(act)

        obs, _rewards, _term, _trunc, _infos = env.step(actions)

        steps += 1

    agent_logs: dict[str, list[LogEntry]] = env.get_agent_logs()

    env.close()
    return agent_logs
