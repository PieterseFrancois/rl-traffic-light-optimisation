from ray.rllib.policy.policy import PolicySpec
from ray.rllib.algorithms.ppo import PPOConfig, PPO

from typing import Callable

from environments.env_wrappers import RLlibPZEnv


from dataclasses import dataclass


@dataclass
class TrainerParameters:
    """
    Parameters for building an independent PPO trainer.

    Attributes:
        num_workers (int): Number of parallel rollout workers. Default 0.
        rollout_fragment_length (int): Number of steps to include in each rollout fragment. Default 200.
        train_batch_size (int): Size of the training batch. Default 4000.
        minibatch_size (int): Size of each minibatch. Default 128.
        num_epochs (int): Number of epochs per training batch. Default 4.
        lr (float): Learning rate. Default 1e-4.
    """

    num_workers: int = 0
    rollout_fragment_length: int = 200
    train_batch_size: int = 4000
    minibatch_size: int = 128
    num_epochs: int = 4
    lr: float = 1e-4


def build_independent_ppo_config(
    *,
    env_kwargs: dict,
    register_fn: Callable[[], None],
    training_model: dict,
    trainer_params: TrainerParameters,
    extra_policy_config: dict | None = None,
    algo_overrides: dict | None = None,
) -> PPOConfig:
    """
    Build an RLlib PPOConfig for independent policies, one per agent in the env.

    Args:
        env_kwargs: kwargs passed to the env constructor.
        register_fn: Function to register any custom models used by the policies.
        training_model: Model config dict used by all policies.
        trainer_params: TrainerParameters: Parameters for the PPO trainer.
        extra_policy_config: Extra config to add to each policy's config dict.
        algo_overrides: Extra config to add to the PPOConfig.

    Returns:
        Configured PPOConfig.
    """

    # Register once
    register_fn()

    # Probe spaces
    tmp_env = RLlibPZEnv({"env_kwargs": env_kwargs})
    try:
        tmp_env.reset()
        obs_spaces = tmp_env.observation_space
        act_spaces = tmp_env.action_space
        agent_ids: list[str] = list(tmp_env.agents)
    finally:
        tmp_env.close()

    # Build independent policies without a per-policy "model"
    if extra_policy_config and "model" in extra_policy_config:
        raise ValueError(
            "extra_policy_config must not include 'model' when using a global training_model"
        )

    policies: dict[str, PolicySpec] = {}
    for aid in agent_ids:
        policies[aid] = PolicySpec(
            observation_space=obs_spaces[aid],
            action_space=act_spaces[aid],
            config=extra_policy_config or {},
        )

    def policy_mapping_fn(agent_id, *args, **kwargs):
        return agent_id

    if not isinstance(training_model, dict) or "custom_model" not in training_model:
        raise ValueError("training_model must be a dict with at least 'custom_model'")

    training_kwargs = dict(
        lr=trainer_params.lr,
        train_batch_size=trainer_params.train_batch_size,
        minibatch_size=trainer_params.minibatch_size,
        num_epochs=trainer_params.num_epochs,
        model=training_model,
    )
    if algo_overrides:
        training_kwargs.update(algo_overrides)

    BATCH_MODE: str = "truncate_episodes"  # "complete_episodes"

    cfg = (
        PPOConfig()
        .api_stack(
            enable_rl_module_and_learner=False,
            enable_env_runner_and_connector_v2=False,  # Uses old API stack
        )
        .environment(env=RLlibPZEnv, env_config={"env_kwargs": env_kwargs})
        .framework("torch")
        .env_runners(
            num_env_runners=trainer_params.num_workers,
            rollout_fragment_length=trainer_params.rollout_fragment_length,
            batch_mode=BATCH_MODE,
        )
        .training(**training_kwargs)
        .multi_agent(
            policies=policies,
            policy_mapping_fn=policy_mapping_fn,
            policies_to_train=list(policies.keys()),
        )
    )
    return cfg


def build_trainer(
    *,
    env_kwargs: dict,
    register_fn: Callable[[], None],
    training_model: dict,
    trainer_params: TrainerParameters,
    extra_policy_config: dict | None = None,
    algo_overrides: dict | None = None,
) -> PPO:

    cfg = build_independent_ppo_config(
        env_kwargs=env_kwargs,
        register_fn=register_fn,
        training_model=training_model,
        trainer_params=trainer_params,
        extra_policy_config=extra_policy_config,
        algo_overrides=algo_overrides,
    )

    return cfg.build_algo()
