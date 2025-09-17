from ray.rllib.env.wrappers.pettingzoo_env import ParallelPettingZooEnv
from environments.pz_multi_tls_env import MultiTLSParallelEnv

from typing import Callable


class RLlibPZEnv(ParallelPettingZooEnv):
    """
    RLlib-compatible env class. RLlib will call RLlibPZEnv(env_config),
    and build the underlying PettingZoo ParallelEnv from env_config["env_kwargs"].
    """

    def __init__(self, env_config=None):
        env_config = env_config or {}
        env_kwargs = env_config.get("env_kwargs", {})
        super().__init__(MultiTLSParallelEnv(**env_kwargs))


def make_rllib_env(env_kwargs) -> Callable:
    """
    For rollouts outside the trainer (e.g. baselines), return a callable
    that constructs the wrapped env with the same kwargs.
    """

    def _env_creator() -> RLlibPZEnv:
        return RLlibPZEnv({"env_kwargs": env_kwargs})

    return _env_creator
