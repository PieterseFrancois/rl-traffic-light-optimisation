# tests/test_policy_sb3.py
"""
SB3PolicyModule Test Suite (Independent Agent)

Covers:
1) Initialisation and predict() on a minimal env
2) Short learn() run to ensure training executes end-to-end
3) Save/Load round-trip preserves behaviour (deterministic predict)
4) Access to features-extractor self-embedding via helper
5) Error handling for unknown algorithm names
6) Late env binding: predict() raises before set_env, then works after
7) MaskablePPO path smoke test (skipped if sb3_contrib not installed)
8) Features extractor wiring: NeighbourGNNFeatures is used
"""

import os
import numpy as np
import torch
import pytest
import gymnasium as gym

from modules.intersection.policy import SB3PolicyModule
from modules.intersection.features_extractor import NeighbourGNNFeatures


# ------------------------
# Minimal single-agent env
# ------------------------


class _TinyIntersectionEnv(gym.Env):
    """
    Minimal env for one intersection.
    - Fixed neighbour count K for this agent (fits SB3's fixed-shape requirement).
    - Publishes zero vector for missing neighbours naturally if you choose to.
    """

    metadata = {"render_modes": []}

    def __init__(self, F_raw=6, K=2, D_emb=8, A=3, horizon=10, seed=123):
        super().__init__()
        self.F_raw, self.K, self.D_emb, self.A = F_raw, K, D_emb, A
        self.horizon = horizon
        self.rng = np.random.default_rng(seed)
        self.t = 0

        self.observation_space = gym.spaces.Dict(
            {
                "self_raw": gym.spaces.Box(
                    low=-np.inf, high=np.inf, shape=(F_raw,), dtype=np.float32
                ),
                "nbr_embed": gym.spaces.Box(
                    low=-np.inf, high=np.inf, shape=(K, D_emb), dtype=np.float32
                ),
            }
        )
        self.action_space = gym.spaces.Discrete(A)

    def _obs(self):
        self_raw = self.rng.standard_normal(self.F_raw).astype(np.float32)
        nbr = self.rng.standard_normal((self.K, self.D_emb)).astype(np.float32)
        return {"self_raw": self_raw, "nbr_embed": nbr}

    def reset(self, *, seed=None, options=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.t = 0
        return self._obs(), {}

    def step(self, action):
        self.t += 1
        reward = float(-0.1 + 0.05 * (action == (self.t % self.A)))
        terminated = self.t >= self.horizon
        truncated = False
        info = {}
        return self._obs(), reward, terminated, truncated, info


# ----------------
# Test parameters
# ----------------

F_RAW = 6
K_NEI = 2
D_EMB = 8
N_ACT = 3

EXTRACTOR_KW = dict(
    self_raw_dim=F_RAW,
    embed_dim=D_EMB,
    gat_hidden=16,
    gat_heads=2,
    edge_index=None,  # default star (0 <-> i)
    device=torch.device("cpu"),
)


# -----------
# The tests
# -----------


@pytest.fixture(autouse=True)
def _seed_all():
    torch.manual_seed(1234)
    np.random.seed(1234)


@pytest.mark.parametrize("algo_name", ["PPO"] + ["MaskablePPO"])
def test_init_and_predict(algo_name):
    """
    1) SB3PolicyModule initialises and predict() returns a valid action.
    """
    env = _TinyIntersectionEnv(F_raw=F_RAW, K=K_NEI, D_emb=D_EMB, A=N_ACT, horizon=5)
    policy = SB3PolicyModule(
        env=env,
        algo_name=algo_name,
        extractor_kwargs=EXTRACTOR_KW,
        net_arch=dict(pi=[32], vf=[32]),
        algo_kwargs=dict(n_steps=8, batch_size=8),  # light
        device="cpu",
        verbose=0,
    )

    obs, _ = env.reset()
    action, _ = policy.predict(obs, deterministic=True)
    assert isinstance(action, np.ndarray)
    a = int(np.asarray(action).squeeze())
    assert 0 <= a < N_ACT


@pytest.mark.parametrize("algo_name", ["PPO"] + ["MaskablePPO"])
def test_learn_short_run(algo_name):
    """
    2) A short learn() run executes without error (sanity check training path).
    For MaskablePPO, wrap the env with ActionMasker providing a boolean mask.
    """
    base_env = _TinyIntersectionEnv(
        F_raw=F_RAW, K=K_NEI, D_emb=D_EMB, A=N_ACT, horizon=8
    )

    if algo_name == "MaskablePPO":
        from sb3_contrib.common.wrappers.action_masker import ActionMasker

        def mask_fn(_obs):
            return np.ones(base_env.action_space.n, dtype=bool)  # type: ignore

        env = ActionMasker(base_env, mask_fn)
    else:
        env = base_env

    policy = SB3PolicyModule(
        env=env,
        algo_name=algo_name,
        extractor_kwargs=EXTRACTOR_KW,
        net_arch=dict(pi=[32], vf=[32]),
        algo_kwargs=dict(n_steps=16, batch_size=16, learning_rate=3e-4),
        device="cpu",
        verbose=0,
    )
    policy.learn(total_timesteps=64)  # tiny run


@pytest.mark.parametrize("algo_name", ["PPO"])
def test_save_and_load_roundtrip(tmp_path, algo_name):
    """
    3) Save/Load preserves deterministic predict output on the same observation.
    """
    env = _TinyIntersectionEnv(
        F_raw=F_RAW, K=K_NEI, D_emb=D_EMB, A=N_ACT, horizon=5, seed=321
    )
    policy = SB3PolicyModule(
        env=env,
        algo_name=algo_name,
        extractor_kwargs=EXTRACTOR_KW,
        net_arch=dict(pi=[16], vf=[16]),
        algo_kwargs=dict(n_steps=8, batch_size=8),
        device="cpu",
        verbose=0,
    )

    obs, _ = env.reset(seed=999)
    act_before, _ = policy.predict(obs, deterministic=True)

    path = os.path.join(tmp_path, "agent_sb3.zip")
    policy._save(path)

    policy2 = SB3PolicyModule._load(
        path=path,
        env=env,
        algo_name=algo_name,
        extractor_kwargs=EXTRACTOR_KW,
        net_arch=dict(pi=[16], vf=[16]),
        algo_kwargs=dict(n_steps=8, batch_size=8),
        device="cpu",
        verbose=0,
    )

    act_after, _ = policy2.predict(obs, deterministic=True)
    assert int(np.asarray(act_before).squeeze()) == int(
        np.asarray(act_after).squeeze()
    ), "Deterministic action should match after load"


@pytest.mark.parametrize("algo_name", ["PPO"])
def test_get_last_self_embedding_helper(algo_name):
    """
    4) After a forward path (predict/learn), get_last_self_embedding returns a (D_emb,) tensor.
    """
    env = _TinyIntersectionEnv(F_raw=F_RAW, K=K_NEI, D_emb=D_EMB, A=N_ACT, horizon=3)
    policy = SB3PolicyModule(
        env=env,
        algo_name=algo_name,
        extractor_kwargs=EXTRACTOR_KW,
        net_arch=dict(pi=[16], vf=[16]),
        algo_kwargs=dict(n_steps=8, batch_size=8),
        device="cpu",
        verbose=0,
    )

    # Before any forward, it should be None
    assert policy.get_last_self_embedding() is None

    obs, _ = env.reset()
    _ = policy.predict(obs, deterministic=True)
    emb = policy.get_last_self_embedding()
    assert emb is not None, "Expected a cached self embedding after predict()"
    assert isinstance(emb, torch.Tensor)
    assert emb.shape == (D_EMB,)


def test_invalid_algo_raises():
    """
    5) Unknown algorithm names should raise a clear ValueError.
    """
    env = _TinyIntersectionEnv()
    with pytest.raises(ValueError):
        _ = SB3PolicyModule(
            env=env,
            algo_name="NotAnAlgo",
            extractor_kwargs=EXTRACTOR_KW,
            device="cpu",
            verbose=0,
        )


def test_late_env_binding_predict_raises_then_ok():
    """
    6) Late env binding:
       - predict() must raise before set_env
       - after set_env, predict() returns a valid action
    """
    env = _TinyIntersectionEnv(F_raw=F_RAW, K=K_NEI, D_emb=D_EMB, A=N_ACT, horizon=5)
    policy = SB3PolicyModule(
        env=None,  # late binding
        algo_name="PPO",
        extractor_kwargs=EXTRACTOR_KW,
        net_arch=dict(pi=[16], vf=[16]),
        device="cpu",
        verbose=0,
    )

    obs, _ = env.reset()
    with pytest.raises(RuntimeError):
        _ = policy.predict(obs, deterministic=True)

    policy.set_env(env)
    action, _ = policy.predict(obs, deterministic=True)
    a = int(np.asarray(action).squeeze())
    assert 0 <= a < N_ACT


def test_maskableppo_smoke_with_trivial_mask():
    """
    7) MaskablePPO path smoke test with a trivial all-True mask.
    """
    from sb3_contrib.common.wrappers.action_masker import ActionMasker

    base_env = _TinyIntersectionEnv(
        F_raw=F_RAW, K=K_NEI, D_emb=D_EMB, A=N_ACT, horizon=5
    )

    def mask_fn(_obs):
        return np.ones(base_env.action_space.n, dtype=bool)  # type: ignore

    env = ActionMasker(base_env, mask_fn)
    policy = SB3PolicyModule(
        env=env,
        algo_name="MaskablePPO",
        extractor_kwargs=EXTRACTOR_KW,
        net_arch=dict(pi=[16], vf=[16]),
        algo_kwargs=dict(n_steps=8, batch_size=8),
        device="cpu",
        verbose=0,
    )
    obs, _ = env.reset()
    action, _ = policy.predict(obs, deterministic=True)
    a = int(np.asarray(action).squeeze())
    assert 0 <= a < N_ACT


def test_extractor_wiring_is_neighbour_gnn():
    """
    8) Features extractor wiring: ensure the model uses NeighbourGNNFeatures.
    """
    env = _TinyIntersectionEnv(F_raw=F_RAW, K=K_NEI, D_emb=D_EMB, A=N_ACT, horizon=3)
    policy = SB3PolicyModule(
        env=env,
        algo_name="PPO",
        extractor_kwargs=EXTRACTOR_KW,
        net_arch=dict(pi=[16], vf=[16]),
        device="cpu",
        verbose=0,
    )
    # one forward to ensure policy is fully built
    obs, _ = env.reset()
    _ = policy.predict(obs, deterministic=True)

    # Inspect the extractor instance
    assert policy.model is not None
    fx = policy.model.policy.features_extractor  # type: ignore[attr-defined]
    assert isinstance(
        fx, NeighbourGNNFeatures
    ), "Expected NeighbourGNNFeatures as the extractor"
