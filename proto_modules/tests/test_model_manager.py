# modules/tests/test_model_manager.py
"""
ModelManager Test Suite (SB3, Independent Agent)

This suite validates the versioned checkpoint manager for one agent:
- Snapshots: creating .zip files and updating the JSON registry
- Latest/list semantics: latest_version(), list_versions()
- Loading: load_sb3(version, env) returns a usable SB3 model
- Restoring: restore_into(policy_module, version, env) swaps the inner model
- Housekeeping: remove(version) deletes artifacts; export(version, path) copies zips
- Error handling: clear messages for missing snapshots or unknown versions

Design notes
------------
- Uses a minimal single-agent Env with a Box observation and Discrete actions,
  so we don't depend on any custom extractors for this test. The ModelManager
  is algo-agnostic, so testing with PPO is sufficient.
- We keep learn() calls extremely short (or skip them entirely) to keep tests fast.
"""

import os
import shutil
import numpy as np
import pytest
import torch

# ---- Gym/Gymnasium shim for spaces and Env base ----
import gymnasium as gym

from stable_baselines3 import PPO

from modules.intersection.model_manager import ModelManager


# ------------------------
# Minimal single-agent env
# ------------------------

class _TinyBoxEnv(gym.Env):
    """
    Minimal SB3-compatible env:
      - Observation: Box(F,) with small Gaussian noise
      - Action: Discrete(A)
      - Reward: simple shaped function of action vs timestep
    """
    metadata = {"render_modes": []}

    def __init__(self, F=4, A=3, horizon=10, seed=123):
        super().__init__()
        self.F, self.A, self.horizon = F, A, horizon
        self.t = 0
        self.rng = np.random.default_rng(seed)
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(F,), dtype=np.float32)
        self.action_space = gym.spaces.Discrete(A)

    def _obs(self):
        return self.rng.standard_normal(self.F).astype(np.float32)

    def reset(self, *, seed=None, options=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.t = 0
        return self._obs(), {}

    def step(self, action):
        self.t += 1
        # Tiny shaped reward that depends on action and time
        reward = float(-0.1 + 0.05 * (action == (self.t % self.A)))
        terminated = self.t >= self.horizon
        truncated = False
        return self._obs(), reward, terminated, truncated, {}


# -------------
# Test fixtures
# -------------

@pytest.fixture(autouse=True)
def _seed_all():
    torch.manual_seed(1234)
    np.random.seed(1234)


@pytest.fixture
def env():
    return _TinyBoxEnv(F=4, A=3, horizon=6, seed=999)


@pytest.fixture
def mgr(tmp_path):
    # Each test gets its own manager rooted in a temp dir; auto-cleaned by pytest
    root = os.path.join(tmp_path, "models")
    os.makedirs(root, exist_ok=True)
    return ModelManager(root_dir=root, agent_id="agent_X")


# -----------
# The tests
# -----------

def test_snapshot_and_registry_update(mgr, env):
    """
    Create a PPO model, snapshot it once, and verify:
      - registry exists and has exactly one entry
      - version number starts at 1
      - referenced .zip file exists on disk
      - metadata fields are present
    """
    model = PPO("MlpPolicy", env, n_steps=8, batch_size=8, verbose=0, device="cpu")
    ver = mgr.snapshot_sb3(model, algo="PPO", meta={"note": "initial"})
    assert ver == 1

    reg = mgr.list_versions()
    assert len(reg) == 1
    rec = reg[0]
    assert rec["version"] == 1
    assert rec["algo"] == "PPO"
    assert isinstance(rec.get("timestamp"), float)
    assert isinstance(rec.get("meta"), dict)
    # .zip path exists
    zip_path = os.path.join(mgr.agent_dir, rec["file"])
    assert os.path.exists(zip_path)


def test_latest_and_multiple_snapshots(mgr, env):
    """
    Snapshot twice and ensure latest_version == 2 and ordering is preserved.
    """
    model = PPO("MlpPolicy", env, n_steps=8, batch_size=8, verbose=0, device="cpu")
    v1 = mgr.snapshot_sb3(model, algo="PPO", meta={"k": 1})
    v2 = mgr.snapshot_sb3(model, algo="PPO", meta={"k": 2})
    assert v1 == 1 and v2 == 2

    assert mgr.latest_version() == 2
    reg = mgr.list_versions()
    assert [r["version"] for r in reg] == [1, 2]


def test_load_sb3_roundtrip_predict(mgr, env):
    """
    Save a model, then load it back via load_sb3 and verify predict works
    and returns a valid action in the correct bounds.
    """
    model = PPO("MlpPolicy", env, n_steps=8, batch_size=8, verbose=0, device="cpu")
    ver = mgr.snapshot_sb3(model, algo="PPO", meta={"run": "rt"})

    loaded = mgr.load_sb3(version=ver, env=env, device="cpu")
    obs, _ = env.reset()
    act, _ = loaded.predict(obs, deterministic=True)
    a = int(np.asarray(act).squeeze())
    assert 0 <= a < env.action_space.n


def test_restore_into_replaces_inner_model(mgr, env):
    """
    Create a wrapper-like object with a 'model' attribute (a PPO instance),
    call restore_into, and verify the attribute gets replaced by a freshly
    loaded model that can predict.
    """
    class Wrapper:
        pass

    w = Wrapper()
    w.model = PPO("MlpPolicy", env, n_steps=8, batch_size=8, verbose=0, device="cpu") # type: ignore

    # Snapshot a separate model to restore from
    model_src = PPO("MlpPolicy", env, n_steps=8, batch_size=8, verbose=0, device="cpu")
    ver = mgr.snapshot_sb3(model_src, algo="PPO", meta={"role": "source"})

    used_ver = mgr.restore_into(w, version=ver, env=env, device="cpu")
    assert used_ver == ver

    obs, _ = env.reset()
    act, _ = w.model.predict(obs, deterministic=True) # type: ignore
    a = int(np.asarray(act).squeeze())
    assert 0 <= a < env.action_space.n


def test_remove_deletes_zip_and_registry_entry(mgr, env):
    """
    After snapshotting, remove(version) should delete the zip and drop the
    registry entry; subsequent attempts to use that version should fail.
    """
    model = PPO("MlpPolicy", env, n_steps=8, batch_size=8, verbose=0, device="cpu")
    ver = mgr.snapshot_sb3(model, algo="PPO", meta={"to_remove": True})

    rec = mgr.list_versions()[0]
    zip_path = os.path.join(mgr.agent_dir, rec["file"])
    assert os.path.exists(zip_path)

    mgr.remove(ver)
    assert mgr.list_versions() == []
    assert not os.path.exists(zip_path)

    # Now loading that version should raise
    with pytest.raises(KeyError):
        _ = mgr.load_sb3(version=ver, env=env)


def test_export_copies_zip(mgr, env, tmp_path):
    """
    export(version, dest) must copy the correct zip to dest, creating folders as needed.
    """
    model = PPO("MlpPolicy", env, n_steps=8, batch_size=8, verbose=0, device="cpu")
    ver = mgr.snapshot_sb3(model, algo="PPO", meta={"export": 1})

    # Compute src and dest paths
    rec = mgr.list_versions()[0]
    src = os.path.join(mgr.agent_dir, rec["file"])

    dest_dir = os.path.join(tmp_path, "export_here")
    dest_path = os.path.join(dest_dir, "copied.zip")
    assert not os.path.exists(dest_path)

    mgr.export(ver, dest_path)
    assert os.path.exists(dest_path)
    # Files should be identical in size at least
    assert os.path.getsize(dest_path) == os.path.getsize(src)


def test_errors_no_snapshots_yet(tmp_path, env):
    """
    When there are no snapshots:
      - latest_version() returns None
      - load_sb3(None, ...) raises KeyError
      - restore_into(None, ...) raises KeyError
    """
    mgr = ModelManager(root_dir=os.path.join(tmp_path, "models_empty"), agent_id="solo")
    assert mgr.latest_version() is None

    with pytest.raises(KeyError):
        _ = mgr.load_sb3(version=None, env=env)

    class W:
        def __init__(self, env):  # have a 'model' to satisfy restore_into path
            self.model = PPO("MlpPolicy", env, n_steps=8, batch_size=8, verbose=0, device="cpu")

    with pytest.raises(KeyError):
        _ = mgr.restore_into(W(env), version=None, env=env)


def test_error_missing_version(mgr, env):
    """
    Requesting a non-existent version should raise a clear KeyError.
    """
    model = PPO("MlpPolicy", env, n_steps=8, batch_size=8, verbose=0, device="cpu")
    _ = mgr.snapshot_sb3(model, algo="PPO")
    with pytest.raises(KeyError):
        _ = mgr.load_sb3(version=999, env=env)
    with pytest.raises(KeyError):
        _ = mgr.restore_into(type("W", (), {"model": model})(), version=999, env=env)
