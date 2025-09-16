# model_manager.py
"""
Versioned checkpoint manager for a single independent agent using SB3.

What it stores
--------------
- One .zip SB3 model file per version (created by SB3's `model.save()`), which
  includes your custom FeaturesExtractor (state embedder + GATv2), policy/value
  heads, and optimiser state if the algo supports it.
- A lightweight JSON registry with metadata per version.

Typical usage per agent
-----------------------
mgr = ModelManager(root_dir="models", agent_id="int_23")

# During/after training:
ver = mgr.snapshot_sb3(policy.model, meta={"algo": "PPO", "note": "after 100k steps"})

# Later, to load:
loaded = mgr.load_sb3(version=ver, env=env)  # returns an SB3 algorithm instance
# or:
policy = SB3PolicyModule(...); mgr.restore_into(policy, version=ver, env=env)
"""

from __future__ import annotations
from typing import Any, Dict, Optional, List, Type
import os
import json
import time
import shutil

import torch
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3 import PPO, A2C

try:
    from sb3_contrib import MaskablePPO

    _HAS_CONTRIB = True
except Exception:
    _HAS_CONTRIB = False

_ALGOS: Dict[str, Type[BaseAlgorithm]] = {
    "PPO": PPO,
    "A2C": A2C,
}
if _HAS_CONTRIB:
    _ALGOS["MaskablePPO"] = MaskablePPO


REGISTRY_NAME = "registry.json"


class ModelManager:
    """
    Versioned SB3 model manager bound to one agent (independent learner).

    Filesystem layout (example):
      root_dir/
        int_23/
          registry.json
          v0001_PPO_2025-08-21T10-30-03.zip
          v0002_PPO_2025-08-21T10-45-10.zip
          ...

    Notes
    -----
    - The .zip produced/consumed is the native SB3 format.
    - `env` must be provided when loading/restoring so SB3 rebinds the policy.
    """

    def __init__(self, root_dir: str = "models", agent_id: str = "agent"):
        self.root_dir = root_dir
        self.agent_id = agent_id
        self.agent_dir = os.path.join(self.root_dir, self.agent_id)
        os.makedirs(self.agent_dir, exist_ok=True)

        self.registry_path = os.path.join(self.agent_dir, REGISTRY_NAME)
        self._reg: Dict[str, Any] = self._load_registry()

    # ---------- registry I/O ----------

    def _load_registry(self) -> Dict[str, Any]:
        if os.path.exists(self.registry_path):
            with open(self.registry_path, "r", encoding="utf-8") as f:
                return json.load(f)
        return {"versions": []}

    def _save_registry(self) -> None:
        tmp = self.registry_path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(self._reg, f, indent=2, sort_keys=True)
        os.replace(tmp, self.registry_path)

    # ---------- version helpers ----------

    def _next_version(self) -> int:
        vers = [int(item["version"]) for item in self._reg["versions"]]
        return (max(vers) + 1) if vers else 1

    def latest_version(self) -> Optional[int]:
        return self._reg["versions"][-1]["version"] if self._reg["versions"] else None

    def list_versions(self) -> List[Dict[str, Any]]:
        return list(self._reg["versions"])

    def _zip_name(self, version: int, algo: str) -> str:
        ts = time.strftime("%Y-%m-%dT%H-%M-%S", time.localtime())
        return f"v{version:04d}_{algo}_{ts}.zip"

    # ---------- SB3 snapshot / load ----------

    def snapshot_sb3(
        self,
        model: BaseAlgorithm,
        *,
        algo: Optional[str] = None,
        meta: Optional[Dict[str, Any]] = None,
    ) -> int:
        """
        Save the given SB3 model to a new versioned .zip and record metadata.

        Parameters
        ----------
        model : BaseAlgorithm
            The SB3 algorithm instance (e.g., PPO/MaskablePPO/A2C).
        algo : str, optional
            Algo name for labelling; if None, tries model.__class__.__name__.
        meta : dict, optional
            Arbitrary metadata to store (e.g., step count, lr, notes).
        """
        version = self._next_version()
        algo_name = str(algo or model.__class__.__name__)
        zip_name = self._zip_name(version, algo_name)
        zip_path = os.path.join(self.agent_dir, zip_name)

        # SB3 native save (includes policy + custom features extractor)
        model.save(zip_path)

        rec = {
            "version": version,
            "file": zip_name,
            "algo": algo_name,
            "timestamp": time.time(),
            "meta": meta or {},
        }
        self._reg["versions"].append(rec)
        self._save_registry()
        return version

    def _version_to_path(self, version: int) -> str:
        for rec in self._reg["versions"]:
            if rec["version"] == version:
                return os.path.join(self.agent_dir, rec["file"])
        raise KeyError(f"Version {version} not found for agent '{self.agent_id}'")

    def load_sb3(
        self, version: Optional[int], env, device: str | torch.device = "auto"
    ) -> BaseAlgorithm:
        """
        Load the SB3 model for 'version' and return the algorithm instance.
        If version is None, the latest snapshot is used.
        """
        if version is None:
            version = self.latest_version()
        if version is None:
            raise KeyError("No snapshots available to load")

        rec = next((r for r in self._reg["versions"] if r["version"] == version), None)
        if rec is None:
            raise KeyError(f"Version {version} not found for agent '{self.agent_id}'")
        algo_name = str(rec.get("algo", "PPO"))
        AlgoClass = _ALGOS.get(algo_name)
        if AlgoClass is None:
            raise ValueError(
                f"Unsupported algo '{algo_name}' in registry for version {version}"
            )

        zip_path = self._version_to_path(version)
        return AlgoClass.load(zip_path, env=env, device=device)

    def restore_into(
        self,
        policy_module,
        version: Optional[int],
        env,
        device: str | torch.device = "auto",
    ) -> int:
        """
        Load the given 'version' directly into an SB3PolicyModule wrapper (in-place).
        Returns the version used.
        """
        if version is None:
            version = self.latest_version()
        if version is None:
            raise KeyError("No snapshots available to restore")

        rec = next((r for r in self._reg["versions"] if r["version"] == version), None)
        if rec is None:
            raise KeyError(f"Version {version} not found for agent '{self.agent_id}'")
        algo_name = str(rec.get("algo", "PPO"))
        AlgoClass = _ALGOS.get(algo_name)
        if AlgoClass is None:
            raise ValueError(
                f"Unsupported algo '{algo_name}' in registry for version {version}"
            )

        zip_path = self._version_to_path(version)
        policy_module.model = AlgoClass.load(zip_path, env=env, device=device)
        return version

    # ---------- housekeeping ----------

    def remove(self, version: int) -> None:
        """Delete a given version (zip + registry entry)."""
        zip_path = self._version_to_path(version)
        if os.path.exists(zip_path):
            os.remove(zip_path)
        self._reg["versions"] = [
            rec for rec in self._reg["versions"] if rec["version"] != version
        ]
        self._save_registry()

    def export(self, version: int, dest_path: str) -> None:
        """Copy a versioned zip to a specific destination path (for deployment)."""
        src = self._version_to_path(version)
        os.makedirs(os.path.dirname(dest_path), exist_ok=True)
        shutil.copy2(src, dest_path)
