import os
import json
import pickle
import time

from pathlib import Path
from typing import Callable

from ray.rllib.algorithms.ppo import PPO

from policy.independent_ppo_setup import build_trainer, TrainerParameters


class ModelManager:

    def __init__(self):
        self.ENV_KWARGS_FILENAME: str = "env_kwargs.pkl"
        self.CHECKPOINT_SUBDIR: str = "checkpoint"
        self.META_FILENAME: str = "meta.json"

    def _save_checkpoint_compat(self, trainer: PPO, out_dir: Path) -> Path:
        """Handle RLlib return types for save()."""
        out_dir.mkdir(parents=True, exist_ok=True)
        ret = trainer.save(checkpoint_dir=str(out_dir))
        if isinstance(ret, str):
            return Path(ret)
        # Fallbacks for object-returning variants
        cp = getattr(ret, "checkpoint", None)
        if cp is not None:
            p = getattr(cp, "path", None)
            if p:
                return Path(p)
        p = getattr(ret, "path", None)
        if p:
            return Path(p)
        raise TypeError(
            f"Unexpected return from trainer.save(): type={type(ret)} value={ret}"
        )

    def save_bundle(
        self,
        *,
        trainer: PPO,
        bundle_dir_rel: Path,
        model_name: str,
        custom_model_config: dict,
        env_kwargs: dict,
        notes: str | None = None,
    ) -> Path:
        """
        Creates a self-contained bundle directory with:
          bundle_dir
            checkpoint/...
            meta.json
            env_kwargs.pkl

        Args:
            trainer: an RLlib Algorithm (e.g. PPO) to save.
            bundle_dir_rel: realtive directory to create (will be made if needed).
            model_name: name of the custom model used by the trainer's policies.
            custom_model_config: config dict for the custom model.
            env_kwargs: dict of kwargs used to create the env (exact Python objects).
            notes: optional string notes to include in meta.json.

        Returns:
            Path to the created bundle directory.
        """
        bundle_dir_rel.mkdir(parents=True, exist_ok=True)

        # 1) Save RLlib checkpoint
        checkpoint_dir = self._save_checkpoint_compat(
            trainer, bundle_dir_rel / self.CHECKPOINT_SUBDIR
        )

        checkpoint_abs = checkpoint_dir.resolve()
        checkpoint_path_rel = os.path.relpath(checkpoint_abs, bundle_dir_rel)

        # 2) Save env_kwargs as pickle (exact Python objects)
        env_pkl_path: Path = Path(bundle_dir_rel) / Path(self.ENV_KWARGS_FILENAME)
        with open(env_pkl_path, "wb") as f:
            pickle.dump(env_kwargs, f, protocol=pickle.HIGHEST_PROTOCOL)

        # 3) Lean meta.json
        meta = {
            "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "model_name": model_name,
            "custom_model_config": custom_model_config,
            "checkpoint_path": str(checkpoint_path_rel),
            "env_kwargs_file": str(env_pkl_path.name),
            "notes": notes or "",
        }

        (bundle_dir_rel / self.META_FILENAME).write_text(
            json.dumps(meta, indent=2), encoding="utf-8"
        )

        return bundle_dir_rel

    def load_bundle_artifacts(
        self,
        *,
        bundle_run_dir: Path,
    ) -> tuple[dict, str, dict, Path]:
        """
        Read the bundle directory and return the raw artifacts without building a trainer.

        Args:
            bundle_run_dir: Path to the bundle directory created by `save_bundle()`.

        Returns:
            env_kwargs : dict of kwargs to create the env.
            model_name : name of the custom model used by the policies.
            custom_model_cfg : config dict for the custom model.
            checkpoint_path_abs : absolute Path to the RLlib checkpoint.
        """
        run_dir = Path(bundle_run_dir)
        meta = json.loads((run_dir / self.META_FILENAME).read_text(encoding="utf-8"))

        # Unpickle env_kwargs (exact Python objects like SUMOConfig)
        env_pkl_path = run_dir / meta["env_kwargs_file"]
        with open(env_pkl_path, "rb") as f:
            env_kwargs = pickle.load(f)

        model_name: str = meta["model_name"]
        custom_model_cfg: dict = meta["custom_model_config"]

        # Get absolute checkpoint path - currently we have realtive to base directory
        base_dir = Path.cwd()
        checkpoint_path_rel: Path = (
            Path(base_dir) / Path(run_dir) / meta["checkpoint_path"]
        )
        checkpoint_path_abs = checkpoint_path_rel.absolute()

        return env_kwargs, model_name, custom_model_cfg, checkpoint_path_abs

    def build_and_restore_from_artifacts(
        self,
        *,
        register_fn: Callable,
        model_name: str,
        custom_model_config: dict,
        ckpt_path: Path,
        env_kwargs: dict,
        trainer_params: TrainerParameters,
    ) -> PPO:
        """
        Build a fresh trainer from the saved bundle artifacts and restore its state.

        Args:
            register_fn: Function to register any custom models used by the policies.
            model_name: name of the custom model used by the policies.
            custom_model_config: config dict for the custom model.
            ckpt_path: absolute Path to the RLlib checkpoint.
            env_kwargs: dict of kwargs to create the env (exact Python objects).
            trainer_params: TrainerParameters: Parameters for the PPO trainer.

        Returns:
            A fully restored PPO trainer.
        """
        register_fn()

        training_model = {
            "custom_model": model_name,
            "custom_model_config": custom_model_config,
        }

        trainer = build_trainer(
            env_kwargs=env_kwargs,
            register_fn=register_fn,
            training_model=training_model,
            trainer_params=trainer_params,
        )

        trainer.restore(str(ckpt_path))

        return trainer
