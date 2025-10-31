import os
import ray
from enum import Enum

from PySide6.QtCore import QObject, Signal

from utils.sumo_helpers import close_sumo
from event_bus import event_bus, EventNames


class RunMode(Enum):
    TRAIN_EVAL = "Train + Eval"
    EVAL_ONLY = "Eval only"
    BATCH_EVAL_ONLY = "Batch eval only"


class SimulationRunner(QObject):
    finished = Signal()
    error = Signal(str)

    def __init__(
        self,
        run_mode: RunMode,
        config_file: str,
        hyperparams_file: str,
        outdir: str,
        bundle_run_dir: str | None = None,
        already_evaluated: bool = False,
    ):
        super().__init__()
        self.run_mode = run_mode
        self.config_file = config_file
        self.hyperparams_file = hyperparams_file
        self.outdir = outdir
        self.bundle_run_dir = bundle_run_dir
        self.already_evaluated = already_evaluated
        self._stop_requested = False

    def request_stop(self):
        # Cooperative stop; the pipeline should listen for this
        self._stop_requested = True
        try:
            close_sumo()
            # Stop ray if running
            if ray.is_initialized():
                ray.shutdown()
        except Exception:
            pass

    def start(self):
        try:
            # ensure no GUI backend when runner is in a worker thread
            os.environ.setdefault("MPLBACKEND", "Agg")
            try:
                import matplotlib as _mpl  # noqa

                _mpl.use("Agg", force=True)
            except Exception:
                pass

            if self._stop_requested:
                event_bus.emit(EventNames.SIMULATION_DONE.value, "Stopped")
                return

            if self.run_mode is RunMode.TRAIN_EVAL:
                from run import run as _rl_run

                _rl_run(
                    config_file=self.config_file,
                    hyperparams_file=self.hyperparams_file,
                    outdir=self.outdir,
                    event_bus=event_bus,
                )
            elif self.run_mode is RunMode.EVAL_ONLY:
                from eval_run import run_eval_only as _rl_eval_run

                _rl_eval_run(
                    bundle_run_dir=self.bundle_run_dir,
                    new_config_file=self.config_file,
                    outdir=self.outdir,
                    event_bus=event_bus,
                )
            elif self.run_mode is RunMode.BATCH_EVAL_ONLY:
                from batch_eval import run_eval_only as _rl_batch_eval_run

                _rl_batch_eval_run(
                    bundle_run_dir=self.bundle_run_dir,
                    new_config_file=self.config_file,
                    outdir=self.outdir,
                    already_evaluated=self.already_evaluated,
                    event_bus=event_bus,
                )

            if self._stop_requested:
                event_bus.emit(EventNames.SIMULATION_DONE.value, "Stopped")
            else:
                event_bus.emit(EventNames.SIMULATION_DONE.value, "Completed")

        except Exception as e:
            msg = str(e)
            # If we stopped on purpose, swallow noisy socket/connection shutdowns
            if self._stop_requested and "connection already closed" in msg.lower():
                event_bus.emit(EventNames.SIMULATION_DONE.value, "Stopped")
            else:
                self.error.emit(msg)
                try:
                    event_bus.emit(
                        EventNames.SIMULATION_FAILED.value, f"Simulation error: {msg}"
                    )
                except Exception:
                    pass
        finally:
            self.finished.emit()
