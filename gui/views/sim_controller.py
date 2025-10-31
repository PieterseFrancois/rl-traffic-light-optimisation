import os
from datetime import datetime, timezone

from PySide6.QtCore import QThread, Signal, Slot, Qt, QTimer
from PySide6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QFileDialog,
    QLineEdit,
    QMessageBox,
    QComboBox,
    QCheckBox,
    QPlainTextEdit,
)

from event_bus import event_bus, EventNames
from gui.components.live_chart import LiveKpiPlot
from gui.runners.sim_runner import RunMode, SimulationRunner


class SimControllerView(QWidget):
    # GUI-thread relays
    sig_log_line = Signal(str)
    sig_status = Signal(str)
    sig_summary = Signal(str)
    sig_enable_start = Signal(bool)
    sig_enable_stop = Signal(bool)
    sig_kpis = Signal(dict, str)

    def __init__(
        self,
        config_file: str,
        hyperparams_file: str,
        plot_stride: int,
    ):
        super().__init__()
        self.setObjectName("SimControllerView")

        self._config_file = config_file
        self._hyperparams_file = hyperparams_file

        self._runner_thread: QThread | None = None
        self._runner: SimulationRunner | None = None
        self._is_running: bool = False

        self._stop_in_progress: bool = False
        self._tearing_down: bool = False

        # KPI throttle
        self._plot_stride = max(1, int(plot_stride))
        self._kpi_counter = 0

        # UI
        root = QVBoxLayout(self)

        # Output dir
        out_row = QHBoxLayout()
        self._dir_edit = QLineEdit()
        self._dir_edit.setPlaceholderText("Output folder for run artefacts")
        out_browse = QPushButton("Browse")
        out_browse.setObjectName("sumoBrowseBtn")
        out_browse.clicked.connect(self._browse_dir)
        out_row.addWidget(self._dir_edit, 1)
        out_row.addWidget(out_browse)
        root.addLayout(out_row)

        # Run mode
        mode_row = QHBoxLayout()
        mode_row.addWidget(QLabel("Run mode:"))
        self._mode = QComboBox()
        self._mode.setObjectName("simRunModeCombo")
        self._mode.addItem(RunMode.TRAIN_EVAL.value, RunMode.TRAIN_EVAL)
        self._mode.addItem(RunMode.EVAL_ONLY.value, RunMode.EVAL_ONLY)
        self._mode.addItem(RunMode.BATCH_EVAL_ONLY.value, RunMode.BATCH_EVAL_ONLY)
        mode_row.addWidget(self._mode, 1)
        root.addLayout(mode_row)

        # bundle_run_dir row (eval modes)
        bundle_row = QHBoxLayout()
        self._bundle_edit = QLineEdit()
        self._bundle_edit.setPlaceholderText("bundle_run_dir (for eval modes)")
        bundle_btn = QPushButton("Browse bundle dir")
        bundle_btn.clicked.connect(self._browse_bundle)
        bundle_row.addWidget(self._bundle_edit, 1)
        bundle_row.addWidget(bundle_btn)
        self._bundle_row_w = QWidget()
        self._bundle_row_w.setLayout(bundle_row)
        root.addWidget(self._bundle_row_w)

        self._already_chk = QCheckBox("Already evaluated")
        root.addWidget(self._already_chk)

        # Controls
        btns = QHBoxLayout()
        self._start = QPushButton("Start")
        self._start.setObjectName("simStartBtn")
        self._start.setProperty("variant", "primary")
        self._stop = QPushButton("Force Stop")
        self._stop.setObjectName("simStopBtn")
        self._stop.setEnabled(False)
        btns.addWidget(self._start)
        btns.addWidget(self._stop)
        root.addLayout(btns)

        # Status + highlighted summary row
        self._status = QLabel("Idle")
        self._status.setObjectName("simStatus")

        self._summary = QLabel(
            "t: -, veh: -, avg_wait: -, time_loss: -, acc_wait: -, "
            "speed: -, arrived: -, departed: -"
        )
        self._summary.setObjectName("simSummaryRow")
        self._summary.setProperty("variant", "highlight")

        root.addWidget(self._status)
        root.addWidget(self._summary)

        # Split view: left = terminal, right = live chart
        split = QHBoxLayout()

        # Left: terminal log
        left = QVBoxLayout()
        self._log = QPlainTextEdit()
        self._log.setObjectName("simLogPane")
        self._log.setReadOnly(True)
        left.addWidget(self._log)
        split.addLayout(left, 1)

        # Right: live KPI overlay chart
        right = QVBoxLayout()
        self._chart = LiveKpiPlot(max_points=4000)
        right.addWidget(self._chart)
        split.addLayout(right, 2)

        root.addLayout(split)

        # Wire buttons/mode
        self._start.clicked.connect(self._on_start_clicked)
        self._stop.clicked.connect(self._on_stop_clicked)
        self._mode.currentIndexChanged.connect(self._apply_mode_visibility)
        self._apply_mode_visibility()

        # GUI-thread relays
        self.sig_log_line.connect(self._append_log, Qt.QueuedConnection)
        self.sig_status.connect(self._status.setText, Qt.QueuedConnection)
        self.sig_summary.connect(self._summary.setText, Qt.QueuedConnection)
        self.sig_enable_start.connect(self._start.setEnabled, Qt.QueuedConnection)
        self.sig_enable_stop.connect(self._stop.setEnabled, Qt.QueuedConnection)
        self.sig_kpis.connect(self._on_kpis, Qt.QueuedConnection)

        # Event bus subscriptions
        event_bus.subscribe(
            EventNames.SIMULATION_INFO.value,
            lambda msg: self.sig_log_line.emit(f"[info] {msg or ''}"),
        )
        event_bus.subscribe(
            EventNames.BASELINE_STARTED.value,
            lambda msg: self.sig_log_line.emit(f"[baseline] {msg or ''}"),
        )
        event_bus.subscribe(
            EventNames.EVALUATION_STARTED.value,
            lambda msg: self.sig_log_line.emit(f"[eval] {msg or ''}"),
        )

        # Live KPI streams
        event_bus.subscribe(
            EventNames.SIMULATION_KPIS_BASELINE.value,
            lambda payload: self.sig_kpis.emit(payload, "baseline"),
        )
        event_bus.subscribe(
            EventNames.SIMULATION_KPIS_EVAL.value,
            lambda payload: self.sig_kpis.emit(payload, "eval"),
        )

        # Teardown on GUI thread
        event_bus.subscribe(
            EventNames.SIMULATION_DONE.value,
            lambda msg: QTimer.singleShot(0, lambda: self._on_sim_done(msg)),
        )
        event_bus.subscribe(
            EventNames.SIMULATION_FAILED.value,
            lambda msg: QTimer.singleShot(0, lambda: self._on_sim_failed(msg)),
        )

    # UI helpers

    def _browse_dir(self):
        d = QFileDialog.getExistingDirectory(self, "Choose output folder")
        if d:
            self._dir_edit.setText(d)

    def _browse_bundle(self):
        d = QFileDialog.getExistingDirectory(self, "Choose bundle_run_dir")
        if d:
            self._bundle_edit.setText(d)

    def _apply_mode_visibility(self):
        mode: RunMode = self._mode.currentData()
        show_bundle = mode in (RunMode.EVAL_ONLY, RunMode.BATCH_EVAL_ONLY)
        show_already = mode is RunMode.BATCH_EVAL_ONLY
        self._bundle_row_w.setVisible(show_bundle)
        self._already_chk.setVisible(show_already)

    # Start/Stop

    def _on_start_clicked(self):
        out_dir = self._dir_edit.text().strip()
        if not out_dir:
            QMessageBox.warning(
                self, "Missing folder", "Please choose an output directory."
            )
            return
        if not os.path.isdir(out_dir):
            try:
                os.makedirs(out_dir, exist_ok=True)
            except Exception as e:
                QMessageBox.critical(
                    self, "Folder error", f"Cannot create output folder:\n{e}"
                )
                return

        # Guard against concurrent runs
        if self._is_running or (
            self._runner_thread and self._runner_thread.isRunning()
        ):
            self.sig_status.emit("Busy shutting down previous run…")
            return

        mode: RunMode = self._mode.currentData()
        bundle = self._bundle_edit.text().strip() or None

        if mode is RunMode.EVAL_ONLY and not bundle:
            QMessageBox.warning(
                self, "Missing input", "bundle_run_dir is required for Eval only."
            )
            return
        if mode is RunMode.BATCH_EVAL_ONLY and not bundle:
            QMessageBox.warning(
                self, "Missing input", "bundle_run_dir is required for Batch eval only."
            )
            return

        # Intro message
        intro = (
            f"---- {mode.value} ----\n"
            f"Starting run at {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S %Z')}."
            f"\nConfig: {self._config_file}"
            f"\nHyperparams: {self._hyperparams_file}"
            f"\nOut: {out_dir}"
            + (f"\nBundle: {bundle}" if bundle else "")
            + "\n--------------------------"
        )
        event_bus.emit(EventNames.SIMULATION_INFO.value, intro)

        # Clear chart and summary for a fresh run
        self._chart.clear()
        self.sig_summary.emit(
            "t: -, veh: -, avg_wait: -, time_loss: -, acc_wait: -, "
            "speed: -, arrived: -, departed: -"
        )
        self._kpi_counter = 0  # reset throttle

        # Runner
        self._runner_thread = QThread(self)
        self._runner = SimulationRunner(
            run_mode=mode,
            config_file=self._config_file,
            hyperparams_file=self._hyperparams_file,
            outdir=out_dir,
            bundle_run_dir=bundle,
            already_evaluated=self._already_chk.isChecked(),
        )
        self._runner.moveToThread(self._runner_thread)
        self._runner_thread.started.connect(self._runner.start)
        self._runner.error.connect(self._on_runner_error)
        self._runner.finished.connect(self._runner_thread.quit)
        self._runner.finished.connect(self._runner.deleteLater)
        self._runner_thread.finished.connect(self._runner_thread.deleteLater)
        self._runner_thread.start()

        self._is_running = True
        self._stop_in_progress = False
        self._tearing_down = False
        self.sig_status.emit("Running")
        self.sig_enable_start.emit(False)
        self.sig_enable_stop.emit(True)

    def _join_thread(self, th: QThread | None, timeout_ms: int) -> bool:
        if not th:
            return True
        try:
            th.quit()
            if th is not QThread.currentThread():
                return th.wait(timeout_ms)
            return True
        except Exception:
            return False

    def _on_stop_clicked(self):
        if self._stop_in_progress:
            return
        self._stop_in_progress = True
        self.sig_enable_start.emit(False)
        self.sig_enable_stop.emit(False)
        self.sig_status.emit("Stopping…")
        self.sig_log_line.emit("[info] Stop requested.")

        # Ask runner to stop and wait
        try:
            if self._runner:
                self._runner.request_stop()
        except Exception:
            pass

        self._join_thread(self._runner_thread, 15000)

        # Release
        self._runner = None
        self._runner_thread = None

        # UI
        self._is_running = False
        self._stop_in_progress = False
        self.sig_status.emit("Stopped")
        self.sig_enable_start.emit(True)
        self.sig_enable_stop.emit(False)
        self.sig_log_line.emit("[info] Stopped.\n")

    # Bus / UI slots

    def _on_runner_error(self, msg: str):
        self.sig_status.emit(f"Run error: {msg}")
        if not self._stop_in_progress:
            QTimer.singleShot(0, lambda: QMessageBox.critical(self, "Run error", msg))

    @Slot(str)
    def _append_log(self, line: str):
        if not line:
            return
        self._log.appendPlainText(line)
        # Rolling buffer
        max_blocks = 5000
        doc = self._log.document()
        while doc.blockCount() > max_blocks:
            cursor = self._log.textCursor()
            cursor.movePosition(cursor.Start)
            cursor.select(cursor.LineUnderCursor)
            cursor.removeSelectedText()
            cursor.deleteChar()

    def _format_summary(self, payload: dict) -> str:
        return (
            f"t: {int(payload.get('sim_time', 0))}, "
            f"veh: {int(payload.get('n_vehicles', 0))}, "
            f"avg_wait: {payload.get('avg_wait', 0.0):.2f}, "
            f"time_loss: {payload.get('avg_time_loss', 0.0):.2f}, "
            f"acc_wait: {payload.get('avg_acc_wait', 0.0):.2f}, "
            f"speed: {payload.get('mean_speed_vehicle', 0.0):.2f}, "
            f"arrived: {int(payload.get('throughput_arrived', 0))}, "
            f"departed: {int(payload.get('throughput_departed', 0))}"
        )

    @Slot(dict, str)
    def _on_kpis(self, payload: dict, phase: str):
        """
        Always runs on the GUI thread. Throttle plot updates to avoid
        overwhelming the UI, but keep the summary row fully real-time.
        """
        # Update summary every tick
        self.sig_summary.emit(self._format_summary(payload))

        # Throttled chart update
        self._kpi_counter += 1
        if (self._kpi_counter % self._plot_stride) != 0:
            return

        t = int(payload.get("sim_time", 0))
        # The LiveKpiPlot component keeps per-metric time series internally.
        # Pass the full payload so it can update all metric buffers and
        # only render the currently selected metric.
        self._chart.update_point(phase, t, payload)

    def _teardown_run(self, clean: bool):
        if self._tearing_down:
            return
        self._tearing_down = True

        self._join_thread(self._runner_thread, 15000)

        self._runner = None
        self._runner_thread = None

        self._is_running = False
        self._stop_in_progress = False
        self.sig_enable_start.emit(True)
        self.sig_enable_stop.emit(False)
        self.sig_status.emit("Idle" if clean else "Error")
        self.sig_summary.emit(
            "t: -, veh: -, avg_wait: -, time_loss: -, acc_wait: -, speed: -, arrived: -, departed: -"
        )
        self._tearing_down = False

    def _on_sim_done(self, msg: str | None):
        self.sig_log_line.emit(f"[done] {msg or 'Simulation finished.'}")
        self._teardown_run(clean=True)

    def _on_sim_failed(self, msg: str | None):
        if msg and "connection already closed" in msg.lower():
            self.sig_log_line.emit("[done] Stopped")
            self._teardown_run(clean=True)
        else:
            self.sig_log_line.emit(f"[error] {msg or 'Simulation failed.'}")
            self._teardown_run(clean=False)
