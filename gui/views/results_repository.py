# ============================================================
#  DISCLAIMER:
#  The CSV metric comparison section of this view was written
#  with the assistance of an AI coding agent and then subsequently
#  manually refining the generated framework.
# ============================================================

import os
from pathlib import Path
import pandas as pd

from PySide6.QtCore import (
    Qt,
    QUrl,
    Slot,
    QSize,
)
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import (
    QWidget,
    QLabel,
    QPushButton,
    QLineEdit,
    QComboBox,
    QGroupBox,
    QFileDialog,
    QStackedLayout,
    QSlider,
    QGridLayout,
    QVBoxLayout,
    QHBoxLayout,
    QStyle,
    QTableView,
    QSplitter,
    QCheckBox,
    QHeaderView,
)
from PySide6.QtMultimedia import QMediaPlayer, QAudioOutput

from gui.components.video_widget import VideoWidget
from gui.components.csv_metrics import CsvTableModel, RowContainsFilter, setup_table


class ResultsView(QWidget):

    IMAGE_EXTS = {".png", ".jpg", ".jpeg"}
    VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv"}

    def __init__(
        self, default_video_dir: str | None = None, parent: QWidget | None = None
    ):
        super().__init__(parent)
        self.setObjectName("ResultsView")

        # State
        self._run_dir: Path | None = None
        self._plots: list[Path] = []
        self._video_dir: Path = Path(default_video_dir).resolve()
        self._videos: list[Path] = []

        # Metrics state
        self._csv_dir_user_override: bool = False
        self._csv_dir: Path | None = None
        self._csv_left_paths: list[Path] = []
        self._csv_right_paths: list[Path] = []

        # Grid layout
        root = QGridLayout(self)
        root.setColumnStretch(0, 1)  # graphs
        root.setColumnStretch(1, 1)  # video

        # Graphs panel (col 0)
        self._graphs_box = QGroupBox("Graphs")
        graphs_col = QVBoxLayout(self._graphs_box)

        # Run dir picker row
        run_row = QHBoxLayout()
        self._run_edit = QLineEdit()
        self._run_edit.setPlaceholderText("Select a run directory that contains /plots")
        btn_set_run = QPushButton("Browse")
        btn_set_run.clicked.connect(self._choose_run_dir)
        run_row.addWidget(self._run_edit, 1)
        run_row.addWidget(btn_set_run)
        graphs_col.addLayout(run_row)

        # Plot selector row
        plots_row = QHBoxLayout()
        plots_row.addWidget(QLabel("Graph:"))
        self._plot_combo = QComboBox()
        self._plot_combo.currentIndexChanged.connect(self._on_plot_selected)
        plots_row.addWidget(self._plot_combo, 1)
        graphs_col.addLayout(plots_row)

        # Image area or no-graph message
        self._image_stack = QStackedLayout()
        self._img_label = QLabel()
        self._img_label.setAlignment(Qt.AlignCenter)
        self._img_label.setMinimumSize(QSize(480, 360))
        self._img_label.setObjectName("resultsGraphImage")

        self._no_graph = QLabel("No graphs available in /plots for the selected run.")
        self._no_graph.setAlignment(Qt.AlignCenter)
        self._no_graph.setObjectName("resultsNoGraphMsg")

        self._image_stack.addWidget(self._img_label)  # index 0
        self._image_stack.addWidget(self._no_graph)  # index 1
        graphs_col.addLayout(self._image_stack)

        # Video panel (col 1)
        self._video_box = QGroupBox("Video")
        video_col = QVBoxLayout(self._video_box)

        # Video directory selector
        vdir_row = QHBoxLayout()
        self._video_dir_edit = QLineEdit(str(self._video_dir))
        self._video_dir_edit.setReadOnly(True)
        btn_change_vdir = QPushButton("Browse")
        btn_change_vdir.clicked.connect(self._choose_video_dir)
        vdir_row.addWidget(self._video_dir_edit, 1)
        vdir_row.addWidget(btn_change_vdir)
        video_col.addLayout(vdir_row)

        # Video selector from directory contents
        sel_row = QHBoxLayout()
        sel_row.addWidget(QLabel("Clip:"))
        self._video_combo = QComboBox()
        self._video_combo.currentIndexChanged.connect(self._on_video_selected)
        sel_row.addWidget(self._video_combo, 1)
        video_col.addLayout(sel_row)

        # The actual video window
        self._video_widget = VideoWidget()
        self._video_widget.setMinimumSize(QSize(480, 360))
        self._video_widget.setObjectName("resultsVideoWidget")

        # Update button text when full screen changes
        if hasattr(self._video_widget, "fullScreenChanged"):
            self._video_widget.fullScreenChanged.connect(self._on_fullscreen_changed)
        video_col.addWidget(self._video_widget)

        # Media controls
        controls = QHBoxLayout()

        icon_size = QSize(24, 24)

        self._btn_playpause = QPushButton()
        self._btn_playpause.setIcon(self.style().standardIcon(QStyle.SP_MediaPlay))
        self._btn_playpause.setIconSize(icon_size)
        self._btn_playpause.setFlat(True)

        self._btn_stop = QPushButton()
        self._btn_stop.setIcon(self.style().standardIcon(QStyle.SP_MediaStop))
        self._btn_stop.setIconSize(icon_size)
        self._btn_stop.setFlat(True)

        self._btn_full = QPushButton()
        self._btn_full.setIcon(self.style().standardIcon(QStyle.SP_TitleBarMaxButton))
        self._btn_full.setIconSize(icon_size)
        self._btn_full.setFlat(True)

        self._slider = QSlider(Qt.Horizontal)
        self._slider.setRange(0, 0)

        self._btn_playpause.clicked.connect(self._on_playpause_clicked)
        self._btn_stop.clicked.connect(self._on_stop)
        self._btn_full.clicked.connect(self._toggle_fullscreen)
        self._slider.sliderMoved.connect(self._on_seek)

        controls.addWidget(self._btn_playpause)
        controls.addWidget(self._btn_stop)
        controls.addWidget(self._slider, 1)
        controls.addWidget(self._btn_full)
        video_col.addLayout(controls)

        # Place Graphs and Video on the grid
        root.addWidget(self._graphs_box, 0, 0)  # row 0, col 0
        root.addWidget(self._video_box, 0, 1)  # row 0, col 1

        # Metrics panel (row 1, span 2)
        metrics_box = QGroupBox("Metrics")
        mbox = QVBoxLayout(metrics_box)

        # Top bar: CSV directory selector & expand toggle
        csvdir_row = QHBoxLayout()
        csvdir_row.addWidget(QLabel("CSV dir:"))
        self._csv_dir_edit = QLineEdit()
        self._csv_dir_edit.setReadOnly(True)
        self._csv_dir_btn = QPushButton("Browse")
        self._csv_dir_btn.clicked.connect(self._choose_csv_dir)

        csvdir_row.addWidget(self._csv_dir_edit, 1)
        csvdir_row.addWidget(self._csv_dir_btn)

        self._metrics_expand_btn = QPushButton("Expand")
        self._metrics_expand_btn.setCheckable(True)
        self._metrics_expand_btn.toggled.connect(self._toggle_metrics_expand)
        csvdir_row.addWidget(self._metrics_expand_btn)
        mbox.addLayout(csvdir_row)

        # Compare toggle
        compare_row = QHBoxLayout()
        self._csv_compare_toggle = QCheckBox("Compare")
        self._csv_compare_toggle.setProperty("role", "boolParam")
        self._csv_compare_toggle.toggled.connect(self._toggle_csv_compare)
        compare_row.addWidget(self._csv_compare_toggle)
        compare_row.addStretch(1)
        mbox.addLayout(compare_row)

        # Splitter with left and right panels
        self._csv_splitter = QSplitter(Qt.Horizontal)

        # Left panel (controls above table)
        self._left_panel = QWidget()
        left_v = QVBoxLayout(self._left_panel)

        left_ctrl = QHBoxLayout()
        left_ctrl.addWidget(QLabel("CSV:"))
        self._csv_left_combo = QComboBox()
        self._csv_left_combo.currentIndexChanged.connect(self._load_left_csv)
        self._csv_left_filter = QLineEdit()
        self._csv_left_filter.setPlaceholderText("Filter rows…")
        left_ctrl.addWidget(self._csv_left_combo, 1)
        left_ctrl.addWidget(self._csv_left_filter, 1)
        left_v.addLayout(left_ctrl)

        self._csv_left_model = CsvTableModel()
        self._csv_left_proxy = RowContainsFilter()
        self._csv_left_proxy.setSourceModel(self._csv_left_model)
        self._csv_left_view = QTableView()
        setup_table(self._csv_left_view, self._csv_left_proxy)
        self._csv_left_filter.textChanged.connect(self._csv_left_proxy.setNeedle)
        left_v.addWidget(self._csv_left_view)

        self._csv_splitter.addWidget(self._left_panel)

        # Right panel (controls above table, hidden until compare)
        self._right_panel = QWidget()
        right_v = QVBoxLayout(self._right_panel)

        right_ctrl = QHBoxLayout()
        right_ctrl.addWidget(QLabel("CSV:"))
        self._csv_right_combo = QComboBox()
        self._csv_right_combo.currentIndexChanged.connect(self._load_right_csv)
        self._csv_right_filter = QLineEdit()
        self._csv_right_filter.setPlaceholderText("Filter rows…")
        right_ctrl.addWidget(self._csv_right_combo, 1)
        right_ctrl.addWidget(self._csv_right_filter, 1)
        right_v.addLayout(right_ctrl)

        self._csv_right_model = CsvTableModel()
        self._csv_right_proxy = RowContainsFilter()
        self._csv_right_proxy.setSourceModel(self._csv_right_model)
        self._csv_right_view = QTableView()
        setup_table(self._csv_right_view, self._csv_right_proxy)
        self._csv_right_filter.textChanged.connect(self._csv_right_proxy.setNeedle)
        right_v.addWidget(self._csv_right_view)

        self._csv_splitter.addWidget(self._right_panel)
        self._right_panel.setVisible(False)

        mbox.addWidget(self._csv_splitter)

        # Place metrics on grid row 1
        root.addWidget(metrics_box, 1, 0, 1, 2)  # row 1, span 2 columns

        # QSS object names
        metrics_box.setObjectName("metricsPanel")
        self._csv_splitter.setObjectName("metricsSplitter")

        self._csv_left_view.setObjectName("metricsTableLeft")
        self._csv_right_view.setObjectName("metricsTableRight")

        self._csv_left_combo.setObjectName("csvComboLeft")
        self._csv_right_combo.setObjectName("csvComboRight")

        self._csv_left_filter.setObjectName("csvFilterLeft")
        self._csv_right_filter.setObjectName("csvFilterRight")

        # Media engine
        self._player = QMediaPlayer(self)
        self._audio = QAudioOutput(self)
        self._player.setAudioOutput(self._audio)
        self._player.setVideoOutput(self._video_widget)

        self._player.playbackStateChanged.connect(self._on_playback_state_changed)

        # Keep seek bar synced
        self._player.positionChanged.connect(self._on_position_changed)
        self._player.durationChanged.connect(self._on_duration_changed)

        # Initial discovery
        self._refresh_plots()
        self._refresh_videos()
        self._refresh_csv_listing()

    # Run directory & plots
    def _choose_run_dir(self):
        d = QFileDialog.getExistingDirectory(self, "Choose run directory")
        if not d:
            return
        self.set_run_dir(Path(d))

    def set_run_dir(self, d: Path):
        self._run_dir = Path(d).resolve()
        self._run_edit.setText(str(self._run_dir))
        self._refresh_plots()

        # Default CSV dir: prioritise <run_dir>/csv then <run_dir>/plots else <run_dir>
        if not self._csv_dir_user_override:
            candidate = None
            for sub in ("csv", "plots"):
                p = self._run_dir / sub
                if p.is_dir():
                    candidate = p
                    break
            self._set_csv_dir(candidate or self._run_dir)

    def _refresh_plots(self):
        self._plot_combo.blockSignals(True)
        self._plot_combo.clear()
        self._plots = []

        plots_dir = None
        if self._run_dir:
            candidate = self._run_dir / "plots"
            if candidate.is_dir():
                plots_dir = candidate

        if plots_dir:
            for name in sorted(os.listdir(plots_dir)):
                p = plots_dir / name
                if p.suffix.lower() in self.IMAGE_EXTS and p.is_file():
                    self._plots.append(p)
                    self._plot_combo.addItem(p.name, str(p))

        self._plot_combo.blockSignals(False)

        if self._plots:
            self._image_stack.setCurrentIndex(0)
            self._plot_combo.setCurrentIndex(0)
            self._load_image(self._plots[0])
        else:
            self._img_label.clear()
            self._image_stack.setCurrentIndex(1)

    @Slot(int)
    def _on_plot_selected(self, idx: int):
        if 0 <= idx < len(self._plots):
            self._load_image(self._plots[idx])

    def _load_image(self, path: Path):
        pm = QPixmap(str(path))
        if pm.isNull():
            self._img_label.setText(f"Unable to load: {path.name}")
            return
        scaled = pm.scaled(
            self._img_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation
        )
        self._img_label.setPixmap(scaled)

    def resizeEvent(self, e):
        super().resizeEvent(e)
        idx = self._plot_combo.currentIndex()
        if 0 <= idx < len(self._plots):
            self._load_image(self._plots[idx])

    # Video directory & player
    def _choose_video_dir(self):
        d = QFileDialog.getExistingDirectory(self, "Choose video results directory")
        if not d:
            return
        self.set_video_dir(Path(d))

    def set_video_dir(self, d: Path):
        self._video_dir = Path(d).resolve()
        self._video_dir_edit.setText(str(self._video_dir))
        self._refresh_videos()

    def _refresh_videos(self):
        self._video_combo.blockSignals(True)
        self._video_combo.clear()
        self._videos = []

        if self._video_dir.is_dir():
            for name in sorted(os.listdir(self._video_dir)):
                p = self._video_dir / name
                if p.suffix.lower() in self.VIDEO_EXTS and p.is_file():
                    self._videos.append(p)
                    self._video_combo.addItem(p.name, str(p))

        self._video_combo.blockSignals(False)

        if self._videos:
            self._video_combo.setCurrentIndex(0)
            self._load_video(self._videos[0])
        else:
            self._player.stop()
            self._slider.setRange(0, 0)

    @Slot(int)
    def _on_video_selected(self, idx: int):
        if 0 <= idx < len(self._videos):
            self._load_video(self._videos[idx])

    def _load_video(self, path: Path):
        self._player.setSource(QUrl.fromLocalFile(str(path)))
        self._slider.setValue(0)
        # self._player.play()
        self._on_playback_state_changed(self._player.playbackState())

    # Transport controls
    def _on_playpause_clicked(self):
        state = self._player.playbackState()
        if state == QMediaPlayer.PlaybackState.PlayingState:
            self._player.pause()
        else:
            self._player.play()

    def _on_playback_state_changed(self, state):
        # Swap icon & tooltip based on current state
        if state == QMediaPlayer.PlaybackState.PlayingState:
            self._btn_playpause.setIcon(self.style().standardIcon(QStyle.SP_MediaPause))
        else:
            self._btn_playpause.setIcon(self.style().standardIcon(QStyle.SP_MediaPlay))

    def _on_stop(self):
        self._player.stop()
        self._slider.setValue(0)
        self._on_playback_state_changed(self._player.playbackState())

    def _on_seek(self, pos: int):
        self._player.setPosition(pos)

    def _on_position_changed(self, pos: int):
        self._slider.blockSignals(True)
        self._slider.setValue(pos)
        self._slider.blockSignals(False)

    def _on_duration_changed(self, dur: int):
        self._slider.setRange(0, dur)

    def _toggle_fullscreen(self):
        self._video_widget.setFullScreen(not self._video_widget.isFullScreen())

    def _on_fullscreen_changed(self, is_fs: bool):
        self._btn_full.setIcon(
            self.style().standardIcon(
                QStyle.SP_TitleBarNormalButton if is_fs else QStyle.SP_TitleBarMaxButton
            )
        )

    #  Metrics CSV
    def _choose_csv_dir(self):
        d = QFileDialog.getExistingDirectory(self, "Choose CSV directory")
        if not d:
            return
        self._csv_dir_user_override = True
        self._set_csv_dir(Path(d))

    def _set_csv_dir(self, d: Path):
        self._csv_dir = d.resolve()
        self._csv_dir_edit.setText(str(self._csv_dir))
        self._refresh_csv_listing()

    def _refresh_csv_listing(self):
        left_idx = self._csv_left_combo.currentIndex()
        right_idx = self._csv_right_combo.currentIndex()

        self._csv_left_combo.blockSignals(True)
        self._csv_right_combo.blockSignals(True)
        self._csv_left_combo.clear()
        self._csv_right_combo.clear()
        self._csv_left_paths = []
        self._csv_right_paths = []

        paths = self._discover_csvs(self._csv_dir) if self._csv_dir else []
        for p in paths:
            rel = (
                str(p.relative_to(self._csv_dir))
                if self._csv_dir and self._is_relative_to(p, self._csv_dir)
                else p.name
            )
            self._csv_left_paths.append(p)
            self._csv_left_combo.addItem(rel, str(p))
            self._csv_right_paths.append(p)
            self._csv_right_combo.addItem(rel, str(p))

        self._csv_left_combo.blockSignals(False)
        self._csv_right_combo.blockSignals(False)

        if self._csv_left_paths:
            self._csv_left_combo.setCurrentIndex(0)
            self._load_left_csv(0)
        else:
            self._csv_left_model.set_df(pd.DataFrame())

        if self._csv_compare_toggle.isChecked():
            if self._csv_right_paths:
                self._csv_right_combo.setCurrentIndex(0)
                self._load_right_csv(0)
            else:
                self._csv_right_model.set_df(pd.DataFrame())

    def _discover_csvs(self, base: Path | None) -> list[Path]:
        out: list[Path] = []
        if not base or not base.is_dir():
            return out
        for root, _dirs, files in os.walk(base):
            for fname in files:
                if fname.lower().endswith(".csv"):
                    out.append(Path(root) / fname)
        out.sort()
        return out

    def _is_relative_to(self, path: Path, base: Path) -> bool:
        try:
            path.relative_to(base)
            return True
        except Exception:
            return False

    @Slot(int)
    def _load_left_csv(self, idx: int):
        path = self._csv_left_combo.itemData(idx)
        if not path:
            self._csv_left_model.set_df(pd.DataFrame())
            return
        self._load_csv_path_left(Path(path))

    @Slot(int)
    def _load_right_csv(self, idx: int):
        path = self._csv_right_combo.itemData(idx)
        if not path:
            self._csv_right_model.set_df(pd.DataFrame())
            return
        self._load_csv_path_right(Path(path))

    def _load_csv_path_left(self, p: Path):
        df = self._read_csv_safe(p)
        self._csv_left_model.set_df(df)
        self._auto_format_table(self._csv_left_view)

    def _load_csv_path_right(self, p: Path):
        df = self._read_csv_safe(p)
        self._csv_right_model.set_df(df)
        self._auto_format_table(self._csv_right_view)

    def _auto_format_table(self, tv: QTableView, max_width: int = 240):
        tv.resizeColumnsToContents()
        # constrain very wide columns
        hh: QHeaderView = tv.horizontalHeader()
        for c in range(hh.count()):
            w = tv.columnWidth(c)
            tv.setColumnWidth(c, min(w, max_width))

    def _read_csv_safe(self, p: Path) -> pd.DataFrame:
        if not p or not p.exists():
            return pd.DataFrame()
        try:
            return pd.read_csv(p, low_memory=False)
        except Exception:
            try:
                return pd.read_csv(p, sep=";", low_memory=False)
            except Exception:
                try:
                    return pd.read_csv(
                        p, engine="python", low_memory=False, on_bad_lines="skip"
                    )
                except Exception:
                    return pd.DataFrame()

    def _toggle_csv_compare(self, on: bool):
        self._right_panel.setVisible(on)
        if on:
            if self._csv_right_paths:
                self._csv_right_combo.setCurrentIndex(0)
                self._load_right_csv(0)
            self._csv_splitter.setSizes([1, 1])
        else:
            self._csv_splitter.setSizes([1, 0])

    def _toggle_metrics_expand(self, on: bool):
        # Hide/show the graphs and video panels to give metrics full page
        self._graphs_box.setVisible(not on)
        self._video_box.setVisible(not on)
        self._metrics_expand_btn.setText("Collapse" if on else "Expand")
