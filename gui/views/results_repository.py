import os
from pathlib import Path

from PySide6.QtCore import Qt, QUrl, Slot, QSize
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
)
from PySide6.QtMultimedia import QMediaPlayer, QAudioOutput

from gui.components.video_widget import VideoWidget


class ResultsView(QWidget):
    """
    Results page:
      - Choose a run directory. If <run_dir>/plots exists, list images; otherwise show 'no graph' message.
      - Show selected graph in an image area.
      - Video area lists files from a configurable videos directory, with transport controls and full screen toggle.
      - Placeholder panel reserved for future CSV-based metrics.
    """

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
        self._video_dir: Path = Path(default_video_dir or "results/videos").resolve()
        self._videos: list[Path] = []

        # Grid layout
        root = QGridLayout(self)
        root.setColumnStretch(0, 1)  # graphs
        root.setColumnStretch(1, 1)  # video

        # Graphs panel (col 0)
        graphs_box = QGroupBox("Graphs")
        graphs_col = QVBoxLayout(graphs_box)

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
        video_box = QGroupBox("Video")
        video_col = QVBoxLayout(video_box)

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

        # Metrics panel (row 1, span 2)
        metrics_box = QGroupBox("Metrics (coming soon)")
        mbox = QVBoxLayout(metrics_box)
        mbox.addWidget(QLabel("CSV-derived KPIs and summaries will appear here."))

        # Place panels on the grid
        root.addWidget(graphs_box, 0, 0)  # row 0, col 0
        root.addWidget(video_box, 0, 1)  # row 0, col 1
        root.addWidget(metrics_box, 1, 0, 1, 2)  # row 1, span 2 columns
    
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
