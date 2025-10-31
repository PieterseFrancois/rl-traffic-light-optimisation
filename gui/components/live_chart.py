from collections import deque

from PySide6.QtCore import QTimer
from PySide6.QtWidgets import QWidget, QVBoxLayout, QLabel, QHBoxLayout, QComboBox
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure


class LiveKpiPlot(QWidget):
    def __init__(self, parent=None, max_points: int = 4000):
        super().__init__(parent)
        self.max_points = max_points

        # Metric registry
        self._metrics = [
            ("avg_wait", "Average waiting time (s)"),
            ("avg_time_loss", "Average time loss (s)"),
            ("avg_acc_wait", "Average accumulated wait (s)"),
            ("mean_speed_vehicle", "Mean speed (m/s)"),
            ("throughput_arrived", "Arrivals per step"),
            ("throughput_departed", "Departures per step"),
            ("n_vehicles", "Vehicles in network"),
        ]
        self._cur_key = self._metrics[0][0]

        # UI
        v = QVBoxLayout(self)
        row = QHBoxLayout()
        row.addWidget(QLabel("Metric:"))
        self.metric_combo = QComboBox()
        for key, label in self._metrics:
            self.metric_combo.addItem(label, userData=key)
        self.metric_combo.currentIndexChanged.connect(self._on_metric_changed)
        row.addWidget(self.metric_combo, 1)
        v.addLayout(row)

        self.fig = Figure(figsize=(5, 3), tight_layout=True)
        self.ax = self.fig.add_subplot(111)
        self.canvas = FigureCanvas(self.fig)
        v.addWidget(self.canvas)

        # Two overlay lines
        (self._line_baseline,) = self.ax.plot([], [], label="baseline")
        (self._line_eval,) = self.ax.plot([], [], label="eval")
        self.ax.legend(loc="upper right")

        # Buffers
        def _make_buffer():
            return {"t": deque(maxlen=max_points), "y": deque(maxlen=max_points)}

        self._buffer = {
            "baseline": {k: _make_buffer() for k, _ in self._metrics},
            "eval": {k: _make_buffer() for k, _ in self._metrics},
        }

        # Redraw
        self._dirty = False

    # Methods

    def clear(self):
        """Clear all data points."""
        for run_mode in self._buffer:
            for k in self._buffer[run_mode]:
                self._buffer[run_mode][k]["t"].clear()
                self._buffer[run_mode][k]["y"].clear()
        self._schedule_redraw()

    def update_point(self, run_mode: str, t: float, payload: dict):
        """
        Update the plot with a new data point.

        Args:
            run_mode: "baseline" | "eval"
            t: simulation time (s)
            payload: KPI dict with keys matching those in self._metrics
        """
        if run_mode not in self._buffer:
            return
        for k, _label in self._metrics:
            y = float(payload.get(k, 0.0))
            self._buffer[run_mode][k]["t"].append(t)
            self._buffer[run_mode][k]["y"].append(y)
        self._schedule_redraw()

    # Private methods

    def _on_metric_changed(self, _idx: int):
        self._cur_key = self.metric_combo.currentData()
        self._schedule_redraw()

    def _schedule_redraw(self):
        if self._dirty:
            return
        self._dirty = True
        QTimer.singleShot(0, self._redraw_current)

    def _redraw_current(self):
        self._dirty = False
        key = self._cur_key
        b = self._buffer["baseline"][key]
        e = self._buffer["eval"][key]

        self._line_baseline.set_data(b["t"], b["y"])
        self._line_eval.set_data(e["t"], e["y"])

        # Axes labels
        self.ax.set_xlabel("sim time (s)")
        self.ax.set_ylabel(next(label for k, label in self._metrics if k == key))

        self.ax.relim()
        self.ax.autoscale_view()
        self.canvas.draw_idle()

        self.canvas.flush_events()
        self.canvas.repaint()
