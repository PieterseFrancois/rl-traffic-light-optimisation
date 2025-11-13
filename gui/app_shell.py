from PySide6.QtWidgets import (
    QMainWindow,
    QWidget,
    QStackedWidget,
    QListWidget,
    QHBoxLayout,
    QListWidgetItem,
)
from PySide6.QtGui import QIcon


class MainWindow(QMainWindow):
    def __init__(
        self,
        scenario_view: QWidget,
        sim_view: QWidget,
        results_view: QWidget,
        home_view: QWidget,
    ):
        super().__init__()
        icon = QIcon("assets/logo/logov2.ico")
        # If you have a QMainWindow or QWidget as the main window:
        self.setWindowIcon(icon)
        self.setWindowTitle("Multi Agent RL Control Center")
        self._stack = QStackedWidget()
        self._pages = {
            "Home": home_view,
            "Scenario Manager": scenario_view,
            "Simulation": sim_view,
            "Results Repository": results_view,
        }
        for w in self._pages.values():
            self._stack.addWidget(w)

        self._nav = QListWidget()
        for name in self._pages.keys():
            QListWidgetItem(name, self._nav)
        self._nav.currentRowChanged.connect(self._stack.setCurrentIndex)
        self._nav.setFixedWidth(180)
        self._nav.setObjectName("sideNav")
        self._nav.setProperty("role", "nav")

        container = QWidget()
        layout = QHBoxLayout(container)
        layout.addWidget(self._nav)
        layout.addWidget(self._stack, 1)
        self.setCentralWidget(container)
        container.setObjectName("rootContainer")
        self._stack.setObjectName("mainStack")

        # Default to Home
        self._nav.setCurrentRow(0)
        self._stack.setCurrentWidget(home_view)
