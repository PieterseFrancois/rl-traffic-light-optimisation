from PySide6.QtWidgets import QApplication

from gui.app_shell import MainWindow
from gui.views.scenario_manager import ScenarioManagerView
from gui.views.results_repository import ResultsView
from gui.views.sim_controller import SimControllerView
from gui.views.home_hub import HomeHubView

import sys
from pathlib import Path


def build_app():
    app = QApplication(sys.argv)
    scenario_view = ScenarioManagerView(
        hyper_path="environments/ingolstadt/hyperparams.yaml",
        config_path="environments/ingolstadt/config.yaml",
        scenarios=[
            {
                "name": "InTAS (Full Scenario)",
                "image_path": "assets/media/InTAS.png",
                "sumocfg": "environments/ingolstadt/InTAS_sumo_files/InTAS_buildings.cfg",
            },
            {
                "name": "RESCO (Reduced Scenario)",
                "image_path": "assets/media/RESCO.png",
                "sumocfg": "environments/ingolstadt/sumo_files/ingolstadt21.sumocfg",
            },
        ],
        editable_params=[
            {"path": "sumo.gui", "component": "bool"},
            {"path": "sumo.seed", "component": "pos_int"},
            {"path": "sumo.time_to_teleport_s", "component": "int"},
            {"path": "sumo.simulation_length", "component": "pos_int"},
            {"path": "sumo.ignore_junction_blocker_s", "component": "int"},
            {"path": "training.max_iterations", "component": "pos_int"},
            {"path": "training.early_stopping_patience", "component": "pos_int"},
            {"path": "training.minimum_improvement", "component": "pos_float"},
            {"path": "training.moving_window_size", "component": "pos_int"},
        ],
    )
    results_view = ResultsView(default_video_dir=".video-repository")
    sim_view = SimControllerView(
        config_file="environments/ingolstadt/config.yaml",
        hyperparams_file="environments/ingolstadt/hyperparams.yaml",
        plot_stride=10,
    )

    # Home hub wired to stack navigation
    # The MainWindow creates the stack - pass lambdas that change its index later
    dummy = HomeHubView(
        lambda: None, lambda: None, lambda: None
    )  # replaced after window exists
    win = MainWindow(scenario_view, sim_view, results_view, dummy)

    # Replace callbacks to target actual indices
    def go(idx):
        return lambda: win._nav.setCurrentRow(idx)

    home = HomeHubView(go(1), go(3), go(2))
    win._stack.removeWidget(dummy)
    win._stack.insertWidget(0, home)
    win._nav.setCurrentRow(0)
    win._stack.setCurrentWidget(home)
    win.showMaximized()
    return app


def load_stylesheets(paths: list[str]) -> str:
    parts = []
    for p in paths:
        fp = Path(p)
        try:
            parts.append(fp.read_text(encoding="utf-8"))
        except FileNotFoundError:
            print(f"[QSS] Not found: {fp}")
    return "\n\n/* ---- next file ---- */\n\n".join(parts)


STYLESHEETS = [
    "gui/styles/app.qss",
    "gui/styles/checkboxes.qss",
    "gui/styles/buttons.qss",
    "gui/styles/spinbox.qss",
    "gui/styles/scenario_cards.qss",
    "gui/styles/scenario_params.qss",
    "gui/styles/message_box.qss",
    "gui/styles/combobox.qss",
    "gui/styles/sim_log.qss",
    "gui/styles/slider.qss",
    "gui/styles/csv.qss",
]

if __name__ == "__main__":
    app = build_app()
    qss = load_stylesheets(STYLESHEETS)
    if qss:
        app.setStyleSheet(qss)
    sys.exit(app.exec())
