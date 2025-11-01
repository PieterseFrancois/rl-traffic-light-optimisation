from PySide6.QtWidgets import QWidget, QVBoxLayout, QPushButton, QLabel, QFrame
from PySide6.QtCore import Qt
from PySide6.QtGui import QKeySequence, QShortcut


class HomeHubView(QWidget):
    def __init__(self, go_scenario, go_results, go_sim):
        super().__init__()
        self.setObjectName("HomeHubView")

        root = QVBoxLayout(self)
        root.setSpacing(16)

        title = QLabel("<h2>Welcome</h2>", alignment=Qt.AlignCenter)
        title.setObjectName("homeTitle")
        subtitle = QLabel(
            "This workbench is a developer-oriented UI. Pick a section below to work with scenarios, run simulations, or inspect results.",
            alignment=Qt.AlignCenter,
        )
        subtitle.setWordWrap(True)
        subtitle.setObjectName("homeSubtitle")
        root.addWidget(title)
        root.addWidget(subtitle)

        # Stacked cards
        cards_col = QVBoxLayout()
        cards_col.setSpacing(12)
        cards_col.setAlignment(Qt.AlignTop | Qt.AlignHCenter)
        root.addLayout(cards_col)

        def card(title_text: str, desc: str, btn_text: str, on_click, obj_suffix: str):
            box = QFrame()
            box.setObjectName(f"homeCard_{obj_suffix}")
            box.setProperty("role", "homeCard")
            box.setFrameShape(QFrame.StyledPanel)
            box.setFrameShadow(QFrame.Raised)

            v = QVBoxLayout(box)
            v.setSpacing(8)

            lbl = QLabel(f"<b>{title_text}</b>")
            lbl.setObjectName(f"homeCardTitle_{obj_suffix}")
            lbl.setProperty("role", "cardTitle")

            blurb = QLabel(desc)
            blurb.setObjectName(f"homeCardDesc_{obj_suffix}")
            blurb.setProperty("role", "cardDesc")
            blurb.setWordWrap(True)

            btn = QPushButton(btn_text)
            btn.setObjectName(f"homeCardBtn_{obj_suffix}")
            btn.setProperty("role", "cardBtn")
            btn.clicked.connect(on_click)

            v.addWidget(lbl)
            v.addWidget(blurb)
            v.addStretch(1)
            v.addWidget(btn, alignment=Qt.AlignRight)
            return box, btn

        scen_card, _ = card(
            "Scenario Manager",
            "Create or edit SUMO/RESCo InTAS scenarios. Choose the network and edit configs/hyperparameters used by the RL runs.",
            "Go to Scenario Manager",
            go_scenario,
            "scenario",
        )
        sim_card, _ = card(
            "Simulation",
            "Launch training or evaluation. Runs SUMO via TraCI with your chosen RL runner. Use this when you want to execute a run.",
            "Go to Simulation",
            go_sim,
            "sim",
        )
        res_card, _ = card(
            "Results Repository",
            "Inspect outputs from runs: graphs from /plots, videos from your results folder and CSV metrics with optional side-by-side comparison.",
            "Go to Results",
            go_results,
            "results",
        )

        # Add cards vertically
        cards_col.addWidget(scen_card)
        cards_col.addWidget(sim_card)
        cards_col.addWidget(res_card)

        # Shortcuts
        QShortcut(QKeySequence("Alt+1"), self, activated=go_scenario)
        QShortcut(QKeySequence("Alt+2"), self, activated=go_sim)
        QShortcut(QKeySequence("Alt+3"), self, activated=go_results)
