from PySide6.QtCore import Qt, Signal, QSize
from PySide6.QtWidgets import QWidget, QVBoxLayout, QLabel

from gui.components.aspect_image import AspectImage


class ClickableCard(QWidget):
    clicked = Signal(str)

    def __init__(self, name: str, image_path: str):
        super().__init__()
        self._name = name

        self.setObjectName("scenarioCard")
        self.setProperty("active", False)

        self.setAttribute(Qt.WA_StyledBackground, True)
        self.setAttribute(Qt.WA_Hover, True)

        self.setCursor(Qt.PointingHandCursor)

        v = QVBoxLayout(self)
        v.setContentsMargins(10, 10, 10, 10)

        # Title
        self._title = QLabel(f"{name}")
        self._title.setAlignment(Qt.AlignCenter)
        self._title.setObjectName("scenarioTitle")
        self._title.setAttribute(Qt.WA_TransparentForMouseEvents, True)
        v.addWidget(self._title)

        # Image with aspect handling
        self._img = AspectImage(image_path, mode="fit", min_size=QSize(260, 180))
        self._img.setObjectName("scenarioImage")
        self._img.setAttribute(Qt.WA_TransparentForMouseEvents, True)
        v.addWidget(self._img)

    # Public API toggle active state
    def set_active(self, active: bool):
        self.setProperty("active", active)
        self.style().unpolish(self)
        self.style().polish(self)
        self.update()

    # Make whole card clickable
    def mousePressEvent(self, e):
        if e.button() == Qt.LeftButton:
            self.clicked.emit(self._name)
        super().mousePressEvent(e)

    # Subtle hover shadow when not active
    def enterEvent(self, e):
        super().enterEvent(e)

    def leaveEvent(self, e):
        super().leaveEvent(e)
