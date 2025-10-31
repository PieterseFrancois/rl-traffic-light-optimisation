from PySide6.QtCore import Qt
from PySide6.QtGui import QMouseEvent, QKeyEvent
from PySide6.QtMultimediaWidgets import QVideoWidget


class VideoWidget(QVideoWidget):
    """QVideoWidget with native full screen toggle (double-click to enter/exit, Esc to exit)."""

    def mouseDoubleClickEvent(self, e: QMouseEvent) -> None:
        (
            self.setFullScreen(not self.isFullScreen())
            if hasattr(self, "isFullScreen")
            else super().mouseDoubleClickEvent(e)
        )

    def keyPressEvent(self, e: QKeyEvent) -> None:
        if e.key() == Qt.Key_Escape and self.isFullScreen():
            self.setFullScreen(False)
            return
        super().keyPressEvent(e)
