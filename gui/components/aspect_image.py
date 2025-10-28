from typing import Literal
import os
from PySide6.QtWidgets import QLabel
from PySide6.QtCore import Qt, QSize
from PySide6.QtGui import QPixmap

AspectMode = Literal["fit", "cover"]


class AspectImage(QLabel):
    """
    QLabel that keeps image aspect ratio.
    - mode="fit": letterbox inside the label
    - mode="cover": fill and crop like CSS object-fit: cover
    """

    def __init__(
        self,
        path: str | None = None,
        *,
        mode: AspectMode = "fit",
        min_size: QSize | None = None
    ):
        super().__init__()
        self._rawpm: QPixmap | None = None
        self._mode: AspectMode = mode
        self.setAlignment(Qt.AlignCenter)
        self.setScaledContents(False)
        if min_size:
            self.setMinimumSize(min_size)
        if path:
            self.set_image(path)

    # ---- public API ----
    def set_image(self, path: str | None = None, *, pixmap: QPixmap | None = None):
        pm = pixmap
        if pm is None and path:
            if os.path.exists(path):
                pm = QPixmap(path)
            else:
                pm = None
        self._rawpm = pm
        if pm is None or pm.isNull():
            self.clear()
            self.setText("[no image]")
        else:
            self.setText("")
            self._rescale()

    def set_mode(self, mode: AspectMode):
        if mode not in ("fit", "cover"):
            return
        self._mode = mode
        self._rescale()

    # ---- internals ----
    def resizeEvent(self, e):
        self._rescale()
        return super().resizeEvent(e)

    def _rescale(self):
        pm = self._rawpm
        if not pm or pm.isNull() or self.width() <= 0 or self.height() <= 0:
            return
        target = self.size()
        if self._mode == "cover":
            scaled = pm.scaled(
                target, Qt.KeepAspectRatioByExpanding, Qt.SmoothTransformation
            )
            if not scaled.isNull():
                x = max(0, (scaled.width() - target.width()) // 2)
                y = max(0, (scaled.height() - target.height()) // 2)
                self.setPixmap(scaled.copy(x, y, target.width(), target.height()))
        else:  # fit
            scaled = pm.scaled(target, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            self.setPixmap(scaled)
