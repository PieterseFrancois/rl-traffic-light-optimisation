import os
from io import StringIO
from math import ceil
from typing import Any

from PySide6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QFormLayout,
    QLineEdit,
    QMessageBox,
    QCheckBox,
    QGroupBox,
    QSpinBox,
    QDoubleSpinBox,
    QGridLayout,
    QSizePolicy,
)
from PySide6.QtCore import Qt

from gui.components.clickable_card import ClickableCard

# Comment-preserving YAML
from ruamel.yaml import YAML
from ruamel.yaml.comments import CommentedMap

_yaml = YAML()
_yaml.preserve_quotes = True
_yaml.indent(sequence=4, offset=2)


class ScenarioManagerView(QWidget):

    def __init__(
        self,
        hyper_path: str,
        config_path: str,
        scenarios: list[dict],
        editable_params: list[dict],
    ):
        super().__init__()
        self.setObjectName("ScenarioManagerView")

        self.hyper_path = hyper_path
        self.config_path = config_path

        # [{"path": str, "component": str}]
        self._editable_params: list[dict] = editable_params or []

        # Per-path widget and metadata
        self._widgets: dict[str, QWidget] = {}  # path - widget
        self._path_component: dict[str, str] = {}  # path - component
        self._path_source: dict[str, str] = {}  # path - "hyper"|"config"

        self._hyper_doc: CommentedMap | None = None
        self._config_doc: CommentedMap | None = None

        root = QVBoxLayout(self)

        # Scenario cards
        self._scenarios = {sc["name"]: sc for sc in scenarios}
        self._cards = {}  # name - ClickableCard

        cards = QHBoxLayout()
        for sc in scenarios:
            card = ClickableCard(sc["name"], sc.get("image_path", ""))
            card.clicked.connect(self._on_card_clicked)
            self._cards[sc["name"]] = card
            cards.addWidget(card)
        root.addLayout(cards)

        # Nested containers per top-level path segment
        self._sections_layout = QVBoxLayout()
        root.addLayout(self._sections_layout)

        # Buttons
        btns = QHBoxLayout()

        self._load_btn = QPushButton("Reload")
        self._load_btn.setObjectName("reloadBtn")
        self._load_btn.setProperty("variant", "ghost")
        self._load_btn.clicked.connect(self.load_yamls)

        self._save_btn = QPushButton("Save")
        self._save_btn.setObjectName("saveBtn")
        self._save_btn.setProperty("variant", "primary")
        self._save_btn.setEnabled(False)
        self._save_btn.clicked.connect(self.save_yamls)

        btns.addWidget(self._load_btn)
        btns.addWidget(self._save_btn)
        root.addLayout(btns)

        self._build_sections()
        self.load_yamls()  # initial load

    # UI: scenario cards

    def _on_select_scenario(self, name: str):
        sc = self._scenarios.get(name, {})
        new_cfg = sc.get("sumocfg")
        if not new_cfg:
            return

        src, current = self._get_sumocfg_source_and_value()
        if current == new_cfg:
            self._set_active_card(name)
            return

        if src == "none":
            src = "hyper"

        if src == "hyper":
            self._set_by_path(self._hyper_doc, "sumo.sumocfg", new_cfg)
            self._del_by_path(self._config_doc, "sumo.sumocfg")
        else:
            self._set_by_path(self._config_doc, "sumo.sumocfg", new_cfg)
            self._del_by_path(self._hyper_doc, "sumo.sumocfg")

        self._write_yaml_doc(self.hyper_path, self._hyper_doc)
        self._write_yaml_doc(self.config_path, self._config_doc)

        self._set_active_card(name)

    def _get_sumocfg_source_and_value(self) -> tuple[str, str | None]:
        ok_h, val_h = self._get_by_path(self._hyper_doc, "sumo.sumocfg")
        ok_c, val_c = self._get_by_path(self._config_doc, "sumo.sumocfg")
        if ok_h:
            return "hyper", str(val_h)
        if ok_c:
            return "config", str(val_c)
        return "none", None

    # Sections & widgets

    def _build_sections(self):
        """
        Create QGroupBox per first segment, laid out in a
        centred grid that shares width evenly across columns.
        """
        # ---- Layout configs ----
        ROWS = 1  # 0 = unlimited rows
        COLS = 2  # 0 = unlimited columns
        HSPC, VSPC = 16, 16
        # ------------------------

        # Clear previous
        while self._sections_layout.count():
            item = self._sections_layout.takeAt(0)
            if w := item.widget():
                w.deleteLater()

        # Group params by first segment
        groups: dict[str, list[dict]] = {}
        for p in self._editable_params:
            path = p.get("path", "").strip()
            comp = p.get("component", "text").strip().lower()
            if not path:
                continue
            top = path.split(".", 1)[0]
            groups.setdefault(top, []).append({"path": path, "component": comp})

        names = sorted(groups.keys())
        n = len(names)

        grid = QGridLayout()
        grid.setHorizontalSpacing(HSPC)
        grid.setVerticalSpacing(VSPC)
        grid.setAlignment(Qt.AlignTop | Qt.AlignHCenter)

        # Compute columns to use
        if ROWS > 0 and (COLS == 0):
            # Fixed rows, unlimited columns (column-major)
            columns_used = ceil(n / ROWS) if ROWS else 1
            place_col_major = True
        elif COLS > 0 and (ROWS == 0):
            # Fixed columns, unlimited rows (row-major)
            columns_used = COLS
            place_col_major = False
        elif ROWS > 0 and COLS > 0:
            # Cap rows, at most COLS columns
            columns_used = min(COLS, ceil(n / ROWS) if ROWS else COLS)
            place_col_major = True
        else:
            # Both zero thus fallback to single column
            columns_used = 1
            place_col_major = False

        # Share width evenly across columns
        for i in range(columns_used):
            grid.setColumnStretch(i, 1)

        def add_box(top_key: str, r: int, c: int):
            items = groups[top_key]
            box = QGroupBox(top_key)
            box.setObjectName("scenarioSection")
            box.setProperty("section", top_key)
            box.setAttribute(Qt.WA_StyledBackground, True)
            box.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Maximum)

            form = QFormLayout()
            form.setObjectName("paramsForm")
            form.setFieldGrowthPolicy(QFormLayout.AllNonFixedFieldsGrow)
            box.setLayout(form)

            for item in items:
                path = item["path"]
                comp = item["component"]
                label_text = path.split(".")[-1]

                lbl = QLabel(label_text)
                lbl.setObjectName("paramLabel")
                lbl.setProperty("section", top_key)

                w = self._create_widget_for_component(comp)
                w.setObjectName("paramInput")
                w.setProperty("component", comp)
                w.setSizePolicy(QSizePolicy.Expanding, w.sizePolicy().verticalPolicy())

                self._wire_change_signal(w)

                self._widgets[path] = w
                self._path_component[path] = comp
                form.addRow(lbl, w)

            grid.addWidget(box, r, c)

        # Place boxes
        if place_col_major:
            # Fill down rows first, then next column
            for idx, name in enumerate(names):
                r = idx % (ROWS if ROWS else max(1, n))
                c = idx // (ROWS if ROWS else max(1, n))
                if c >= columns_used:
                    # Overflow: start a new column beyond columns_used
                    grid.setColumnStretch(c, 1)
                add_box(name, r, c)
        else:
            # Row fill across columns, then next row
            for idx, name in enumerate(names):
                c = idx % max(1, columns_used)
                r = idx // max(1, columns_used)
                add_box(name, r, c)

        self._sections_layout.addLayout(grid)

    def _on_card_clicked(self, name: str):
        # Delegate to existing logic (updates YAML + UI)
        self._on_select_scenario(name)

    def _set_active_card(self, name: str | None):
        for n, card in self._cards.items():
            card.setProperty("active", bool(name and n == name))
            card.style().unpolish(card)
            card.style().polish(card)

    def _sync_active_card(self):
        _src, current = self._get_sumocfg_source_and_value()
        active_name = None
        for name, sc in self._scenarios.items():
            if current and str(sc.get("sumocfg", "")) == current:
                active_name = name
                break
        self._set_active_card(active_name)

    def _create_widget_for_component(self, comp: str) -> QWidget:
        comp = comp.lower()
        if comp == "bool":
            cb = QCheckBox()
            cb.setTristate(False)
            cb.setText("")  # no label
            cb.setProperty("role", "boolParam")
            return cb
        if comp in ("int", "pos_int"):
            sb = QSpinBox()
            sb.setRange(1 if comp == "pos_int" else -2_147_483_648, 2_147_483_647)
            sb.setSingleStep(1)
            return sb
        if comp in ("float", "pos_float"):
            dsb = QDoubleSpinBox()
            dsb.setDecimals(2)
            dsb.setRange(0.0 if comp == "pos_float" else -1e12, 1e12)
            dsb.setSingleStep(0.1)
            return dsb
        le = QLineEdit()
        le.setClearButtonEnabled(True)
        return le

    def _wire_change_signal(self, w: QWidget):
        if isinstance(w, QCheckBox):
            w.stateChanged.connect(lambda *_: self._save_btn.setEnabled(True))
        elif isinstance(w, (QSpinBox, QDoubleSpinBox)):
            w.valueChanged.connect(lambda *_: self._save_btn.setEnabled(True))
        elif isinstance(w, QLineEdit):
            w.textEdited.connect(lambda *_: self._save_btn.setEnabled(True))

    # YAML IO

    def _read_yaml_doc(self, path: str) -> CommentedMap:
        if not os.path.exists(path):
            return _yaml.load("{}")
        with open(path, "r", encoding="utf-8") as f:
            return _yaml.load(f) or _yaml.load("{}")

    def _write_yaml_doc(self, path: str, doc: CommentedMap):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            _yaml.dump(doc, f)

    # Path helpers for nested dicts

    def _parse_path(self, path: str) -> list[str]:
        return [p for p in path.split(".") if p]

    def _get_by_path(self, doc: CommentedMap, path: str) -> tuple[bool, Any]:
        cur: Any = doc
        parts = self._parse_path(path)
        for p in parts:
            if not isinstance(cur, (dict, CommentedMap)) or p not in cur:
                return False, None
            cur = cur[p]
        return True, cur

    def _ensure_parent(self, doc: CommentedMap, parts: list[str]) -> CommentedMap:
        cur: CommentedMap = doc
        for p in parts:
            if p not in cur or not isinstance(cur[p], (dict, CommentedMap)):
                cur[p] = CommentedMap()
            cur = cur[p]
        return cur

    def _set_by_path(self, doc: CommentedMap, path: str, value: Any):
        parts = self._parse_path(path)
        parent = self._ensure_parent(doc, parts[:-1])
        parent[parts[-1]] = value

    def _del_by_path(self, doc: CommentedMap, path: str):
        parts = self._parse_path(path)
        cur: Any = doc
        for p in parts[:-1]:
            if p not in cur or not isinstance(cur[p], (dict, CommentedMap)):
                return
            cur = cur[p]
        if isinstance(cur, (dict, CommentedMap)):
            cur.pop(parts[-1], None)

    # Load & populate

    def load_yamls(self):
        self._hyper_doc = self._read_yaml_doc(self.hyper_path)
        self._config_doc = self._read_yaml_doc(self.config_path)

        # Populate each widget and remember its source file
        for path, w in self._widgets.items():
            ok_h, val_h = self._get_by_path(self._hyper_doc, path)
            ok_c, val_c = self._get_by_path(self._config_doc, path)

            # Choose source and value
            if ok_h:
                val = val_h
                src = "hyper"
            elif ok_c:
                val = val_c
                src = "config"
            else:
                val = None
                src = "hyper"  # default

            self._path_source[path] = src
            self._set_widget_value(w, self._path_component[path], val)

        self._save_btn.setEnabled(False)

        self._sync_active_card()

    def _set_widget_value(self, w: QWidget, comp: str, value: Any):
        comp = comp.lower()
        if isinstance(w, QCheckBox):
            w.setChecked(bool(value) if value is not None else False)
            return
        if isinstance(w, QSpinBox):
            if value is None:
                value = w.minimum() if "pos" in comp else 0
            try:
                w.setValue(int(value))
            except Exception:
                w.setValue(w.minimum())
            return
        if isinstance(w, QDoubleSpinBox):
            if value is None:
                value = max(0.0, w.minimum()) if "pos" in comp else 0.0
            try:
                w.setValue(float(value))
            except Exception:
                w.setValue(w.minimum())
            return
        if isinstance(w, QLineEdit):
            w.setText("" if value is None else self._scalar_to_str(value))

    # Save

    def save_yamls(self):
        for path, w in self._widgets.items():
            comp = self._path_component[path]
            src = self._path_source.get(path, "hyper")
            value, is_empty = self._value_from_widget(w, comp)

            if is_empty:
                # Remove from both
                self._del_by_path(self._hyper_doc, path)
                self._del_by_path(self._config_doc, path)
                continue

            if src == "hyper":
                self._set_by_path(self._hyper_doc, path, value)
                self._del_by_path(self._config_doc, path)
            else:
                self._set_by_path(self._config_doc, path, value)
                self._del_by_path(self._hyper_doc, path)

        self._write_yaml_doc(self.hyper_path, self._hyper_doc)
        self._write_yaml_doc(self.config_path, self._config_doc)
        self._save_btn.setEnabled(False)
        QMessageBox.information(self, "Saved", "Parameters updated.")

    def _value_from_widget(self, w: QWidget, comp: str) -> tuple[Any, bool]:
        comp = comp.lower()
        if isinstance(w, QCheckBox):
            return (w.isChecked(), False)
        if isinstance(w, QSpinBox):
            v = int(w.value())
            if comp == "pos_int" and v < 1:
                v = 1
            return (v, False)
        if isinstance(w, QDoubleSpinBox):
            v = float(w.value())
            if comp == "pos_float" and v <= 0:
                v = max(0.000001, v)
            return (v, False)
        if isinstance(w, QLineEdit):
            txt = w.text().strip()
            if txt == "":
                return ("", True)
            try:
                parsed = _yaml.load(txt)
            except Exception:
                parsed = txt
            return (parsed, False)
        return (None, True)

    # Utils

    def _scalar_to_str(self, v: Any) -> str:
        if isinstance(v, (dict, list, CommentedMap)):
            buf = StringIO()
            _yaml.dump(v, buf)
            return " ".join(buf.getvalue().splitlines())
        return str(v)
