import pandas as pd
from PySide6.QtCore import Qt, QAbstractTableModel, QModelIndex, QSortFilterProxyModel
from PySide6.QtWidgets import QTableView, QHeaderView


class CsvTableModel(QAbstractTableModel):
    def __init__(self, df: pd.DataFrame | None = None, parent=None):
        super().__init__(parent)
        self._df = df if df is not None else pd.DataFrame()

    def set_df(self, df: pd.DataFrame):
        self.beginResetModel()
        self._df = df
        self.endResetModel()

    def rowCount(self, parent=QModelIndex()):
        return 0 if parent.isValid() else len(self._df)

    def columnCount(self, parent=QModelIndex()):
        return 0 if parent.isValid() else len(self._df.columns)

    def data(self, index, role):
        if not index.isValid() or role not in (Qt.DisplayRole, Qt.EditRole):
            return None
        val = self._df.iat[index.row(), index.column()]
        return "" if pd.isna(val) else str(val)

    def headerData(self, section, orientation, role):
        if role != Qt.DisplayRole:
            return None
        if orientation == Qt.Horizontal:
            return (
                str(self._df.columns[section])
                if 0 <= section < len(self._df.columns)
                else ""
            )
        return str(section + 1)

    def sort(self, column, order):
        if not (0 <= column < len(self._df.columns)):
            return
        self.layoutAboutToBeChanged.emit()
        ascending = order == Qt.AscendingOrder
        try:
            self._df.sort_values(
                self._df.columns[column],
                ascending=ascending,
                inplace=True,
                kind="mergesort",
            )
            self._df.reset_index(drop=True, inplace=True)
        finally:
            self.layoutChanged.emit()


class RowContainsFilter(QSortFilterProxyModel):
    """Case-insensitive substring filter over the whole row."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._needle = ""

    def setNeedle(self, text: str):
        self._needle = text.lower().strip()
        self.invalidateFilter()

    def filterAcceptsRow(self, src_row, src_parent):
        if not self._needle:
            return True
        model = self.sourceModel()
        cols = model.columnCount()
        for c in range(cols):
            idx = model.index(src_row, c, src_parent)
            val = model.data(idx, Qt.DisplayRole)
            if val and self._needle in str(val).lower():
                return True
        return False


def setup_table(table_view: QTableView, proxy: RowContainsFilter):
    table_view.setModel(proxy)
    table_view.setSortingEnabled(True)
    table_view.setAlternatingRowColors(True)
    table_view.setAcceptDrops(False)
    table_view.horizontalHeader().setSectionResizeMode(QHeaderView.Interactive)
    table_view.horizontalHeader().setDefaultSectionSize(140)
    table_view.horizontalHeader().setStretchLastSection(False)