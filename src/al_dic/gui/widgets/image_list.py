"""Multi-column image list with per-frame ROI buttons.

QTreeWidget with 3 columns:
1. # (Frame number, "00", "01", ...) — fixed 32px
2. Filename — stretch
3. ROI — painted action: red "Need" / green "Edit" / gray "Add"

Ref frames get bold + accent-colored filenames for visibility.

Supports:
- Extended selection (Ctrl+click, Shift+click)
- Right-click context menu: Import/Clear ROI, Delete Selected
- ROI button indicators based on tracking mode / ref-frame status
"""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import QEvent, QSize, Qt, Signal
from PySide6.QtGui import QAction, QColor, QFont
from PySide6.QtWidgets import (
    QHeaderView,
    QMenu,
    QStyledItemDelegate,
    QStyle,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

from al_dic.gui.app_state import AppState
from al_dic.gui.controllers.image_controller import ImageController
from al_dic.gui.theme import COLORS
from al_dic.gui.theme_manager import connect_theme, themed

# Column indices
_COL_FRAME = 0
_COL_FILENAME = 1
_COL_ROI = 2

class _RoiDelegate(QStyledItemDelegate):
    """Paint only visible ROI actions; never allocate a widget per frame."""

    activated = Signal(int)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._pressed_row = None

    def paint(self, painter, option, index) -> None:
        painter.save()
        rect = option.rect.adjusted(2, 3, -2, -3)
        status = index.data(Qt.ItemDataRole.UserRole)
        color = {"has": COLORS.SUCCESS, "need": COLORS.DANGER}.get(
            status, COLORS.BG_INPUT,
        )
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QColor(color))
        painter.drawRoundedRect(rect, 3, 3)
        painter.setPen(QColor("#ffffff" if status in ("has", "need")
                              else COLORS.TEXT_MUTED))
        painter.drawText(rect, Qt.AlignmentFlag.AlignCenter,
                         str(index.data(Qt.ItemDataRole.DisplayRole) or ""))
        if option.state & QStyle.StateFlag.State_HasFocus:
            painter.setPen(QColor(COLORS.ACCENT))
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.drawRoundedRect(rect, 3, 3)
        painter.restore()

    def sizeHint(self, option, index):  # noqa: N802
        return QSize(44, max(26, option.fontMetrics.height() + 6))

    def editorEvent(self, event, model, option, index):  # noqa: N802
        if event.type() == QEvent.Type.MouseButtonRelease:
            pressed, self._pressed_row = self._pressed_row, None
            if (event.button() == Qt.MouseButton.LeftButton
                    and pressed == index.row()):
                self.activated.emit(index.row())
                return True
        if event.type() == QEvent.Type.MouseButtonPress:
            if event.button() == Qt.MouseButton.LeftButton:
                self._pressed_row = index.row()
                return True
        if event.type() == QEvent.Type.KeyPress and event.key() in (
            Qt.Key.Key_Return, Qt.Key.Key_Enter, Qt.Key.Key_Space,
        ):
            self.activated.emit(index.row())
            return True
        return False


class ImageList(QWidget):
    """Scrollable multi-column image list with ROI indicators."""

    # Emitted when images are removed (indices list)
    images_removed = Signal(list)

    # Emitted when user clicks an ROI button to edit that frame's ROI
    roi_edit_requested = Signal(int)

    # Emitted when user imports ROI masks for specific frames via context menu
    # Carries dict {frame_idx: file_path}
    roi_import_for_frames = Signal(object)

    def __init__(
        self,
        state: AppState,
        image_ctrl: ImageController,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._state = state
        self._image_ctrl = image_ctrl

        # Suppress _sync_selection when frame change originates from this widget
        self._suppress_sync = False

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        self._tree = QTreeWidget()
        self._tree.setColumnCount(3)
        # "Region" fits the narrow 50px column better than the full
        # "Region of Interest"; the button text inside the column
        # ("Need"/"Edit"/"Add") gives the unambiguous context.
        self._tree.setHeaderLabels([
            self.tr("#", "Image list column: frame index"),
            self.tr("Filename"),
            self.tr("Region", "Image list column: ROI status"),
        ])
        self._tree.setSelectionMode(QTreeWidget.SelectionMode.ExtendedSelection)
        self._tree.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self._tree.customContextMenuRequested.connect(self._show_context_menu)
        connect_theme(self._on_theme_changed)
        self._tree.setRootIsDecorated(False)
        self._tree.setIndentation(0)
        self._tree.setUniformRowHeights(True)
        self._roi_delegate = _RoiDelegate(self._tree)
        self._roi_delegate.activated.connect(self.roi_edit_requested)
        self._tree.setItemDelegateForColumn(_COL_ROI, self._roi_delegate)
        # Navigation scrolls explicitly; hovering an ROI action must not.
        self._tree.setAutoScroll(False)

        # Column sizing
        header = self._tree.header()
        header.setMinimumSectionSize(10)
        header.setStretchLastSection(False)
        header.resizeSection(_COL_FRAME, 32)
        # Region column only holds short button labels (Add / Edit / Need),
        # so give more horizontal space to the Filename (stretch) column.
        header.resizeSection(_COL_ROI, 44)
        header.setSectionResizeMode(_COL_FRAME, QHeaderView.ResizeMode.Fixed)
        header.setSectionResizeMode(_COL_FILENAME, QHeaderView.ResizeMode.Stretch)
        header.setSectionResizeMode(_COL_ROI, QHeaderView.ResizeMode.Fixed)

        # Styling
        themed(self._tree,
            """
            QTreeWidget {{
                background: {BG_SIDEBAR};
                border: none;
                outline: none;
            }}
            QTreeWidget::item {{
                border-bottom: 1px solid {BORDER};
                padding: 2px 0px;
            }}
            QTreeWidget::item:selected {{
                background: {BG_HOVER};
            }}
            QTreeWidget::item:hover:!selected {{
                background: {BG_INPUT};
            }}
            QHeaderView::section {{
                background: {BG_PANEL};
                color: {TEXT_MUTED};
                border: none;
                border-bottom: 1px solid {BORDER};
                border-right: 1px solid {BORDER};
                padding: 3px 4px;
                font-size: 10px;
                font-weight: bold;
            }}
            """
        )
        layout.addWidget(self._tree)

        # Connect state signals
        self._state.images_changed.connect(self._rebuild_list)
        self._state.current_frame_changed.connect(self._sync_selection)
        self._state.roi_changed.connect(self._update_roi_indicators)
        self._state.params_changed.connect(self._update_roi_indicators)

        # Tree selection -> state update (keyboard: eventFilter, mouse: itemClicked)
        self._tree.installEventFilter(self)
        self._tree.viewport().setMouseTracking(True)
        self._tree.viewport().installEventFilter(self)
        self._tree.itemClicked.connect(self._on_item_clicked)

    def _rebuild_list(self) -> None:
        """Insert lightweight rows in one batch; ROI actions are painted."""
        blocked = self._tree.blockSignals(True)
        self._tree.setUpdatesEnabled(False)
        try:
            self._tree.clear()
            items = []
            for i, filepath in enumerate(self._state.image_files):
                item = QTreeWidgetItem([f"{i:02d}", Path(filepath).name, ""])
                item.setTextAlignment(_COL_FRAME, Qt.AlignmentFlag.AlignCenter)
                item.setData(_COL_FRAME, Qt.ItemDataRole.UserRole, i)
                item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable)
                items.append(item)
            self._tree.addTopLevelItems(items)
            self._update_roi_indicators()
            if items:
                idx = min(self._state.current_frame, len(items) - 1)
                self._tree.setCurrentItem(items[idx])
        finally:
            self._tree.blockSignals(blocked)
            self._tree.setUpdatesEnabled(True)

    def _on_theme_changed(self, _name: str) -> None:
        """Reference-frame rows are tinted with the accent colour: tint again."""
        self._update_roi_indicators()

    def _update_roi_indicators(self) -> None:
        """Update ROI button text/style and ref-frame row highlighting."""
        n = len(self._state.image_files)
        if n == 0:
            return

        ref_frames = self._get_ref_frames()

        for i in range(n):
            item = self._tree.topLevelItem(i)
            if item is None:
                continue

            has_roi = i in self._state.per_frame_rois
            is_ref = i in ref_frames

            if has_roi:
                label, status = self.tr("Edit"), "has"
            elif is_ref:
                label, status = self.tr("Need"), "need"
            else:
                label, status = self.tr("Add"), "add"
            item.setText(_COL_ROI, label)
            item.setData(_COL_ROI, Qt.ItemDataRole.UserRole, status)

            # Ref-frame highlighting: subtle row background + bold accent filename
            n_cols = 3  # frame, filename, ROI
            if is_ref:
                highlight = QColor(COLORS.ACCENT)
                highlight = highlight.lighter(180)
                for col in range(n_cols):
                    item.setBackground(col, highlight)
                # Bold + accent color on filename for ref frames
                bold_font = QFont()
                bold_font.setBold(True)
                item.setFont(_COL_FILENAME, bold_font)
                item.setForeground(_COL_FILENAME, QColor(COLORS.ACCENT))
            else:
                for col in range(n_cols):
                    item.setData(col, Qt.ItemDataRole.BackgroundRole, None)
                # Reset filename styling
                item.setFont(_COL_FILENAME, QFont())
                item.setData(
                    _COL_FILENAME, Qt.ItemDataRole.ForegroundRole, None
                )

    def _get_ref_frames(self) -> set[int]:
        """Compute which frames are reference frames based on tracking settings."""
        n = len(self._state.image_files)
        if n == 0:
            return set()

        mode = self._state.tracking_mode

        if mode == "accumulative":
            return {0}

        # Incremental modes
        sub_mode = self._state.inc_ref_mode

        if sub_mode == "every_frame":
            # Every frame except the last is a reference
            return set(range(n - 1))

        if sub_mode == "every_n":
            interval = max(1, self._state.inc_ref_interval)
            return {i for i in range(0, n - 1, interval)}

        if sub_mode == "custom":
            custom = set(self._state.inc_custom_refs) | {0}
            return {i for i in custom if i < n - 1}

        # Fallback
        return {0}

    def _sync_selection(self, idx: int) -> None:
        """Sync tree selection when frame is changed externally."""
        if self._suppress_sync:
            return
        self._tree.blockSignals(True)
        item = self._tree.topLevelItem(idx)
        if item is not None:
            self._tree.setCurrentItem(item)
            # autoScroll is disabled tree-wide to stop hover-on-button from
            # scrolling the list, so we have to do explicit scrolling here
            # to keep the externally-selected frame visible (default hint
            # is EnsureVisible).
            self._tree.scrollToItem(item)
        self._tree.blockSignals(False)

    def eventFilter(self, obj, event):
        """Handle keyboard navigation in the tree — switch frame on arrow keys."""
        if obj is self._tree.viewport() and event.type() == QEvent.Type.MouseMove:
            index = self._tree.indexAt(event.position().toPoint())
            cursor = (Qt.CursorShape.PointingHandCursor if index.column() == _COL_ROI
                      else Qt.CursorShape.ArrowCursor)
            self._tree.viewport().setCursor(cursor)
        if obj is self._tree and event.type() == QEvent.Type.KeyRelease:
            if event.key() in (
                Qt.Key.Key_Up, Qt.Key.Key_Down,
                Qt.Key.Key_PageUp, Qt.Key.Key_PageDown,
                Qt.Key.Key_Home, Qt.Key.Key_End,
            ):
                current = self._tree.currentItem()
                if current is not None:
                    frame = current.data(_COL_FRAME, Qt.ItemDataRole.UserRole)
                    if frame is not None:
                        self._suppress_sync = True
                        self._state.set_current_frame(frame)
                        self._suppress_sync = False
                        # autoScroll is disabled tree-wide; scroll
                        # explicitly so keyboard nav still keeps the
                        # newly current row visible.
                        self._tree.scrollToItem(current)
        return super().eventFilter(obj, event)

    def _on_item_clicked(
        self,
        item: QTreeWidgetItem,
        column: int,
    ) -> None:
        """Handle mouse click — only switch frame for frame#/filename columns."""
        if column > _COL_FILENAME:
            return
        frame = item.data(_COL_FRAME, Qt.ItemDataRole.UserRole)
        if frame is not None:
            self._suppress_sync = True
            self._state.set_current_frame(frame)
            self._suppress_sync = False

    def _show_context_menu(self, pos) -> None:
        """Show right-click context menu with ROI and delete options."""
        menu = QMenu(self)
        selected = self._tree.selectedItems()
        sel_frames = self._get_selected_frames()
        n_sel = len(sel_frames)

        # --- ROI batch operations (when frames are selected) ---
        if n_sel > 0:
            import_label = self.tr(
                "Import Region of Interest for %n frame(s)", "", n_sel)
            import_action = QAction(import_label, self)
            import_action.triggered.connect(self._import_roi_selected)
            menu.addAction(import_action)

            n_with_roi = sum(
                1 for f in sel_frames if f in self._state.per_frame_rois
            )
            if n_with_roi:
                from al_dic.i18n import tr_args
                clear_label = tr_args(
                    self.tr("Clear Region of Interest (%1 with region)"),
                    n_with_roi,
                )
            else:
                clear_label = self.tr("Clear Region of Interest")
            clear_action = QAction(clear_label, self)
            clear_action.setEnabled(n_with_roi > 0)
            clear_action.triggered.connect(self._clear_roi_selected)
            menu.addAction(clear_action)

            menu.addSeparator()

        # --- Delete option ---
        if selected:
            count = len(selected)
            label = self.tr("Delete %n image(s)", "", count)
            delete_action = QAction(label, self)
            delete_action.triggered.connect(self._delete_selected)
            menu.addAction(delete_action)

        if not menu.actions():
            return
        menu.popup(self._tree.mapToGlobal(pos))

    def _delete_selected(self) -> None:
        """Remove selected images from the list and update state."""
        selected_items = self._tree.selectedItems()
        if not selected_items:
            return

        # Collect frame indices to delete (high indices first)
        rows_to_delete = sorted(
            {
                item.data(_COL_FRAME, Qt.ItemDataRole.UserRole)
                for item in selected_items
                if item.data(_COL_FRAME, Qt.ItemDataRole.UserRole) is not None
            },
            reverse=True,
        )
        if not rows_to_delete:
            return

        # Remove from file list (high indices first to preserve ordering)
        files = list(self._state.image_files)
        original_count = len(files)
        for row in rows_to_delete:
            if 0 <= row < len(files):
                del files[row]

        if not files:
            self._image_ctrl.set_files([])
            self._state.per_frame_rois.clear()
            self._state.set_image_files([])
            return

        # Re-key per_frame_rois after deletion.
        # Build old->new index mapping for surviving frames.
        deleted_set = set(rows_to_delete)
        old_to_new: dict[int, int] = {}
        new_idx = 0
        for old_idx in range(original_count):
            if old_idx not in deleted_set:
                old_to_new[old_idx] = new_idx
                new_idx += 1

        new_rois: dict[int, object] = {}
        for old_key, mask in list(self._state.per_frame_rois.items()):
            if old_key in old_to_new:
                new_rois[old_to_new[old_key]] = mask
        self._state.per_frame_rois = new_rois  # type: ignore[assignment]

        # Q6: image-list mutation invalidates pipeline results
        # (frame count / indices changed).
        from al_dic.gui.app_state import RunState
        had_results = self._state.results is not None
        if had_results:
            self._state.results = None
            self._state.deformed_masks = None
            self._state.run_state = RunState.IDLE
            self._state.show_deformed = False

        # Keep the controller's own list in step; clear_cache() alone would
        # leave it holding the deleted frames.
        self._image_ctrl.set_files(files)

        # Update state (triggers _rebuild_list via images_changed signal)
        self._state.image_files = files
        self._state.current_frame = min(
            self._state.current_frame, len(files) - 1
        )
        if had_results:
            self._state.results_changed.emit()
            self._state.run_state_changed.emit(RunState.IDLE)
        self._state.images_changed.emit()

    def _get_selected_frames(self) -> set[int]:
        """Extract frame indices from currently selected tree items."""
        frames = set()
        for item in self._tree.selectedItems():
            frame = item.data(_COL_FRAME, Qt.ItemDataRole.UserRole)
            if frame is not None:
                frames.add(frame)
        return frames

    def _import_roi_selected(self) -> None:
        """Open file dialog and import ROI masks for selected frames."""
        frames = sorted(self._get_selected_frames())
        if not frames:
            return
        from PySide6.QtWidgets import QFileDialog

        n_frames = len(frames)
        title = self.tr("Select %n Mask File(s)", "", n_frames)
        paths, _ = QFileDialog.getOpenFileNames(
            self,
            title,
            "",
            self.tr("Images") + " (*.png *.bmp *.tif *.tiff *.jpg *.jpeg);;"
            + self.tr("All Files") + " (*)",
        )
        if not paths:
            return
        if len(paths) != len(frames):
            from al_dic.i18n import tr_args
            self._state.log_message.emit(
                tr_args(
                    self.tr(
                        "Selected %1 files for %2 frames "
                        "— count must match"
                    ),
                    len(paths), len(frames),
                ),
                "warn",
            )
            return
        self.roi_import_for_frames.emit(dict(zip(frames, paths)))

    def _clear_roi_selected(self) -> None:
        """Clear ROI masks for all selected frames.

        If frame 0 is among the cleared frames, also drop the brush
        refinement mask -- it lives only on frame 0 and is meaningless
        without an ROI to scope it (Brush#5).
        """
        frames = self._get_selected_frames()
        changed = False
        for f in frames:
            if f in self._state.per_frame_rois:
                del self._state.per_frame_rois[f]
                changed = True
        if 0 in frames and self._state.refine_brush_mask is not None:
            self._state.set_refine_brush_mask(None)
            changed = True
        if changed:
            self._state.roi_changed.emit()
