"""Behavioral regressions for imports and bounded memory.

Run with pytest or python -m unittest tests.test_large_sequence_support.
The tests need no particular image resolution, machine RAM, or real dataset.
"""

from __future__ import annotations

import tempfile
import threading
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
from PIL import Image
from PySide6.QtCore import QEvent, QPointF, QThread, QTimer, Qt
from PySide6.QtGui import QMouseEvent
from PySide6.QtWidgets import QApplication, QPushButton, QStyleOptionViewItem

from al_dic.core.data_structures import DICPara, GridxyROIRange
from al_dic.gui.app_state import AppState, RunState
from al_dic.gui.controllers.frame_provider import StreamingFrameProvider
from al_dic.gui.controllers.image_controller import ImageController, _decode_grayscale_float64
from al_dic.gui.controllers.pipeline_controller import PipelineController, _build_masks
from al_dic.gui.widgets.image_list import ImageList
from al_dic.io.image_ops import ListFrameProvider
from al_dic.utils.array_cache import ArrayLRU, cache_budget


def pump_until(predicate, timeout=8):
    deadline = time.monotonic() + timeout
    while not predicate():
        QApplication.processEvents()
        if time.monotonic() > deadline:
            raise AssertionError("Asynchronous operation did not finish")
        time.sleep(0.002)
    QApplication.processEvents()


def make_frames(folder, count=3, shape=(32, 48), dtype=np.uint8):
    folder.mkdir(parents=True, exist_ok=True)
    paths = []
    base = (np.random.default_rng(5).random(shape) * 200).astype(dtype)
    for i in range(count):
        path = folder / f"image_{i:05d}.tif"
        Image.fromarray(base + i).save(path)
        paths.append(str(path))
    return paths


class TestArrayLRU(unittest.TestCase):
    def test_byte_limit_and_recency(self):
        c = ArrayLRU(100, 32)
        for i in (0, 1):
            c[i] = np.zeros(16, np.uint8)
        c[0]
        c[2] = np.ones(16, np.uint8)
        self.assertEqual(list(c), [0, 2])
        self.assertEqual(c.nbytes, 32)

    def test_oversized_and_disabled_cache(self):
        for budget in (0, 8):
            c = ArrayLRU(4, budget)
            c[0] = np.zeros(16, np.uint8)
            self.assertEqual(len(c), 0)
            self.assertEqual(c.nbytes, 0)

    def test_replace_clear_and_item_limit(self):
        c = ArrayLRU(2, 100)
        for i in range(100):
            c[i] = np.zeros(4, np.uint8)
        self.assertEqual(len(c), 2)
        c[99] = np.zeros(8, np.uint8)
        self.assertEqual(c.nbytes, 12)
        c.clear()
        self.assertEqual(c.nbytes, 0)

    def test_budget_uses_available_ram_and_has_fallback(self):
        fake = SimpleNamespace(virtual_memory=lambda: SimpleNamespace(available=1000))
        with patch.dict("sys.modules", {"psutil": fake}):
            self.assertEqual(cache_budget(200, 0.05), 50)
        with patch.dict("sys.modules", {"psutil": None}):
            self.assertEqual(cache_budget(1024**3), 64 * 1024**2)


class TestSharedMasks(unittest.TestCase):
    def test_ten_thousand_references_have_two_buffers(self):
        roi = np.ones((8, 11), bool)
        roi[0] = False
        masks = _build_masks({0: roi}, 10000, roi.shape, set(range(9999)))
        self.assertEqual(len({id(m) for m in masks}), 2)
        self.assertIs(masks[0], masks[9998])
        self.assertFalse(masks[0].flags.writeable)
        self.assertFalse(masks[-1].flags.writeable)
        np.testing.assert_array_equal(masks[0], roi)
        roi[:] = False
        self.assertTrue(masks[0][1, 0])  # worker snapshot is isolated

    def test_custom_masks_preserve_values_and_aliases(self):
        a = np.eye(8, dtype=bool)
        b = np.ones((8, 8), bool)
        masks = _build_masks({0: a, 2: b, 3: b}, 5, a.shape, {0, 1})
        self.assertIs(masks[0], masks[1])
        self.assertIs(masks[2], masks[3])
        np.testing.assert_array_equal(masks[2], b)
        self.assertEqual(masks[-1].sum(), 64)


class QtCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.folder = Path(self.temp.name)
        self.state = AppState()
        self.ctrl = ImageController(self.state)
        self.widgets = []

    def tearDown(self):
        self.ctrl.cancel_loading()
        pump_until(lambda: not self.ctrl.is_loading)
        for widget in self.widgets:
            widget.close()
            widget.deleteLater()
        self.state.deleteLater()
        self.app.sendPostedEvents(None, QEvent.Type.DeferredDelete)
        self.app.processEvents()
        self.temp.cleanup()


class TestImageListScale(QtCase):
    def test_ten_thousand_rows_without_button_widgets(self):
        view = ImageList(self.state, self.ctrl)
        self.widgets.append(view)
        self.state.set_image_files([f"frame_{i}.tif" for i in range(10000)])
        self.assertEqual(view._tree.topLevelItemCount(), 10000)
        self.assertEqual(view.findChildren(QPushButton), [])
        self.assertIsNone(view._tree.itemWidget(view._tree.topLevelItem(0), 2))

    def test_roi_status_and_click_route_to_correct_frame(self):
        view = ImageList(self.state, self.ctrl)
        self.widgets.append(view)
        self.state.set_image_files(["a.tif", "b.tif", "c.tif"])
        self.assertEqual(view._tree.topLevelItem(0).text(2), "Need")
        self.assertEqual(view._tree.topLevelItem(1).text(2), "Add")
        self.state.per_frame_rois[1] = np.ones((2, 2), bool)
        self.state.roi_changed.emit()
        self.assertEqual(view._tree.topLevelItem(1).text(2), "Edit")
        seen = []
        view.roi_edit_requested.connect(seen.append)
        press = QMouseEvent(
            QEvent.Type.MouseButtonPress, QPointF(5, 5), QPointF(5, 5),
            Qt.MouseButton.LeftButton, Qt.MouseButton.LeftButton,
            Qt.KeyboardModifier.NoModifier,
        )
        event = QMouseEvent(
            QEvent.Type.MouseButtonRelease, QPointF(5, 5), QPointF(5, 5),
            Qt.MouseButton.LeftButton, Qt.MouseButton.NoButton,
            Qt.KeyboardModifier.NoModifier,
        )
        index = view._tree.model().index(1, 2)
        view._roi_delegate.editorEvent(event, view._tree.model(), QStyleOptionViewItem(), index)
        self.assertEqual(seen, [])  # a release without a press is not a click
        view._roi_delegate.editorEvent(press, view._tree.model(), QStyleOptionViewItem(), index)
        view._roi_delegate.editorEvent(event, view._tree.model(), QStyleOptionViewItem(), index)
        self.assertEqual(seen, [1])

    def test_selection_and_delete_all_do_not_resurrect_frames(self):
        view = ImageList(self.state, self.ctrl)
        self.widgets.append(view)
        make_frames(self.folder)
        self.ctrl.load_folder(str(self.folder))
        self.state.set_current_frame(2)
        self.assertEqual(view._tree.currentItem().text(0), "02")
        view._tree.selectAll()
        view._delete_selected()
        self.ctrl.set_natural_sort(True)
        self.assertEqual(self.state.image_files, [])


class TestAsyncImport(QtCase):
    def test_dropped_file_imports_parent_folder(self):
        paths = make_frames(self.folder)
        self.ctrl.load_folder_async(paths[1])
        pump_until(lambda: not self.ctrl.is_loading)
        self.assertEqual(self.state.image_files, paths)
        self.assertEqual(self.state.image_folder, self.folder)

    def test_first_preview_is_decoded_in_worker_even_without_cache(self):
        from al_dic.gui.controllers import image_scan
        paths = make_frames(self.folder)
        self.ctrl._cache_rgb.max_bytes = 0
        threads, images = [], []
        real_decode = image_scan._decode_rgb

        def decode(path):
            threads.append(QThread.currentThread())
            return real_decode(path)

        self.state.images_changed.connect(lambda: images.append(self.ctrl.read_image_rgb(0)))
        with patch.object(image_scan, "_decode_rgb", side_effect=decode), patch(
            "al_dic.gui.controllers.image_controller._decode_rgb",
            side_effect=AssertionError("preview decoded on GUI thread"),
        ):
            self.ctrl.load_folder_async(str(self.folder))
            pump_until(lambda: not self.ctrl.is_loading)
        self.assertEqual(self.state.image_files, paths)
        self.assertEqual(len(images), 1)
        self.assertEqual(len(threads), 1)
        self.assertNotEqual(threads[0], self.app.thread())

    def test_gui_heartbeat_continues_during_slow_scan(self):
        from al_dic.gui.controllers import image_scan
        make_frames(self.folder)
        entered, release = threading.Event(), threading.Event()
        original = image_scan._read_image_size
        ticks = []
        timer = QTimer()
        timer.setInterval(5)
        timer.timeout.connect(lambda: ticks.append(1))

        def slow(path):
            entered.set()
            release.wait(5)
            return original(path)

        try:
            with patch.object(image_scan, "_read_image_size", side_effect=slow):
                timer.start()
                self.ctrl.load_folder_async(str(self.folder))
                pump_until(entered.is_set)
                pump_until(lambda: len(ticks) >= 5)
                self.assertTrue(self.ctrl.is_loading)
                release.set()
                pump_until(lambda: not self.ctrl.is_loading)
        finally:
            release.set()
            timer.stop()

    def test_latest_folder_wins_and_worker_count_is_bounded(self):
        from al_dic.gui.controllers import image_scan
        a, b = self.folder / "a", self.folder / "b"
        make_frames(a)
        expected = make_frames(b, 2)
        entered, release = threading.Event(), threading.Event()
        original = image_scan._read_image_size

        def slow(path):
            if str(a) in path:
                entered.set()
                release.wait(5)
            return original(path)

        try:
            with patch.object(image_scan, "_read_image_size", side_effect=slow):
                self.ctrl.load_folder_async(str(a))
                pump_until(entered.is_set)
                old = self.ctrl._scan_worker
                for _ in range(20):
                    self.ctrl.load_folder_async(str(b))
                self.assertIs(self.ctrl._scan_worker, old)
                release.set()
                pump_until(lambda: not self.ctrl.is_loading)
            self.assertEqual(self.state.image_files, expected)
        finally:
            release.set()

    def test_cancel_and_shutdown_preserve_existing_sequence(self):
        from al_dic.gui.controllers import image_scan
        make_frames(self.folder)
        self.state.set_image_files(["old.tif"])
        entered, release = threading.Event(), threading.Event()
        original = image_scan._read_image_size

        def slow(path):
            entered.set()
            release.wait(5)
            return original(path)

        try:
            with patch.object(image_scan, "_read_image_size", side_effect=slow):
                self.ctrl.load_folder_async(str(self.folder))
                pump_until(entered.is_set)
                self.assertFalse(self.ctrl.shutdown())
                release.set()
                pump_until(lambda: not self.ctrl.is_loading)
            self.assertTrue(self.ctrl.shutdown())
            self.assertEqual(self.state.image_files, ["old.tif"])
        finally:
            release.set()

    def test_failure_preserves_existing_sequence(self):
        self.state.set_image_files(["old.tif"])
        errors = []
        self.ctrl.load_failed.connect(errors.append)
        self.ctrl.load_folder_async(str(self.folder / "missing"))
        pump_until(lambda: not self.ctrl.is_loading)
        self.assertEqual(len(errors), 1)
        self.assertEqual(self.state.image_files, ["old.tif"])

    def test_queued_result_is_discarded_after_cancel(self):
        make_frames(self.folder)
        self.state.set_image_files(["old.tif"])
        self.ctrl.load_folder_async(str(self.folder))
        self.assertTrue(self.ctrl._scan_worker.wait(5000))
        # The worker is done, but its result has not reached the GUI yet.
        self.ctrl.cancel_loading()
        pump_until(lambda: not self.ctrl.is_loading)
        self.assertEqual(self.state.image_files, ["old.tif"])

    def test_unicode_folder_mismatches_and_unreadable_files(self):
        folder = self.folder / "试样 α"
        make_frames(folder, 2)
        Image.new("L", (19, 17)).save(folder / "image_00002.tif")
        (folder / "image_00003.tif").write_bytes(b"invalid image")
        messages, gui_threads = [], []
        self.state.log_message.connect(lambda text, level: messages.append(text))
        self.state.images_changed.connect(lambda: gui_threads.append(QThread.currentThread()))
        self.ctrl.load_folder_async(str(folder))
        pump_until(lambda: not self.ctrl.is_loading)
        self.assertEqual(len(self.state.image_files), 4)
        self.assertTrue(any("size mismatch" in m for m in messages))
        self.assertTrue(any("Could not read 1" in m for m in messages))
        self.assertEqual(gui_threads, [self.app.thread()])

    def test_empty_folder_clears_sequence(self):
        self.state.set_image_files(["old.tif"])
        self.ctrl.load_folder_async(str(self.folder))
        pump_until(lambda: not self.ctrl.is_loading)
        self.assertEqual(self.state.image_files, [])

    def test_import_is_rejected_while_solver_runs(self):
        self.state.run_state = RunState.RUNNING
        errors = []
        self.ctrl.load_failed.connect(errors.append)
        self.ctrl.load_folder_async(str(self.folder))
        self.assertFalse(self.ctrl.is_loading)
        self.assertEqual(len(errors), 1)

    def test_main_window_close_waits_without_blocking_gui(self):
        from al_dic.gui.app import MainWindow
        from al_dic.gui.controllers import image_scan
        make_frames(self.folder)
        entered, release = threading.Event(), threading.Event()
        original = image_scan._read_image_size

        def slow(path):
            entered.set()
            release.wait(5)
            return original(path)

        with patch.object(AppState, "instance", return_value=self.state):
            win = MainWindow()
        self.widgets.append(win)
        win.show()
        try:
            with patch.object(image_scan, "_read_image_size", side_effect=slow):
                win._image_ctrl.load_folder_async(str(self.folder))
                pump_until(entered.is_set)
                win.close()
                self.assertTrue(win.isVisible())
                release.set()
                pump_until(lambda: not win.isVisible())
            self.assertFalse(win._image_ctrl.is_loading)
        finally:
            release.set()
            win._image_ctrl.cancel_loading()
            pump_until(lambda: not win._image_ctrl.is_loading)

    def test_sort_toggle_does_not_scan_again(self):
        make_frames(self.folder)
        self.ctrl.load_folder(str(self.folder))
        with patch("al_dic.gui.controllers.image_controller._read_image_size",
                   side_effect=AssertionError("unexpected rescan")):
            self.ctrl.set_natural_sort(True)
        self.assertEqual(len(self.state.image_files), 3)

    def test_run_cannot_start_during_import(self):
        proxy = SimpleNamespace(is_loading=True)
        pipeline = PipelineController(self.state, proxy)
        pipeline.start()
        self.assertIsNone(pipeline._worker)
        self.assertEqual(self.state.run_state, RunState.IDLE)


class TestFrameCacheParity(QtCase):
    def test_grayscale_cache_is_bounded_and_reload_exact(self):
        paths = make_frames(self.folder, 10)
        self.ctrl.load_folder(str(self.folder))
        first = self.ctrl.read_image(0).copy()
        for i in range(len(paths)):
            self.ctrl.read_image(i)
        self.assertLessEqual(len(self.ctrl._cache), 4)
        self.assertNotIn(0, self.ctrl._cache)
        np.testing.assert_array_equal(first, self.ctrl.read_image(0))

    def test_stream_matches_eager_for_depths_and_tiny_budgets(self):
        for dtype in (np.uint8, np.uint16):
            paths = make_frames(self.folder / str(dtype), dtype=dtype)
            roi = GridxyROIRange(gridx=(3, 40), gridy=(2, 28))
            eager = ListFrameProvider([_decode_grayscale_float64(p) for p in paths], roi)
            for budget in (0, 64, 32 * 48 * 8 * 2):
                stream = StreamingFrameProvider(paths, roi, max_cache_bytes=budget)
                for i in (0, 2, 1, 0, 2):
                    np.testing.assert_array_equal(stream.get_normalized(i), eager.get_normalized(i))
                    self.assertLessEqual(stream._cache.nbytes, budget)

    def test_solver_results_match_eager_with_shared_readonly_masks(self):
        from al_dic.core.pipeline import run_aldic
        paths = make_frames(self.folder, count=3, shape=(64, 64))
        roi = GridxyROIRange(gridx=(10, 54), gridy=(10, 54))
        para = DICPara(
            winstepsize=16, winsize=20, winsize_min=8, tol=1e-2,
            mu=1e-3, admm_max_iter=2, admm_tol=1e-2, gauss_pt_order=2,
            alpha=0.0, use_global_step=False, disp_smoothness=0.0,
            strain_smoothness=0.0, smoothness=0.0, method_to_compute_strain=3,
            strain_type=0, gridxy_roi_range=roi, img_size=(64, 64),
            icgn_max_iter=50, reference_mode="incremental",
        )
        mask = np.ones((64, 64), bool)
        a = run_aldic(para, [_decode_grayscale_float64(p) for p in paths],
                      [mask.astype(float) for _ in paths], compute_strain=False)
        b = run_aldic(para, StreamingFrameProvider(paths, roi, max_cache_bytes=0),
                      _build_masks({0: mask}, len(paths), mask.shape, {0, 1}),
                      compute_strain=False)
        self.assertEqual(len(a.result_disp), len(b.result_disp))
        for x, y in zip(a.result_disp, b.result_disp):
            np.testing.assert_array_equal(x.U, y.U)
            np.testing.assert_array_equal(x.U_accum, y.U_accum)
        for x, y in zip(a.result_def_grad, b.result_def_grad):
            np.testing.assert_array_equal(x.F, y.F)


if __name__ == "__main__":
    unittest.main()
