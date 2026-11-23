"""Cancellable folder import. Workers return data, never mutate GUI state."""

from __future__ import annotations

import os
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PySide6.QtCore import QThread, Signal

from al_dic.gui.controllers.image_controller import (
    SUPPORTED_EXTENSIONS, _decode_rgb, _lexicographic_sort_key,
    _natural_sort_key, _read_image_size,
)


@dataclass
class FolderScan:
    folder: Path
    files: list[str]
    preview: np.ndarray | None
    warnings: list[str]


class FolderScanWorker(QThread):
    result_ready = Signal(int, object)
    progress = Signal(int, int, int)
    failed = Signal(int, str)

    def __init__(self, generation: int, folder: str, natural: bool, parent=None):
        super().__init__(parent)
        self.generation = generation
        self.folder = folder
        self.natural = natural
        self._cancel = threading.Event()

    def cancel(self) -> None:
        self._cancel.set()

    def run(self) -> None:
        try:
            files = []
            # Resolving a dropped file versus a directory can itself block on
            # a network filesystem, so keep this probe in the worker too.
            folder = Path(self.folder)
            if folder.is_file():
                folder = folder.parent
            with os.scandir(folder) as entries:
                for entry in entries:
                    if self._cancel.is_set():
                        return
                    if (Path(entry.name).suffix.lower() in SUPPORTED_EXTENSIONS
                            and entry.is_file()):
                        files.append(entry.path)
            key = _natural_sort_key if self.natural else _lexicographic_sort_key
            files.sort(key=lambda p: key(Path(p).name))
            ref_shape = None
            mismatches, unreadable = [], []
            mismatch_count = unreadable_count = 0
            last_progress = 0.0
            for i, path in enumerate(files):
                if self._cancel.is_set():
                    return
                try:
                    shape = _read_image_size(path)
                    if i == 0:
                        ref_shape = shape
                    elif ref_shape is not None and shape != ref_shape:
                        mismatch_count += 1
                        if len(mismatches) < 5:
                            mismatches.append(f"{Path(path).name} ({shape[1]}x{shape[0]})")
                except Exception as exc:
                    if i == 0:
                        raise ValueError(f"Cannot read reference image: {path}: {exc}") from exc
                    unreadable_count += 1
                    if len(unreadable) < 5:
                        unreadable.append(Path(path).name)
                now = time.monotonic()
                if now - last_progress >= 0.1 or i + 1 == len(files):
                    self.progress.emit(self.generation, i + 1, len(files))
                    last_progress = now
            if self._cancel.is_set():
                return
            # First display decode also belongs off the GUI thread.
            preview = _decode_rgb(files[0]) if files else None
            warnings = []
            if mismatch_count:
                warnings.append(
                    f"Image size mismatch! Reference is {ref_shape[1]}x{ref_shape[0]}, "
                    f"but {mismatch_count} image(s) differ: " + ", ".join(mismatches)
                )
            if unreadable_count:
                warnings.append(
                    f"Could not read {unreadable_count} image(s): " + ", ".join(unreadable)
                )
            if not self._cancel.is_set():
                self.result_ready.emit(
                    self.generation,
                    FolderScan(folder, files, preview, warnings),
                )
        except Exception as exc:
            if not self._cancel.is_set():
                self.failed.emit(self.generation, f"{type(exc).__name__}: {exc}")
