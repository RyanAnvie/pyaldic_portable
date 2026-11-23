"""Byte- and item-bounded image caches, independent of Qt and the platform."""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Iterator, MutableMapping

import numpy as np


def cache_budget(ceiling: int, fraction: float = 0.02) -> int:
    """Use a small fraction of available RAM, or a conservative fallback.

    psutil is optional. Unknown memory availability never means unlimited
    caching. The budget bounds retained arrays, not the solver's working set.
    """
    fallback = min(ceiling, 64 * 1024**2)
    try:
        import psutil

        available = int(psutil.virtual_memory().available)
        if available > 0:
            return max(0, min(ceiling, int(available * fraction)))
    except (ImportError, OSError, AttributeError, ValueError):
        pass
    return fallback


class ArrayLRU(MutableMapping):
    """Retain at most max_items and max_bytes of array payloads.

    An oversized frame is served by its caller without being retained here.
    Composition avoids OrderedDict subclass eviction differences across
    Python versions. Each instance is confined to its owning thread.
    """

    def __init__(self, max_items: int, max_bytes: int) -> None:
        self.max_items = max(0, int(max_items))
        self.max_bytes = max(0, int(max_bytes))
        self._data: OrderedDict[int, np.ndarray] = OrderedDict()
        self.nbytes = 0

    def __getitem__(self, key: int) -> np.ndarray:
        value = self._data[key]
        self._data.move_to_end(key)
        return value

    def __setitem__(self, key: int, value: np.ndarray) -> None:
        if key in self._data:
            del self[key]
        size = value.nbytes
        if not self.max_items or size > self.max_bytes:
            return
        while self._data and (
            len(self._data) >= self.max_items
            or self.nbytes + size > self.max_bytes
        ):
            del self[next(iter(self._data))]
        self._data[key] = value
        self.nbytes += size

    def __delitem__(self, key: int) -> None:
        self.nbytes -= self._data.pop(key).nbytes

    def __iter__(self) -> Iterator:
        return iter(self._data)

    def __len__(self) -> int:
        return len(self._data)

    def __contains__(self, key: object) -> bool:
        return key in self._data

    def clear(self) -> None:
        self._data.clear()
        self.nbytes = 0
