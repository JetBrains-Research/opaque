"""Thread-safe LRU cache with a byte budget for native (Rust) handles.

Contributor-internal: not re-exported from the ``opaque.accounting`` façade.
"""

from __future__ import annotations

import os
import threading
import warnings
from collections import OrderedDict
from collections.abc import Callable

_registry: list["NativeCache"] = []


def _read_max_bytes(max_bytes_env: str | None, default_max_bytes: int) -> int:
    if max_bytes_env is None:
        return default_max_bytes
    raw = os.environ.get(max_bytes_env, "")
    if not raw.strip():
        return default_max_bytes
    try:
        return max(0, int(raw))
    except ValueError:
        return default_max_bytes


def _clear_all_native_caches() -> None:
    for cache in list(_registry):
        cache.clear()


class NativeCache:
    """LRU cache of native handles with a total-byte cap and entry cap."""

    def __init__(
        self,
        *,
        name: str,
        max_bytes_env: str | None,
        default_max_bytes: int,
        max_entries: int,
        nbytes_estimate: Callable[[tuple], int],
        destructor: Callable[[int], None],
    ) -> None:
        self._name = name
        self._max_bytes_env = max_bytes_env
        self._default_max_bytes = default_max_bytes
        self._max_entries = max_entries
        self._nbytes_estimate = nbytes_estimate
        self._destructor = destructor
        self._max_bytes = _read_max_bytes(max_bytes_env, default_max_bytes)
        self._cache: OrderedDict[tuple, int] = OrderedDict()
        self._lock = threading.Lock()
        _registry.append(self)

    @property
    def max_bytes(self) -> int:
        """Current byte cap (from env at construction, or default)."""
        return self._max_bytes

    def get_or_create(self, key: tuple, factory: Callable[[], int]) -> int | None:
        if self._max_bytes == 0:
            return None
        nbytes = self._nbytes_estimate(key)
        if nbytes > self._max_bytes:
            return None

        with self._lock:
            if key in self._cache:
                self._cache.move_to_end(key)
                return self._cache[key]

            current_bytes = sum(self._nbytes_estimate(k) for k in self._cache)
            while self._cache and (
                len(self._cache) >= self._max_entries
                or current_bytes + nbytes > self._max_bytes
            ):
                old_key, old_h = self._cache.popitem(last=False)
                current_bytes -= self._nbytes_estimate(old_key)
                self._safe_destruct(old_h)

            handle = factory()
            self._cache[key] = handle
            return handle

    def _safe_destruct(self, handle: int) -> None:
        try:
            self._destructor(handle)
        except Exception as exc:  # noqa: BLE001
            warnings.warn(
                f"NativeCache({self._name!r}): destructor failed for handle "
                f"{handle}: {exc}",
                stacklevel=2,
            )

    def clear(self) -> None:
        with self._lock:
            while self._cache:
                _k, h = self._cache.popitem(last=False)
                self._safe_destruct(h)

    def __len__(self) -> int:
        with self._lock:
            return len(self._cache)

    def __contains__(self, key: object) -> bool:
        with self._lock:
            return key in self._cache


def native_cache(
    *,
    name: str,
    max_bytes_env: str | None,
    default_max_bytes: int,
    max_entries: int,
    nbytes_estimate: Callable[[tuple], int],
    destructor: Callable[[int], None],
) -> NativeCache:
    """Construct a :class:`NativeCache` (readable call sites for adapters)."""
    return NativeCache(
        name=name,
        max_bytes_env=max_bytes_env,
        default_max_bytes=default_max_bytes,
        max_entries=max_entries,
        nbytes_estimate=nbytes_estimate,
        destructor=destructor,
    )
