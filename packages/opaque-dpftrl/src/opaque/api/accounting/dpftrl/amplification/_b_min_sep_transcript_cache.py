"""Rust-backed transcript corpora for b-min-sep MC (compact memory, reuse across σ)."""

from __future__ import annotations

import os
import threading
from collections import OrderedDict

from opaque.api.accounting.core import _native


def _max_registry_bytes() -> int:
    """Upper bound for registering one corpus (raw f64 storage ~3×S×n×8 bytes).

    Default 4 GiB fits realistic calibration (e.g. n=2000, S=50k → ~2.4 GiB).
    Set ``OPAQUE_B_MIN_SEP_TRANSCRIPT_CACHE_MAX_BYTES`` to cap lower on small VMs,
    or ``0`` to disable transcript reuse (always one-shot MC).
    """
    raw = os.environ.get("OPAQUE_B_MIN_SEP_TRANSCRIPT_CACHE_MAX_BYTES", "")
    if raw.strip():
        try:
            return max(0, int(raw))
        except ValueError:
            return 4 * 1024 * 1024 * 1024
    return 4 * 1024 * 1024 * 1024


_MAX_ENTRIES = 4

_lock = threading.Lock()
# key -> native corpus handle (u64)
_cache: OrderedDict[tuple, int] = OrderedDict()


def _estimate_raw_bytes(num_mc_samples: int, n_steps: int) -> int:
    return 3 * num_mc_samples * n_steps * 8


def get_handle_or_none(
    strategy_coef: tuple[float, ...],
    n_steps: int,
    p: float,
    num_mc_samples: int,
    mc_seed: int,
) -> int | None:
    """Return a Rust corpus handle for reuse, or None to use one-shot MC."""
    max_b = _max_registry_bytes()
    if max_b == 0:
        return None
    nbytes = _estimate_raw_bytes(num_mc_samples, n_steps)
    if nbytes > max_b:
        return None

    key = (strategy_coef, n_steps, p, num_mc_samples, mc_seed)
    with _lock:
        if key in _cache:
            _cache.move_to_end(key)
            return _cache[key]
        # Evict LRU entries BEFORE allocating the new corpus so peak native
        # memory stays under `max_b`; otherwise a nearly-full cache plus a
        # large new entry can OOM before any eviction runs.
        current_bytes = sum(_estimate_raw_bytes(k[3], k[1]) for k in _cache)
        while _cache and (
            len(_cache) >= _MAX_ENTRIES or current_bytes + nbytes > max_b
        ):
            old_key, old_h = _cache.popitem(last=False)
            current_bytes -= _estimate_raw_bytes(old_key[3], old_key[1])
            _native.drop_b_min_sep_transcript_corpus(old_h)
        try:
            hid = _native.register_b_min_sep_transcript_corpus(
                list(strategy_coef),
                n_steps,
                p,
                num_mc_samples,
                mc_seed,
            )
        except ValueError:
            return None
        _cache[key] = hid
        return hid
