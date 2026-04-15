"""Internal cache of b-min-sep MC transcripts for faster calibration (no public API)."""

from __future__ import annotations

import os
import threading
from collections import OrderedDict

from opaque_accounting import opaque_accounting as _native

# Cap resident transcript RAM: each cache entry holds ~3 * num_samples * n_steps * 8 bytes.
# Override with OPAQUE_B_MIN_SEP_TRANSCRIPT_CACHE_MAX_BYTES (e.g. 3000000000 for large runs).
def _max_cache_bytes() -> int:
    raw = os.environ.get("OPAQUE_B_MIN_SEP_TRANSCRIPT_CACHE_MAX_BYTES", "")
    if raw.strip():
        try:
            return max(0, int(raw))
        except ValueError:
            return 512 * 1024 * 1024
    return 512 * 1024 * 1024


_MAX_ENTRIES = 2

_lock = threading.Lock()
_cache: OrderedDict[tuple, tuple[list[float], list[float], list[float]]] = OrderedDict()


def _estimate_bytes(
    rx: list[float], rz: list[float], ae: list[float]
) -> int:
    return 8 * (len(rx) + len(rz) + len(ae))


def get_or_prepare(
    strategy_coef: tuple[float, ...],
    n_steps: int,
    p: float,
    num_mc_samples: int,
    mc_seed: int,
) -> tuple[list[float], list[float], list[float]] | None:
    """Return transcripts for PLD-from-transcripts, or None if too large to retain.

    When None, callers should use ``bandmf_b_min_sep_warm_mc_pld`` (one-shot MC)
    to avoid allocating multi-GB transcript buffers on every calibration probe.
    """
    max_b = _max_cache_bytes()
    nbytes = 3 * num_mc_samples * n_steps * 8
    if max_b == 0 or nbytes > max_b:
        return None

    key = (strategy_coef, n_steps, p, num_mc_samples, mc_seed)
    with _lock:
        if key in _cache:
            _cache.move_to_end(key)
            return _cache[key]
        rx, rz, ae = _native.bandmf_b_min_sep_prepare_transcripts(
            list(strategy_coef),
            n_steps,
            p,
            num_mc_samples,
            mc_seed,
        )
        nbytes = _estimate_bytes(rx, rz, ae)
        if nbytes > max_b:
            return None
        while _cache and (
            len(_cache) >= _MAX_ENTRIES
            or sum(_estimate_bytes(*v) for v in _cache.values()) + nbytes > max_b
        ):
            _cache.popitem(last=False)
        _cache[key] = (rx, rz, ae)
        _cache.move_to_end(key)
        return (rx, rz, ae)
