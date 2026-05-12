"""Rust-backed transcript corpora for b-min-sep MC (compact memory, reuse across σ)."""

from __future__ import annotations

import os
import threading
from collections import OrderedDict

from opaque.api.accounting.core import _native


_HARD_CAP_BYTES = 96 * 1024 * 1024 * 1024


def _max_registry_bytes() -> int:
    """Upper bound for registering one corpus (raw f64 storage ~3×S×n×8 bytes).

    Default 4 GiB fits small horizons.  Long DP-FTRL runs (large ``n_steps`` ×
    ``num_mc_samples``) need a larger working set so :func:`get_handle_or_none`
    can register a transcript at all — otherwise every noise-calibration probe
    rebuilds a fresh Monte Carlo corpus (very slow).  See
    :func:`_effective_registry_budget` which bumps the allowance up to this
    corpus size (capped at ``_HARD_CAP_BYTES``) when the configured cap is too
    small.

    Set ``OPAQUE_B_MIN_SEP_TRANSCRIPT_CACHE_MAX_BYTES`` to cap lower on small
    VMs, or ``0`` to disable transcript reuse (always one-shot MC).
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


def _effective_registry_budget(requested_nbytes: int) -> int:
    """Bytes we are willing to spend to hold *this* transcript once.

    Noise calibration calls :func:`get_handle_or_none` with identical
    ``(coefs, n_steps, p, S, seed)`` while varying only ``σ``; reusing the
    registered corpus is critical.  When the raw corpus exceeds the
    user-configured cap (default 4 GiB) but still fits under
    ``_HARD_CAP_BYTES``, bump the budget for this registration attempt so
    large-horizon jobs do not silently fall back to per-σ one-shot MC.
    """
    configured = _max_registry_bytes()
    if configured == 0:
        return 0
    return min(_HARD_CAP_BYTES, max(configured, requested_nbytes))


def get_handle_or_none(
    strategy_coef: tuple[float, ...],
    n_steps: int,
    p: float,
    num_mc_samples: int,
    mc_seed: int,
) -> int | None:
    """Return a Rust corpus handle for reuse, or None to use one-shot MC."""
    if _max_registry_bytes() == 0:
        return None
    nbytes = _estimate_raw_bytes(num_mc_samples, n_steps)
    max_b = _effective_registry_budget(nbytes)
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
