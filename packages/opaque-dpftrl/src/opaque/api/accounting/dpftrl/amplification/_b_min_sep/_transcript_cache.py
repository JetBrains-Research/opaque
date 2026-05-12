"""Rust-backed transcript corpora for b-min-sep MC (compact memory, reuse across σ)."""

from __future__ import annotations

from opaque.api.accounting.core import _native
from opaque.api.accounting.core._native_cache import native_cache

_cache = native_cache(
    name="b_min_sep_transcripts",
    max_bytes_env="OPAQUE_B_MIN_SEP_TRANSCRIPT_CACHE_MAX_BYTES",
    default_max_bytes=4 * 1024 * 1024 * 1024,
    max_entries=4,
    nbytes_estimate=lambda k: 3 * k[3] * k[1] * 8,  # 3 * num_mc_samples * n_steps * 8
    destructor=_native.drop_b_min_sep_transcript_corpus,
)


def get_handle_or_none(
    strategy_coef: tuple[float, ...],
    n_steps: int,
    p: float,
    num_mc_samples: int,
    mc_seed: int,
) -> int | None:
    """Return a Rust corpus handle for reuse, or None to use one-shot MC."""
    key = (strategy_coef, n_steps, p, num_mc_samples, mc_seed)

    def factory() -> int:
        return _native.register_b_min_sep_transcript_corpus(
            list(strategy_coef),
            n_steps,
            p,
            num_mc_samples,
            mc_seed,
        )

    try:
        return _cache.get_or_create(key, factory)
    except ValueError:
        return None
