"""Tests for b_min_sep BandMF amplification."""

import opaque.accounting as acc
import opaque.dpftrl.accounting as ftrl_acc
from opaque.dpftrl.accounting.amplification._b_min_sep import (
    _participation_p_from_per_example_rate,
)


def test_p_conversion():
    p0 = 0.05
    b = 8
    p = _participation_p_from_per_example_rate(p0, b)
    assert p > p0
    assert abs(1.0 / p0 - (1.0 / p + (b - 1))) < 1e-9


def test_b_min_sep_smoke_pld():
    inner = ftrl_acc.band_mf(1.0, sensitivity=0.5, num_groups=10)
    coef = (0.8**0.5, 0.2**0.5, 0.0, 0.0)
    proc = ftrl_acc.b_min_sep(
        inner,
        strategy_coefficients=coef,
        n_steps=40,
        p0=0.02,
    )
    eps = proc.pld(num_mc_samples=5000, seed=123).epsilon_at(1e-3)
    assert eps > 0.0 and eps < 500.0


def test_transcript_cache_reuses_same_handle():
    """Repeated cache lookup returns the same Rust corpus handle."""
    from opaque.dpftrl.accounting.amplification._b_min_sep_transcript_cache import (
        get_handle_or_none,
    )

    h1 = get_handle_or_none((1.0, 0.0), 12, 0.08, 80, 7)
    h2 = get_handle_or_none((1.0, 0.0), 12, 0.08, 80, 7)
    assert h1 is not None and h2 is not None
    assert h1 == h2


def _drain_cache(tc) -> None:
    """Drop any leftover native handles so tests start from a clean slate."""
    with tc._lock:
        while tc._cache:
            _k, h = tc._cache.popitem(last=False)
            tc._native.drop_b_min_sep_transcript_corpus(h)


def test_transcript_cache_evicts_lru(monkeypatch):
    """Cache caps entries and drops LRU native handle *before* registering new."""
    from opaque.dpftrl.accounting.amplification import _b_min_sep_transcript_cache as tc

    monkeypatch.delenv("OPAQUE_B_MIN_SEP_TRANSCRIPT_CACHE_MAX_BYTES", raising=False)
    monkeypatch.setattr(tc, "_MAX_ENTRIES", 2)
    _drain_cache(tc)

    calls: list[tuple[str, int]] = []
    orig_drop = tc._native.drop_b_min_sep_transcript_corpus
    orig_register = tc._native.register_b_min_sep_transcript_corpus

    def tracking_drop(handle: int) -> None:
        calls.append(("drop", handle))
        orig_drop(handle)

    def tracking_register(*args, **kwargs) -> int:
        hid = orig_register(*args, **kwargs)
        calls.append(("register", hid))
        return hid

    monkeypatch.setattr(tc._native, "drop_b_min_sep_transcript_corpus", tracking_drop)
    monkeypatch.setattr(
        tc._native, "register_b_min_sep_transcript_corpus", tracking_register
    )

    h1 = tc.get_handle_or_none((1.0, 0.0), 10, 0.05, 64, 1)
    h2 = tc.get_handle_or_none((1.0, 0.0), 10, 0.05, 64, 2)
    assert h1 is not None and h2 is not None
    assert len(tc._cache) == 2
    assert calls == [("register", h1), ("register", h2)]

    # The third distinct entry must drop the LRU BEFORE registering the new
    # corpus — that ordering is the regression this PR fixes.
    calls.clear()
    h3 = tc.get_handle_or_none((1.0, 0.0), 10, 0.05, 64, 3)
    assert h3 is not None
    assert len(tc._cache) == 2
    assert calls == [("drop", h1), ("register", h3)], (
        f"eviction must precede registration; got {calls}"
    )

    # Touching h2 promotes it to MRU; next insert evicts h3, not h2.
    assert tc.get_handle_or_none((1.0, 0.0), 10, 0.05, 64, 2) == h2
    calls.clear()
    h4 = tc.get_handle_or_none((1.0, 0.0), 10, 0.05, 64, 4)
    assert h4 is not None
    assert calls == [("drop", h3), ("register", h4)]
    assert len(tc._cache) == 2

    _drain_cache(tc)


def test_transcript_cache_evicts_for_byte_cap(monkeypatch):
    """Byte budget forces eviction even when entry count is below the cap."""
    from opaque.dpftrl.accounting.amplification import _b_min_sep_transcript_cache as tc

    monkeypatch.setattr(tc, "_MAX_ENTRIES", 16)
    _drain_cache(tc)

    # Each entry is 3 * 64 * 10 * 8 = 15360 bytes; budget fits exactly one.
    monkeypatch.setenv(
        "OPAQUE_B_MIN_SEP_TRANSCRIPT_CACHE_MAX_BYTES",
        str(tc._estimate_raw_bytes(64, 10)),
    )

    calls: list[tuple[str, int]] = []
    orig_drop = tc._native.drop_b_min_sep_transcript_corpus
    orig_register = tc._native.register_b_min_sep_transcript_corpus

    def tracking_drop(handle: int) -> None:
        calls.append(("drop", handle))
        orig_drop(handle)

    def tracking_register(*args, **kwargs) -> int:
        hid = orig_register(*args, **kwargs)
        calls.append(("register", hid))
        return hid

    monkeypatch.setattr(tc._native, "drop_b_min_sep_transcript_corpus", tracking_drop)
    monkeypatch.setattr(
        tc._native, "register_b_min_sep_transcript_corpus", tracking_register
    )

    h1 = tc.get_handle_or_none((1.0, 0.0), 10, 0.05, 64, 10)
    assert h1 is not None
    calls.clear()
    h2 = tc.get_handle_or_none((1.0, 0.0), 10, 0.05, 64, 11)
    assert h2 is not None
    # Drop-before-register also holds on the byte-cap path.
    assert calls == [("drop", h1), ("register", h2)]
    assert len(tc._cache) == 1

    _drain_cache(tc)


def test_b_min_sep_stricter_than_mf_only():
    """Subsampling should lower ε at fixed σ vs unamplified BandMF PLD."""
    from opaque.accounting import _native as native
    from opaque.accounting.discretization import get_discretization

    inner = ftrl_acc.band_mf(1.0, sensitivity=0.7, num_groups=5)
    coef = (1.0, 0.0, 0.0)
    bms = ftrl_acc.b_min_sep(
        inner,
        strategy_coefficients=coef,
        n_steps=20,
        p0=0.1,
    )
    cfg = get_discretization()
    pld_mf = native.mf_gaussian_pld(1.0, 0.7, cfg.to_native())
    eps_mf = pld_mf.epsilon_at(1e-3)
    eps_bms = bms.pld(num_mc_samples=8000, seed=1).epsilon_at(1e-3)
    assert eps_bms < eps_mf
