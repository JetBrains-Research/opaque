"""Tests for b_min_sep BandMF amplification."""

import opaque_accounting as acc
from opaque_accounting.amplification.b_min_sep import (
    _participation_p_from_per_example_rate,
)


def test_p_conversion():
    p0 = 0.05
    b = 8
    p = _participation_p_from_per_example_rate(p0, b)
    assert p > p0
    assert abs(1.0 / p0 - (1.0 / p + (b - 1))) < 1e-9


def test_b_min_sep_smoke_pld():
    inner = acc.band_mf(1.0, sensitivity=0.5, num_groups=10)
    coef = (0.8**0.5, 0.2**0.5, 0.0, 0.0)
    proc = acc.b_min_sep(
        inner,
        strategy_coefficients=coef,
        n_steps=40,
        participation_rate_p0=0.02,
        num_mc_samples=5000,
        mc_seed=123,
    )
    eps = proc.epsilon_at(1e-3)
    assert eps > 0.0 and eps < 500.0


def test_transcript_cache_reuses_same_buffers():
    """Small (n, S) transcripts are cached; repeated get_or_prepare returns same lists."""
    from opaque_accounting.amplification._b_min_sep_transcript_cache import get_or_prepare

    t1 = get_or_prepare((1.0, 0.0), 12, 0.08, 80, 7)
    t2 = get_or_prepare((1.0, 0.0), 12, 0.08, 80, 7)
    assert t1 is not None and t2 is not None
    assert t1[0] is t2[0]
    assert t1[1] is t2[1]
    assert t1[2] is t2[2]


def test_b_min_sep_stricter_than_mf_only():
    """Subsampling should lower ε at fixed σ vs unamplified BandMF PLD."""
    from opaque_accounting import opaque_accounting as native
    from opaque_accounting.discretization import get_discretization

    inner = acc.band_mf(1.0, sensitivity=0.7, num_groups=5)
    coef = (1.0, 0.0, 0.0)
    bms = acc.b_min_sep(
        inner,
        strategy_coefficients=coef,
        n_steps=20,
        participation_rate_p0=0.1,
        num_mc_samples=8000,
        mc_seed=1,
    )
    cfg = get_discretization()
    pld_mf = native.mf_gaussian_pld(1.0, 0.7, cfg.to_native())
    eps_mf = pld_mf.epsilon_at(1e-3)
    eps_bms = bms.epsilon_at(1e-3)
    assert eps_bms < eps_mf
