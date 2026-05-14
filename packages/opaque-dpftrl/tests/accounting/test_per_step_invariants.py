"""K-prefix invariants for :meth:`DpFtrlProcess._pld_at_horizon` (= ``per_step(proc) * K``).

The K-prefix bound is the deployed-and-stopped-early mechanism: the
K-step PLD is evaluated using the N-tuned strategy coefficients and the
post-processing inequality on the K-prefix projection of the N-step
output stream.  This file asserts the invariants that flow from that
construction:

- ``ε(_pld_at_horizon(0))         == 0`` (empty accountant).
- ``ε(_pld_at_horizon(N))         == proc.epsilon_at(δ)``.
- ``K1 ≤ K2 ⇒ ε(K1) ≤ ε(K2)`` (monotone; MC paths within slack).
- ``ε(K) ≤ ε(self)`` for ``K ≤ N`` (bounded by full).
- For K = G · atomic_unit + r with r ∈ [0, atomic_unit):
  ``ε(G·M) ≤ ε(K) ≤ ε((G+1)·M)`` (sandwich; closed-form paths only).

Implemented as ``(per_step(proc) * K).epsilon_at(δ)`` since
``_pld_at_horizon`` is internal — but the assertions probe the same
worker.
"""

from __future__ import annotations

import math
from collections.abc import Callable

import pytest

import opaque.dpftrl.accounting as ftrl_acc
from opaque.api.accounting.core.mechanisms.types import Identity
from opaque.dpftrl.accounting.types import (
    BallsInBins,
    BMinSep,
    CyclicPoisson,
    DpFtrlProcess,
)
from opaque.dpftrl.noise import (
    band_mf_strategy,
    bisr_strategy,
    blt_strategy,
    bsr_strategy,
    identity_strategy,
    lambda_cgd_strategy,
)

_DELTA = 1e-5
_MC_KW = {"num_mc_samples": 4000, "seed": 17}


def _atomic_unit(proc: DpFtrlProcess) -> int:
    """Diagnostic — the K-rounding granularity for this amplifier."""
    return int(proc.atomic_unit)


def _eps_at(proc: DpFtrlProcess, K: int, delta: float) -> float:
    """K-step ε via ``per_step(proc) * K`` — the public API surface.

    MC kwargs are read off the global ``acc.set_discretization`` config
    when the underlying amplifier is MC-based; tests pin them with the
    :func:`_seed_mc` fixture.
    """
    if K <= 0:
        return Identity().epsilon_at(delta)
    step = ftrl_acc.per_step(proc)
    return (step * K).epsilon_at(delta)


@pytest.fixture
def _seed_mc():
    """Pin global MC discretization for MC-based amplifiers.

    ``Repeated.pld()`` → ``repeated_pld(count)`` only accepts the core
    discretization kwargs; MC sample budget / seed must come from the
    global config.  Set globally for the duration of the test.
    """
    import opaque.accounting as acc

    acc.set_discretization(num_mc_samples=_MC_KW["num_mc_samples"], seed=_MC_KW["seed"])
    yield
    acc.set_discretization()  # restore defaults


# ---------------------------------------------------------------------------
# CyclicPoisson(IdentityMf): per-step granularity, closed-form PLD,
# exact match to underlying subsampled-Gaussian composition.
# ---------------------------------------------------------------------------


class TestCyclicPoissonIdentity:
    def _proc(self, n_steps: int = 200) -> CyclicPoisson:
        return ftrl_acc.poisson(
            ftrl_acc.mf_gaussian(1.0, identity_strategy()),
            sample_rate=0.01,
            n_steps=n_steps,
        )

    def test_atomic_unit_is_one(self):
        assert _atomic_unit(self._proc()) == 1

    def test_zero_step_returns_identity_eps(self):
        # per_step(proc) * 0 is not constructible; the empty accountant
        # already pins ε=0 at K=0.
        assert _eps_at(self._proc(), 0, _DELTA) == 0.0

    def test_full_step_matches_proc(self):
        proc = self._proc(100)
        e_full = proc.epsilon_at(_DELTA)
        e_via_step = _eps_at(proc, 100, _DELTA)
        assert math.isclose(e_full, e_via_step, rel_tol=1e-9)

    def test_overshoot_step_raises(self):
        proc = self._proc(100)
        step = ftrl_acc.per_step(proc)
        with pytest.raises(ValueError, match="exceeds n_steps"):
            (step * 10_000).epsilon_at(_DELTA)

    def test_monotonic(self):
        proc = self._proc(200)
        prev = -math.inf
        for k in [1, 5, 25, 50, 100, 150, 199, 200]:
            e = _eps_at(proc, k, _DELTA)
            assert e >= prev - 1e-10, f"non-monotone at k={k}: {e} < {prev}"
            prev = e

    def test_bounded_by_full(self):
        proc = self._proc(200)
        e_full = proc.epsilon_at(_DELTA)
        for k in [1, 50, 100, 199]:
            assert _eps_at(proc, k, _DELTA) <= e_full + 1e-10

    def test_truncated_poisson_works(self):
        """K-prefix preserves the truncated-Poisson kwarg pair."""
        proc = ftrl_acc.poisson(
            ftrl_acc.mf_gaussian(1.0, identity_strategy()),
            sample_rate=0.01,
            n_steps=200,
            truncated_batch_size=50,
            dataset_size=5000,
        )
        e_at_75 = _eps_at(proc, 75, _DELTA)
        assert math.isfinite(e_at_75)
        assert e_at_75 <= proc.epsilon_at(_DELTA) + 1e-10


# ---------------------------------------------------------------------------
# CyclicPoisson(BandMf): band-quantised granularity, sandwich per band.
# ---------------------------------------------------------------------------


class TestCyclicPoissonBand:
    def _proc(self, n_steps: int = 100, bands: int = 8) -> CyclicPoisson:
        return ftrl_acc.poisson(
            ftrl_acc.mf_gaussian(1.0, band_mf_strategy(bands=bands)),
            sample_rate=0.01,
            n_steps=n_steps,
        )

    def test_atomic_unit_equals_bands(self):
        assert _atomic_unit(self._proc(bands=8)) == 8
        assert _atomic_unit(self._proc(bands=4)) == 4

    def test_plateau_within_band(self):
        """ε(K) is constant for K in the same band — rounds up to next band."""
        proc = self._proc(n_steps=100, bands=8)
        # K=7 and K=8 both round up to 8.
        e_7 = _eps_at(proc, 7, _DELTA)
        e_8 = _eps_at(proc, 8, _DELTA)
        assert math.isclose(e_7, e_8, rel_tol=1e-12)

    def test_endpoints(self):
        proc = self._proc(n_steps=64, bands=8)
        assert _eps_at(proc, 0, _DELTA) == 0.0
        e_full = proc.epsilon_at(_DELTA)
        assert math.isclose(_eps_at(proc, 64, _DELTA), e_full, rel_tol=1e-12)

    def test_sandwich_at_every_step(self):
        """For K = G·M + r: ε(G·M) ≤ ε(K) ≤ ε((G+1)·M) for r ∈ [0, M)."""
        proc = self._proc(n_steps=80, bands=8)
        M = _atomic_unit(proc)
        for K in range(1, proc.n_steps + 1):
            G, r = divmod(K, M)
            e_lo = _eps_at(proc, G * M, _DELTA) if G > 0 else 0.0
            e_K = _eps_at(proc, K, _DELTA)
            e_hi_step = min((G + 1) * M, proc.n_steps)
            e_hi = _eps_at(proc, e_hi_step, _DELTA)
            assert e_lo - 1e-10 <= e_K <= e_hi + 1e-10, (
                f"sandwich broken at K={K} (G={G}, r={r}): {e_lo} ≤ {e_K} ≤ {e_hi}"
            )

    def test_monotonic_across_bands(self):
        proc = self._proc(n_steps=100, bands=8)
        eps_seq = [_eps_at(proc, k, _DELTA) for k in range(8, 101, 8)]
        for a, b in zip(eps_seq, eps_seq[1:]):
            assert b >= a - 1e-10


# ---------------------------------------------------------------------------
# BMinSep(BandMf): MC-based; endpoints + trend with seeded MC.
# ---------------------------------------------------------------------------


class TestBMinSep:
    def _proc(self, n_steps: int = 32, bands: int = 4) -> BMinSep:
        return ftrl_acc.b_min_sep(
            ftrl_acc.mf_gaussian(1.0, band_mf_strategy(bands=bands)),
            n_steps=n_steps,
            p0=0.02,
        )

    def test_atomic_unit_equals_bands(self):
        assert _atomic_unit(self._proc(bands=4)) == 4
        assert _atomic_unit(self._proc(bands=8)) == 8

    def test_endpoints(self, _seed_mc):
        proc = self._proc(n_steps=32)
        assert _eps_at(proc, 0, _DELTA) == 0.0
        e_full = proc.pld(**_MC_KW).epsilon_at(_DELTA)
        e_via_step = _eps_at(proc, 32, _DELTA)
        assert math.isclose(e_full, e_via_step, rel_tol=1e-9)

    def test_trend_increases(self, _seed_mc):
        """Aggregate ε increases between K=bands and K=N."""
        proc = self._proc(n_steps=32, bands=4)
        e_full = _eps_at(proc, 32, _DELTA)
        e_small = _eps_at(proc, 4, _DELTA)
        assert e_small < e_full, (
            f"expected ε(K=4) < ε(K=32) in expectation, got {e_small} ≮ {e_full}"
        )


# ---------------------------------------------------------------------------
# BallsInBins(IdentityMf): epoch-quantised granularity, Gram-free.
# ---------------------------------------------------------------------------


class TestBallsInBinsIdentity:
    def _proc(self, num_bins: int = 10, num_epochs: int = 10) -> BallsInBins:
        return ftrl_acc.balls_in_bins(
            ftrl_acc.mf_gaussian(1.0, identity_strategy()),
            num_bins=num_bins,
            n_steps=num_bins * num_epochs,
        )

    def test_atomic_unit_equals_num_bins(self):
        assert _atomic_unit(self._proc(num_bins=10)) == 10
        assert _atomic_unit(self._proc(num_bins=4)) == 4

    def test_endpoints(self, _seed_mc):
        proc = self._proc(num_bins=10, num_epochs=10)
        assert _eps_at(proc, 0, _DELTA) == 0.0
        e_full = _eps_at(proc, 100, _DELTA)
        e_direct = proc.pld(**_MC_KW).epsilon_at(_DELTA)
        assert math.isclose(e_full, e_direct, rel_tol=1e-9)

    def test_monotonic(self, _seed_mc):
        proc = self._proc(num_bins=10, num_epochs=10)
        e_full = _eps_at(proc, 100, _DELTA)
        prev = 0.0
        for k in range(10, 101, 10):
            e = _eps_at(proc, k, _DELTA)
            # 15% slack: bnb_mc_pld_identity uses an importance-sampled MC.
            assert e >= prev - 0.15 * max(e_full, 1.0), (
                f"non-monotone at k={k}: {e} < {prev}"
            )
            assert e <= e_full + 0.15 * max(e_full, 1.0)
            prev = e


# ---------------------------------------------------------------------------
# Cross-product of (amplification, inner mechanism) supported by K-prefix.
#
# DP-FTRL accounting is defined by combining an amplification factory with
# an inner mechanism and dispatching by ``match self.inner.strategy``
# inside the amplification's ``_pld_at_horizon``.  These tests mirror that
# structure: every (amp, inner) pair that ``_pld_at_horizon`` should
# support is exercised against the same invariants.
#
# To add a new supported pair: register the amp / inner factory below and
# append the tuple to ``_SUPPORTED_PAIRS``.
# ---------------------------------------------------------------------------


_AMPLIFICATIONS: dict[str, tuple[Callable[..., DpFtrlProcess], bool]] = {
    "CyclicPoisson": (
        lambda inner: ftrl_acc.poisson(inner, sample_rate=0.01, n_steps=64),
        False,
    ),
    "BMinSep": (
        lambda inner: ftrl_acc.b_min_sep(inner, n_steps=32, p0=0.02),
        True,
    ),
    "BallsInBins": (
        lambda inner: ftrl_acc.balls_in_bins(inner, num_bins=4, n_steps=16),
        True,  # bnb_mc_pld is MC-based.
    ),
}


_MECHANISMS: dict[str, Callable[[], object]] = {
    "IdentityMf": lambda: ftrl_acc.mf_gaussian(1.0, identity_strategy()),
    "BandMf": lambda: ftrl_acc.mf_gaussian(1.0, band_mf_strategy(bands=4)),
    "Blt": lambda: ftrl_acc.mf_gaussian(1.0, blt_strategy()),
    "LambdaCgd": lambda: ftrl_acc.mf_gaussian(1.0, lambda_cgd_strategy(lambda_=0.5)),
    "Bisr": lambda: ftrl_acc.mf_gaussian(1.0, bisr_strategy(bandwidth=2)),
    "Bsr": lambda: ftrl_acc.mf_gaussian(
        1.0, bsr_strategy(bandwidth=2, alpha=1.0, beta=0.5)
    ),
}


# Pairs where ``_pld_at_horizon`` returns a usable K-step PLD for any K.
_SUPPORTED_PAIRS: list[tuple[str, str]] = [
    ("CyclicPoisson", "IdentityMf"),
    ("CyclicPoisson", "BandMf"),
    ("BMinSep", "BandMf"),
    ("BallsInBins", "IdentityMf"),
    ("BallsInBins", "Blt"),
    ("BallsInBins", "LambdaCgd"),
    ("BallsInBins", "Bisr"),
    ("BallsInBins", "Bsr"),
]


def _build(amp: str, mech: str) -> DpFtrlProcess:
    factory, _ = _AMPLIFICATIONS[amp]
    return factory(_MECHANISMS[mech]())


def _is_mc(amp: str) -> bool:
    return _AMPLIFICATIONS[amp][1]


def _pld_kwargs(amp: str) -> dict:
    return _MC_KW if _is_mc(amp) else {}


def _eps_via_step(proc: DpFtrlProcess, K: int, delta: float) -> float:
    """K-step ε via ``per_step(proc) * K`` — the public idiom."""
    if K <= 0:
        return Identity().epsilon_at(delta)
    step = ftrl_acc.per_step(proc)
    return (step * K).epsilon_at(delta)


def _is_mc_proc(proc: DpFtrlProcess) -> bool:
    return isinstance(proc, (BMinSep, BallsInBins))


def _eps_full(proc: DpFtrlProcess, delta: float) -> float:
    """Full-horizon ε direct off ``proc`` (MC kwargs forwarded on the leaf)."""
    if _is_mc_proc(proc):
        return proc.pld(**_MC_KW).epsilon_at(delta)
    return proc.epsilon_at(delta)


def _pair_id(pair: tuple[str, str]) -> str:
    return f"{pair[0]}({pair[1]})"


_SUPPORTED_IDS = [_pair_id(p) for p in _SUPPORTED_PAIRS]


@pytest.mark.parametrize("amp,mech", _SUPPORTED_PAIRS, ids=_SUPPORTED_IDS)
class TestKPrefixInvariants:
    """For every supported (amp, inner) pair, ``per_step(proc) * K`` satisfies
    the documented contract: endpoint identities, monotone non-decreasing,
    bounded by ε of the full process, sandwich at atomic-unit boundaries
    (closed-form paths only), and the result is a real :class:`DpProcess`
    (composes with ``|``, ``*``).

    MC paths read their sample budget / seed off the global
    discretization config (pinned via :func:`_seed_mc`), since
    ``Repeated.pld()`` only forwards the four core discretization kwargs
    — MC kwargs cannot be threaded through ``(step * K).epsilon_at(δ)``.
    """

    def test_inherits_dp_ftrl_process(self, amp: str, mech: str, _seed_mc):
        proc = _build(amp, mech)
        assert isinstance(proc, DpFtrlProcess)

    def test_step_zero_is_identity_eps(self, amp: str, mech: str, _seed_mc):
        proc = _build(amp, mech)
        assert _eps_via_step(proc, 0, _DELTA) == 0.0

    def test_full_horizon_matches_proc(self, amp: str, mech: str, _seed_mc):
        proc = _build(amp, mech)
        e_full_via_step = _eps_via_step(proc, proc.n_steps, _DELTA)
        e_full = _eps_full(proc, _DELTA)
        assert math.isclose(e_full_via_step, e_full, rel_tol=1e-9)

    def test_overshoot_step_raises(self, amp: str, mech: str, _seed_mc):
        proc = _build(amp, mech)
        step = ftrl_acc.per_step(proc)
        with pytest.raises(ValueError, match="exceeds n_steps"):
            (step * (proc.n_steps + 10_000)).epsilon_at(_DELTA)

    def test_bounded_by_full(self, amp: str, mech: str, _seed_mc):
        proc = _build(amp, mech)
        e_full = _eps_full(proc, _DELTA)
        # MC paths have transcript noise: 10% slack at the test budget.
        slack = 0.10 * max(e_full, 1.0) if _is_mc_proc(proc) else 1e-9
        for K in (1, proc.atomic_unit, proc.n_steps // 2, proc.n_steps - 1):
            assert _eps_via_step(proc, K, _DELTA) <= e_full + slack

    def test_trend_increases(self, amp: str, mech: str, _seed_mc):
        proc = _build(amp, mech)
        e_small = _eps_via_step(proc, proc.atomic_unit, _DELTA)
        e_full = _eps_full(proc, _DELTA)
        assert e_small < e_full

    def test_monotone_at_unit_boundaries(self, amp: str, mech: str, _seed_mc):
        proc = _build(amp, mech)
        n, M = proc.n_steps, proc.atomic_unit
        e_full = _eps_full(proc, _DELTA)
        # MC paths regenerate independent transcripts at each n_steps, so
        # the per-K ε curve has MC noise; the monotone bound holds in
        # expectation, with seed-dependent per-step gaps.  15% slack is
        # empirically robust at the test's sample budget.
        slack = 0.15 * max(e_full, 1.0) if _is_mc_proc(proc) else 1e-9
        prev = 0.0
        steps = list(range(M, n + 1, M))
        if steps[-1] != n:
            steps.append(n)
        for K in steps:
            e_K = _eps_via_step(proc, K, _DELTA)
            assert e_K >= prev - slack
            prev = e_K

    def test_sandwich_at_intermediate_K(self, amp: str, mech: str, _seed_mc):
        if _is_mc_proc(_build(amp, mech)):
            pytest.skip("MC path: sandwich holds in expectation only")
        proc = _build(amp, mech)
        n, M = proc.n_steps, proc.atomic_unit
        K = max(1, min(n - 1, n // 2 + M // 2))
        G, _r = divmod(K, M)
        e_lo = _eps_via_step(proc, G * M, _DELTA) if G > 0 else 0.0
        e_K = _eps_via_step(proc, K, _DELTA)
        e_hi = _eps_via_step(proc, min((G + 1) * M, n), _DELTA)
        assert e_lo - 1e-9 <= e_K <= e_hi + 1e-9

    def test_supports_composition(self, amp: str, mech: str, _seed_mc):
        proc = _build(amp, mech)
        step = ftrl_acc.per_step(proc)
        # step * K1 | step * K2 composes (same proc → merges to Repeated).
        combined = (step * (proc.n_steps // 2)) | (step * proc.atomic_unit)
        assert math.isfinite(combined.epsilon_at(_DELTA))


# ---------------------------------------------------------------------------
# BallsInBins(correlated MF): K-prefix gram matches a freshly-built strategy
# at the shorter horizon for recipe-driven strategies (BSR, BiSR, λ-CGD).
# For BLT (the only retuning inner) the K-prefix uses the N-tuned Toeplitz
# first column, so the values may differ from a K-tuned re-build — they
# stay bounded by ε(self) and monotone in K, which the parametrised
# invariants above already check.
# ---------------------------------------------------------------------------


_REGEN_NUM_BINS = 4
_REGEN_N_FULL = 16
_REGEN_K = 8
_REGEN_TOL = 0.05  # generous: bnb_mc_pld is MC-based.


def _bnb(mechanism, n_steps: int) -> BallsInBins:
    return ftrl_acc.balls_in_bins(mechanism, num_bins=_REGEN_NUM_BINS, n_steps=n_steps)


class TestRecipeDrivenGramRegen:
    """``per_step(proc) * K`` matches a strategy built directly at horizon ``K``.

    For each recipe-driven MF (BSR / BiSR / λ-CGD), build BnB at full
    horizon and ``per_step * K``; independently build BnB at horizon ``K``
    from a fresh strategy.  Both paths use the same ``bnb_mc_pld``
    primitive on the same Gram structure, so ε agrees up to MC variance.
    """

    def test_bsr(self, _seed_mc):
        full = bsr_strategy(bandwidth=2, alpha=1.0, beta=0.5)
        direct = bsr_strategy(bandwidth=2, alpha=1.0, beta=0.5)
        proc = _bnb(ftrl_acc.mf_gaussian(1.0, full), _REGEN_N_FULL)
        step = ftrl_acc.per_step(proc)
        e_at = (step * _REGEN_K).epsilon_at(_DELTA)
        e_dir = (
            _bnb(ftrl_acc.mf_gaussian(1.0, direct), _REGEN_K)
            .pld(**_MC_KW)
            .epsilon_at(_DELTA)
        )
        assert math.isclose(e_at, e_dir, rel_tol=_REGEN_TOL)

    def test_bisr(self, _seed_mc):
        full = bisr_strategy(bandwidth=2)
        direct = bisr_strategy(bandwidth=2)
        proc = _bnb(ftrl_acc.mf_gaussian(1.0, full), _REGEN_N_FULL)
        step = ftrl_acc.per_step(proc)
        e_at = (step * _REGEN_K).epsilon_at(_DELTA)
        e_dir = (
            _bnb(ftrl_acc.mf_gaussian(1.0, direct), _REGEN_K)
            .pld(**_MC_KW)
            .epsilon_at(_DELTA)
        )
        assert math.isclose(e_at, e_dir, rel_tol=_REGEN_TOL)

    def test_lambda_cgd(self, _seed_mc):
        full = lambda_cgd_strategy(lambda_=0.5)
        direct = lambda_cgd_strategy(lambda_=0.5)
        proc = _bnb(ftrl_acc.mf_gaussian(1.0, full), _REGEN_N_FULL)
        step = ftrl_acc.per_step(proc)
        e_at = (step * _REGEN_K).epsilon_at(_DELTA)
        e_dir = (
            _bnb(ftrl_acc.mf_gaussian(1.0, direct), _REGEN_K)
            .pld(**_MC_KW)
            .epsilon_at(_DELTA)
        )
        assert math.isclose(e_at, e_dir, rel_tol=_REGEN_TOL)
