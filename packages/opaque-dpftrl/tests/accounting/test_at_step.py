"""Contract tests for :meth:`DpFtrlProcess.approx_at_step`.

The contract is documented on :class:`DpFtrlProcess`:

- ``ε(approx_at_step(0))         == 0`` (returns Identity).
- ``ε(approx_at_step(N))         == self.epsilon_at(δ)`` (returns self).
- ``K1 ≤ K2 ⇒ ε(approx_at_step(K1)) ≤ ε(approx_at_step(K2))`` (monotone).
- ``ε(approx_at_step(G·M)) ≤ ε(approx_at_step(K)) ≤ ε(approx_at_step((G+1)·M))`` (sandwich).
- ``ε(approx_at_step(K)) ≤ ε(self)`` for ``K ≤ N``.

Tests cover all three amplifications (CyclicPoisson, BMinSep, BallsInBins)
and type preservation.  For BallsInBins with correlated-MF inners, an
oracle test verifies that ``approx_at_step`` matches a freshly-built
strategy at the shorter horizon.
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
    bisr_strategy,
    blt_strategy,
    bsr_strategy,
    identity_strategy,
    lambda_cgd_strategy,
)
from opaque.dpftrl.noise.types import BandMfStrategy

_DELTA = 1e-5
_MC_KW = {"num_mc_samples": 4000, "seed": 17}


# ---------------------------------------------------------------------------
# CyclicPoisson(IdentityMf): per-step granularity, exact match to underlying
# subsampled-Gaussian composition.
# ---------------------------------------------------------------------------


class TestCyclicPoissonIdentity:
    def _proc(self, n_steps: int = 200) -> CyclicPoisson:
        return ftrl_acc.poisson(
            ftrl_acc.mf_gaussian(1.0, identity_strategy()),
            sample_rate=0.01,
            n_steps=n_steps,
        )

    def test_atomic_unit_is_one(self):
        assert self._proc().atomic_unit == 1

    def test_zero_step_returns_identity(self):
        sub = self._proc().approx_at_step(0)
        assert isinstance(sub, Identity)
        assert sub.epsilon_at(_DELTA) == 0.0

    def test_negative_step_returns_identity(self):
        assert isinstance(self._proc().approx_at_step(-5), Identity)

    def test_full_step_returns_self(self):
        proc = self._proc(100)
        assert proc.approx_at_step(100) is proc

    def test_overshoot_step_returns_self(self):
        proc = self._proc(100)
        assert proc.approx_at_step(10_000) is proc

    def test_type_preserved(self):
        sub = self._proc().approx_at_step(50)
        assert type(sub) is CyclicPoisson
        assert isinstance(sub, DpFtrlProcess)

    def test_n_steps_replaced(self):
        proc = self._proc(200)
        assert proc.approx_at_step(73).n_steps == 73  # M=1, no rounding

    def test_exact_match_to_explicit_construction(self):
        """``proc.approx_at_step(K)`` ≡ same factory called with ``n_steps=K``."""
        K = 137
        full = self._proc(200)
        partial = full.approx_at_step(K)
        explicit = self._proc(K)
        assert math.isclose(
            partial.epsilon_at(_DELTA),
            explicit.epsilon_at(_DELTA),
            rel_tol=1e-9,
        )

    def test_monotonic(self):
        proc = self._proc(200)
        prev = -math.inf
        for k in [0, 1, 5, 25, 50, 100, 150, 199, 200]:
            e = proc.approx_at_step(k).epsilon_at(_DELTA)
            assert e >= prev - 1e-10, f"non-monotone at k={k}: {e} < {prev}"
            prev = e

    def test_bounded_by_full(self):
        proc = self._proc(200)
        e_full = proc.epsilon_at(_DELTA)
        for k in [1, 50, 100, 199]:
            assert proc.approx_at_step(k).epsilon_at(_DELTA) <= e_full + 1e-10

    def test_truncated_poisson_works(self):
        """at_step preserves the truncated-Poisson kwarg pair."""
        proc = ftrl_acc.poisson(
            ftrl_acc.mf_gaussian(1.0, identity_strategy()),
            sample_rate=0.01,
            n_steps=200,
            truncated_batch_size=50,
            dataset_size=5000,
        )
        sub = proc.approx_at_step(75)
        assert isinstance(sub, CyclicPoisson)
        assert sub.truncated_batch_size == 50
        assert sub.dataset_size == 5000
        assert math.isfinite(sub.epsilon_at(_DELTA))


# ---------------------------------------------------------------------------
# CyclicPoisson(BandMf): band-quantised granularity, sandwich verified per band.
# ---------------------------------------------------------------------------


class TestCyclicPoissonBand:
    def _proc(self, n_steps: int = 100, bands: int = 8) -> CyclicPoisson:
        coeffs = tuple(1.0 for _ in range(bands))
        return ftrl_acc.poisson(
            ftrl_acc.mf_gaussian(
                1.0,
                BandMfStrategy(sensitivity=float(bands) ** 0.5, coefficients=coeffs),
            ),
            sample_rate=0.01,
            n_steps=n_steps,
        )

    def test_atomic_unit_equals_bands(self):
        assert self._proc(bands=8).atomic_unit == 8
        assert self._proc(bands=4).atomic_unit == 4

    def test_at_step_rounds_up_to_band(self):
        proc = self._proc(n_steps=100, bands=8)
        # K=7 → ceil(7/8)*8 = 8
        assert proc.approx_at_step(7).n_steps == 8
        # K=8 → 8
        assert proc.approx_at_step(8).n_steps == 8
        # K=9 → 16
        assert proc.approx_at_step(9).n_steps == 16

    def test_at_step_clamps_at_n_steps(self):
        proc = self._proc(n_steps=100, bands=8)
        # 100 is not a multiple of 8 (12*8=96, 13*8=104).
        # at_step(99) would round to 104 but is clamped to 100.
        assert proc.approx_at_step(99).n_steps == 100

    def test_endpoints(self):
        proc = self._proc(n_steps=64, bands=8)
        assert proc.approx_at_step(0).epsilon_at(_DELTA) == 0.0
        assert proc.approx_at_step(64) is proc

    def test_sandwich_at_every_step(self):
        """For K = G·M + r: ε(G·M) ≤ ε(K) ≤ ε((G+1)·M) for r ∈ [0, M)."""
        proc = self._proc(n_steps=80, bands=8)
        M = proc.atomic_unit
        for K in range(1, proc.n_steps + 1):
            G, r = divmod(K, M)
            e_lo = proc.approx_at_step(G * M).epsilon_at(_DELTA) if G > 0 else 0.0
            e_K = proc.approx_at_step(K).epsilon_at(_DELTA)
            e_hi_step = min((G + 1) * M, proc.n_steps)
            e_hi = proc.approx_at_step(e_hi_step).epsilon_at(_DELTA)
            assert e_lo - 1e-10 <= e_K <= e_hi + 1e-10, (
                f"sandwich broken at K={K} (G={G}, r={r}): {e_lo} ≤ {e_K} ≤ {e_hi}"
            )

    def test_monotonic_across_bands(self):
        proc = self._proc(n_steps=100, bands=8)
        eps_seq = [proc.approx_at_step(k).epsilon_at(_DELTA) for k in range(0, 101, 8)]
        for a, b in zip(eps_seq, eps_seq[1:]):
            assert b >= a - 1e-10


# ---------------------------------------------------------------------------
# BMinSep(BandMf): MC-based; verify endpoints, rounding, type preservation.
# Monotonicity tested with a fixed seed to make the MC reproducible.
# ---------------------------------------------------------------------------


class TestBMinSep:
    def _proc(self, n_steps: int = 32, bands: int = 4) -> BMinSep:
        coeffs = tuple(1.0 / bands**0.5 for _ in range(bands))
        return ftrl_acc.b_min_sep(
            ftrl_acc.mf_gaussian(
                1.0, BandMfStrategy(sensitivity=1.0, coefficients=coeffs)
            ),
            n_steps=n_steps,
            p0=0.02,
        )

    def test_atomic_unit_equals_bands(self):
        assert self._proc(bands=4).atomic_unit == 4
        assert self._proc(bands=8).atomic_unit == 8

    def test_endpoints(self):
        proc = self._proc(n_steps=32)
        assert isinstance(proc.approx_at_step(0), Identity)
        assert proc.approx_at_step(32) is proc

    def test_at_step_rounds_up_to_band(self):
        proc = self._proc(n_steps=32, bands=4)
        # K=5 → ceil(5/4)*4 = 8
        assert proc.approx_at_step(5).n_steps == 8
        # K=4 → 4
        assert proc.approx_at_step(4).n_steps == 4

    def test_type_preserved(self):
        sub = self._proc(n_steps=32).approx_at_step(16)
        assert type(sub) is BMinSep
        assert isinstance(sub, DpFtrlProcess)

    def test_trend_increases(self):
        """Aggregate ε increases between K=bands and K=N.

        The b-min-sep MC re-samples an independent transcript for every
        ``n_steps``, so the per-step ε curve is noisy under any single
        seed.  We assert only the global trend (small K significantly
        less private than full).
        """
        proc = self._proc(n_steps=32, bands=4)
        e_full = proc.pld(**_MC_KW).epsilon_at(_DELTA)
        e_small = proc.approx_at_step(4).pld(**_MC_KW).epsilon_at(_DELTA)
        assert e_small < e_full, (
            f"expected ε(K=4) < ε(K=32) in expectation, got {e_small} ≮ {e_full}"
        )

    def test_endpoint_full_is_self(self):
        """``at_step(N)`` returns ``self`` so ε is exactly the full PLD."""
        proc = self._proc(n_steps=32)
        assert proc.approx_at_step(32) is proc


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
        assert self._proc(num_bins=10).atomic_unit == 10
        assert self._proc(num_bins=4).atomic_unit == 4

    def test_endpoints(self):
        proc = self._proc(num_bins=10, num_epochs=10)
        assert isinstance(proc.approx_at_step(0), Identity)
        assert proc.approx_at_step(100) is proc

    def test_at_step_rounds_up_to_epoch(self):
        proc = self._proc(num_bins=10, num_epochs=10)
        # K=15 → ceil(15/10)*10 = 20
        assert proc.approx_at_step(15).n_steps == 20
        # K=10 → 10
        assert proc.approx_at_step(10).n_steps == 10

    def test_type_preserved(self):
        sub = self._proc(num_bins=10, num_epochs=10).approx_at_step(50)
        assert type(sub) is BallsInBins
        assert isinstance(sub, DpFtrlProcess)

    def test_n_steps_remains_multiple_of_num_bins(self):
        """The constructor's ``n_steps % num_bins == 0`` invariant is preserved."""
        proc = self._proc(num_bins=10, num_epochs=10)
        for K in [1, 5, 10, 11, 50, 99, 100]:
            sub = proc.approx_at_step(K)
            if isinstance(sub, BallsInBins):
                assert sub.n_steps % sub.num_bins == 0

    def test_monotonic(self):
        proc = self._proc(num_bins=10, num_epochs=10)
        e_full = proc.epsilon_at(_DELTA)
        prev = 0.0
        for k in range(0, 101, 10):
            e = proc.approx_at_step(k).epsilon_at(_DELTA)
            assert e >= prev - 1e-10, f"non-monotone at k={k}: {e} < {prev}"
            assert e <= e_full + 1e-10
            prev = e


# ---------------------------------------------------------------------------
# BallsInBins(correlated MF): Gram regeneration oracle.
#
# ``inner.with_horizon`` is the load-bearing path: ``at_step(K)`` must match a
# strategy built directly at the shorter horizon ``K`` (up to MC variance).
# ---------------------------------------------------------------------------


_REGEN_NUM_BINS = 4
_REGEN_N_FULL = 16
_REGEN_K = 8
_REGEN_TOL = 0.05  # generous: bnb_mc_pld is MC-based.


def _bnb(mechanism, n_steps: int) -> BallsInBins:
    return ftrl_acc.balls_in_bins(mechanism, num_bins=_REGEN_NUM_BINS, n_steps=n_steps)


class TestGramRegenMatchesDirect:
    """``at_step(K)`` must match a strategy built directly at horizon ``K``.

    For each correlated MF, build BnB at full horizon and call ``at_step(K)``;
    independently build BnB at horizon ``K`` from a fresh strategy.  Both
    paths use the same ``bnb_mc_pld`` primitive on the same Gram structure,
    so ε agrees up to MC variance.
    """

    def test_blt(self):
        full = blt_strategy(
            n_steps=_REGEN_N_FULL, min_sep=_REGEN_NUM_BINS, max_participations=4
        )
        direct = blt_strategy(
            n_steps=_REGEN_K, min_sep=_REGEN_NUM_BINS, max_participations=2
        )
        e_at = _bnb(ftrl_acc.mf_gaussian(1.0, full), _REGEN_N_FULL).approx_at_step(_REGEN_K)
        e_dir = _bnb(ftrl_acc.mf_gaussian(1.0, direct), _REGEN_K)
        assert math.isclose(
            e_at.epsilon_at(_DELTA),
            e_dir.epsilon_at(_DELTA),
            rel_tol=_REGEN_TOL,
        )

    def test_bsr(self):
        full = bsr_strategy(
            bandwidth=2,
            n_steps=_REGEN_N_FULL,
            min_sep=_REGEN_NUM_BINS,
            max_participations=4,
            alpha=1.0,
            beta=0.5,
        )
        direct = bsr_strategy(
            bandwidth=2,
            n_steps=_REGEN_K,
            min_sep=_REGEN_NUM_BINS,
            max_participations=2,
            alpha=1.0,
            beta=0.5,
        )
        e_at = _bnb(ftrl_acc.mf_gaussian(1.0, full), _REGEN_N_FULL).approx_at_step(_REGEN_K)
        e_dir = _bnb(ftrl_acc.mf_gaussian(1.0, direct), _REGEN_K)
        assert math.isclose(
            e_at.epsilon_at(_DELTA),
            e_dir.epsilon_at(_DELTA),
            rel_tol=_REGEN_TOL,
        )

    def test_bisr(self):
        full = bisr_strategy(
            bandwidth=2,
            n_steps=_REGEN_N_FULL,
            min_sep=_REGEN_NUM_BINS,
            max_participations=4,
        )
        direct = bisr_strategy(
            bandwidth=2,
            n_steps=_REGEN_K,
            min_sep=_REGEN_NUM_BINS,
            max_participations=2,
        )
        e_at = _bnb(ftrl_acc.mf_gaussian(1.0, full), _REGEN_N_FULL).approx_at_step(_REGEN_K)
        e_dir = _bnb(ftrl_acc.mf_gaussian(1.0, direct), _REGEN_K)
        assert math.isclose(
            e_at.epsilon_at(_DELTA),
            e_dir.epsilon_at(_DELTA),
            rel_tol=_REGEN_TOL,
        )

    def test_lambda_cgd(self):
        full = lambda_cgd_strategy(
            0.5,
            n_steps=_REGEN_N_FULL,
            min_sep=_REGEN_NUM_BINS,
            max_participations=4,
        )
        direct = lambda_cgd_strategy(
            0.5,
            n_steps=_REGEN_K,
            min_sep=_REGEN_NUM_BINS,
            max_participations=2,
        )
        e_at = _bnb(ftrl_acc.mf_gaussian(1.0, full), _REGEN_N_FULL).approx_at_step(_REGEN_K)
        e_dir = _bnb(ftrl_acc.mf_gaussian(1.0, direct), _REGEN_K)
        assert math.isclose(
            e_at.epsilon_at(_DELTA),
            e_dir.epsilon_at(_DELTA),
            rel_tol=_REGEN_TOL,
        )


# ---------------------------------------------------------------------------
# Serialization registry hardening: abstract bases must NOT be registered.
# ---------------------------------------------------------------------------


class TestSerializationRegistryHardening:
    """Abstract intermediates (``DpFtrlProcess``, etc.) are not dataclasses
    and have no fields to serialize; they must not pollute the registry."""

    def test_abstract_bases_not_registered(self):
        from opaque.api.accounting.core._base import _PROCESS_REGISTRY

        assert "DpFtrlProcess" not in _PROCESS_REGISTRY
        # Concrete classes still register normally.
        assert "CyclicPoisson" in _PROCESS_REGISTRY
        assert "BallsInBins" in _PROCESS_REGISTRY


# ---------------------------------------------------------------------------
# Composition: result of approx_at_step is a full DpProcess, supports |/* operators.
# ---------------------------------------------------------------------------


class TestCompositionOnTruncated:
    def test_truncated_supports_or_composition(self):
        proc = ftrl_acc.poisson(
            ftrl_acc.mf_gaussian(1.0, identity_strategy()), sample_rate=0.01, n_steps=100
        )
        a = proc.approx_at_step(30)
        b = proc.approx_at_step(40)
        composed = a | b
        # Sanity: composing two truncated processes yields a finite ε.
        assert math.isfinite(composed.epsilon_at(_DELTA))

    def test_truncated_supports_self_compose(self):
        proc = ftrl_acc.poisson(
            ftrl_acc.mf_gaussian(1.0, identity_strategy()), sample_rate=0.01, n_steps=100
        )
        sub = proc.approx_at_step(50)
        repeated = sub * 2
        assert math.isfinite(repeated.epsilon_at(_DELTA))


# ---------------------------------------------------------------------------
# atomic_unit validation.
# ---------------------------------------------------------------------------


class TestAtomicUnitValidation:
    def test_zero_atomic_unit_raises_in_at_step(self):
        """Sanity: an implementation returning a 0/negative atomic_unit fails."""
        from opaque.api.accounting.dpftrl._base import DpFtrlProcess
        from dataclasses import dataclass

        # Construct a minimal subclass that intentionally violates the contract.
        @dataclass(frozen=True)
        class _Broken(DpFtrlProcess):
            n_steps: int = 100

            @property
            def atomic_unit(self) -> int:
                return 0

            def pld(self, **kw):  # pragma: no cover - not exercised
                raise NotImplementedError

        with pytest.raises(ValueError, match="atomic_unit"):
            _Broken().approx_at_step(5)


# ---------------------------------------------------------------------------
# Cross-product of (amplification, inner mechanism) supported by ``at_step``.
#
# DP-FTRL accounting is defined by combining an amplification factory with
# an inner mechanism and dispatching by ``match self.inner`` inside the
# amplification.  These tests mirror that structure: every (amp, inner)
# pair that ``at_step`` should support is exercised against the same
# invariants.  Illegal combos (inner not in amp's pairing matrix) are the
# amp factory's responsibility and are covered by their own tests — not
# here.
#
# To add a new supported pair: register the amp / inner factory below and
# append the tuple to ``_SUPPORTED_PAIRS``.
# ---------------------------------------------------------------------------


# Amplification name → (factory(inner) -> DpFtrlProcess, is_monte_carlo).
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


# Inner mechanism name → factory() -> mechanism dataclass.
_MECHANISMS: dict[str, Callable[[], object]] = {
    "IdentityMf": lambda: ftrl_acc.mf_gaussian(1.0, identity_strategy()),
    "BandMf": lambda: ftrl_acc.mf_gaussian(
        1.0, BandMfStrategy(sensitivity=2.0, coefficients=(1.0, 1.0, 1.0, 1.0))
    ),
    "Blt": lambda: ftrl_acc.mf_gaussian(
        1.0,
        blt_strategy(n_steps=16, min_sep=4, max_participations=4),
    ),
    "LambdaCgd": lambda: ftrl_acc.mf_gaussian(
        1.0,
        lambda_cgd_strategy(0.5, n_steps=16, min_sep=4, max_participations=4),
    ),
    "Bisr": lambda: ftrl_acc.mf_gaussian(
        1.0,
        bisr_strategy(bandwidth=2, n_steps=16, min_sep=4, max_participations=4),
    ),
    "Bsr": lambda: ftrl_acc.mf_gaussian(
        1.0,
        bsr_strategy(
            bandwidth=2,
            n_steps=16,
            min_sep=4,
            max_participations=4,
            alpha=1.0,
            beta=0.5,
        ),
    ),
}


# Pairs where ``at_step`` returns a usable truncated process for any K.
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


def _pld_kwargs(amp: str) -> dict:
    return _MC_KW if _AMPLIFICATIONS[amp][1] else {}


def _eps(process, delta: float, amp: str) -> float:
    if isinstance(process, Identity):
        return process.epsilon_at(delta)
    return process.pld(**_pld_kwargs(amp)).epsilon_at(delta)


def _pair_id(pair: tuple[str, str]) -> str:
    return f"{pair[0]}({pair[1]})"


_SUPPORTED_IDS = [_pair_id(p) for p in _SUPPORTED_PAIRS]


@pytest.mark.parametrize("amp,mech", _SUPPORTED_PAIRS, ids=_SUPPORTED_IDS)
class TestAtStepInvariants:
    """For every supported (amp, inner) pair, ``at_step`` satisfies the
    documented contract: endpoint identities, monotone non-decreasing,
    bounded by ε of the full process, sandwich at atomic-unit boundaries,
    and the result is a real :class:`DpProcess` (composes with ``|``, ``*``).

    Monte-Carlo paths (``BMinSep``) skip the per-K sandwich because
    independent transcripts per ``n_steps`` mean the bound holds only in
    expectation; the trend assertion still covers the global direction.
    """

    def test_inherits_dp_ftrl_process(self, amp: str, mech: str):
        proc = _build(amp, mech)
        assert isinstance(proc, DpFtrlProcess)

    def test_zero_step_returns_identity(self, amp: str, mech: str):
        sub = _build(amp, mech).approx_at_step(0)
        assert isinstance(sub, Identity)
        assert sub.epsilon_at(_DELTA) == 0.0

    def test_full_step_returns_self(self, amp: str, mech: str):
        proc = _build(amp, mech)
        assert proc.approx_at_step(proc.n_steps) is proc
        assert proc.approx_at_step(proc.n_steps + 10_000) is proc

    def test_intermediate_type_preserved(self, amp: str, mech: str):
        proc = _build(amp, mech)
        sub = proc.approx_at_step(proc.n_steps // 2)
        assert type(sub) is type(proc)
        assert isinstance(sub, DpFtrlProcess)

    def test_n_steps_aligned_to_atomic_unit(self, amp: str, mech: str):
        proc = _build(amp, mech)
        for K in (1, max(1, proc.atomic_unit - 1), proc.n_steps // 2):
            sub = proc.approx_at_step(K)
            if isinstance(sub, Identity):
                continue
            assert sub.n_steps % proc.atomic_unit == 0 or sub.n_steps == proc.n_steps
            assert K <= sub.n_steps <= proc.n_steps

    def test_bounded_by_full(self, amp: str, mech: str):
        proc = _build(amp, mech)
        e_full = _eps(proc, _DELTA, amp)
        slack = 0.10 * max(e_full, 1.0) if _AMPLIFICATIONS[amp][1] else 1e-9
        for K in (1, proc.atomic_unit, proc.n_steps // 2, proc.n_steps - 1):
            assert _eps(proc.approx_at_step(K), _DELTA, amp) <= e_full + slack

    def test_trend_increases(self, amp: str, mech: str):
        proc = _build(amp, mech)
        e_small = _eps(proc.approx_at_step(proc.atomic_unit), _DELTA, amp)
        e_full = _eps(proc, _DELTA, amp)
        assert e_small < e_full

    def test_monotone_at_unit_boundaries(self, amp: str, mech: str):
        proc = _build(amp, mech)
        n, M = proc.n_steps, proc.atomic_unit
        e_full = _eps(proc, _DELTA, amp)
        slack = 0.10 * max(e_full, 1.0) if _AMPLIFICATIONS[amp][1] else 1e-9
        prev = 0.0
        steps = list(range(0, n + 1, M))
        if steps[-1] != n:
            steps.append(n)
        for K in steps:
            sub = Identity() if K == 0 else proc if K >= n else proc.approx_at_step(K)
            e_K = _eps(sub, _DELTA, amp)
            assert e_K >= prev - slack
            prev = e_K

    def test_sandwich_at_intermediate_K(self, amp: str, mech: str):
        if _AMPLIFICATIONS[amp][1]:
            pytest.skip("MC path: sandwich holds in expectation only")
        proc = _build(amp, mech)
        n, M = proc.n_steps, proc.atomic_unit
        K = max(1, min(n - 1, n // 2 + M // 2))
        G, _r = divmod(K, M)
        e_lo = _eps(proc.approx_at_step(G * M) if G > 0 else Identity(), _DELTA, amp)
        e_K = _eps(proc.approx_at_step(K), _DELTA, amp)
        e_hi = _eps(proc.approx_at_step(min((G + 1) * M, n)), _DELTA, amp)
        assert e_lo - 1e-9 <= e_K <= e_hi + 1e-9

    def test_supports_composition(self, amp: str, mech: str):
        proc = _build(amp, mech)
        sub = proc.approx_at_step(proc.n_steps // 2)
        composed = sub | proc.approx_at_step(proc.atomic_unit)
        if not isinstance(composed, Identity):
            assert math.isfinite(composed.epsilon_at(_DELTA))
        assert math.isfinite((sub * 2).epsilon_at(_DELTA))
