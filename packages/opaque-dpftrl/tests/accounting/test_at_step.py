"""Contract tests for :meth:`DpFtrlProcess.at_step` and :func:`at_step`.

The contract is documented on :class:`DpFtrlProcess`:

- ``ε(at_step(0))         == 0`` (returns Identity).
- ``ε(at_step(N))         == self.epsilon_at(δ)`` (returns self).
- ``K1 ≤ K2 ⇒ ε(at_step(K1)) ≤ ε(at_step(K2))`` (monotone).
- ``ε(at_step(G·M)) ≤ ε(at_step(K)) ≤ ε(at_step((G+1)·M))`` (sandwich).
- ``ε(at_step(K)) ≤ ε(self)`` for ``K ≤ N``.

Tests cover all three amplifications (CyclicPoisson, BMinSep, BallsInBins),
both the method and the free-function form, type preservation, and the
documented error path for BallsInBins with correlated-MF inners.
"""

from __future__ import annotations

import math

import pytest

import opaque.dpftrl.accounting as ftrl_acc
from opaque.api.accounting.core.mechanisms.types import Identity
from opaque.dpftrl.accounting.types import (
    BallsInBins,
    BMinSep,
    CyclicPoisson,
    DpFtrlProcess,
)

_DELTA = 1e-5
_MC_KW = {"num_mc_samples": 4000, "seed": 17}


# ---------------------------------------------------------------------------
# CyclicPoisson(IdentityMf): per-step granularity, exact match to underlying
# subsampled-Gaussian composition.
# ---------------------------------------------------------------------------


class TestCyclicPoissonIdentity:
    def _proc(self, n_steps: int = 200) -> CyclicPoisson:
        return ftrl_acc.poisson(
            ftrl_acc.identity_mf(1.0),
            sample_rate=0.01,
            n_steps=n_steps,
        )

    def test_atomic_unit_is_one(self):
        assert self._proc().atomic_unit == 1

    def test_zero_step_returns_identity(self):
        sub = self._proc().at_step(0)
        assert isinstance(sub, Identity)
        assert sub.epsilon_at(_DELTA) == 0.0

    def test_negative_step_returns_identity(self):
        assert isinstance(self._proc().at_step(-5), Identity)

    def test_full_step_returns_self(self):
        proc = self._proc(100)
        assert proc.at_step(100) is proc

    def test_overshoot_step_returns_self(self):
        proc = self._proc(100)
        assert proc.at_step(10_000) is proc

    def test_type_preserved(self):
        sub = self._proc().at_step(50)
        assert type(sub) is CyclicPoisson
        assert isinstance(sub, DpFtrlProcess)

    def test_n_steps_replaced(self):
        proc = self._proc(200)
        assert proc.at_step(73).n_steps == 73  # M=1, no rounding

    def test_exact_match_to_explicit_construction(self):
        """``proc.at_step(K)`` ≡ same factory called with ``n_steps=K``."""
        K = 137
        full = self._proc(200)
        partial = full.at_step(K)
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
            e = proc.at_step(k).epsilon_at(_DELTA)
            assert e >= prev - 1e-10, f"non-monotone at k={k}: {e} < {prev}"
            prev = e

    def test_bounded_by_full(self):
        proc = self._proc(200)
        e_full = proc.epsilon_at(_DELTA)
        for k in [1, 50, 100, 199]:
            assert proc.at_step(k).epsilon_at(_DELTA) <= e_full + 1e-10

    def test_truncated_poisson_works(self):
        """at_step preserves the truncated-Poisson kwarg pair."""
        proc = ftrl_acc.poisson(
            ftrl_acc.identity_mf(1.0),
            sample_rate=0.01,
            n_steps=200,
            truncated_batch_size=50,
            dataset_size=5000,
        )
        sub = proc.at_step(75)
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
            ftrl_acc.band_mf(1.0, sensitivity=float(bands) ** 0.5, coefficients=coeffs),
            sample_rate=0.01,
            n_steps=n_steps,
        )

    def test_atomic_unit_equals_bands(self):
        assert self._proc(bands=8).atomic_unit == 8
        assert self._proc(bands=4).atomic_unit == 4

    def test_at_step_rounds_up_to_band(self):
        proc = self._proc(n_steps=100, bands=8)
        # K=7 → ceil(7/8)*8 = 8
        assert proc.at_step(7).n_steps == 8
        # K=8 → 8
        assert proc.at_step(8).n_steps == 8
        # K=9 → 16
        assert proc.at_step(9).n_steps == 16

    def test_at_step_clamps_at_n_steps(self):
        proc = self._proc(n_steps=100, bands=8)
        # 100 is not a multiple of 8 (12*8=96, 13*8=104).
        # at_step(99) would round to 104 but is clamped to 100.
        assert proc.at_step(99).n_steps == 100

    def test_endpoints(self):
        proc = self._proc(n_steps=64, bands=8)
        assert proc.at_step(0).epsilon_at(_DELTA) == 0.0
        assert proc.at_step(64) is proc

    def test_sandwich_at_every_step(self):
        """For K = G·M + r: ε(G·M) ≤ ε(K) ≤ ε((G+1)·M) for r ∈ [0, M)."""
        proc = self._proc(n_steps=80, bands=8)
        M = proc.atomic_unit
        for K in range(1, proc.n_steps + 1):
            G, r = divmod(K, M)
            e_lo = (
                proc.at_step(G * M).epsilon_at(_DELTA) if G > 0 else 0.0
            )
            e_K = proc.at_step(K).epsilon_at(_DELTA)
            e_hi_step = min((G + 1) * M, proc.n_steps)
            e_hi = proc.at_step(e_hi_step).epsilon_at(_DELTA)
            assert e_lo - 1e-10 <= e_K <= e_hi + 1e-10, (
                f"sandwich broken at K={K} (G={G}, r={r}): "
                f"{e_lo} ≤ {e_K} ≤ {e_hi}"
            )

    def test_monotonic_across_bands(self):
        proc = self._proc(n_steps=100, bands=8)
        eps_seq = [proc.at_step(k).epsilon_at(_DELTA) for k in range(0, 101, 8)]
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
            ftrl_acc.band_mf(1.0, sensitivity=1.0, coefficients=coeffs),
            n_steps=n_steps,
            p0=0.02,
        )

    def test_atomic_unit_equals_bands(self):
        assert self._proc(bands=4).atomic_unit == 4
        assert self._proc(bands=8).atomic_unit == 8

    def test_endpoints(self):
        proc = self._proc(n_steps=32)
        assert isinstance(proc.at_step(0), Identity)
        assert proc.at_step(32) is proc

    def test_at_step_rounds_up_to_band(self):
        proc = self._proc(n_steps=32, bands=4)
        # K=5 → ceil(5/4)*4 = 8
        assert proc.at_step(5).n_steps == 8
        # K=4 → 4
        assert proc.at_step(4).n_steps == 4

    def test_type_preserved(self):
        sub = self._proc(n_steps=32).at_step(16)
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
        e_small = proc.at_step(4).pld(**_MC_KW).epsilon_at(_DELTA)
        assert e_small < e_full, (
            f"expected ε(K=4) < ε(K=32) in expectation, got {e_small} ≮ {e_full}"
        )

    def test_endpoint_full_is_self(self):
        """``at_step(N)`` returns ``self`` so ε is exactly the full PLD."""
        proc = self._proc(n_steps=32)
        assert proc.at_step(32) is proc


# ---------------------------------------------------------------------------
# BallsInBins(IdentityMf): epoch-quantised granularity, Gram-free.
# ---------------------------------------------------------------------------


class TestBallsInBinsIdentity:
    def _proc(
        self, num_bins: int = 10, num_epochs: int = 10
    ) -> BallsInBins:
        return ftrl_acc.balls_in_bins(
            ftrl_acc.identity_mf(1.0),
            num_bins=num_bins,
            n_steps=num_bins * num_epochs,
        )

    def test_atomic_unit_equals_num_bins(self):
        assert self._proc(num_bins=10).atomic_unit == 10
        assert self._proc(num_bins=4).atomic_unit == 4

    def test_endpoints(self):
        proc = self._proc(num_bins=10, num_epochs=10)
        assert isinstance(proc.at_step(0), Identity)
        assert proc.at_step(100) is proc

    def test_at_step_rounds_up_to_epoch(self):
        proc = self._proc(num_bins=10, num_epochs=10)
        # K=15 → ceil(15/10)*10 = 20
        assert proc.at_step(15).n_steps == 20
        # K=10 → 10
        assert proc.at_step(10).n_steps == 10

    def test_type_preserved(self):
        sub = self._proc(num_bins=10, num_epochs=10).at_step(50)
        assert type(sub) is BallsInBins
        assert isinstance(sub, DpFtrlProcess)

    def test_n_steps_remains_multiple_of_num_bins(self):
        """The constructor's ``n_steps % num_bins == 0`` invariant is preserved."""
        proc = self._proc(num_bins=10, num_epochs=10)
        for K in [1, 5, 10, 11, 50, 99, 100]:
            sub = proc.at_step(K)
            if isinstance(sub, BallsInBins):
                assert sub.n_steps % sub.num_bins == 0

    def test_monotonic(self):
        proc = self._proc(num_bins=10, num_epochs=10)
        e_full = proc.epsilon_at(_DELTA)
        prev = 0.0
        for k in range(0, 101, 10):
            e = proc.at_step(k).epsilon_at(_DELTA)
            assert e >= prev - 1e-10, f"non-monotone at k={k}: {e} < {prev}"
            assert e <= e_full + 1e-10
            prev = e


# ---------------------------------------------------------------------------
# BallsInBins(correlated MF): NotImplementedError until Gram regen lands.
# ---------------------------------------------------------------------------


class TestBallsInBinsCorrelatedNotSupported:
    @pytest.fixture
    def proc(self) -> BallsInBins:
        gram = (1.0,) * (10 * 10)
        return ftrl_acc.balls_in_bins(
            ftrl_acc.blt(1.0, sensitivity=1.0, gram_matrix=gram),
            num_bins=10,
            n_steps=100,
        )

    def test_endpoints_still_work(self, proc: BallsInBins):
        # K=0 returns Identity (no Gram needed); K=N returns self.
        assert isinstance(proc.at_step(0), Identity)
        assert proc.at_step(100) is proc

    def test_partial_step_raises(self, proc: BallsInBins):
        with pytest.raises(NotImplementedError, match="gram_matrix"):
            proc.at_step(50)

    def test_message_mentions_inner_class(self, proc: BallsInBins):
        with pytest.raises(NotImplementedError, match="Blt"):
            proc.at_step(50)

    def test_other_correlated_inners_raise_too(self):
        gram = (1.0,) * (4 * 4)
        for factory_name in ("bsr", "bisr", "lambda_cgd"):
            inner = getattr(ftrl_acc, factory_name)(
                1.0, sensitivity=1.0, gram_matrix=gram
            )
            proc = ftrl_acc.balls_in_bins(inner, num_bins=4, n_steps=16)
            with pytest.raises(NotImplementedError):
                proc.at_step(8)


# ---------------------------------------------------------------------------
# Free-function form parity, error paths.
# ---------------------------------------------------------------------------


class TestFreeFunction:
    def test_at_step_function_matches_method(self):
        proc = ftrl_acc.poisson(
            ftrl_acc.identity_mf(1.0), sample_rate=0.01, n_steps=100
        )
        for k in [0, 25, 50, 99, 100]:
            assert ftrl_acc.at_step(proc, k).epsilon_at(_DELTA) == pytest.approx(
                proc.at_step(k).epsilon_at(_DELTA)
            )

    def test_at_step_rejects_non_dpftrl_process(self):
        from opaque.api.accounting.core.mechanisms.types import Identity

        # Identity is a DpProcess but not a DpFtrlProcess.
        with pytest.raises(TypeError, match="DpFtrlProcess"):
            ftrl_acc.at_step(Identity(), 5)


# ---------------------------------------------------------------------------
# Composition: result of at_step is a full DpProcess, supports |/* operators.
# ---------------------------------------------------------------------------


class TestCompositionOnTruncated:
    def test_truncated_supports_or_composition(self):
        proc = ftrl_acc.poisson(
            ftrl_acc.identity_mf(1.0), sample_rate=0.01, n_steps=100
        )
        a = proc.at_step(30)
        b = proc.at_step(40)
        composed = a | b
        # Sanity: composing two truncated processes yields a finite ε.
        assert math.isfinite(composed.epsilon_at(_DELTA))

    def test_truncated_supports_self_compose(self):
        proc = ftrl_acc.poisson(
            ftrl_acc.identity_mf(1.0), sample_rate=0.01, n_steps=100
        )
        sub = proc.at_step(50)
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
            _Broken().at_step(5)


# ---------------------------------------------------------------------------
# Cross-product compliance: parametrise the contract over every supported
# (amplification, mechanism) combination so adding a new pair is one line.
#
# Positive paths (at_step returns a DpFtrlProcess / Identity):
#   CyclicPoisson(IdentityMf)        — band=1, exact per-step
#   CyclicPoisson(BandMf)            — band=4
#   BMinSep(BandMf)                  — Monte Carlo
#   BallsInBins(IdentityMf)          — epoch=10
#
# Negative paths (at_step raises NotImplementedError):
#   BallsInBins(Blt | LambdaCgd | Bisr | Bsr) — Gram-sized to original n_steps.
# ---------------------------------------------------------------------------


# (label, factory, expected_amp_cls, expected_atomic_unit, n_steps, is_monte_carlo)
_POSITIVE_CASES: list[
    tuple[str, callable, type[DpFtrlProcess], int, int, bool]
] = [
    (
        "CyclicPoisson(IdentityMf)",
        lambda: ftrl_acc.poisson(
            ftrl_acc.identity_mf(1.0), sample_rate=0.01, n_steps=80
        ),
        CyclicPoisson,
        1,
        80,
        False,
    ),
    (
        "CyclicPoisson(BandMf)",
        lambda: ftrl_acc.poisson(
            ftrl_acc.band_mf(
                1.0, sensitivity=2.0, coefficients=(1.0, 1.0, 1.0, 1.0)
            ),
            sample_rate=0.01,
            n_steps=80,
        ),
        CyclicPoisson,
        4,
        80,
        False,
    ),
    (
        "BMinSep(BandMf)",
        lambda: ftrl_acc.b_min_sep(
            ftrl_acc.band_mf(
                1.0, sensitivity=2.0, coefficients=(1.0, 1.0, 1.0, 1.0)
            ),
            n_steps=32,
            p0=0.02,
        ),
        BMinSep,
        4,
        32,
        True,
    ),
    (
        "BallsInBins(IdentityMf)",
        lambda: ftrl_acc.balls_in_bins(
            ftrl_acc.identity_mf(1.0), num_bins=10, n_steps=100
        ),
        BallsInBins,
        10,
        100,
        False,
    ),
]


def _pld_kwargs(is_mc: bool) -> dict:
    return _MC_KW if is_mc else {}


def _eps(process, delta: float, is_mc: bool) -> float:
    """Compute ε with the right kwargs (MC paths need samples + seed)."""
    if isinstance(process, Identity):
        return process.epsilon_at(delta)
    return process.pld(**_pld_kwargs(is_mc)).epsilon_at(delta)


@pytest.mark.parametrize(
    "label,factory,amp_cls,expected_unit,n_steps,is_mc",
    _POSITIVE_CASES,
    ids=[case[0] for case in _POSITIVE_CASES],
)
class TestContractAcrossAllSupportedPairs:
    """Every supported (amp, mech) pair satisfies the at_step contract."""

    def test_inherits_dp_ftrl_process(
        self, label, factory, amp_cls, expected_unit, n_steps, is_mc
    ):
        proc = factory()
        assert isinstance(proc, DpFtrlProcess), f"{label} is not a DpFtrlProcess"
        assert isinstance(proc, amp_cls), f"{label}: expected {amp_cls.__name__}"

    def test_atomic_unit_value(
        self, label, factory, amp_cls, expected_unit, n_steps, is_mc
    ):
        assert factory().atomic_unit == expected_unit, label

    def test_zero_step_returns_identity(
        self, label, factory, amp_cls, expected_unit, n_steps, is_mc
    ):
        sub = factory().at_step(0)
        assert isinstance(sub, Identity), f"{label}: at_step(0) is not Identity"
        assert sub.epsilon_at(_DELTA) == 0.0

    def test_full_step_returns_self(
        self, label, factory, amp_cls, expected_unit, n_steps, is_mc
    ):
        proc = factory()
        assert proc.at_step(n_steps) is proc, f"{label}: at_step(N) ≠ self"
        # And one step past N also returns self.
        assert proc.at_step(n_steps + 1_000) is proc

    def test_intermediate_step_preserves_type(
        self, label, factory, amp_cls, expected_unit, n_steps, is_mc
    ):
        proc = factory()
        K = n_steps // 2
        sub = proc.at_step(K)
        assert type(sub) is amp_cls, (
            f"{label}: at_step(N/2) returned {type(sub).__name__}, expected "
            f"{amp_cls.__name__}"
        )
        assert isinstance(sub, DpFtrlProcess)

    def test_n_steps_respects_atomic_unit(
        self, label, factory, amp_cls, expected_unit, n_steps, is_mc
    ):
        proc = factory()
        for K in (1, expected_unit - 1 if expected_unit > 1 else 1, n_steps // 2):
            sub = proc.at_step(K)
            if isinstance(sub, Identity):
                continue
            # n_steps on the rebuilt process is on an atomic-unit boundary,
            # and ≥ K (round-up), and ≤ original n_steps (clamp).
            assert sub.n_steps % expected_unit == 0 or sub.n_steps == n_steps, (
                f"{label}: at_step({K}).n_steps={sub.n_steps} not aligned"
            )
            assert sub.n_steps >= K, label
            assert sub.n_steps <= n_steps, label

    def test_bounded_by_full(
        self, label, factory, amp_cls, expected_unit, n_steps, is_mc
    ):
        proc = factory()
        e_full = _eps(proc, _DELTA, is_mc)
        slack = 0.10 * max(e_full, 1.0) if is_mc else 1e-9
        for K in (1, expected_unit, n_steps // 2, n_steps - 1):
            sub = proc.at_step(K)
            e_K = _eps(sub, _DELTA, is_mc)
            assert e_K <= e_full + slack, (
                f"{label}: ε(K={K})={e_K} exceeds ε(N={n_steps})={e_full} "
                f"(slack {slack})"
            )

    def test_trend_increases_from_small_to_full(
        self, label, factory, amp_cls, expected_unit, n_steps, is_mc
    ):
        proc = factory()
        e_small = _eps(proc.at_step(expected_unit), _DELTA, is_mc)
        e_full = _eps(proc, _DELTA, is_mc)
        assert e_small < e_full, (
            f"{label}: ε at first atomic unit ({e_small}) "
            f"≮ ε at full ({e_full})"
        )

    def test_monotone_at_atomic_unit_boundaries(
        self, label, factory, amp_cls, expected_unit, n_steps, is_mc
    ):
        """At G·M boundaries the curve is monotone non-decreasing.

        For Monte-Carlo paths a single realisation can fluctuate; we
        allow a per-step slack proportional to ε_full.
        """
        proc = factory()
        e_full = _eps(proc, _DELTA, is_mc)
        slack = 0.10 * max(e_full, 1.0) if is_mc else 1e-9
        steps = list(range(0, n_steps + 1, expected_unit))
        if steps[-1] != n_steps:
            steps.append(n_steps)
        prev = 0.0
        for K in steps:
            sub = proc.at_step(K) if 0 < K < n_steps else (
                Identity() if K == 0 else proc
            )
            e_K = _eps(sub, _DELTA, is_mc)
            assert e_K >= prev - slack, (
                f"{label}: ε({K})={e_K} < ε(prev)={prev} (slack {slack})"
            )
            prev = e_K

    def test_sandwich_at_random_K(
        self, label, factory, amp_cls, expected_unit, n_steps, is_mc
    ):
        """ε(G·M) ≤ ε(K) ≤ ε((G+1)·M) for K = G·M + r, r ∈ [0, M).

        Skipped for Monte-Carlo paths where independent transcripts per
        n_steps break the per-K bound (verified in expectation only).
        """
        if is_mc:
            pytest.skip("MC path: bound holds in expectation, not per-sample")
        proc = factory()
        # Pick a K mid-way through the process.
        K = max(1, min(n_steps - 1, n_steps // 2 + expected_unit // 2))
        G, r = divmod(K, expected_unit)
        e_lo = _eps(
            proc.at_step(G * expected_unit) if G > 0 else Identity(),
            _DELTA,
            is_mc,
        )
        e_K = _eps(proc.at_step(K), _DELTA, is_mc)
        hi_step = min((G + 1) * expected_unit, n_steps)
        e_hi = _eps(proc.at_step(hi_step), _DELTA, is_mc)
        assert e_lo - 1e-9 <= e_K <= e_hi + 1e-9, (
            f"{label}: sandwich broken at K={K} (G={G},r={r}): "
            f"{e_lo} ≤ {e_K} ≤ {e_hi}"
        )

    def test_supports_or_composition(
        self, label, factory, amp_cls, expected_unit, n_steps, is_mc
    ):
        """Result is a real DpProcess: composes with ``|`` and ``*``.

        We only check that composition constructs and evaluates — the
        composed PLD uses default discretization (MC paths fall back to
        their default sample count, which is fine for a structural check).
        """
        proc = factory()
        sub = proc.at_step(n_steps // 2)
        composed = sub | proc.at_step(expected_unit)
        if isinstance(composed, Identity):
            return
        e = composed.epsilon_at(_DELTA)
        assert math.isfinite(e), f"{label}: composed.epsilon_at = {e}"
        # And ``*`` also works.
        repeated = sub * 2
        assert math.isfinite(repeated.epsilon_at(_DELTA)), label


# ---------------------------------------------------------------------------
# Negative parametrised path: every correlated-MF inner under BallsInBins
# raises NotImplementedError at intermediate K but still handles endpoints.
# ---------------------------------------------------------------------------


_BNB_CORRELATED_FACTORIES = [
    ("Blt", lambda gram: ftrl_acc.blt(1.0, sensitivity=1.0, gram_matrix=gram)),
    ("Bsr", lambda gram: ftrl_acc.bsr(1.0, sensitivity=1.0, gram_matrix=gram)),
    ("Bisr", lambda gram: ftrl_acc.bisr(1.0, sensitivity=1.0, gram_matrix=gram)),
    (
        "LambdaCgd",
        lambda gram: ftrl_acc.lambda_cgd(1.0, sensitivity=1.0, gram_matrix=gram),
    ),
]


@pytest.mark.parametrize(
    "mech_label,mech_factory",
    _BNB_CORRELATED_FACTORIES,
    ids=[m[0] for m in _BNB_CORRELATED_FACTORIES],
)
class TestBallsInBinsCorrelatedRaisesUniformly:
    """Every correlated-MF inner under BallsInBins handles the boundary cases
    and raises NotImplementedError for intermediate K (until Gram regen lands).
    """

    _NUM_BINS = 4
    _N_STEPS = 16
    _GRAM = (1.0,) * (_NUM_BINS * _NUM_BINS)

    def _proc(self, mech_factory) -> BallsInBins:
        return ftrl_acc.balls_in_bins(
            mech_factory(self._GRAM),
            num_bins=self._NUM_BINS,
            n_steps=self._N_STEPS,
        )

    def test_inherits_dp_ftrl_process(self, mech_label, mech_factory):
        proc = self._proc(mech_factory)
        assert isinstance(proc, DpFtrlProcess)
        assert proc.atomic_unit == self._NUM_BINS

    def test_endpoint_zero_returns_identity(self, mech_label, mech_factory):
        assert isinstance(self._proc(mech_factory).at_step(0), Identity)

    def test_endpoint_full_returns_self(self, mech_label, mech_factory):
        proc = self._proc(mech_factory)
        assert proc.at_step(self._N_STEPS) is proc

    def test_intermediate_step_raises(self, mech_label, mech_factory):
        proc = self._proc(mech_factory)
        with pytest.raises(NotImplementedError, match="gram_matrix"):
            proc.at_step(self._N_STEPS // 2)

    def test_error_message_names_inner_class(self, mech_label, mech_factory):
        proc = self._proc(mech_factory)
        with pytest.raises(NotImplementedError, match=mech_label):
            proc.at_step(self._N_STEPS // 2)
