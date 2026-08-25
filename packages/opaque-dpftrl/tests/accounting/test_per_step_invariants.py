"""K-prefix invariants for :meth:`DpHorizonProcess.pld_at`.

The K-prefix bound is the deployed-and-stopped-early mechanism: the
K-step PLD is evaluated using the N-tuned strategy coefficients and the
post-processing inequality on the K-prefix projection of the N-step
output stream.  This file asserts the invariants that flow from that
construction:

- ``ε(pld_at(0))         == 0`` (empty accountant).
- ``ε(pld_at(N))         == proc.epsilon_at(δ)``.
- ``K1 ≤ K2 ⇒ ε(K1) ≤ ε(K2)`` (monotone).
- ``ε(K) ≤ ε(self)`` for ``K ≤ N`` (bounded by full).
- For K = G · atomic_unit + r with r ∈ [0, atomic_unit):
  ``ε(G·M) ≤ ε(K) ≤ ε((G+1)·M)`` (sandwich; closed-form paths only).

Implemented as ``(per_step(proc) * K).epsilon_at(δ)`` through the generic
horizon adapter.
"""

from __future__ import annotations

import itertools
import math
from typing import TYPE_CHECKING

import pytest

import opaque.accounting as acc
import opaque.dpftrl.accounting as ftrl_acc
from opaque.api.accounting.core._horizon import DpHorizonProcess
from opaque.api.accounting.core.mechanisms.types import Identity
from opaque.api.dpftrl.noise._identity import IdentityStrategy
from opaque.dpftrl.accounting.types import (
    BallsInBins,
    BMinSep,
    CyclicPoisson,
)
from opaque.dpftrl.noise import (
    band_mf_strategy,
    bisr_strategy,
    blt_strategy,
    bsr_strategy,
    identity_strategy,
    lambda_cgd_strategy,
)

if TYPE_CHECKING:
    from collections.abc import Callable

_DELTA = 1e-5
_MC_DELTA = 1e-2
_MC_KW = {
    "seed": 17,
    "mc_resolution": 5e-3,
    "mc_failure_probability": 1e-2,
}


def _atomic_unit(proc: DpHorizonProcess) -> int:
    """Diagnostic — the K-rounding granularity for this amplifier."""
    return int(proc.atomic_unit)


def _eps_at(proc: DpHorizonProcess, K: int, delta: float) -> float:
    """K-step ε via ``per_step(proc) * K`` — the public API surface.

    MC kwargs could be passed per call; these tests pin them globally with the
    :func:`_seed_mc` fixture so every call site resolves the same config.
    """
    if K <= 0:
        return Identity().epsilon_at(delta)
    step = acc.per_step(proc)
    return (step * K).epsilon_at(delta)


@pytest.fixture
def _seed_mc():
    """Pin global MC discretization for MC-based amplifiers.

    ``Repeated.pld()`` → ``repeated_pld(count)`` also forwards confidence
    settings per call; pinning them globally keeps
    the shared helpers here parameter-free.  Set for the duration of the
    test.
    """
    acc.set_discretization(**_MC_KW)
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
        step = acc.per_step(proc)
        with pytest.raises(ValueError, match="exceeds n_steps"):
            (step * 10_000).epsilon_at(_DELTA)

    @pytest.mark.slow
    def test_monotonic_and_bounded_by_full(self):
        proc = self._proc(200)
        e_full = proc.epsilon_at(_DELTA)
        points = [1, 5, 25, 50, 100, 150, 199, 200]
        epsilons = {k: _eps_at(proc, k, _DELTA) for k in points}
        prev = -math.inf
        for k in points:
            e = epsilons[k]
            assert e >= prev - 1e-10, f"non-monotone at k={k}: {e} < {prev}"
            prev = e

        for k in [1, 50, 100, 199]:
            assert epsilons[k] <= e_full + 1e-10

    @pytest.mark.slow
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

    @pytest.mark.slow
    def test_sandwich_at_every_step(self):
        """For K = G·M + r: ε(G·M) ≤ ε(K) ≤ ε((G+1)·M) for r ∈ [0, M)."""
        proc = self._proc(n_steps=80, bands=8)
        M = _atomic_unit(proc)
        epsilons: dict[int, float] = {}

        def epsilon_at(K: int) -> float:
            if K not in epsilons:
                epsilons[K] = _eps_at(proc, K, _DELTA)
            return epsilons[K]

        for K in range(1, proc.n_steps + 1):
            G, r = divmod(K, M)
            e_lo = epsilon_at(G * M) if G > 0 else 0.0
            e_K = epsilon_at(K)
            e_hi_step = min((G + 1) * M, proc.n_steps)
            e_hi = epsilon_at(e_hi_step)
            assert e_lo - 1e-10 <= e_K <= e_hi + 1e-10, (
                f"sandwich broken at K={K} (G={G}, r={r}): {e_lo} ≤ {e_K} ≤ {e_hi}"
            )

    def test_monotonic_across_bands(self):
        proc = self._proc(n_steps=100, bands=8)
        eps_seq = [_eps_at(proc, k, _DELTA) for k in range(8, 101, 8)]
        for a, b in itertools.pairwise(eps_seq):
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

    @pytest.mark.usefixtures("_seed_mc")
    def test_endpoints(self):
        proc = self._proc(n_steps=32)
        assert _eps_at(proc, 0, _MC_DELTA) == 0.0
        e_full = proc.pld(**_MC_KW).epsilon_at(_MC_DELTA)
        e_via_step = _eps_at(proc, 32, _MC_DELTA)
        assert math.isclose(e_full, e_via_step, rel_tol=1e-9)

    @pytest.mark.usefixtures("_seed_mc")
    def test_nonzero_prefix_charges_full_horizon(self):
        """Fail-closed MC prefixes use the full-horizon bound."""
        proc = self._proc(n_steps=32, bands=4)
        e_full = _eps_at(proc, 32, _MC_DELTA)
        e_small = _eps_at(proc, 4, _MC_DELTA)
        assert e_small == pytest.approx(e_full)


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

    def test_small_horizon_prefix_matches_full_process(self):
        proc = self._proc(num_bins=2, num_epochs=1)

        assert _eps_via_step(proc, 2, _DELTA) == pytest.approx(proc.epsilon_at(_DELTA))

    @pytest.mark.slow
    @pytest.mark.usefixtures("_seed_mc")
    def test_endpoints_and_monotonic_prefixes(self):
        # A three-epoch horizon covers first, interior, and final epoch
        # boundaries without re-running the deterministic PLD transform ten
        # times at a 100-step horizon.
        proc = self._proc(num_bins=4, num_epochs=3)
        assert _eps_at(proc, 0, _DELTA) == 0.0
        e_full = _eps_at(proc, 12, _DELTA)
        e_direct = proc.pld(**_MC_KW).epsilon_at(_DELTA)
        assert math.isclose(e_full, e_direct, rel_tol=1e-9)

        prev = 0.0
        for k in range(4, 13, 4):
            e = e_full if k == 12 else _eps_at(proc, k, _DELTA)
            # The identity path is deterministic (random-allocation PLD
            # transform), so monotonicity holds exactly rather than in
            # expectation.
            assert e >= prev - 1e-9, f"non-monotone at k={k}: {e} < {prev}"
            assert e <= e_full + 1e-9
            prev = e


# ---------------------------------------------------------------------------
# Cross-product of (amplification, inner mechanism) supported by K-prefix.
#
# DP-FTRL accounting is defined by combining an amplification factory with
# an inner mechanism and dispatching by ``match self.inner.strategy``
# inside the amplification's ``pld_at``. These tests mirror that structure:
# every (amp, inner) pair that ``pld_at`` should
# support is exercised against the same invariants.
#
# To add a new supported pair: register the amp / inner factory below and
# append the tuple to ``_SUPPORTED_PAIRS``.
# ---------------------------------------------------------------------------


_AMPLIFICATIONS: dict[str, tuple[Callable[..., DpHorizonProcess], bool]] = {
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


# Pairs where ``pld_at`` returns a usable K-step PLD for any K.
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


def _build(amp: str, mech: str) -> DpHorizonProcess:
    factory, _ = _AMPLIFICATIONS[amp]
    return factory(_MECHANISMS[mech]())


def _is_mc(amp: str) -> bool:
    return _AMPLIFICATIONS[amp][1]


def _pld_kwargs(amp: str) -> dict:
    return _MC_KW if _is_mc(amp) else {}


def _eps_via_step(proc: DpHorizonProcess, K: int, delta: float) -> float:
    """K-step ε via ``per_step(proc) * K`` — the public idiom."""
    if K <= 0:
        return Identity().epsilon_at(delta)
    step = acc.per_step(proc)
    return (step * K).epsilon_at(delta)


def _is_mc_proc(proc: DpHorizonProcess) -> bool:
    """Whether ``proc`` evaluates its PLD by Monte Carlo.

    ``BallsInBins`` is strategy-dependent: the identity inner uses the
    deterministic random-allocation transform, while correlated strategies
    still sample the Lemma 3.2 pair via ``bnb_mc_pld``.
    """
    if isinstance(proc, BallsInBins):
        return not isinstance(proc.inner.strategy, IdentityStrategy)
    return isinstance(proc, BMinSep)


def _eps_full(proc: DpHorizonProcess, delta: float) -> float:
    """Full-horizon ε direct off ``proc`` (MC kwargs forwarded on the leaf)."""
    if _is_mc_proc(proc):
        return proc.pld(**_MC_KW).epsilon_at(delta)
    return proc.epsilon_at(delta)


def _pair_id(pair: tuple[str, str]) -> str:
    return f"{pair[0]}({pair[1]})"


_SUPPORTED_IDS = [_pair_id(p) for p in _SUPPORTED_PAIRS]

# Committed vectors cover the deterministic entries of the supported matrix.
# Correlated BallsInBins and BMinSep remain MC-backed and use the invariant
# tests below instead.
_DETERMINISTIC_VECTORS: dict[tuple[str, str], float] = {
    ("CyclicPoisson", "IdentityMf"): 0.6260223959034013,
    ("CyclicPoisson", "BandMf"): 0.4292229170296121,
    ("BallsInBins", "IdentityMf"): 8.612106227069155,
}


@pytest.mark.parametrize(
    ("amp", "mech"),
    [
        pytest.param(
            *pair,
            id=_pair_id(pair),
            marks=pytest.mark.slow if pair == ("BallsInBins", "IdentityMf") else (),
        )
        for pair in _DETERMINISTIC_VECTORS
    ],
)
def test_deterministic_epsilon_matches_committed_vector(amp: str, mech: str):
    actual = _build(amp, mech).epsilon_at(_DELTA)
    expected = _DETERMINISTIC_VECTORS[(amp, mech)]

    assert actual == pytest.approx(expected, rel=1e-9, abs=3e-9), (
        f"{_pair_id((amp, mech))}, delta={_DELTA}: epsilon drifted; "
        f"committed={expected:.17g}, observed={actual:.17g}"
    )


@pytest.mark.parametrize(("amp", "mech"), _SUPPORTED_PAIRS, ids=_SUPPORTED_IDS)
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

    @pytest.mark.usefixtures("_seed_mc")
    def test_inherits_dp_ftrl_process(self, amp: str, mech: str):
        proc = _build(amp, mech)
        assert isinstance(proc, DpHorizonProcess)

    @pytest.mark.usefixtures("_seed_mc")
    def test_step_zero_is_identity_eps(self, amp: str, mech: str):
        proc = _build(amp, mech)
        assert _eps_via_step(proc, 0, _DELTA) == 0.0

    @pytest.mark.slow
    @pytest.mark.usefixtures("_seed_mc")
    def test_prefix_invariants(self, amp: str, mech: str):
        proc = _build(amp, mech)
        n, M = proc.n_steps, proc.atomic_unit
        delta = _MC_DELTA if _is_mc_proc(proc) else _DELTA
        e_full = _eps_full(proc, delta)
        checked_steps = {
            1,
            M,
            n // 2,
            n,
        }
        epsilons = {K: _eps_via_step(proc, K, delta) for K in checked_steps}

        assert math.isclose(epsilons[n], e_full, rel_tol=1e-9)

        for K in (1, M, n // 2):
            assert epsilons[K] <= e_full + 1e-9

        if _is_mc_proc(proc):
            assert epsilons[M] == pytest.approx(e_full)
        else:
            assert epsilons[M] < e_full

        prev = 0.0
        for K in sorted({M, n // 2, n}):
            assert epsilons[K] >= prev - 1e-9
            prev = epsilons[K]

    @pytest.mark.usefixtures("_seed_mc")
    def test_overshoot_step_raises(self, amp: str, mech: str):
        proc = _build(amp, mech)
        step = acc.per_step(proc)
        with pytest.raises(ValueError, match="exceeds n_steps"):
            (step * (proc.n_steps + 10_000)).epsilon_at(_DELTA)

    @pytest.mark.slow
    @pytest.mark.usefixtures("_seed_mc")
    def test_sandwich_at_intermediate_K(self, amp: str, mech: str):
        proc = _build(amp, mech)
        delta = _MC_DELTA if _is_mc_proc(proc) else _DELTA
        n, M = proc.n_steps, proc.atomic_unit
        K = max(1, min(n - 1, n // 2 + M // 2))
        G, _r = divmod(K, M)
        e_lo = _eps_via_step(proc, G * M, delta) if G > 0 else 0.0
        e_K = _eps_via_step(proc, K, delta)
        e_hi = _eps_via_step(proc, min((G + 1) * M, n), delta)
        assert e_lo - 1e-9 <= e_K <= e_hi + 1e-9

    @pytest.mark.usefixtures("_seed_mc")
    def test_supports_composition(self, amp: str, mech: str):
        if (amp, mech) == ("BallsInBins", "IdentityMf"):
            pytest.skip("covered by the slow identity-composition case")
        proc = _build(amp, mech)
        delta = _MC_DELTA if _is_mc_proc(proc) else _DELTA
        step = acc.per_step(proc)
        # step * K1 | step * K2 composes (same proc → merges to Repeated).
        combined = (step * (proc.n_steps // 2)) | (step * proc.atomic_unit)
        assert math.isfinite(combined.epsilon_at(delta))


@pytest.mark.slow
@pytest.mark.usefixtures("_seed_mc")
def test_balls_in_bins_identity_supports_composition():
    proc = _build("BallsInBins", "IdentityMf")
    step = acc.per_step(proc)
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


def _bnb(mechanism, n_steps: int) -> BallsInBins:
    return ftrl_acc.balls_in_bins(mechanism, num_bins=_REGEN_NUM_BINS, n_steps=n_steps)


class TestRecipeDrivenGramRegen:
    """Every correlated BnB nonzero prefix charges the full horizon."""

    @pytest.mark.usefixtures("_seed_mc")
    def test_bsr(self):
        full = bsr_strategy(bandwidth=2, alpha=1.0, beta=0.5)
        proc = _bnb(ftrl_acc.mf_gaussian(1.0, full), _REGEN_N_FULL)
        step = acc.per_step(proc)
        e_at = (step * _REGEN_K).epsilon_at(_MC_DELTA)
        e_full = proc.pld(**_MC_KW).epsilon_at(_MC_DELTA)
        assert e_at == pytest.approx(e_full)

    @pytest.mark.usefixtures("_seed_mc")
    def test_bisr(self):
        full = bisr_strategy(bandwidth=2)
        proc = _bnb(ftrl_acc.mf_gaussian(1.0, full), _REGEN_N_FULL)
        step = acc.per_step(proc)
        e_at = (step * _REGEN_K).epsilon_at(_MC_DELTA)
        e_full = proc.pld(**_MC_KW).epsilon_at(_MC_DELTA)
        assert e_at == pytest.approx(e_full)

    @pytest.mark.usefixtures("_seed_mc")
    def test_lambda_cgd(self):
        full = lambda_cgd_strategy(lambda_=0.5)
        proc = _bnb(ftrl_acc.mf_gaussian(1.0, full), _REGEN_N_FULL)
        step = acc.per_step(proc)
        e_at = (step * _REGEN_K).epsilon_at(_MC_DELTA)
        e_full = proc.pld(**_MC_KW).epsilon_at(_MC_DELTA)
        assert e_at == pytest.approx(e_full)
