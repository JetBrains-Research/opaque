"""Contract tests for :class:`PerStep` / :func:`opaque.accounting.per_step`.

``PerStep`` wraps a whole-process :class:`DpHorizonProcess` so the K-fold
:class:`Repeated` node materialises as ``proc.pld_at(K)`` (the
K-prefix bound on the deployed N-step mechanism) rather than the K-fold
self-composition of a single-step PLD.

These tests cover the contract:

- ``PerStep(proc).pld() == proc.pld_at(1)``.
- ``Repeated(PerStep(proc), K).pld() == proc.pld_at(K)`` for
  any 1 ≤ K ≤ ``n_steps``.
- ``Repeated(PerStep(proc), n_steps).pld() == proc.pld()`` (full-process
  equivalence).
- ``Accountant() |= PerStep(proc)`` advances count by 1 and merge-optimises
  to ``Repeated`` after the second composition.
- ``per_step(proc) * (n_steps + 1)`` raises at materialisation.
- ``per_step(procA) | per_step(procB)`` raises when ``procA != procB``.
"""

from __future__ import annotations

import math
from dataclasses import asdict
from typing import TYPE_CHECKING

import pytest

import opaque.accounting as acc
import opaque.dpftrl.accounting as ftrl_acc
from opaque.accounting import Accountant
from opaque.api.accounting.core.composition._per_step import PerStep
from opaque.api.accounting.core.composition.types import Repeated
from opaque.dpftrl.noise import (
    band_mf_strategy,
    blt_strategy,
    identity_strategy,
)

if TYPE_CHECKING:
    from collections.abc import Callable

    from opaque.api.accounting.core._horizon import DpHorizonProcess


_DELTA = 1e-5
_MC_DELTA = 1e-2
_MC_KW = {
    "seed": 17,
    "mc_resolution": 5e-3,
    "mc_failure_probability": 1e-2,
}


# ---------------------------------------------------------------------------
# Deterministic equivalence: CyclicPoisson(IdentityMf) — closed-form PLD, no MC.
# ---------------------------------------------------------------------------


class TestPerStepDeterministic:
    """Exact equivalences between ``PerStep`` materialisation and
    ``proc.pld_at(K)`` on a non-MC path.
    """

    def _proc(self, n_steps: int = 100) -> DpHorizonProcess:
        return ftrl_acc.poisson(
            ftrl_acc.mf_gaussian(1.0, identity_strategy()),
            sample_rate=0.01,
            n_steps=n_steps,
        )

    def test_factory_returns_per_step(self):
        proc = self._proc()
        step = acc.per_step(proc)
        assert isinstance(step, PerStep)
        assert step.process is proc

    def test_factory_rejects_non_horizon_process(self):
        with pytest.raises(TypeError, match="DpHorizonProcess"):
            acc.per_step(acc.identity())

    def test_pld_equals_pld_at_horizon_1(self):
        proc = self._proc(100)
        step = acc.per_step(proc)
        e_step = step.epsilon_at(_DELTA)
        e_at_1 = proc.pld_at(1).epsilon_at(_DELTA)
        assert math.isclose(e_step, e_at_1, rel_tol=1e-9)

    @pytest.mark.parametrize("K", [1, 2, 7, 50, 99])
    def test_repeated_equals_pld_at_horizon_K(self, K: int):
        proc = self._proc(100)
        step = acc.per_step(proc)
        e_repeated = (step * K).epsilon_at(_DELTA)
        e_at_K = proc.pld_at(K).epsilon_at(_DELTA)
        assert math.isclose(e_repeated, e_at_K, rel_tol=1e-9)

    def test_full_horizon_equals_full_proc(self):
        proc = self._proc(100)
        step = acc.per_step(proc)
        e_full_step = (step * 100).epsilon_at(_DELTA)
        e_full_proc = proc.epsilon_at(_DELTA)
        assert math.isclose(e_full_step, e_full_proc, rel_tol=1e-9)


# ---------------------------------------------------------------------------
# Accountant idiom: ``acc |= step`` and ``acc |= step * K`` match.
# ---------------------------------------------------------------------------


class TestPerStepAccountant:
    def _proc(self, n_steps: int = 50) -> DpHorizonProcess:
        return ftrl_acc.poisson(
            ftrl_acc.mf_gaussian(1.0, identity_strategy()),
            sample_rate=0.01,
            n_steps=n_steps,
        )

    def test_loop_matches_explicit_repeat(self):
        proc = self._proc(50)
        step = acc.per_step(proc)

        loop_acc = Accountant()
        for _ in range(50):
            loop_acc |= step

        bulk_acc = Accountant()
        bulk_acc |= step * 50

        assert math.isclose(
            loop_acc.epsilon_at(_DELTA),
            bulk_acc.epsilon_at(_DELTA),
            rel_tol=1e-9,
        )

    def test_loop_matches_full_proc(self):
        proc = self._proc(50)
        step = acc.per_step(proc)

        loop_acc = Accountant()
        for _ in range(50):
            loop_acc |= step

        assert math.isclose(
            loop_acc.epsilon_at(_DELTA),
            proc.epsilon_at(_DELTA),
            rel_tol=1e-9,
        )

    def test_loop_preserves_correlated_horizon_prefix(self):
        proc = ftrl_acc.poisson(
            ftrl_acc.mf_gaussian(1.0, band_mf_strategy(bands=2)),
            sample_rate=0.01,
            n_steps=10,
        )
        step = acc.per_step(proc)
        proc.pld_at.cache_clear()

        loop_acc = Accountant()
        for _ in range(5):
            loop_acc |= step
        prefix_pld = loop_acc.process.pld(discretization=0.1)

        assert prefix_pld is proc.pld_at(5, discretization=0.1)
        assert proc.pld_at.cache_info().misses == 1

        for _ in range(5):
            loop_acc |= step
        full_pld = loop_acc.process.pld(discretization=0.1)

        assert full_pld is proc.pld_at(10, discretization=0.1)
        assert proc.pld_at.cache_info().misses == 2

    def test_cached_horizon_step_preserves_prefix(self):
        proc = self._proc(10)
        step = acc.per_step(proc)
        cached_step = acc.cached(step)

        assert cached_step is not step
        assert (cached_step * 5).pld() is proc.pld_at(5)

    def test_cached_correlated_full_horizon_preserves_epsilon(self):
        n_steps = 128
        proc = ftrl_acc.poisson(
            ftrl_acc.mf_gaussian(1.0, band_mf_strategy(bands=16)),
            sample_rate=0.01,
            n_steps=n_steps,
        )
        step = acc.per_step(proc)

        expected = (step * n_steps).epsilon_at(_DELTA)
        actual = (acc.cached(step) * n_steps).epsilon_at(_DELTA)

        assert actual == expected

    def test_cached_horizon_accountant_warns_and_skips_boundary(self):
        proc = self._proc(10)
        step = acc.per_step(proc)
        accountant = Accountant()
        for _ in range(5):
            accountant |= step

        with pytest.warns(RuntimeWarning, match="whole-horizon"):
            cached_accountant = acc.cached(accountant)

        assert cached_accountant is accountant
        assert cached_accountant.process.pld() is proc.pld_at(5)

    def test_empty_accountant_is_zero(self):
        # Step-0 eval: empty accountant returns ε=0 regardless of step_proc.
        accnt = Accountant()
        assert accnt.epsilon_at(_DELTA) == 0.0

    def test_merge_collapses_to_repeated(self):
        proc = self._proc(50)
        step = acc.per_step(proc)
        # Two ``|`` compositions of the same leaf collapse via merge.
        composed = step | step
        assert isinstance(composed, Repeated)
        assert composed.count == 2
        assert composed.inner == step


# ---------------------------------------------------------------------------
# Overflow and heterogeneous mixing.
# ---------------------------------------------------------------------------


class TestPerStepErrors:
    def _proc(self, n_steps: int = 50, nm: float = 1.0) -> DpHorizonProcess:
        return ftrl_acc.poisson(
            ftrl_acc.mf_gaussian(nm, identity_strategy()),
            sample_rate=0.01,
            n_steps=n_steps,
        )

    def test_overflow_at_materialisation(self):
        proc = self._proc(50)
        step = acc.per_step(proc)
        # The Repeated node builds fine; the raise happens at .pld() /
        # .epsilon_at() when ``pld_at(count)`` is called.
        with pytest.raises(ValueError, match="exceeds n_steps"):
            (step * 51).epsilon_at(_DELTA)

    def test_heterogeneous_compose_preserves_both_processes(self):
        step_a = acc.per_step(self._proc(nm=1.0))
        step_b = acc.per_step(self._proc(nm=2.0))
        composed = step_a | step_b
        assert math.isfinite(composed.epsilon_at(_DELTA))

    def test_distinct_objects_with_equal_fields_still_compose(self):
        # Structural equality, not identity: two separately-constructed procs
        # with the same dataclass fields compare ``==``, so per_step adapters
        # built from them merge under the ``__or__`` rule (procs match).
        proc_a = self._proc(nm=1.0)
        proc_b = self._proc(nm=1.0)
        assert proc_a == proc_b
        assert proc_a is not proc_b
        step_a = acc.per_step(proc_a)
        step_b = acc.per_step(proc_b)
        result = step_a | step_b
        assert isinstance(result, Repeated)
        assert result.count == 2

    def test_same_proc_object_composes(self):
        proc = self._proc(nm=1.0)
        step_a = acc.per_step(proc)
        step_b = acc.per_step(proc)
        result = step_a | step_b
        assert isinstance(result, Repeated)
        assert result.count == 2

    def test_zero_count_raises(self):
        proc = self._proc(50)
        step = acc.per_step(proc)
        # __mul__ rejects count < 1 at construction.
        with pytest.raises(ValueError, match="count must be >= 1"):
            step * 0


# ---------------------------------------------------------------------------
# Cross-amplification smoke: PerStep wraps every supported amplifier and
# Repeated routes through repeated_pld correctly.  MC paths use the same
# seeded config to keep results reproducible.
# ---------------------------------------------------------------------------


_AMPLIFICATIONS: dict[str, tuple[Callable[..., DpHorizonProcess], bool]] = {
    "CyclicPoisson": (
        lambda: ftrl_acc.poisson(
            ftrl_acc.mf_gaussian(1.0, identity_strategy()),
            sample_rate=0.01,
            n_steps=64,
        ),
        False,
    ),
    "BMinSep": (
        lambda: ftrl_acc.b_min_sep(
            ftrl_acc.mf_gaussian(1.0, band_mf_strategy(bands=4)),
            n_steps=32,
            p0=0.02,
        ),
        True,
    ),
    "BallsInBins": (
        lambda: ftrl_acc.balls_in_bins(
            ftrl_acc.mf_gaussian(1.0, blt_strategy()),
            num_bins=4,
            n_steps=16,
        ),
        True,
    ),
}


@pytest.fixture
def _seed_mc():
    """Pin global MC discretization for MC-based amplifiers.

    ``Repeated.pld()`` also accepts confidence settings per call (see
    :class:`TestQueryTimeMcParams`); pinning them globally keeps the
    many ``epsilon_at`` call sites in these tests unchanged.  Set for the
    duration of the test.
    """
    prev = asdict(acc.get_discretization())
    acc.set_discretization(**_MC_KW)
    yield
    acc.set_discretization(**prev)


@pytest.mark.parametrize("amp", list(_AMPLIFICATIONS))
class TestPerStepCrossAmp:
    """``PerStep`` and ``Repeated(PerStep, K)`` work for every supported amp."""

    @pytest.mark.usefixtures("_seed_mc")
    def test_repeated_at_full_horizon_matches_proc(self, amp: str):
        factory, is_mc = _AMPLIFICATIONS[amp]
        proc = factory()
        step = acc.per_step(proc)
        delta = _MC_DELTA if is_mc else _DELTA

        e_full_via_step = (step * proc.n_steps).epsilon_at(delta)
        e_full_direct = proc.epsilon_at(delta)
        assert math.isclose(e_full_via_step, e_full_direct, rel_tol=1e-9)

    @pytest.mark.usefixtures("_seed_mc")
    def test_repeated_at_intermediate_matches_pld_at_horizon(self, amp: str):
        factory, is_mc = _AMPLIFICATIONS[amp]
        proc = factory()
        K = max(1, proc.n_steps // 2)
        step = acc.per_step(proc)
        delta = _MC_DELTA if is_mc else _DELTA

        e_repeated = (step * K).epsilon_at(delta)
        e_at_K = proc.pld_at(K).epsilon_at(delta)
        # Both paths route through the same pld_at(K) call
        # (PerStep.repeated_pld is exactly that), so deterministic
        # equivalence; MC paths use the same global seed.
        assert math.isclose(e_repeated, e_at_K, rel_tol=1e-9)


# ---------------------------------------------------------------------------
# Serialization registry hardening: abstract bases must NOT be registered.
# ---------------------------------------------------------------------------


class TestSerializationRegistryHardening:
    """Abstract intermediates (``DpHorizonProcess``, etc.) are not dataclasses
    and have no fields to serialize; they must not pollute the registry."""

    def test_abstract_bases_not_registered(self):
        from opaque.api.accounting.core._base import _PROCESS_REGISTRY

        assert "DpHorizonProcess" not in _PROCESS_REGISTRY
        # Concrete classes still register normally.
        assert "CyclicPoisson" in _PROCESS_REGISTRY
        assert "BallsInBins" in _PROCESS_REGISTRY
        assert "PerStep" in _PROCESS_REGISTRY


# ---------------------------------------------------------------------------
# Query-time MC confidence and seed propagate through the
# composition algebra to MC leaves — issue #479.
# ---------------------------------------------------------------------------


class TestQueryTimeMcParams:
    """Confidence settings and ``seed`` flow from any ``pld()``-family call
    through ``Repeated`` / ``CachedProcess`` / ``PerStep`` / ``Composed``
    down to the MC leaf's ``get_discretization`` — no global
    ``set_discretization`` mutation required.
    """

    def _mc_proc(self) -> DpHorizonProcess:
        return ftrl_acc.b_min_sep(
            ftrl_acc.mf_gaussian(1.0, band_mf_strategy(bands=4)),
            n_steps=32,
            p0=0.02,
        )

    def test_chain_forwards_to_pld_at(self):
        """``per_step(p) * K`` resolves the same config as ``pld_at(K)``."""
        proc = self._mc_proc()
        chain = acc.per_step(proc) * 8

        e_chain = chain.epsilon_at(_MC_DELTA, **_MC_KW)
        e_direct = proc.pld_at(8, **_MC_KW).epsilon_at(_MC_DELTA)
        assert e_chain == e_direct

    def test_chain_honors_derived_resolution(self):
        proc = self._mc_proc()
        chain = acc.per_step(proc) * 8

        pld = chain.pld(**_MC_KW)
        assert 0.0 < pld.mc_resolution <= _MC_KW["mc_resolution"]
        assert pld.mc_failure_probability == pytest.approx(
            _MC_KW["mc_failure_probability"]
        )

    def test_full_horizon_seed_reproducible(self):
        proc = self._mc_proc()
        kw = {**_MC_KW, "seed": 1}
        e_a = proc.pld(**kw).epsilon_at(_MC_DELTA)
        e_a_again = proc.pld(**kw).epsilon_at(_MC_DELTA)
        assert e_a == e_a_again

    def test_per_call_matches_global_config(self):
        """``pld(**mc)`` and ``set_discretization(**mc); pld()`` resolve the
        same native config, hence identical PLDs (broadcast semantics)."""
        proc = self._mc_proc()
        # The explicit query and equivalent global setting resolve to the same
        # DiscretizationConfig. ``horizon_pld_cache`` keys PLDs by that config,
        # the process cache key, and the horizon, so both calls reuse one PLD.
        pld_per_call = proc.pld(**_MC_KW)
        e_per_call = proc.epsilon_at(_MC_DELTA, **_MC_KW)
        prev = asdict(acc.get_discretization())
        acc.set_discretization(**_MC_KW)
        try:
            pld_global = proc.pld()
            e_global = proc.epsilon_at(_MC_DELTA)
        finally:
            acc.set_discretization(**prev)
        assert pld_per_call is pld_global
        assert e_per_call == e_global
