"""Contracts for explicit horizon-run accounting.

``HorizonRun`` is a deployment handle for one whole-horizon process.
Multiplication materializes a :class:`HorizonPrefix`, and an
:class:`Accountant` advances that same prefix by stable ``run_id``. Equal
configuration is deliberately insufficient evidence that two releases belong
to the same correlated deployment.

These tests cover the contract:

- ``HorizonRun(proc).pld() == proc.pld_at(1)``.
- ``(HorizonRun(proc) * K).pld() == proc.pld_at(K)`` for any
  1 ≤ K ≤ ``n_steps``.
- ``(HorizonRun(proc) * n_steps).pld() == proc.pld()`` (full-process
  equivalence).
- ``Accountant() |= run`` replaces prefix K by prefix K+1 for that run.
- caching never materializes the active horizon frontier.
- fresh equal-configured runs compose independently rather than merging.
- serialized run IDs preserve continuation lineage across restore.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import pytest

import opaque.accounting as acc
import opaque.dpftrl.accounting as ftrl_acc
from opaque.accounting import Accountant
from opaque.accounting.types import HorizonPrefix, HorizonRun
from opaque.api.accounting.core.composition.types import (
    CachedProcess,
    Composed,
    Repeated,
)
from opaque.dpftrl.noise import (
    band_mf_strategy,
    blt_strategy,
    identity_strategy,
)
from opaque.serialization import from_state_dict, state_dict

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


class TestHorizonRunDeterministic:
    """Exact equivalences between ``HorizonRun`` materialisation and
    ``proc.pld_at(K)`` on a non-MC path.
    """

    def _proc(self, n_steps: int = 100) -> DpHorizonProcess:
        return ftrl_acc.poisson(
            ftrl_acc.mf_gaussian(1.0, identity_strategy()),
            sample_rate=0.01,
            n_steps=n_steps,
        )

    def test_factory_returns_horizon_run(self):
        proc = self._proc()
        step = acc.horizon_run(proc)
        assert isinstance(step, HorizonRun)
        assert step.process is proc

    def test_each_factory_call_creates_a_fresh_deployment(self):
        proc = self._proc()
        first = acc.horizon_run(proc)
        second = acc.horizon_run(proc)

        assert first.process is second.process
        assert first.run_id != second.run_id

    def test_factory_rejects_non_horizon_process(self):
        with pytest.raises(TypeError, match="DpHorizonProcess"):
            acc.horizon_run(acc.identity())

    def test_pld_equals_pld_at_horizon_1(self):
        proc = self._proc(100)
        step = acc.horizon_run(proc)
        e_step = step.epsilon_at(_DELTA)
        e_at_1 = proc.pld_at(1).epsilon_at(_DELTA)
        assert math.isclose(e_step, e_at_1, rel_tol=1e-9)

    @pytest.mark.parametrize("K", [1, 2, 7, 50, 99])
    def test_repeated_equals_pld_at_horizon_K(self, K: int):
        proc = self._proc(100)
        step = acc.horizon_run(proc)
        prefix = step * K
        assert isinstance(prefix, HorizonPrefix)
        assert prefix.steps == K
        assert prefix.run_id == step.run_id
        e_repeated = prefix.epsilon_at(_DELTA)
        e_at_K = proc.pld_at(K).epsilon_at(_DELTA)
        assert math.isclose(e_repeated, e_at_K, rel_tol=1e-9)

    def test_full_horizon_equals_full_proc(self):
        proc = self._proc(100)
        step = acc.horizon_run(proc)
        e_full_step = (step * 100).epsilon_at(_DELTA)
        e_full_proc = proc.epsilon_at(_DELTA)
        assert math.isclose(e_full_step, e_full_proc, rel_tol=1e-9)


# ---------------------------------------------------------------------------
# Accountant idiom: ``acc |= step`` and ``acc |= step * K`` match.
# ---------------------------------------------------------------------------


class TestHorizonRunAccountant:
    def _proc(self, n_steps: int = 50, nm: float = 1.0) -> DpHorizonProcess:
        return ftrl_acc.poisson(
            ftrl_acc.mf_gaussian(nm, identity_strategy()),
            sample_rate=0.01,
            n_steps=n_steps,
        )

    def test_loop_matches_explicit_repeat(self):
        proc = self._proc(50)
        step = acc.horizon_run(proc)

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
        step = acc.horizon_run(proc)

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
        step = acc.horizon_run(proc)
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
        step = acc.horizon_run(proc)
        cached_step = acc.cached(step)

        # A handle has no closed PLD to cache; prefix PLDs are already keyed by
        # K on the horizon process itself.
        assert cached_step is step
        assert (cached_step * 5).pld() is proc.pld_at(5)
        with pytest.raises(TypeError, match="deployment handle"):
            cached_step | cached_step

    @pytest.mark.parametrize("K", range(1, 6))
    def test_cached_horizon_accountant_continues_prefix(self, K: int):
        proc = ftrl_acc.poisson(
            ftrl_acc.mf_gaussian(1.0, band_mf_strategy(bands=2)),
            sample_rate=0.01,
            n_steps=6,
        )
        step = acc.cached(acc.horizon_run(proc))
        accountant = Accountant()
        for _ in range(K):
            accountant |= step

        continued = acc.cached(accountant) | step

        assert isinstance(continued.process, HorizonPrefix)
        assert continued.process.steps == K + 1
        assert continued.process.run_id == step.run_id
        assert continued.process.pld(discretization=0.1) is proc.pld_at(
            K + 1, discretization=0.1
        )

    def test_cached_horizon_continues_across_boundaries_and_restore(self):
        proc = ftrl_acc.poisson(
            ftrl_acc.mf_gaussian(1.0, band_mf_strategy(bands=2)),
            sample_rate=0.01,
            n_steps=8,
        )
        step = acc.cached(acc.horizon_run(proc))
        accountant = Accountant()

        for K in range(1, 9):
            accountant |= step
            if K in {3, 5, 7}:
                accountant = acc.cached(accountant)
            if K == 5:
                accountant = from_state_dict(Accountant(), state_dict(accountant))

        assert isinstance(accountant.process, HorizonPrefix)
        assert accountant.process.steps == 8
        assert accountant.process.run_id == step.run_id
        assert accountant.epsilon_at(_DELTA) == pytest.approx(
            proc.pld_at(8).epsilon_at(_DELTA)
        )

    def test_cached_heterogeneous_prefix_keeps_only_prefix_opaque(self):
        proc = ftrl_acc.poisson(
            ftrl_acc.mf_gaussian(1.0, band_mf_strategy(bands=2)),
            sample_rate=0.01,
            n_steps=6,
        )
        step = acc.cached(acc.horizon_run(proc))
        prefix = acc.eps_delta(0.1, 1e-6)
        accountant = Accountant() | prefix
        for _ in range(3):
            accountant |= step

        continued = acc.cached(accountant) | step

        assert isinstance(continued.process, Composed)
        assert isinstance(continued.process.left, CachedProcess)
        assert continued.process.left.inner == prefix
        assert isinstance(continued.process.right, HorizonPrefix)
        assert continued.process.right.steps == 4
        assert continued.process.right.run_id == step.run_id
        assert continued.process.right.pld(discretization=0.1) is proc.pld_at(
            4, discretization=0.1
        )

    def test_fresh_horizon_run_is_an_independent_sequence(self):
        first = acc.cached(acc.horizon_run(self._proc(n_steps=6, nm=1.0)))
        second = acc.cached(acc.horizon_run(self._proc(n_steps=6, nm=2.0)))
        accountant = acc.cached(Accountant() | (first * 3))

        continued = accountant | second

        assert isinstance(continued.process, Composed)
        assert continued.process.left == first.prefix(3)
        assert continued.process.right == second.prefix(1)
        assert first.run_id != second.run_id

    def test_same_run_cannot_cross_an_intervening_release(self):
        step = acc.cached(acc.horizon_run(self._proc(n_steps=6)))
        intervening = acc.eps_delta(0.1, 1e-6)
        accountant = acc.cached(Accountant() | (step * 2) | intervening)

        with pytest.raises(ValueError, match="intervening release"):
            accountant | step

        fresh = acc.horizon_run(step.process)
        continued = accountant | fresh
        assert isinstance(continued.process, Composed)
        assert isinstance(continued.process.left, CachedProcess)
        assert continued.process.right == fresh.prefix(1)

    def test_empty_accountant_is_zero(self):
        # Step-0 eval: empty accountant returns ε=0 regardless of step_proc.
        accnt = Accountant()
        assert accnt.epsilon_at(_DELTA) == 0.0

    def test_accountant_advances_explicit_prefix(self):
        proc = self._proc(50)
        step = acc.horizon_run(proc)
        accountant = Accountant().advance(step, 2)

        assert accountant.process == step.prefix(2)


# ---------------------------------------------------------------------------
# Overflow and heterogeneous mixing.
# ---------------------------------------------------------------------------


class TestHorizonRunErrors:
    def _proc(self, n_steps: int = 50, nm: float = 1.0) -> DpHorizonProcess:
        return ftrl_acc.poisson(
            ftrl_acc.mf_gaussian(nm, identity_strategy()),
            sample_rate=0.01,
            n_steps=n_steps,
        )

    def test_overflow_rejected_at_prefix_construction(self):
        proc = self._proc(50)
        step = acc.horizon_run(proc)
        with pytest.raises(ValueError, match="exceeds n_steps"):
            step * 51

    def test_run_handles_are_not_generic_processes(self):
        step_a = acc.horizon_run(self._proc(nm=1.0))
        step_b = acc.horizon_run(self._proc(nm=2.0))
        with pytest.raises(TypeError, match="deployment handle"):
            step_a | step_b

    def test_equal_configured_fresh_runs_compose_independently(self):
        proc_a = self._proc(nm=1.0)
        proc_b = self._proc(nm=1.0)
        assert proc_a == proc_b
        assert proc_a is not proc_b
        step_a = acc.horizon_run(proc_a)
        step_b = acc.horizon_run(proc_b)
        accountant = Accountant() | step_a
        accountant |= step_b

        assert isinstance(accountant.process, Composed)
        assert accountant.process.left == step_a.prefix(1)
        assert accountant.process.right == step_b.prefix(1)
        assert step_a.run_id != step_b.run_id
        expected = proc_a.pld_at(1).compose(proc_b.pld_at(1))
        assert accountant.epsilon_at(_DELTA) == pytest.approx(
            expected.epsilon_at(_DELTA)
        )

    def test_equal_band_mf_deployments_do_not_collapse_into_one_prefix(self):
        proc = ftrl_acc.poisson(
            ftrl_acc.mf_gaussian(1.0, band_mf_strategy(bands=2)),
            sample_rate=0.1,
            n_steps=6,
        )
        accountant = Accountant() | acc.horizon_run(proc)
        accountant |= acc.horizon_run(proc)

        independent_epsilon = accountant.process.epsilon_at(
            _DELTA,
            discretization=0.1,
        )
        expected_independent = proc.pld_at(1, discretization=0.1).self_compose(2)
        one_run_epsilon = proc.pld_at(2, discretization=0.1).epsilon_at(_DELTA)

        # Concrete regression for the structural-equality bug: these bounds
        # differ materially, so merging equal configuration into one run is
        # not a harmless tree optimization.
        assert independent_epsilon == pytest.approx(
            expected_independent.epsilon_at(_DELTA)
        )
        assert independent_epsilon - one_run_epsilon > 0.2

    def test_same_process_object_still_creates_distinct_runs(self):
        proc = self._proc(nm=1.0)
        step_a = acc.horizon_run(proc)
        step_b = acc.horizon_run(proc)
        accountant = Accountant().advance(step_a).advance(step_b)

        assert isinstance(accountant.process, Composed)
        assert step_a.run_id != step_b.run_id

    def test_same_run_prefix_cannot_be_composed_twice(self):
        step = acc.horizon_run(self._proc())
        prefix = step.prefix(2)

        with pytest.raises(ValueError, match="same horizon run"):
            prefix | prefix
        with pytest.raises(ValueError, match="same horizon run"):
            (acc.eps_delta(0.1, 0.0) | prefix) | prefix
        with pytest.raises(ValueError, match="same horizon run"):
            (acc.eps_delta(0.1, 0.0) | prefix) | (acc.eps_delta(0.2, 0.0) | prefix)

    def test_materialized_prefix_cannot_be_repeated(self):
        prefix = acc.horizon_run(self._proc()).prefix(2)

        with pytest.raises(TypeError, match="deployed transcript"):
            prefix * 2
        with pytest.raises(TypeError, match="cannot be self-composed"):
            Repeated(prefix, 2).pld()

    def test_same_run_id_cannot_change_configuration(self):
        first = acc.horizon_run(self._proc(nm=1.0))
        spoofed = HorizonRun(self._proc(nm=2.0), run_id=first.run_id)
        accountant = Accountant() | first

        with pytest.raises(ValueError, match="configuration changed"):
            accountant | spoofed

    def test_zero_count_raises(self):
        proc = self._proc(50)
        step = acc.horizon_run(proc)
        # __mul__ rejects count < 1 at construction.
        with pytest.raises(ValueError, match=r"steps .* must be >= 1"):
            step * 0


# ---------------------------------------------------------------------------
# Cross-amplification smoke: every supported amplifier materializes an
# explicit prefix through ``step * K``. MC paths use the same seeded config to
# keep results reproducible.
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
    acc.set_discretization(**_MC_KW)
    yield
    acc.set_discretization()  # restore defaults


@pytest.mark.parametrize("amp", list(_AMPLIFICATIONS))
class TestHorizonRunCrossAmp:
    """``HorizonRun`` and ``HorizonPrefix`` work for every supported amp."""

    @pytest.mark.usefixtures("_seed_mc")
    def test_repeated_at_full_horizon_matches_proc(self, amp: str):
        factory, is_mc = _AMPLIFICATIONS[amp]
        proc = factory()
        step = acc.horizon_run(proc)
        delta = _MC_DELTA if is_mc else _DELTA

        e_full_via_step = (step * proc.n_steps).epsilon_at(delta)
        e_full_direct = proc.epsilon_at(delta)
        assert math.isclose(e_full_via_step, e_full_direct, rel_tol=1e-9)

    @pytest.mark.usefixtures("_seed_mc")
    def test_repeated_at_intermediate_matches_pld_at_horizon(self, amp: str):
        factory, is_mc = _AMPLIFICATIONS[amp]
        proc = factory()
        K = max(1, proc.n_steps // 2)
        step = acc.horizon_run(proc)
        delta = _MC_DELTA if is_mc else _DELTA

        e_repeated = (step * K).epsilon_at(delta)
        e_at_K = proc.pld_at(K).epsilon_at(delta)
        # Both paths route through the same pld_at(K) call, so deterministic
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
        assert "HorizonRun" in _PROCESS_REGISTRY
        assert "HorizonPrefix" in _PROCESS_REGISTRY


# ---------------------------------------------------------------------------
# Query-time MC confidence and seed propagate through the
# composition algebra to MC leaves — issue #479.
# ---------------------------------------------------------------------------


class TestQueryTimeMcParams:
    """Confidence settings and ``seed`` flow from any ``pld()``-family call
    through ``HorizonPrefix`` / ``CachedProcess`` / ``Composed``
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
        """``horizon_run(p) * K`` resolves the same config as ``pld_at(K)``."""
        proc = self._mc_proc()
        chain = acc.horizon_run(proc) * 8

        e_chain = chain.epsilon_at(_MC_DELTA, **_MC_KW)
        e_direct = proc.pld_at(8, **_MC_KW).epsilon_at(_MC_DELTA)
        assert e_chain == e_direct

    def test_chain_honors_derived_resolution(self):
        proc = self._mc_proc()
        chain = acc.horizon_run(proc) * 8

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
        # ``pld_at``'s lru_cache keys on (process, kwargs), not the resolved
        # global config: the no-kwargs entry for an equal-valued process must
        # not have been populated under a different global config by an
        # earlier test (all files pin the same _MC_KW).
        e_per_call = proc.epsilon_at(_MC_DELTA, **_MC_KW)
        acc.set_discretization(**_MC_KW)
        try:
            e_global = proc.epsilon_at(_MC_DELTA)
        finally:
            acc.set_discretization()  # restore defaults
        assert e_per_call == e_global
