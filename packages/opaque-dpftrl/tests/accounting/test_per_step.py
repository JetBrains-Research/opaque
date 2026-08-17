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
_MC_KW = {"num_mc_samples": 4000, "seed": 17}


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

    ``Repeated.pld()`` also accepts ``num_mc_samples`` / ``seed`` per call
    (see :class:`TestQueryTimeMcParams`); pinning them globally keeps the
    many ``epsilon_at`` call sites in these tests unchanged.  Set for the
    duration of the test.
    """
    acc.set_discretization(num_mc_samples=_MC_KW["num_mc_samples"], seed=_MC_KW["seed"])
    yield
    acc.set_discretization()  # restore defaults


@pytest.mark.parametrize("amp", list(_AMPLIFICATIONS))
class TestPerStepCrossAmp:
    """``PerStep`` and ``Repeated(PerStep, K)`` work for every supported amp."""

    @pytest.mark.usefixtures("_seed_mc")
    def test_repeated_at_full_horizon_matches_proc(self, amp: str):
        factory, _ = _AMPLIFICATIONS[amp]
        proc = factory()
        step = acc.per_step(proc)

        e_full_via_step = (step * proc.n_steps).epsilon_at(_DELTA)
        e_full_direct = proc.epsilon_at(_DELTA)
        assert math.isclose(e_full_via_step, e_full_direct, rel_tol=1e-9)

    @pytest.mark.usefixtures("_seed_mc")
    def test_repeated_at_intermediate_matches_pld_at_horizon(self, amp: str):
        factory, _ = _AMPLIFICATIONS[amp]
        proc = factory()
        K = max(1, proc.n_steps // 2)
        step = acc.per_step(proc)

        e_repeated = (step * K).epsilon_at(_DELTA)
        e_at_K = proc.pld_at(K).epsilon_at(_DELTA)
        # Both paths route through the same pld_at(K) call
        # (PerStep.repeated_pld is exactly that), so deterministic
        # equivalence; MC paths use the same global seed.
        assert math.isclose(e_repeated, e_at_K, rel_tol=1e-9)


# ---------------------------------------------------------------------------
# CachedProcess must relay repeated_pld — otherwise DPTrainer's
# ``acc.cached(per_step(proc))`` path silently falls back to K-fold
# single-step composition and inflates ε for every MF run.
# ---------------------------------------------------------------------------


class TestCachedPerStepRepeatedPld:
    """``cached(per_step(p)) * K`` must equal ``per_step(p) * K``.

    Band-MF with ``bands > 1`` is required so the Identity degeneracy
    (where K-fold single-step composition happens to match the horizon
    PLD) cannot mask a missing ``CachedProcess.repeated_pld`` relay.
    """

    def _band_proc(self, *, n_steps: int = 128, bands: int = 16) -> DpHorizonProcess:
        return ftrl_acc.poisson(
            ftrl_acc.mf_gaussian(1.0, band_mf_strategy(bands=bands)),
            sample_rate=0.01,
            n_steps=n_steps,
        )

    def test_cached_repeated_matches_uncached(self):
        proc = self._band_proc()
        step = acc.per_step(proc)
        K = proc.n_steps

        e_uncached = (step * K).epsilon_at(_DELTA)
        e_cached = (acc.cached(step) * K).epsilon_at(_DELTA)
        assert math.isclose(e_uncached, e_cached, rel_tol=0.0, abs_tol=0.0)

    def test_cached_repeated_pld_is_horizon_not_self_compose(self):
        """Guard against the pre-fix failure mode (~2× inflated ε)."""
        proc = self._band_proc()
        step = acc.per_step(proc)
        K = proc.n_steps

        e_cached = (acc.cached(step) * K).epsilon_at(_DELTA)
        e_horizon = proc.pld_at(K).epsilon_at(_DELTA)
        # Wrong path would be step.pld().self_compose(K) ≈ 2× horizon.
        e_wrong = step.pld().self_compose(K).epsilon_at(_DELTA)
        assert math.isclose(e_cached, e_horizon, rel_tol=0.0, abs_tol=0.0)
        assert e_wrong > e_horizon * 1.5


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
# Query-time MC params (num_mc_samples / seed) propagate through the
# composition algebra to MC leaves — issue #479.
# ---------------------------------------------------------------------------


class TestQueryTimeMcParams:
    """``num_mc_samples`` / ``seed`` flow from any ``pld()``-family call
    through ``Repeated`` / ``CachedProcess`` / ``PerStep`` / ``Composed``
    down to the MC leaf's ``get_discretization`` — no global
    ``set_discretization`` mutation required.

    Inequality assertions use the MC sample *budget* (500 vs 4000), whose
    effect on ε is far larger than one discretization grid bucket.  Seed
    changes are asserted for reproducibility only: ε is grid-quantized
    and the native sampler seeds per-thread streams as ``seed + tid``, so
    nearby seeds can produce bit-identical ε on any given machine.
    """

    def _mc_proc(self) -> DpHorizonProcess:
        return ftrl_acc.b_min_sep(
            ftrl_acc.mf_gaussian(1.0, band_mf_strategy(bands=4)),
            n_steps=32,
            p0=0.02,
        )

    def test_chain_forwards_to_pld_at(self):
        """``cached(per_step(p)) * K`` resolves the same config as a direct
        ``pld_at(K, **mc_kw)`` call — byte-identical PLD."""
        proc = self._mc_proc()
        chain = acc.cached(acc.per_step(proc)) * 8

        e_chain = chain.epsilon_at(_DELTA, **_MC_KW)
        e_direct = proc.pld_at(8, **_MC_KW).epsilon_at(_DELTA)
        assert e_chain == e_direct

    def test_chain_honors_mc_budget(self):
        proc = self._mc_proc()
        chain = acc.cached(acc.per_step(proc)) * 8

        e_small = chain.epsilon_at(_DELTA, num_mc_samples=500, seed=_MC_KW["seed"])
        e_large = chain.epsilon_at(_DELTA, **_MC_KW)
        assert e_small != e_large

    def test_composed_broadcasts_to_mc_leaves(self):
        proc = self._mc_proc()
        composed = acc.eps_delta(0.5, 0.0) | acc.per_step(proc) * 4

        e_small = composed.epsilon_at(_DELTA, num_mc_samples=500, seed=_MC_KW["seed"])
        e_large = composed.epsilon_at(_DELTA, **_MC_KW)
        assert e_small != e_large

    def test_full_horizon_seed_reproducible(self):
        proc = self._mc_proc()
        e_a = proc.pld(num_mc_samples=3000, seed=1).epsilon_at(_DELTA)
        e_a_again = proc.pld(num_mc_samples=3000, seed=1).epsilon_at(_DELTA)
        assert e_a == e_a_again

    def test_per_call_matches_global_config(self):
        """``pld(**mc)`` and ``set_discretization(**mc); pld()`` resolve the
        same native config, hence identical PLDs (broadcast semantics)."""
        proc = self._mc_proc()
        # ``pld_at``'s lru_cache keys on (process, kwargs), not the resolved
        # global config: the no-kwargs entry for an equal-valued process must
        # not have been populated under a different global config by an
        # earlier test (all files pin the same _MC_KW).
        e_per_call = proc.epsilon_at(_DELTA, **_MC_KW)
        acc.set_discretization(**_MC_KW)
        try:
            e_global = proc.epsilon_at(_DELTA)
        finally:
            acc.set_discretization()  # restore defaults
        assert e_per_call == e_global
