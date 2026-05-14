"""Contract tests for :class:`PerStep` / :func:`per_step`.

``PerStep`` wraps a whole-process :class:`DpFtrlProcess` so the K-fold
:class:`Repeated` node materialises as ``proc.approx_at_step(K).pld()``
rather than the K-fold self-composition of a single-step PLD.

These tests cover the contract:

- ``PerStep(proc).pld() == proc.approx_at_step(1).pld()``.
- ``Repeated(PerStep(proc), K).pld() == proc.approx_at_step(K).pld()``
  for any 1 ≤ K ≤ ``n_steps``.
- ``Repeated(PerStep(proc), n_steps).pld() == proc.pld()`` (full-process
  equivalence).
- ``Accountant() |= PerStep(proc) `` advances count by 1 and merge-optimises
  to ``Repeated`` after the second composition.
- ``per_step(proc) * (n_steps + 1)`` raises at materialisation.
- ``per_step(procA) | per_step(procB)`` raises when ``procA != procB``.
"""

from __future__ import annotations

import math
from collections.abc import Callable

import pytest

import opaque.accounting as acc
import opaque.dpftrl.accounting as ftrl_acc
from opaque.accounting import Accountant
from opaque.api.accounting.core.composition.types import Repeated
from opaque.api.accounting.dpftrl.composition import PerStep
from opaque.dpftrl.accounting.types import DpFtrlProcess
from opaque.dpftrl.noise import (
    band_mf_strategy,
    blt_strategy,
    identity_strategy,
)

_DELTA = 1e-5
_MC_KW = {"num_mc_samples": 4000, "seed": 17}


# ---------------------------------------------------------------------------
# Deterministic equivalence: CyclicPoisson(IdentityMf) — closed-form PLD, no MC.
# ---------------------------------------------------------------------------


class TestPerStepDeterministic:
    """Exact equivalences between ``PerStep`` materialisation and
    ``proc.approx_at_step(K).pld()`` on a non-MC path.
    """

    def _proc(self, n_steps: int = 100) -> DpFtrlProcess:
        return ftrl_acc.poisson(
            ftrl_acc.mf_gaussian(1.0, identity_strategy()),
            sample_rate=0.01,
            n_steps=n_steps,
        )

    def test_factory_returns_per_step(self):
        proc = self._proc()
        step = ftrl_acc.per_step(proc)
        assert isinstance(step, PerStep)
        assert step.proc is proc

    def test_factory_rejects_non_dpftrl(self):
        with pytest.raises(TypeError, match="DpFtrlProcess"):
            ftrl_acc.per_step(acc.identity())

    def test_pld_equals_approx_at_step_1(self):
        proc = self._proc(100)
        step = ftrl_acc.per_step(proc)
        e_step = step.epsilon_at(_DELTA)
        e_at_1 = proc.approx_at_step(1).epsilon_at(_DELTA)
        assert math.isclose(e_step, e_at_1, rel_tol=1e-9)

    @pytest.mark.parametrize("K", [1, 2, 7, 50, 99])
    def test_repeated_equals_approx_at_step_K(self, K: int):
        proc = self._proc(100)
        step = ftrl_acc.per_step(proc)
        e_repeated = (step * K).epsilon_at(_DELTA)
        e_at_K = proc.approx_at_step(K).epsilon_at(_DELTA)
        assert math.isclose(e_repeated, e_at_K, rel_tol=1e-9)

    def test_full_horizon_equals_full_proc(self):
        proc = self._proc(100)
        step = ftrl_acc.per_step(proc)
        e_full_step = (step * 100).epsilon_at(_DELTA)
        e_full_proc = proc.epsilon_at(_DELTA)
        assert math.isclose(e_full_step, e_full_proc, rel_tol=1e-9)


# ---------------------------------------------------------------------------
# Accountant idiom: ``acc |= step`` and ``acc |= step * K`` match.
# ---------------------------------------------------------------------------


class TestPerStepAccountant:
    def _proc(self, n_steps: int = 50) -> DpFtrlProcess:
        return ftrl_acc.poisson(
            ftrl_acc.mf_gaussian(1.0, identity_strategy()),
            sample_rate=0.01,
            n_steps=n_steps,
        )

    def test_loop_matches_explicit_repeat(self):
        proc = self._proc(50)
        step = ftrl_acc.per_step(proc)

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
        step = ftrl_acc.per_step(proc)

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
        step = ftrl_acc.per_step(proc)
        # Two ``|`` compositions of the same leaf collapse via merge.
        composed = step | step
        assert isinstance(composed, Repeated)
        assert composed.count == 2
        assert composed.inner == step


# ---------------------------------------------------------------------------
# Overflow and heterogeneous mixing: errors raised at the right boundary.
# ---------------------------------------------------------------------------


class TestPerStepErrors:
    def _proc(self, n_steps: int = 50, nm: float = 1.0) -> DpFtrlProcess:
        return ftrl_acc.poisson(
            ftrl_acc.mf_gaussian(nm, identity_strategy()),
            sample_rate=0.01,
            n_steps=n_steps,
        )

    def test_overflow_at_materialisation(self):
        proc = self._proc(50)
        step = ftrl_acc.per_step(proc)
        # The Repeated node builds fine; the raise happens at .pld() /
        # .epsilon_at() when ``approx_at_step(count)`` is called.
        with pytest.raises(ValueError, match="exceeds n_steps"):
            (step * 51).epsilon_at(_DELTA)

    def test_heterogeneous_compose_raises(self):
        step_a = ftrl_acc.per_step(self._proc(nm=1.0))
        step_b = ftrl_acc.per_step(self._proc(nm=2.0))
        with pytest.raises(ValueError, match="different process"):
            step_a | step_b

    def test_distinct_objects_with_equal_fields_still_raise(self):
        # Identity, not structural equality.  Two separately-constructed
        # procs may compare ``==`` (some strategy fields are excluded from
        # equality, e.g. ``BltStrategy.lr_schedule``) yet have different
        # privacy behaviour — composing them silently is unsafe.
        proc_a = self._proc(nm=1.0)
        proc_b = self._proc(nm=1.0)
        assert proc_a == proc_b and proc_a is not proc_b
        step_a = ftrl_acc.per_step(proc_a)
        step_b = ftrl_acc.per_step(proc_b)
        with pytest.raises(ValueError, match="different process"):
            step_a | step_b

    def test_same_proc_object_composes(self):
        # Wrapping the same ``proc`` object in two ``per_step`` calls
        # merges into a single ``Repeated(step, 2)`` via the merge
        # optimizer — this is the supported trainer pattern.
        proc = self._proc(nm=1.0)
        step_a = ftrl_acc.per_step(proc)
        step_b = ftrl_acc.per_step(proc)
        result = step_a | step_b
        assert isinstance(result, Repeated)
        assert result.count == 2


# ---------------------------------------------------------------------------
# Cross-amplification smoke: PerStep wraps every supported amplifier and
# Repeated routes through repeated_pld correctly.  MC paths use the same
# seeded config to keep results reproducible.
# ---------------------------------------------------------------------------


_AMPLIFICATIONS: dict[str, tuple[Callable[..., DpFtrlProcess], bool]] = {
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
    """Pin global MC discretization so PLDs flow through ``Repeated.pld()``.

    ``Repeated.pld()`` only accepts the core discretization kwargs; the MC
    sample budget / seed must come from the global config (or per-call on
    the leaf, which is not what ``Repeated`` does).  Set globally for the
    duration of the test.
    """
    acc.set_discretization(num_mc_samples=_MC_KW["num_mc_samples"], seed=_MC_KW["seed"])
    yield
    acc.set_discretization()  # restore defaults


@pytest.mark.parametrize("amp", list(_AMPLIFICATIONS))
class TestPerStepCrossAmp:
    """``PerStep`` and ``Repeated(PerStep, K)`` work for every supported amp."""

    def test_repeated_at_full_horizon_matches_proc(self, amp: str, _seed_mc):
        factory, _ = _AMPLIFICATIONS[amp]
        proc = factory()
        step = ftrl_acc.per_step(proc)

        e_full_via_step = (step * proc.n_steps).epsilon_at(_DELTA)
        e_full_direct = proc.epsilon_at(_DELTA)
        assert math.isclose(e_full_via_step, e_full_direct, rel_tol=1e-9)

    def test_repeated_at_intermediate_matches_approx_at_step(self, amp: str, _seed_mc):
        factory, _ = _AMPLIFICATIONS[amp]
        proc = factory()
        K = max(1, proc.n_steps // 2)
        step = ftrl_acc.per_step(proc)

        e_repeated = (step * K).epsilon_at(_DELTA)
        e_approx = proc.approx_at_step(K).epsilon_at(_DELTA)
        # Both paths route through the same approx_at_step.pld() call
        # (PerStep.repeated_pld is exactly that), so deterministic
        # equivalence; MC paths use the same global seed.
        assert math.isclose(e_repeated, e_approx, rel_tol=1e-9)
