"""Random allocation across the DP-SGD and DP-FTRL stacks.

Both stacks call the same native primitive, but they parameterise it
differently, and the difference is exactly one square root:

- ``dpftrl_acc.balls_in_bins(mf_gaussian(σ, identity), b, b·E)`` accounts the
  whole run in one shot, at ``(σ/√E, t=b)``.  With ``C = I`` the Lemma 3.2
  mixture means are orthogonal with norm ``√E``, so the ``E`` epochs collapse
  into a single allocation against effective noise ``σ/√E``.
- ``dpsgd_acc.random_allocation(gaussian(σ), num_bins=b) * E`` accounts one
  epoch at ``(σ, t=b)`` and composes ``E`` of them, which is valid because the
  DP-SGD sampler redraws its assignment every epoch.

At ``E = 1`` the two reduce to the same call, so they must agree to the last
bit.  That is the one cheap check that catches a ``σ_eff`` mix-up in either
stack — a mistake that would otherwise show up only as a quietly wrong ε.
"""

from __future__ import annotations

import pytest

import opaque.dpftrl.accounting as ftrl_acc
import opaque.dpsgd.accounting as dpsgd_acc
from opaque.dpftrl.noise import identity_strategy

_DELTAS = (1e-5, 1e-8)


def _bnb(sigma: float, b: int, epochs: int):
    """Fixed-assignment balls-in-bins over the whole run (scheme A)."""
    return ftrl_acc.balls_in_bins(
        ftrl_acc.mf_gaussian(sigma, identity_strategy()),
        num_bins=b,
        n_steps=b * epochs,
    )


def _alloc(sigma: float, b: int, epochs: int):
    """Per-epoch redrawn random allocation (scheme B)."""
    return dpsgd_acc.random_allocation(dpsgd_acc.gaussian(sigma), num_bins=b) * epochs


@pytest.mark.slow
@pytest.mark.parametrize(("sigma", "b"), [(1.0, 16), (2.0, 16), (0.8, 32)])
def test_single_epoch_paths_coincide(sigma: float, b: int):
    """At ``E = 1`` the two stacks must produce the same ε, not merely a
    similar one — the underlying primitive call is identical."""
    a = _alloc(sigma, b, 1).epsilon_at(1e-8)
    c = _bnb(sigma, b, 1).epsilon_at(1e-8)
    assert a == pytest.approx(c, rel=1e-12, abs=1e-12)


@pytest.mark.slow
@pytest.mark.parametrize("epochs", [1, 2, 4])
@pytest.mark.parametrize(("sigma", "b"), [(1.0, 16), (2.0, 16)])
def test_redraw_never_costs_more_than_fixed_assignment(
    sigma: float, b: int, epochs: int
):
    """Per-epoch redraw dominates fixed assignment, at every δ.

    This is a theorem, not a measurement: for ``C = I`` the redrawn pair is a
    mixture ``P_B = E_d[T_d # P_A]`` of shifts of the fixed-assignment pair,
    and the hockey-stick divergence is jointly convex, so no ``ε`` can be
    larger.  Asserted as a hard inequality rather than an expected gap —
    the gap itself ranges from 0% at ``E = 1`` to tens of percent, and
    pinning a band would just be recording today's grid.
    """
    fixed = _bnb(sigma, b, epochs)
    redrawn = _alloc(sigma, b, epochs)
    for delta in _DELTAS:
        assert redrawn.epsilon_at(delta) <= fixed.epsilon_at(delta) + 1e-9


@pytest.mark.slow
def test_redraw_strictly_better_beyond_one_epoch():
    """The inequality above is not vacuous — it is strict once ``E > 1``."""
    sigma, b, delta = 1.0, 16, 1e-8
    assert _alloc(sigma, b, 4).epsilon_at(delta) < _bnb(sigma, b, 4).epsilon_at(delta)
