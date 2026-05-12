"""DP-FTRL process base class — adds ``approx_at_step`` to whole-process accountants.

Unlike DP-SGD where a per-step factory composes externally with
``* num_steps``, every DP-FTRL accountant returns a process spanning
the full training run.  ``DpFtrlProcess`` adds intermediate-step
accounting on top of :class:`DpProcess`: ``process.approx_at_step(K)``
returns a fresh :class:`DpProcess` representing the privacy budget for
a fresh K-step run.

Semantics.  ``approx_at_step(K)`` is **an upper bound** on the smooth
ε-curve at K, and it reports the budget *for a fresh K-step run* —
**not** the actual privacy cost of the original N-step process stopped
at step K (which would generally be larger, since the original strategy
is tuned for N steps).  Tight at K = G·atomic_unit; off-boundary it
rounds K up to the next atomic-unit multiple.

Contract (sandwich form).  Let ``M = atomic_unit``, ``N = n_steps``,
``G = K // M``, ``r = K - G·M``.  Implementations satisfy:

- ``ε(approx_at_step(0))         == 0`` (returns :class:`Identity`).
- ``ε(approx_at_step(N))         == self.epsilon_at(δ)`` (returns ``self``).
- ``K1 ≤ K2 ⇒ ε(approx_at_step(K1)) ≤ ε(approx_at_step(K2))`` (monotone).
- ``ε(approx_at_step(G·M)) ≤ ε(approx_at_step(K)) ≤ ε(approx_at_step((G+1)·M))`` (sandwich).
- ``ε(approx_at_step(K)) ≤ ε(self)`` for ``K ≤ N``.

The default ``approx_at_step`` rounds ``K`` up to the next multiple of
``atomic_unit`` and rebuilds the dataclass with the smaller ``n_steps``.
This is exact when ``M`` divides ``K`` and an upper bound otherwise —
within an atomic unit ``ε(K)`` plateaus, which trades strict ``<`` on
the LHS of the sandwich for ``≤`` (still monotone).  Subclasses may
override for tighter or per-inner behaviour.
"""

from __future__ import annotations

import dataclasses
from abc import abstractmethod

from opaque.api.accounting.core._base import DpProcess
from opaque.api.accounting.core.mechanisms.types import Identity

__all__ = ["DpFtrlProcess"]


class DpFtrlProcess(DpProcess):
    """Whole-process accountant for DP-FTRL — exposes intermediate-step accounting.

    Subclasses MUST be frozen dataclasses with an ``n_steps: int`` field
    and implement :attr:`atomic_unit`.
    """

    n_steps: int  # required dataclass field on every subclass

    @property
    @abstractmethod
    def atomic_unit(self) -> int:
        """Step granularity at which ``pld()`` factors exactly.

        Within an atomic unit (band, epoch, ...) the default
        ``approx_at_step`` rounds up — ``ε(K)`` plateaus until the next
        multiple of ``atomic_unit``.  Implementations should pick the
        largest unit that keeps the existing accountant correct after a
        simple ``n_steps`` substitution (1 for per-step Identity-style;
        ``bands`` for BandMF; ``num_bins`` for BallsInBins; etc.).
        """

    def approx_at_step(self, step: int) -> DpProcess:
        """Process truncated to its first ``step`` of ``n_steps`` rounds.

        ``step`` is rounded up to the nearest multiple of
        :attr:`atomic_unit` and capped at ``n_steps``.  Returns
        :class:`Identity` for ``step <= 0`` and ``self`` for
        ``step >= n_steps``.

        This is an **upper bound** on the smooth ε-curve at ``step`` and
        reports the budget for a fresh K-step run, **not** the actual
        privacy cost of the original N-step process stopped at step K.

        Returns a fresh dataclass instance of the same concrete type
        (so the full :class:`DpProcess` API — ``epsilon_at``, ``pld``,
        composition operators ``|`` / ``*`` — works on the result), or
        :class:`Identity` at the zero endpoint.

        Raises:
            ValueError: If ``atomic_unit`` is not positive.
        """
        if step <= 0:
            return Identity()
        if step >= self.n_steps:
            return self
        unit = self.atomic_unit
        if unit < 1:
            raise ValueError(
                f"{type(self).__name__}.atomic_unit must be >= 1, got {unit}"
            )
        rounded = min(-(-step // unit) * unit, self.n_steps)
        if rounded == self.n_steps:
            return self
        return dataclasses.replace(self, n_steps=rounded)
