"""``PerStep`` — wrap a DP-FTRL whole-process accountant as a per-step atom.

DP-FTRL accountants describe whole training runs; their K-step privacy cost
is not the K-fold composition of a single-step PLD because the strategy is
tuned for a particular horizon.  ``PerStep`` adapts a :class:`DpFtrlProcess`
to the :class:`Accountant`'s ``acc |= step`` idiom by overriding
:meth:`DpProcess.repeated_pld` to call ``proc.approx_at_step(count).pld()``,
so a :class:`Repeated` node ``PerStep(proc) * K`` materialises as the true
K-step PLD rather than the K-fold self-composition of a single-step PLD.
"""

from __future__ import annotations

import functools
from dataclasses import dataclass

from opaque.api.accounting.core._base import DpProcess, Pld
from opaque.api.accounting.dpftrl._base import DpFtrlProcess


@dataclass(frozen=True, slots=True, eq=False)
class PerStep(DpProcess):
    """One atomic step of a :class:`DpFtrlProcess`, composable with ``|`` / ``*``.

    Wraps a whole-process DP-FTRL accountant so it behaves as a
    per-step factory under the accountant algebra:

    - ``PerStep(proc).pld()``           ≡ ``proc.approx_at_step(1).pld()``
    - ``Repeated(PerStep(proc), K).pld()`` ≡ ``proc.approx_at_step(K).pld()``

    The :class:`Repeated` node is built by the standard ``__mul__`` /
    ``__or__`` merge optimizations — ``PerStep`` simply overrides
    :meth:`DpProcess.repeated_pld` so the K-fold materialisation calls
    the wrapped process's strategy-aware ``approx_at_step``.

    Heterogeneous mixing of ``PerStep`` instances with different
    underlying processes is rejected: the per-step PLD only makes sense
    relative to one whole-process accountant.

    Equality is by object identity of :attr:`proc` (not structural
    equality).  Some DP-FTRL strategies exclude privacy-relevant fields
    from ``__eq__`` (e.g. ``BltStrategy.lr_schedule`` is ``compare=False``
    yet shifts the coefficients used by accounting), so a structural
    ``==`` could silently merge two distinct-by-privacy procs into one
    :class:`Repeated` leaf and report an underestimate.  Identity
    semantics force the trainer to thread one ``proc`` object through
    every ``per_step(...)`` call, matching its actual single-process
    accounting intent.
    """

    proc: DpFtrlProcess

    def __eq__(self, other: object) -> bool:
        return isinstance(other, PerStep) and other.proc is self.proc

    def __hash__(self) -> int:
        return hash((PerStep, id(self.proc)))

    @functools.lru_cache(maxsize=8)
    def pld(
        self,
        *,
        discretization: float | None = None,
        log_x_mass_truncation_bound: float | None = None,
        pessimistic_estimate: bool | None = None,
        max_grid_size: int | None = None,
    ) -> Pld:
        return self.proc.approx_at_step(1).pld(
            discretization=discretization,
            log_x_mass_truncation_bound=log_x_mass_truncation_bound,
            pessimistic_estimate=pessimistic_estimate,
            max_grid_size=max_grid_size,
        )

    def repeated_pld(
        self,
        count: int,
        *,
        discretization: float | None = None,
        log_x_mass_truncation_bound: float | None = None,
        pessimistic_estimate: bool | None = None,
        max_grid_size: int | None = None,
    ) -> Pld:
        return self.proc.approx_at_step(count).pld(
            discretization=discretization,
            log_x_mass_truncation_bound=log_x_mass_truncation_bound,
            pessimistic_estimate=pessimistic_estimate,
            max_grid_size=max_grid_size,
        )

    def __or__(self, other: DpProcess) -> DpProcess:
        if isinstance(other, PerStep) and other.proc is not self.proc:
            raise ValueError(
                "PerStep cannot be composed with a PerStep wrapping a different "
                "process; the per-step PLD is defined only relative to one "
                "whole-process accountant.  Wrap the same ``proc`` object in "
                "both calls — equality is by object identity to avoid silent "
                "merges when strategies exclude privacy-relevant fields from "
                "``__eq__`` (e.g. ``BltStrategy.lr_schedule``)."
            )
        # Explicit base call: zero-arg ``super()`` doesn't work in slotted
        # dataclasses (the @dataclass decorator rebuilds the class for slots,
        # invalidating the implicit ``__class__`` cell).
        return DpProcess.__or__(self, other)


def per_step(proc: DpFtrlProcess) -> PerStep:
    """Wrap a whole-process DP-FTRL accountant as a composable per-step atom.

    Lets DP-FTRL training loops use the ``acc |= step`` accountant idiom
    (as in DP-SGD trainers) while still getting strategy-aware K-step
    privacy accounting — the wrapped ``proc.approx_at_step(K)`` materialises
    the true K-step PLD when the :class:`Repeated` node is asked for it.

    Args:
        proc: A whole-process DP-FTRL accountant (CyclicPoisson, BMinSep,
            BallsInBins, ...).

    Example::

        proc = ftrl_acc.balls_in_bins(
            ftrl_acc.mf_gaussian(1.0, strategy),
            num_bins=100, n_steps=1000,
        )
        step = ftrl_acc.per_step(proc)

        acc = Accountant()
        for _ in range(1000):
            acc |= step
        eps = acc.epsilon_at(1e-5)
    """
    if not isinstance(proc, DpFtrlProcess):
        raise TypeError(
            f"per_step() requires a DpFtrlProcess, got {type(proc).__name__}."
        )
    return PerStep(proc=proc)
