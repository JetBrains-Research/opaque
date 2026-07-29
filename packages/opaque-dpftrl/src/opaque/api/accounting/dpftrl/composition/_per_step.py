"""``PerStep`` — wrap a whole-process DP-FTRL accountant as a per-step atom.

DP-FTRL accountants describe whole training runs; their K-step privacy
cost is not the K-fold composition of a single-step PLD because the
strategy is tuned for a particular horizon.  ``PerStep`` adapts a
:class:`DpFtrlProcess` to the :class:`Accountant`'s ``acc |= step``
idiom by overriding :meth:`DpProcess.repeated_pld` to call the wrapped
process's :meth:`DpFtrlProcess._pld_at_horizon` — so a :class:`Repeated`
node ``PerStep(proc) * K`` materialises as the **true** K-step PLD of
the deployed N-step mechanism (evaluated on its first K rows of
output), rather than the K-fold self-composition of a single-step PLD.

By the post-processing inequality on the K-prefix projection,
``ε(per_step(proc) * K) ≤ ε(proc)`` and is monotone in K.  At
``K == proc.n_steps`` the per-step formulation is bit-exact equal to
``proc.epsilon_at(δ)``.
"""

from __future__ import annotations

import functools
from dataclasses import dataclass

from opaque.api.accounting.core._base import DpProcess, Pld
from opaque.api.accounting.dpftrl._base import DpFtrlProcess


@dataclass(frozen=True, slots=True)
class PerStep(DpProcess):
    """One atomic step of a :class:`DpFtrlProcess`, composable with ``|`` / ``*``.

    Wraps a whole-process DP-FTRL accountant so it behaves as a
    per-step factory under the accountant algebra:

    - ``PerStep(proc).pld()``           ≡ ``proc._pld_at_horizon(1)``
    - ``Repeated(PerStep(proc), K).pld()`` ≡ ``proc._pld_at_horizon(K)``

    The :class:`Repeated` node is built by the standard ``__mul__`` /
    ``__or__`` merge optimizations — ``PerStep`` simply overrides
    :meth:`DpProcess.repeated_pld` so the K-fold materialisation calls
    the wrapped process's K-prefix worker.

    Heterogeneous mixing of ``PerStep`` instances with different
    underlying processes is rejected: the per-step PLD only makes sense
    relative to one whole-process accountant.
    """

    proc: DpFtrlProcess

    @functools.lru_cache(maxsize=8)
    def pld(
        self,
        *,
        discretization: float | None = None,
        log_x_mass_truncation_bound: float | None = None,
        max_grid_size: int | None = None,
    ) -> Pld:
        return self.proc._pld_at_horizon(
            1,
            discretization=discretization,
            log_x_mass_truncation_bound=log_x_mass_truncation_bound,
            max_grid_size=max_grid_size,
        )

    def repeated_pld(
        self,
        count: int,
        *,
        discretization: float | None = None,
        log_x_mass_truncation_bound: float | None = None,
        max_grid_size: int | None = None,
    ) -> Pld:
        if count <= 0:
            raise ValueError(f"count ({count}) must be >= 1")
        if count > self.proc.n_steps:
            raise ValueError(
                f"count ({count}) exceeds n_steps ({self.proc.n_steps}); "
                f"{type(self.proc).__name__} is undefined beyond its declared "
                f"horizon."
            )
        return self.proc._pld_at_horizon(
            count,
            discretization=discretization,
            log_x_mass_truncation_bound=log_x_mass_truncation_bound,
            max_grid_size=max_grid_size,
        )

    def __or__(self, other: DpProcess) -> DpProcess:
        if isinstance(other, PerStep) and other.proc != self.proc:
            raise ValueError(
                "PerStep cannot be composed with a PerStep wrapping a different "
                "process; the per-step PLD is defined only relative to one "
                "whole-process accountant."
            )
        # Explicit base call: zero-arg ``super()`` doesn't work in slotted
        # dataclasses (the @dataclass decorator rebuilds the class for slots,
        # invalidating the implicit ``__class__`` cell).
        return DpProcess.__or__(self, other)


def per_step(proc: DpFtrlProcess) -> PerStep:
    """Wrap a whole-process DP-FTRL accountant as a composable per-step atom.

    Lets DP-FTRL training loops use the ``acc |= step`` accountant idiom
    (as in DP-SGD trainers) while still getting strategy-aware K-step
    privacy accounting — ``Repeated(PerStep(proc), K).pld()`` materialises
    via ``proc._pld_at_horizon(K)``: the K-prefix projection of the
    deployed N-step mechanism (post-processing-bounded above by
    ``proc.epsilon_at(δ)``).

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
