"""DP-FTRL process base — whole-run accountant with a per-step PLD worker.

DP-FTRL accountants describe whole training runs.  Unlike DP-SGD, where
a per-step factory composes externally with ``* num_steps``, every
DP-FTRL factory in :mod:`opaque.dpftrl.accounting` returns a
:class:`DpProcess` spanning the full horizon ``n_steps``.

To compose a DP-FTRL accountant into the standard ``acc |= step``
accountant loop, wrap it with :func:`opaque.dpftrl.accounting.per_step`
— a thin adapter that exposes one step of the run as an algebraic atom.
``Repeated(per_step(proc), K)`` materialises as the true K-step PLD via
:meth:`DpFtrlProcess._pld_at_horizon`, which evaluates the deployed
N-step mechanism on its first K rows of output (the post-processing
inequality on the K-prefix projection gives
``ε(K) ≤ ε(N)`` and monotonicity in K).

Subclass contract:

- Subclasses MUST be frozen dataclasses with an ``n_steps: int`` field.
- Subclasses MUST implement :meth:`_pld_at_horizon` — the K-step PLD
  using N-tuned strategy quantities — and inherit the default
  :meth:`pld` which delegates ``_pld_at_horizon(self.n_steps)``.
- Within an "atomic unit" (band, epoch, ...) the K-step ε plateaus at
  the next-multiple boundary; ``_pld_at_horizon`` rounds up internally
  and the upper-bounding semantics are preserved.
"""

from __future__ import annotations

from abc import abstractmethod

from opaque.api.accounting.core._base import DpProcess, Pld

__all__ = ["DpFtrlProcess"]


class DpFtrlProcess(DpProcess):
    """Whole-process accountant for DP-FTRL.

    Subclasses MUST be frozen dataclasses with an ``n_steps: int`` field
    and implement :meth:`_pld_at_horizon`.  ``pld()`` delegates to
    ``_pld_at_horizon(self.n_steps)`` by default; subclasses with
    extra mechanism-specific PLD knobs (Monte Carlo seed / sample count
    on b-min-sep and balls-in-bins) override ``pld()`` to forward those
    knobs to ``_pld_at_horizon``.
    """

    n_steps: int  # required dataclass field on every subclass

    @abstractmethod
    def _pld_at_horizon(self, n_steps: int, **kwargs) -> Pld:
        """K-step PLD using N-tuned strategy quantities (the "K-prefix" bound).

        ``n_steps`` may be any positive integer ``≤ self.n_steps`` and
        is rounded up to the implementation's natural granularity (one
        band / one epoch / 1).  Strategy coefficients, sensitivity, and
        gram-matrix data are evaluated at ``self.n_steps`` — the N-tuned
        deployed mechanism — and ``n_steps`` only changes the number of
        rows / compositions / epochs the privacy bound is evaluated on.

        For analytic PLDs, the post-processing inequality on the K-prefix
        projection implies ``ε(_pld_at_horizon(K)) ≤ ε(self)`` and
        monotonicity in K. Monte Carlo implementations return empirical
        point estimates for which that reported-epsilon ordering is not
        guaranteed.

        Raises:
            ValueError: If ``n_steps`` is outside ``[1, self.n_steps]``.
        """

    def pld(
        self,
        *,
        discretization: float | None = None,
        log_x_mass_truncation_bound: float | None = None,
        max_grid_size: int | None = None,
    ) -> Pld:
        """Return the full-horizon PLD.

        Monte Carlo subclasses return empirical point estimates rather than
        upper confidence bounds; conservative grid discretization does not
        account for Monte Carlo sampling error.
        """
        return self._pld_at_horizon(
            self.n_steps,
            discretization=discretization,
            log_x_mass_truncation_bound=log_x_mass_truncation_bound,
            max_grid_size=max_grid_size,
        )
