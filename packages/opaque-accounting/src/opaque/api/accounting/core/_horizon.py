"""Whole-horizon DP process with prefix privacy accounting."""

from __future__ import annotations

from abc import abstractmethod

from opaque.api.accounting.core._base import DpProcess, Pld

__all__ = ["DpHorizonProcess"]


class DpHorizonProcess(DpProcess):
    """A DP process defined over a declared sequence of ``n_steps`` releases.

    Subclasses are frozen dataclasses with an ``n_steps`` field and implement
    :meth:`pld_at`. The returned PLD may be exact or a conservative prefix
    bound at the mechanism's natural granularity.
    """

    n_steps: int

    @abstractmethod
    def pld_at(self, n_steps: int, **kwargs) -> Pld:
        """Return a privacy-loss bound for the first ``n_steps`` releases."""

    def pld(
        self,
        *,
        discretization: float | None = None,
        log_x_mass_truncation_bound: float | None = None,
        max_grid_size: int | None = None,
    ) -> Pld:
        """Return the full-horizon PLD."""
        return self.pld_at(
            self.n_steps,
            discretization=discretization,
            log_x_mass_truncation_bound=log_x_mass_truncation_bound,
            max_grid_size=max_grid_size,
        )
