"""Whole-horizon DP process marker."""

from __future__ import annotations

from abc import abstractmethod

from opaque.api.accounting.core._base import DpProcess, Pld

__all__ = ["DpHorizonProcess"]


class DpHorizonProcess(DpProcess):
    """A complete fixed-horizon process that cannot be accounted by prefixes.

    Subclasses implement the ordinary :meth:`DpProcess.pld` contract for their
    full declared ``n_steps`` and are lifecycle markers for trainer integrations.
    """

    n_steps: int

    @abstractmethod
    def pld(
        self,
        *,
        discretization: float | None = None,
        log_x_mass_truncation_bound: float | None = None,
        max_grid_size: int | None = None,
        max_conv_grid: int | None = None,
        seed: int | None = None,
        mc_resolution: float | None = None,
        mc_failure_probability: float | None = None,
    ) -> Pld:
        """Return the privacy-loss distribution for the complete horizon."""
