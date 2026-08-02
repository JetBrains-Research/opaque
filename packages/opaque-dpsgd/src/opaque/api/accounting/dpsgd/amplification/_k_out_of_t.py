"""Whole-horizon global k-out-of-t random allocation."""

from __future__ import annotations

import functools
from dataclasses import dataclass
from typing import TYPE_CHECKING

from opaque.api.accounting.core import _native
from opaque.api.accounting.core._horizon import DpHorizonProcess
from opaque.api.accounting.core.mechanisms._nonprivate import NonPrivate
from opaque.api.accounting.dpsgd.mechanisms._adaclip import AdaClip
from opaque.api.accounting.dpsgd.mechanisms._gaussian import Gaussian

if TYPE_CHECKING:
    from opaque.api.accounting.core._base import Pld

_Inner = Gaussian | AdaClip | NonPrivate


@dataclass(frozen=True, slots=True)
class KOutOfT(DpHorizonProcess):
    """Each record participates in exactly ``k`` uniform steps of the horizon."""

    inner: _Inner
    total_participations: int
    n_steps: int

    def __post_init__(self) -> None:
        if self.n_steps < 1:
            raise ValueError(f"n_steps must be >= 1, got {self.n_steps}")
        if not 1 <= self.total_participations <= self.n_steps:
            raise ValueError(
                "total_participations must be in "
                f"[1, n_steps={self.n_steps}], got {self.total_participations}"
            )

    def _noise_multiplier(self) -> float | None:
        match self.inner:
            case NonPrivate() | Gaussian(noise_multiplier=0):
                return None
            case Gaussian(noise_multiplier=nm):
                return nm
            case AdaClip(inner=NonPrivate() | Gaussian(noise_multiplier=0)):
                return None
            case AdaClip(inner=Gaussian()) as ac:
                return ac.effective_noise_multiplier
            case _:
                raise TypeError(
                    "KOutOfT requires Gaussian, AdaClip(Gaussian), or NonPrivate"
                )

    @functools.lru_cache(maxsize=16)
    def pld_at(
        self,
        n_steps: int,
        *,
        discretization: float | None = None,
        log_x_mass_truncation_bound: float | None = None,
        max_grid_size: int | None = None,
    ) -> Pld:
        if n_steps < 1 or n_steps > self.n_steps:
            raise ValueError(f"n_steps ({n_steps}) must be in [1, {self.n_steps}]")
        from opaque.api.accounting.core.discretization import get_discretization

        config = get_discretization(
            discretization=discretization,
            log_x_mass_truncation_bound=log_x_mass_truncation_bound,
            max_grid_size=max_grid_size,
        ).to_native()
        noise_multiplier = self._noise_multiplier()
        if noise_multiplier is None:
            return _native.non_private_pld(config)
        return _native.k_out_of_t_gaussian_prefix_pld(
            noise_multiplier,
            self.n_steps,
            self.total_participations,
            n_steps,
            config,
        )


def k_out_of_t(
    inner: _Inner,
    *,
    total_participations: int,
    n_steps: int,
) -> KOutOfT:
    """Create a global k-out-of-t whole-horizon process."""
    if not isinstance(inner, (Gaussian, AdaClip, NonPrivate)):
        raise TypeError(
            "k_out_of_t() requires Gaussian, AdaClip, or NonPrivate inner, got "
            f"{type(inner).__name__}."
        )
    return KOutOfT(
        inner=inner,
        total_participations=int(total_participations),
        n_steps=int(n_steps),
    )


__all__ = ["KOutOfT", "k_out_of_t"]
