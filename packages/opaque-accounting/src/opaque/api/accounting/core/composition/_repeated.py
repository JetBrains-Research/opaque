"""Homogeneous k-fold self-composition of a DP process."""

from __future__ import annotations

from dataclasses import dataclass

from opaque.api.accounting.core._base import DpProcess, Pld
from opaque.api.accounting.core._pld_cache import pld_cache


@dataclass(frozen=True, slots=True, eq=False)
class Repeated(DpProcess):
    """Homogeneous k-fold self-composition."""

    inner: DpProcess
    count: int

    def __hash__(self) -> int:
        # Iterative tree walk — depth bounded by heap, not stack.
        # See ``_iter_hash``.
        from ._iter_hash import iter_hash

        return iter_hash(self)

    def __eq__(self, other: object) -> bool:
        # Iterative tree walk (dataclass semantics preserved) — depth
        # bounded by heap, not stack.  See ``_iter_eq``.
        if not isinstance(other, DpProcess):
            return NotImplemented
        from ._iter_eq import iter_eq

        return iter_eq(self, other)

    def __repr__(self) -> str:
        # Iterative tree walk, string-identical to the dataclass repr.
        from ._iter_repr import iter_repr

        return iter_repr(self)

    def _leaf_and_count(self) -> tuple[DpProcess, int]:
        return (self.inner, self.count)

    def _pld_cache_key(self, *, n_steps: int | None = None) -> tuple[object, ...]:
        from ._iter_cache_key import iter_cache_key

        return iter_cache_key(self, n_steps=n_steps)

    @pld_cache(maxsize=8)
    def pld(
        self,
        *,
        discretization: float | None = None,
        log_x_mass_truncation_bound: float | None = None,
        max_grid_size: int | None = None,
        max_conv_grid: int | None = None,
        num_mc_samples: int | None = None,
        seed: int | None = None,
    ) -> Pld:
        return self.inner.repeated_pld(
            self.count,
            discretization=discretization,
            log_x_mass_truncation_bound=log_x_mass_truncation_bound,
            max_grid_size=max_grid_size,
            max_conv_grid=max_conv_grid,
            num_mc_samples=num_mc_samples,
            seed=seed,
        )
