"""BandMF strategy — banded Toeplitz MF mechanism.

Computes optimized banded Toeplitz coefficients on demand from the
strategy's recipe (``bands``, ``momentum``, ``lr_schedule``) and the
amplifier-supplied ``n_steps``.

References:
    - BandMF: https://arxiv.org/abs/2306.08153
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from functools import lru_cache

import torch

from opaque.api.dpftrl.noise._strategy_codec import register_strategy

from ._streaming_matrix import StreamingMatrix
from ._toeplitz import inverse_as_streaming_matrix
from ._toeplitz import optimize as optimize_toeplitz


def _momentum_workload_coef(
    momentum: float,
    n: int,
    lr_schedule: torch.Tensor | None = None,
) -> torch.Tensor:
    """Compute Toeplitz workload coefficients for momentum-SGD + LR schedule.

    For momentum β and per-step learning rates η_t, the workload matrix W
    has entries W[t,s] = η_t · β^{t-s} for s ≤ t.  The Toeplitz
    coefficients are [η_0, η_1·β, η_2·β², ...].
    """
    if momentum < 0:
        raise ValueError(f"momentum must be >= 0, got {momentum}")
    if momentum == 0.0:
        warnings.warn(
            "momentum=0.0 produces an identity workload — MF noise will "
            "reduce to independent noise with no benefit over standard "
            "Gaussian (DP-SGD). This is useful for testing but not for "
            "production training.",
            stacklevel=3,
        )
        coef = torch.zeros(n, dtype=torch.float64)
        coef[0] = 1.0
        if lr_schedule is not None:
            lr = torch.as_tensor(lr_schedule, dtype=torch.float64)
            coef[0] = lr[0]
        return coef

    base = torch.tensor([momentum**i for i in range(n)], dtype=torch.float64)

    if lr_schedule is not None:
        lr = torch.as_tensor(lr_schedule, dtype=torch.float64)
        if lr.shape[0] != n:
            raise ValueError(f"lr_schedule length ({lr.shape[0]}) must equal n ({n})")
        return lr * base

    return base


# ---------------------------------------------------------------------------
# Cached coefficient computation
# ---------------------------------------------------------------------------


def _lr_key(lr_schedule: torch.Tensor | None) -> tuple[float, ...] | None:
    return None if lr_schedule is None else tuple(lr_schedule.tolist())


@lru_cache(maxsize=32)
def _band_mf_coefficients_cached(
    n_steps: int,
    bands: int,
    momentum: float,
    lr_key: tuple[float, ...] | None,
) -> torch.Tensor:
    """Run the BandMF Toeplitz optimization for the given recipe + horizon."""
    if n_steps < 1:
        raise ValueError(f"n_steps must be >= 1, got {n_steps}")
    if bands < 1 or bands > n_steps:
        raise ValueError(f"bands must be in [1, n_steps={n_steps}], got {bands}")
    lr = torch.tensor(lr_key, dtype=torch.float64) if lr_key is not None else None
    workload_coef = _momentum_workload_coef(momentum, n_steps, lr_schedule=lr)
    return optimize_toeplitz(n_steps, bands, workload_coef=workload_coef)


# ---------------------------------------------------------------------------
# Strategy dataclass and factory
# ---------------------------------------------------------------------------


@register_strategy
@dataclass(frozen=True, slots=True)
class BandMfStrategy:
    """BandMF banded Toeplitz strategy — recipe only.

    All derived data (coefficients, sensitivity, streaming matrix) is
    computed on demand via the strategy methods, keyed on the
    amplifier-supplied ``n_steps``.  No gram matrix: BandMF uses cyclic
    Poisson / b-min-sep amplification, not BnB.
    """

    bands: int
    momentum: float = 1.0
    lr_schedule: torch.Tensor | None = field(default=None, compare=False)

    def coefficients(self, *, n_steps: int, **_) -> torch.Tensor:
        return _band_mf_coefficients_cached(
            n_steps, self.bands, self.momentum, _lr_key(self.lr_schedule)
        )

    def gram_matrix(self, **_) -> tuple[float, ...]:
        # BandMF uses Poisson / b-min-sep amplification, not BnB.
        raise NotImplementedError(
            "BandMfStrategy does not provide a gram matrix (uses Poisson / "
            "b-min-sep amplification, not BnB)."
        )

    def streaming_matrix(self, *, n_steps: int, **_) -> StreamingMatrix:
        coefs = self.coefficients(n_steps=n_steps)
        return inverse_as_streaming_matrix(coefs)

    def sensitivity(self, *, n_steps: int, **_) -> float:
        coefs = self.coefficients(n_steps=n_steps)
        return float(coefs.norm())


def band_mf_strategy(
    *,
    bands: int,
    momentum: float = 1.0,
    lr_schedule: torch.Tensor | None = None,
) -> BandMfStrategy:
    """Create a BandMF strategy recipe.

    The factory only validates static recipe knobs.  The Toeplitz
    optimization happens lazily when a strategy method is first called
    with an amplifier-supplied ``n_steps``.

    Args:
        bands: Number of bands in the Toeplitz matrix (>= 1).
        momentum: Polyak momentum coefficient (default 1.0 = prefix-sum).
        lr_schedule: Optional per-step learning rates (length must match
            the amplifier's ``n_steps`` at use time).

    Returns:
        A :class:`BandMfStrategy` recipe.
    """
    if bands < 1:
        raise ValueError(f"bands must be >= 1, got {bands}")
    return BandMfStrategy(
        bands=bands,
        momentum=momentum,
        lr_schedule=(
            torch.as_tensor(lr_schedule, dtype=torch.float64).clone()
            if lr_schedule is not None
            else None
        ),
    )


__all__ = ["BandMfStrategy", "band_mf_strategy"]
