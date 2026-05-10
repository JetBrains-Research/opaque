"""BandMF strategy — banded Toeplitz MF mechanism.

Computes optimized banded Toeplitz coefficients and wraps their
sensitivity / streaming matrix into a frozen strategy dataclass.

Use ``mf_noise(band_mf_strategy(...), ...)`` to create the noise function.

References:
    - BandMF: https://arxiv.org/abs/2306.08153
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass

import torch

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

    When ``lr_schedule=None``, assumes constant η=1 everywhere (the
    original behavior).

    Special cases:
        β = 0.0 → [η_0, 0, 0, ...] (identity workload)
        β = 0.95, lr=None → [1, 0.95, 0.9025, ...] (momentum-SGD)
        β = 1.0, lr=None → [1, 1, 1, ...] (prefix-sum workload, true FTRL)

    Args:
        momentum: Polyak momentum β (must be >= 0).
        n: Number of steps.
        lr_schedule: Optional per-step learning rates, shape [n].
            If None, assumes constant LR (implicit η=1).

    Raises:
        ValueError: If momentum < 0 or lr_schedule has wrong length.
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


__all__ = ["BandMfStrategy", "band_mf_strategy"]


# ---------------------------------------------------------------------------
# Strategy dataclass and factory
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class BandMfStrategy:
    """BandMF banded Toeplitz strategy.

    BandMF uses cyclic Poisson amplification (not BnB), so
    ``gram_matrix`` is always ``None``.
    """

    sensitivity: float
    coefficients: tuple[float, ...]
    gram_matrix: None = None
    _streaming_matrix: StreamingMatrix | None = None
    _n_steps: int = 0
    _bands: int = 0
    _lr_schedule: torch.Tensor | None = None

    @property
    def _max_column_norm(self) -> float:
        """Max column L2 norm ‖C‖_{1→2} (equals sensitivity for BandMF)."""
        return self.sensitivity

    @property
    def num_groups(self) -> int:
        """Number of independent cyclic groups: ceil(n_steps / bands)."""
        return math.ceil(self._n_steps / self._bands) if self._bands > 0 else 0


def band_mf_strategy(
    n_steps: int,
    bands: int,
    *,
    momentum: float = 1.0,
    lr_schedule: torch.Tensor | None = None,
) -> BandMfStrategy:
    """Create a BandMF strategy by optimizing banded Toeplitz coefficients.

    BandMF uses cyclic Poisson amplification, not BnB, so ``gram_matrix``
    is always ``None``.

    Args:
        n_steps: Number of training iterations.
        bands: Number of bands in the Toeplitz matrix.
        momentum: Polyak momentum coefficient (default 1.0 = prefix-sum).
        lr_schedule: Optional per-step learning rates, shape [n_steps].

    Returns:
        A :class:`BandMfStrategy` with optimized coefficients.
    """
    if n_steps < 1:
        raise ValueError(f"n_steps must be >= 1, got {n_steps}")
    if bands < 1 or bands > n_steps:
        raise ValueError(f"bands must be in [1, n_steps={n_steps}], got {bands}")

    workload_coef = _momentum_workload_coef(momentum, n_steps, lr_schedule=lr_schedule)
    coefs = optimize_toeplitz(n_steps, bands, workload_coef=workload_coef)

    sensitivity = float(coefs.norm())
    coefficients = tuple(coefs.tolist())
    streaming = inverse_as_streaming_matrix(coefs)

    return BandMfStrategy(
        sensitivity=sensitivity,
        coefficients=coefficients,
        _streaming_matrix=streaming,
        _n_steps=n_steps,
        _bands=bands,
        _lr_schedule=(
            torch.as_tensor(lr_schedule, dtype=torch.float64).clone()
            if lr_schedule is not None
            else None
        ),
    )
