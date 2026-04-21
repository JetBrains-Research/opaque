"""BLT (Buffered Linear Toeplitz) strategy — multi-epoch MF mechanism.

Computes optimized BLT parameters, sensitivity under participation
patterns, and pre-computed BnB Gram matrix.

Use ``mf_noise(blt_strategy(...), ...)`` to create the noise function.

References:
    - BLT: https://arxiv.org/abs/2404.16706
    - Multi-epoch BLT: https://arxiv.org/abs/2408.08868
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


def _native():
    from opaque.accounting import _native as _n

    return _n


from .band_mf import _momentum_workload_coef
from ._blt_math import (
    inverse_as_streaming_matrix,
    optimize,
    sensitivity_squared as _blt_sensitivity_squared,
    toeplitz_coefs as _blt_toeplitz_coefs,
)
from ._sensitivity import (
    minsep_true_max_participations,
)
from ._streaming_matrix import StreamingMatrix
from ._toeplitz import (
    minsep_sensitivity_squared as _toeplitz_minsep_sensitivity_squared,
)


__all__ = ["BltStrategy", "blt_strategy"]


# ---------------------------------------------------------------------------
# Strategy dataclass and factory
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class BltStrategy:
    """BLT (Buffered Linear Toeplitz) strategy."""

    sensitivity: float
    coefficients: tuple[float, ...]
    gram_matrix: tuple[float, ...] | None = None
    _streaming_matrix: StreamingMatrix | None = None
    _max_column_norm: float = 0.0
    _n_steps: int = 0
    _min_sep: int = 1
    _max_participations: int | None = 1
    _max_buffers: int = 10
    _lr_schedule: torch.Tensor | None = None


def blt_strategy(
    n_steps: int,
    min_sep: int,
    max_participations: int | None = 1,
    *,
    max_buffers: int = 10,
    momentum: float = 1.0,
    lr_schedule: torch.Tensor | None = None,
) -> BltStrategy:
    """Create a BLT strategy by optimizing Buffered Linear Toeplitz parameters.

    Computes sensitivity under the given participation pattern and
    pre-computes the BnB Gram matrix.

    Args:
        n_steps: Number of training iterations.
        min_sep: Minimum separation between participations.
        max_participations: Maximum participations per user (default 1).
        max_buffers: Maximum number of BLT buffers to try (default 10).
        momentum: Polyak momentum coefficient (default 1.0 = prefix-sum).
        lr_schedule: Optional per-step learning rates, shape [n_steps].

    Returns:
        A :class:`BltStrategy` with optimized parameters and Gram matrix.
    """
    if n_steps < 1:
        raise ValueError(f"n_steps must be >= 1, got {n_steps}")

    workload_coef = _momentum_workload_coef(momentum, n_steps, lr_schedule=lr_schedule)
    blt = optimize(
        n=n_steps,
        min_sep=min_sep,
        max_participations=max_participations,
        error="max",
        max_buffers=max_buffers,
        workload_coef=workload_coef,
    )

    # Sensitivity
    # max_column_norm = ‖C‖_{1→2} (single-participation, always needed for JME)
    max_col_norm_sq = _blt_sensitivity_squared(blt, n=n_steps)
    max_column_norm = float(max_col_norm_sq.sqrt())

    k = minsep_true_max_participations(
        n=n_steps,
        min_sep=min_sep,
        max_participations=max_participations,
    )
    if k == 1:
        sensitivity = max_column_norm
    else:
        coefs_tensor = _blt_toeplitz_coefs(blt, n_steps)
        sens_sq = _toeplitz_minsep_sensitivity_squared(
            strategy_coef=coefs_tensor,
            min_sep=min_sep,
            max_participations=max_participations,
            skip_checks=True,
        )
        sensitivity = float(sens_sq.sqrt())

    # Coefficients
    coefs_tensor = _blt_toeplitz_coefs(blt, n_steps)
    coefficients = tuple(coefs_tensor.tolist())

    # Gram matrix
    gram = _native().toeplitz_gram_matrix(
        list(coefficients),
        n_steps,
        min_sep,
        max_participations,
        True,
    )
    gram_matrix = tuple(gram)

    # Streaming matrix
    streaming = inverse_as_streaming_matrix(blt)

    return BltStrategy(
        sensitivity=sensitivity,
        coefficients=coefficients,
        gram_matrix=gram_matrix,
        _streaming_matrix=streaming,
        _max_column_norm=max_column_norm,
        _n_steps=n_steps,
        _min_sep=min_sep,
        _max_participations=max_participations,
        _max_buffers=max_buffers,
        _lr_schedule=(
            torch.as_tensor(lr_schedule, dtype=torch.float64).clone()
            if lr_schedule is not None
            else None
        ),
    )
