"""BISR strategy -- Banded Inverse Square Root MF mechanism.

BISR (Kalinin et al., ICLR 2026) generalises lambda-CGD to arbitrary
bandwidth p.  The inverse strategy matrix C^{-1} is banded Toeplitz
with p coefficients.

Use ``mf_noise(bisr_strategy(...), ...)`` to create the noise function.

References:
    - Kalinin, McKenna, Upadhyay, Lampert (2026) "Back to Square Roots"
      https://arxiv.org/abs/2505.12128
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import torch

from opaque_accounting import opaque_accounting as _native
from ._streaming_matrix import StreamingMatrix
from ._toeplitz import (
    inverse_as_streaming_matrix,
)


def _bisr_inverse_coefficients(bandwidth: int, beta: float = 0.0) -> list[float]:
    """Compute BISR inverse square-root coefficients (Lemma 1, arxiv:2505.12128).

    For alpha=1: c_k = sum_{j=0}^{k} r_j * beta^j * r_{k-j}
    where r_0 = 1, r_j = ((j - 3/2) / j) * r_{j-1}.
    """
    r_tilde = [0.0] * bandwidth
    r_tilde[0] = 1.0
    for j in range(1, bandwidth):
        r_tilde[j] = ((j - 1.5) / j) * r_tilde[j - 1]

    if beta == 0.0:
        return r_tilde

    coefs = [0.0] * bandwidth
    for k in range(bandwidth):
        s = 0.0
        for j in range(k + 1):
            s += r_tilde[j] * (beta**j) * r_tilde[k - j]
        coefs[k] = s
    return coefs


def _recover_strategy_coefficients(inv_coefs: Sequence[float], n: int) -> list[float]:
    """Recover strategy matrix first-column entries from C^{-1} coefficients.

    C[0] = 1 / c_0
    C[t] = -sum_{k=1}^{min(t,p-1)} c_k * C[t-k] / c_0   for t > 0

    Returns first n entries.
    """
    p = len(inv_coefs)
    alpha0 = inv_coefs[0]
    col = [0.0] * n
    col[0] = 1.0 / alpha0
    for t in range(1, n):
        s = 0.0
        k_max = min(t, p - 1)
        for k in range(1, k_max + 1):
            s += inv_coefs[k] * col[t - k]
        col[t] = -s / alpha0
    return col


__all__ = ["BisrStrategy", "bisr_strategy"]


# ---------------------------------------------------------------------------
# Strategy dataclass and factory
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class BisrStrategy:
    """BISR (Banded Inverse Square Root) strategy."""

    sensitivity: float
    coefficients: tuple[float, ...]
    gram_matrix: tuple[float, ...] | None = None
    _streaming_matrix: StreamingMatrix | None = None
    _max_column_norm: float = 0.0
    _bandwidth: int = 2
    _n_steps: int = 1
    _min_sep: int = 1
    _max_participations: int | None = 1


def bisr_strategy(
    bandwidth: int,
    n_steps: int,
    min_sep: int,
    max_participations: int | None = 1,
    *,
    normalized: bool = True,
    momentum: float = 0.0,
    coefficients: Sequence[float] | None = None,
) -> BisrStrategy:
    """Create a BISR (Banded Inverse Square Root) strategy.

    Uses Rust functions for sensitivity, Gram matrix, and strategy
    coefficient recovery. Builds a StreamingMatrix for noise generation.

    Args:
        bandwidth: BISR bandwidth p (>= 2).
        n_steps: Total training steps.
        min_sep: Minimum separation between participations.
        max_participations: Maximum participations per user (default 1).
        normalized: Use column-normalized matrix (default True).
        momentum: Optimizer momentum in [0, 1). Default 0.
        coefficients: Explicit C^{-1} coefficients. Default: BISR optimal.

    Returns:
        A :class:`BisrStrategy` with pre-computed Gram matrix.
    """
    if bandwidth < 2:
        raise ValueError(f"bandwidth must be >= 2, got {bandwidth}")
    if n_steps < 1:
        raise ValueError(f"n_steps must be >= 1, got {n_steps}")

    # Compute inverse coefficients
    if coefficients is not None:
        inv_coefs = list(coefficients)
        if len(inv_coefs) != bandwidth:
            raise ValueError(
                f"coefficients length ({len(inv_coefs)}) must equal "
                f"bandwidth ({bandwidth})"
            )
    else:
        inv_coefs = _bisr_inverse_coefficients(bandwidth, beta=momentum)

    # Max column norm ‖C‖_{1→2} (single-participation sensitivity)
    if normalized:
        max_column_norm = 1.0  # all columns have unit norm after normalization
    else:
        mcn_sq = _native.bisr_sensitivity_squared(
            inv_coefs, n_steps, n_steps, 1,
        )
        max_column_norm = float(mcn_sq**0.5)

    # Sensitivity under actual participation pattern
    if normalized:
        sens_sq = _native.bisr_normalized_sensitivity_squared(
            inv_coefs,
            n_steps,
            min_sep,
            max_participations,
        )
    else:
        sens_sq = _native.bisr_sensitivity_squared(
            inv_coefs,
            n_steps,
            min_sep,
            max_participations,
        )
    sensitivity = float(sens_sq**0.5)

    # Strategy coefficients (Rust)
    strategy_coefs = _native.bisr_strategy_coefficients(inv_coefs, bandwidth)
    full_coefs = _recover_strategy_coefficients(inv_coefs, n_steps)
    coefs_tuple = tuple(full_coefs)

    # Gram matrix (Rust)
    gram = _native.bisr_gram_matrix(
        inv_coefs,
        n_steps,
        min_sep,
        max_participations,
        normalized,
    )
    gram_matrix = tuple(gram)

    # Streaming matrix
    streaming = inverse_as_streaming_matrix(
        torch.tensor(strategy_coefs, dtype=torch.float64),
        column_normalize_for_n=n_steps if normalized else None,
    )

    return BisrStrategy(
        sensitivity=sensitivity,
        coefficients=coefs_tuple,
        gram_matrix=gram_matrix,
        _streaming_matrix=streaming,
        _max_column_norm=max_column_norm,
        _bandwidth=bandwidth,
        _n_steps=n_steps,
        _min_sep=min_sep,
        _max_participations=max_participations,
    )
