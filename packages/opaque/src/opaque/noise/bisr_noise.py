"""BISR noise mechanism — Banded Inverse Square Root via StreamingMatrix.

BISR (Kalinin et al., ICLR 2026) generalises λCGD to arbitrary bandwidth p.
The inverse strategy matrix C^{-1} is banded Toeplitz with p coefficients.

Noise generation uses the existing StreamingMatrix infrastructure:
  1. Compute BISR inverse coefficients (c̃_0, ..., c̃_{p-1})
  2. Recover strategy coefficients (c_0, ..., c_{p-1}) via recurrence
  3. Build StreamingMatrix via inverse_as_streaming_matrix(strategy_coefs)
  4. Delegate to _matrix_factorization_noise()

References:
    - Kalinin, McKenna, Upadhyay, Lampert (2026) "Back to Square Roots"
      https://arxiv.org/abs/2505.12128
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

import torch

from opaque_accounting import opaque_accounting as _native
from opaque.noise.matrix_factorization.noise import (
    MFNoiseState,
    _matrix_factorization_noise,
)
from opaque.noise.matrix_factorization.toeplitz import (
    inverse_as_streaming_matrix,
)
from opaque.random import RngKey


def _bisr_inverse_coefficients(bandwidth: int, beta: float = 0.0) -> list[float]:
    """Compute BISR inverse square-root coefficients (Lemma 1, arxiv:2505.12128).

    For α=1: c̃_k = Σ_{j=0}^{k} r̃_j · β^j · r̃_{k-j}
    where r̃_0 = 1, r̃_j = ((j - 3/2) / j) · r̃_{j-1}.
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

    C[0] = 1 / c̃_0
    C[t] = -Σ_{k=1}^{min(t,p-1)} c̃_k · C[t-k] / c̃_0   for t > 0

    Returns first min(n, enough_for_convergence) entries.
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


def bisr_noise(
    grad_template: Any,
    n_steps: int,
    *,
    stddev: float,
    key: RngKey,
    bandwidth: int,
    momentum: float = 0.0,
    coefficients: Sequence[float] | None = None,
    column_normalize: bool = True,
    dtype: torch.dtype | None = None,
) -> tuple[
    Callable[[Any, MFNoiseState], tuple[Any, MFNoiseState]],
    MFNoiseState,
]:
    """Create BISR noise mechanism via StreamingMatrix.

    Computes BISR inverse coefficients (accounting for optimizer momentum
    in the workload), recovers strategy coefficients, and delegates to
    the standard matrix factorization noise infrastructure.

    For bandwidth=2, this is equivalent to lambda_cgd_noise with λ=1/2.
    For bandwidth>2, the StreamingMatrix maintains a buffer of p-1 vectors.

    Args:
        grad_template: Pytree with same structure/shapes as gradients.
        n_steps: Total training steps.
        stddev: Standard deviation for base noise.
        key: Explicit RNG key.
        bandwidth: BISR bandwidth p (>= 2).
        momentum: Optimizer momentum β (default 0.0). Affects the BISR
            coefficient computation via the workload A_{1,β}.
        coefficients: Explicit C^{-1} coefficients. Default: BISR optimal.
        column_normalize: If True (default), apply column-norm scaling.
        dtype: Optional dtype for intermediate computation.

    Returns:
        (noise_fn, state) tuple.
    """
    if bandwidth < 2:
        raise ValueError(f"bandwidth must be >= 2, got {bandwidth}")

    if coefficients is not None:
        inv_coefs = list(coefficients)
        if len(inv_coefs) != bandwidth:
            raise ValueError(
                f"coefficients length ({len(inv_coefs)}) must equal bandwidth ({bandwidth})"
            )
    else:
        inv_coefs = _bisr_inverse_coefficients(bandwidth, beta=momentum)

    # Recover strategy coefficients from inverse coefficients (via Rust)
    strategy_coefs = _native.bisr_strategy_coefficients(inv_coefs, bandwidth)

    # Build StreamingMatrix: inverse_as_streaming_matrix takes strategy coefs
    # and builds a streaming C^{-1} that applies the banded inverse.
    noising = inverse_as_streaming_matrix(
        torch.tensor(strategy_coefs, dtype=torch.float64),
        column_normalize_for_n=n_steps if column_normalize else None,
    )

    return _matrix_factorization_noise(
        grad_template,
        noising,
        stddev=stddev,
        key=key,
        dtype=dtype,
    )


__all__ = ["bisr_noise"]
