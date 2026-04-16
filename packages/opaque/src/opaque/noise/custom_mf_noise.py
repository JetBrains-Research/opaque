"""Custom matrix factorization noise mechanism.

Bring-your-own-matrix entry point for DP-FTRL. Accepts a dense tensor or
``StreamingMatrix`` representing the noising matrix C^{-1} and returns
``(noise_fn, state)`` ready for a training loop.

For pre-built strategies, use the strategy factories (``band_mf_strategy``,
``blt_strategy``, etc.) with ``mf_noise`` instead.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import torch

from opaque.noise.mf._engine import (
    MFNoiseState,
    _matrix_factorization_noise,
)
from opaque.random import RngKey


def custom_mf_noise(
    grad_template: Any,
    noising: torch.Tensor | Any,
    *,
    stddev: float,
    key: RngKey,
    dtype: torch.dtype | None = None,
) -> tuple[
    Callable[[Any, MFNoiseState], tuple[Any, MFNoiseState]],
    MFNoiseState,
]:
    """Create a custom matrix factorization noise mechanism.

    This is the bring-your-own-matrix entry point for DP-FTRL. The
    ``noising`` argument represents C^{-1} in the factorization A = B @ C.

    For pre-built strategies, use :func:`~opaque.noise.mf.mf_noise` with a
    strategy factory (e.g. ``band_mf_strategy``, ``identity_strategy``) instead.

    Args:
        grad_template: A pytree with the same structure and shapes as the
            gradients that will be passed to ``noise_fn``.
        noising: Either a dense 2D tensor (``torch.Tensor``) or a
            ``StreamingMatrix`` representing C^{-1}.
        stddev: Standard deviation for the base noise.
        key: Explicit RNG key for deterministic, functional randomness.
        dtype: Optional dtype for intermediate noise computation.

    Returns:
        A tuple ``(noise_fn, state)``.

    Example:
        >>> import torch
        >>> from opaque.noise import custom_mf_noise
        >>> from opaque.noise.mf._streaming_matrix import identity
        >>> from opaque.random import key
        >>>
        >>> grad_template = torch.zeros(10)
        >>> noise_fn, state = custom_mf_noise(
        ...     grad_template, identity(), stddev=1.0, key=key(42),
        ... )
        >>> noisy_grad, state = noise_fn(torch.zeros(10), state)
    """
    return _matrix_factorization_noise(
        grad_template,
        noising,
        stddev=stddev,
        key=key,
        dtype=dtype,
    )


__all__ = ["custom_mf_noise"]
