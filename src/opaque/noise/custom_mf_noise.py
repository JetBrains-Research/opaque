"""Custom matrix factorization noise mechanism.

Bring-your-own-matrix entry point for DP-FTRL. Accepts a dense tensor or
``StreamingMatrix`` representing the noising matrix C^{-1} and returns
``(noise_fn, state)`` ready for a training loop.

For pre-built strategies, use ``band_mf_noise``, ``blt_mf_noise``,
``dense_mf_noise``, or ``identity_mf_noise`` instead.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import torch

from opaque.noise.gaussian_noise import _create_rng_state
from opaque.random import RngKey
from opaque.noise.matrix_factorization.noise import (
    MFNoiseState,
    _matrix_factorization_noise,
)


def custom_mf_noise(
    grad_template: Any,
    noising: torch.Tensor | Any,
    *,
    stddev: float,
    key: RngKey | None = None,
    synchronized: str | bool = "auto",
    dtype: torch.dtype | None = None,
) -> tuple[
    Callable[[Any, MFNoiseState], tuple[Any, MFNoiseState]],
    MFNoiseState,
]:
    """Create a custom matrix factorization noise mechanism.

    This is the bring-your-own-matrix entry point for DP-FTRL. The
    ``noising`` argument represents C^{-1} in the factorization A = B @ C.

    For pre-built strategies, use :func:`band_mf_noise`,
    :func:`blt_mf_noise`, :func:`dense_mf_noise`, or
    :func:`identity_mf_noise` instead.

    Args:
        grad_template: A pytree with the same structure and shapes as the
            gradients that will be passed to ``noise_fn``. Used to
            initialize internal state.
        noising: Either a dense 2D tensor (``torch.Tensor``) or a
            ``StreamingMatrix`` representing C^{-1}.
        stddev: Standard deviation for the base noise.
                key: Optional RNG key (primary API) for explicit functional randomness.
                        - ``None``: Non-deterministic in single-device mode; fixed key in
                            distributed mode with ``synchronized="auto"``
                        - ``RngKey``: Explicit key for reproducibility
        synchronized: Synchronization mode for distributed training:
            - ``"auto"`` (default): Auto-detect and sync if distributed
            - ``True``: Force synchronized noise (same seed across devices)
            - ``False``: Independent noise per device (seed + rank offset)
        dtype: Optional dtype for intermediate noise computation.

    Returns:
        A tuple ``(noise_fn, state)`` where:

        - ``noise_fn(grads, state) -> (noisy_grads, new_state)``
        - ``state`` is a :class:`~opaque.noise.matrix_factorization.noise.MFNoiseState`

    Example:
        >>> import torch
        >>> from opaque.noise import custom_mf_noise
        >>> from opaque.random import key
        >>> from opaque.noise.matrix_factorization import identity
        >>>
        >>> grad_template = torch.zeros(10)
        >>> noise_fn, state = custom_mf_noise(
        ...     grad_template, identity(), stddev=1.0, key=key(42),
        ... )
        >>> noisy_grad, state = noise_fn(torch.zeros(10), state)
    """
    gen, resolved_seed, is_sync = _create_rng_state(key, synchronized)
    return _matrix_factorization_noise(
        grad_template, noising, stddev=stddev, gen=gen, seed=resolved_seed, synchronized=is_sync, dtype=dtype
    )


__all__ = ["custom_mf_noise"]
