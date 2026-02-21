"""Gaussian noise generation for differential privacy.

This module provides a higher-order function for adding calibrated Gaussian noise
to gradients in DP-SGD (Differentially Private Stochastic Gradient Descent).

The API returns ``(noise_fn, state)`` where state is always immutable:

    >>> from opaque.random import key
    >>> noise_fn, state = gaussian_noise(stddev=1.0, key=key(42))
    >>> noisy_grads, state = noise_fn(grads, state)

Auto-distributed support:
- When distributed mode is detected (via torch.distributed.is_initialized()), and
  synchronized="auto" (default), all devices automatically use the SAME seed for
  synchronized noise.
- This prevents model divergence while keeping the API simple.
"""

import dataclasses
from collections.abc import Callable
from typing import Any

import torch

from opaque.distributed import get_rank, is_distributed
from opaque.random import RngKey, generator_from_key
from opaque.random import (
    fold_in as rng_fold_in,
)
from opaque.utils.pytree import tree_map


@dataclasses.dataclass(frozen=True)
class GaussianNoiseState:
    """Immutable state for Gaussian noise generation.

    Holds immutable metadata plus a JAX-style key for deterministic per-step
    derivation. Noise for step ``t`` is generated from ``fold_in(rng_key, t)``.

    Attributes:
        seed: Canonical seed metadata derived from the base key.
        synchronized: Whether noise is synchronized across devices in distributed mode.
        step_counter: Number of noise_fn calls made.
        rng_key: Immutable RNG key for deterministic derivation.
    """

    seed: int
    synchronized: bool
    step_counter: int
    rng_key: RngKey


def _create_rng_state(
    key: RngKey,
    synchronized: str | bool,
) -> tuple[RngKey, int, bool]:
    """Create RNG state with appropriate seed for current distributed configuration.

    Args:
        key: Explicit RNG key (required API).
        synchronized: Synchronization mode:
            - ``"auto"``: Auto-detect distributed mode and sync if detected
            - ``True``: Force synchronized noise (same seed on all devices)
            - ``False``: Independent noise per device (seed shifts by rank)

    Returns:
        Tuple of (base_key, resolved_seed, is_synchronized):
            - base_key: Resolved RNG key
            - resolved_seed: Canonical seed metadata value
            - is_synchronized: Whether noise is synchronized across devices
    """
    # Resolve synchronized mode
    if synchronized == "auto":
        is_sync = is_distributed()
    elif isinstance(synchronized, bool):
        is_sync = synchronized
    else:
        raise ValueError(
            f"synchronized must be 'auto', True, or False, got {synchronized!r}"
        )

    if not isinstance(key, RngKey):
        raise TypeError(f"key must be RngKey, got {type(key)}")

    rank = get_rank() if is_distributed() else 0

    base_key = key if is_sync else rng_fold_in(key, f"rank:{rank}")
    return base_key, int(base_key.seed), is_sync


def gaussian_noise(
    stddev: float,
    *,
    key: RngKey,
    synchronized: str | bool = "auto",
) -> tuple[
    Callable[[Any, GaussianNoiseState], tuple[Any, GaussianNoiseState]],
    GaussianNoiseState,
]:
    """Create a Gaussian noise function with immutable state.

    Returns ``(noise_fn, state)`` where ``noise_fn`` adds calibrated Gaussian
    noise N(0, stddev²) to gradients and returns updated state.

    **Automatic distributed support**: When ``synchronized="auto"`` (default) and
    distributed mode is detected (via ``torch.distributed.is_initialized()``),
    automatically uses the SAME seed across all devices. This provides synchronized
    noise for model convergence.

    Args:
        stddev: Standard deviation of Gaussian noise
            (usually ``noise_multiplier * clip_norm``).
        key: Explicit RNG key for deterministic, functional randomness.
        synchronized: Synchronization mode for distributed training:
            - ``"auto"`` (default): Auto-detect and sync if distributed
            - ``True``: Force synchronized noise (same seed across devices)
            - ``False``: Independent noise per device (seed + rank offset)

    Returns:
        A tuple ``(noise_fn, state)`` where:

        - ``noise_fn(grads, state) -> (noisy_grads, new_state)``
        - ``state`` is a :class:`GaussianNoiseState`

    Example (typical use - auto-detected synchronization):
        >>> import torch
        >>> from opaque.noise import gaussian_noise
        >>> from opaque.random import key
        >>>
        >>> # When distributed is detected, automatically synchronizes noise across devices
        >>> noise_fn, state = gaussian_noise(stddev=1.1, key=key(0))
        >>> grads = torch.zeros(10)
        >>> noisy_grads, state = noise_fn(grads, state)

    Example (reproducible with explicit key):
        >>> # Provide explicit key for deterministic training
        >>> from opaque.random import key
        >>> noise_fn, state = gaussian_noise(stddev=1.1, key=key(42))
        >>> noisy_grads, state = noise_fn(grads, state)

    Example (independent noise per device):
        >>> # Each device gets different noise (key folded with rank)
        >>> from opaque.random import key
        >>> noise_fn, state = gaussian_noise(stddev=1.1, key=key(42), synchronized=False)
        >>> noisy_grads, state = noise_fn(grads, state)
    """
    if stddev < 0:
        raise ValueError(f"stddev must be non-negative, got {stddev}")

    base_key, resolved_seed, is_sync = _create_rng_state(key, synchronized)
    state = GaussianNoiseState(
        seed=resolved_seed,
        synchronized=is_sync,
        step_counter=0,
        rng_key=base_key,
    )

    if stddev == 0:

        def zero_noise_fn(grads, st):
            return grads, st

        return zero_noise_fn, state

    def noise_fn(grads, st):
        """Add Gaussian noise to gradients."""
        step_key = rng_fold_in(st.rng_key, st.step_counter)
        g = generator_from_key(step_key)

        def add_noise_to_tensor(tensor: torch.Tensor) -> torch.Tensor:
            # torch.Generator is CPU-only; generate on CPU and move if needed
            noise = torch.randn(
                tensor.shape,
                dtype=tensor.dtype,
                generator=g,
            ).to(device=tensor.device)
            return tensor + noise * stddev

        noisy = tree_map(add_noise_to_tensor, grads)

        # Return updated state with incremented step counter
        return noisy, GaussianNoiseState(
            seed=st.seed,
            synchronized=st.synchronized,
            step_counter=st.step_counter + 1,
            rng_key=st.rng_key,
        )

    return noise_fn, state


__all__ = ["gaussian_noise", "GaussianNoiseState"]
