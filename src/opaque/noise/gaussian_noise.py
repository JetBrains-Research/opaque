"""Gaussian noise generation for differential privacy.

This module provides a higher-order function for adding calibrated Gaussian noise
to gradients in DP-SGD (Differentially Private Stochastic Gradient Descent).

The API returns ``(noise_fn, state)`` where state is always immutable:

    >>> noise_fn, state = gaussian_noise(stddev=1.0, seed=42)
    >>> noisy_grads, state = noise_fn(grads, state)

Auto-distributed support:
- When distributed mode is detected (via torch.distributed.is_initialized()), and
  synchronized="auto" (default), all devices automatically use the SAME seed for
  synchronized noise.
- This prevents model divergence while keeping the API simple - just pass seed=None.
"""

import dataclasses
from collections.abc import Callable
from typing import Any

import torch

from opaque.distributed import get_rank, is_distributed
from opaque.utils.pytree import tree_map


@dataclasses.dataclass(frozen=True)
class GaussianNoiseState:
    """Immutable state for Gaussian noise generation.

    Wraps a ``torch.Generator`` for reproducible noise with additional metadata
    for per-step RNG derivation (Phase 2). Although the generator itself is
    mutable internally, the state object is frozen and users must always pass
    back the state they received.

    Attributes:
        rng_state: Random number generator.
        seed: Base seed used for initialization (None if unseeded).
        synchronized: Whether noise is synchronized across devices in distributed mode.
        step_counter: Number of noise_fn calls made (for future per-step derivation).
    """

    rng_state: torch.Generator
    seed: int | None = None
    synchronized: bool = False
    step_counter: int = 0


def _create_rng_state(
    seed: int | None,
    synchronized: str | bool,
) -> tuple[torch.Generator, int | None, bool]:
    """Create RNG state with appropriate seed for current distributed configuration.

    Args:
        seed: Base seed. ``None`` for unseeded (random), ``int`` for deterministic.
        synchronized: Synchronization mode:
            - ``"auto"``: Auto-detect distributed mode and sync if detected
            - ``True``: Force synchronized noise (same seed on all devices)
            - ``False``: Independent noise per device (seed shifts by rank)

    Returns:
        Tuple of (generator, resolved_seed, is_synchronized):
            - generator: Configured torch.Generator
            - resolved_seed: Actual seed used (None if unseeded)
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

    # Determine effective seed
    if seed is None:
        if is_sync and is_distributed():
            # Auto-sync in distributed: use fixed seed (0) across all devices
            effective_seed = 0
        else:
            # Unseeded: non-reproducible
            gen = torch.Generator()
            gen.seed()
            return gen, None, is_sync
    else:
        if not isinstance(seed, int):
            raise TypeError(f"seed must be None or int, got {type(seed)}")
        if is_sync:
            # Synchronized: use same seed on all devices
            effective_seed = seed
        else:
            # Independent: shift seed by rank for diversity
            rank = get_rank() if is_distributed() else 0
            effective_seed = seed + rank

    gen = torch.Generator().manual_seed(effective_seed)
    return gen, (seed if seed is not None else 0 if is_sync else None), is_sync


def gaussian_noise(
    stddev: float,
    *,
    seed: int | None = None,
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
        seed: Base seed for RNG:
            - ``None``: Unseeded in single-device mode, fixed seed (0) in distributed
              mode with ``synchronized="auto"``
            - ``int``: Explicit seed for reproducibility
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
        >>>
        >>> # When distributed is detected, automatically synchronizes noise across devices
        >>> noise_fn, state = gaussian_noise(stddev=1.1)  # No seed needed!
        >>> grads = torch.zeros(10)
        >>> noisy_grads, state = noise_fn(grads, state)

    Example (reproducible with explicit seed):
        >>> # Provide explicit seed for deterministic training
        >>> noise_fn, state = gaussian_noise(stddev=1.1, seed=42)
        >>> noisy_grads, state = noise_fn(grads, state)

    Example (independent noise per device):
        >>> # Each device gets different noise (seed + rank)
        >>> noise_fn, state = gaussian_noise(stddev=1.1, seed=42, synchronized=False)
        >>> noisy_grads, state = noise_fn(grads, state)
    """
    if stddev < 0:
        raise ValueError(f"stddev must be non-negative, got {stddev}")

    gen, resolved_seed, is_sync = _create_rng_state(seed, synchronized)
    state = GaussianNoiseState(
        rng_state=gen,
        seed=resolved_seed,
        synchronized=is_sync,
        step_counter=0,
    )

    if stddev == 0:

        def zero_noise_fn(grads, st):
            return grads, st

        return zero_noise_fn, state

    def noise_fn(grads, st):
        """Add Gaussian noise to gradients."""
        g = st.rng_state

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
            rng_state=g,
            seed=st.seed,
            synchronized=st.synchronized,
            step_counter=st.step_counter + 1,
        )

    return noise_fn, state


__all__ = ["gaussian_noise", "GaussianNoiseState"]
