"""Gaussian noise generation for differential privacy.

This module provides a higher-order function for adding calibrated Gaussian noise
to gradients in DP-SGD (Differentially Private Stochastic Gradient Descent).

The API returns ``(noise_fn, state)`` where state is always immutable:

    >>> noise_fn, state = gaussian_noise(stddev=1.0, generator=42)
    >>> noisy_grads, state = noise_fn(grads, state)

Auto-distributed support:
- When distributed mode is detected (via torch.distributed.is_initialized()), and no
  generator is provided, all devices automatically use the SAME seed for synchronized noise.
- This prevents model divergence while keeping the API simple - just pass generator=None.
"""

import dataclasses
from collections.abc import Callable
from typing import Any

import torch

from opaque.distributed import is_distributed
from opaque.utils.pytree import tree_map


@dataclasses.dataclass(frozen=True)
class GaussianNoiseState:
    """Immutable state for Gaussian noise generation.

    Wraps a ``torch.Generator`` for reproducible noise. Although the
    generator itself is mutable internally, the state object is frozen
    and users must always pass back the state they received.

    Attributes:
        rng_state: Random number generator.
    """

    rng_state: torch.Generator


def _resolve_generator(
    generator: None | int | torch.Generator,
) -> torch.Generator:
    """Resolve a generator specification to a torch.Generator.

    Args:
        generator: One of:
            - ``None``: In distributed mode, creates a generator seeded with
              a fixed seed (same across all devices). In single device mode,
              creates an unseeded generator (non-reproducible).
            - ``int``: create a generator seeded with this value (reproducible)
            - ``torch.Generator``: use directly
    """
    if generator is None:
        # Auto-detect distributed mode for synchronized noise
        if is_distributed():
            # Distributed mode: use same seed everywhere for synchronized noise
            # Use a fixed seed (e.g., 0) that doesn't depend on rank
            gen = torch.Generator().manual_seed(0)
        else:
            # Single device: unseeded generator (non-reproducible)
            gen = torch.Generator()
            gen.seed()
        return gen
    elif isinstance(generator, int):
        return torch.Generator().manual_seed(generator)
    elif isinstance(generator, torch.Generator):
        return generator
    else:
        raise TypeError(
            f"generator must be None, int, or torch.Generator, got {type(generator)}"
        )


def gaussian_noise(
    stddev: float,
    *,
    generator: None | int | torch.Generator = None,
) -> tuple[
    Callable[[Any, GaussianNoiseState], tuple[Any, GaussianNoiseState]],
    GaussianNoiseState,
]:
    """Create a Gaussian noise function with immutable state.

    Returns ``(noise_fn, state)`` where ``noise_fn`` adds calibrated Gaussian
    noise N(0, stddev²) to gradients and returns updated state.

    **Automatic distributed support**: When ``generator=None`` and distributed
    mode is detected (via RANK/WORLD_SIZE env vars), automatically uses the SAME
    seed across all devices. This provides synchronized noise for model convergence.

    Args:
        stddev: Standard deviation of Gaussian noise
            (usually ``noise_multiplier * clip_norm``).
        generator: RNG configuration:
            - ``None``:
              - **Distributed (world_size > 1)**: Use fixed seed (seed=0) across all devices
                for synchronized noise. Prevents model divergence.
              - **Single device**: Unseeded generator (non-reproducible)
            - ``int``: Seeded generator (reproducible, used as-is)
            - ``torch.Generator``: Use directly

    Returns:
        A tuple ``(noise_fn, state)`` where:

        - ``noise_fn(grads, state) -> (noisy_grads, new_state)``
        - ``state`` is a :class:`GaussianNoiseState`

    Example (typical use - no seed management needed):
        >>> import torch
        >>> from opaque.noise import gaussian_noise
        >>>
        >>> # When distributed is detected, automatically synchronizes noise across devices
        >>> noise_fn, state = gaussian_noise(stddev=1.1)  # No seed needed!
        >>> grads = torch.zeros(10)
        >>> noisy_grads, state = noise_fn(grads, state)

    Example (reproducible):
        >>> # Provide explicit seed for reproducibility
        >>> noise_fn, state = gaussian_noise(stddev=1.1, generator=42)
        >>> noisy_grads, state = noise_fn(grads, state)
    """
    if stddev < 0:
        raise ValueError(f"stddev must be non-negative, got {stddev}")

    gen = _resolve_generator(generator)
    state = GaussianNoiseState(rng_state=gen)

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
        return noisy, GaussianNoiseState(rng_state=g)

    return noise_fn, state


__all__ = ["gaussian_noise", "GaussianNoiseState"]
