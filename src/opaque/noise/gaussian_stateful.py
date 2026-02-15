"""Stateful Gaussian noise generation for differential privacy.

This module provides a helper compatible with older tests that expect a
stateful noise function returning only noisy gradients.
"""

import dataclasses
from typing import Any

import torch

from opaque.distributed import get_rank, is_initialized
from opaque.utils.pytree import tree_map


@dataclasses.dataclass(frozen=True)
class GaussianStatefulState:
    """Immutable state for Gaussian noise generation.

    Attributes:
        rng_state: Random number generator.
    """

    rng_state: torch.Generator


def _resolve_generator(seed: None | int | torch.Generator) -> torch.Generator:
    """Resolve seed/generator to a torch.Generator."""
    if seed is None:
        gen = torch.Generator()
        gen.seed()
        return gen
    if isinstance(seed, int):
        return torch.Generator().manual_seed(seed)
    if isinstance(seed, torch.Generator):
        return seed
    raise TypeError(f"seed must be None, int, or torch.Generator, got {type(seed)}")


def gaussian_stateful(
    stddev: float,
    seed: None | int | torch.Generator = None,
    *,
    distributed: bool = False,
) -> tuple[callable, GaussianStatefulState]:
    """Create a stateful Gaussian noise function.

    Args:
        stddev: Standard deviation of Gaussian noise.
        seed: RNG seed or generator. If distributed=True and seed is int,
            uses seed + rank for per-rank determinism.
        distributed: If True, require distributed initialization and
            offset seed by rank.

    Returns:
        Tuple of (noise_fn, state) where noise_fn(grads, state) -> noisy_grads.
    """
    if stddev < 0:
        raise ValueError(f"stddev must be non-negative, got {stddev}")

    if distributed:
        if not is_initialized():
            raise RuntimeError("Distributed is not initialized")
        if isinstance(seed, int):
            seed = seed + get_rank()

    gen = _resolve_generator(seed)
    state = GaussianStatefulState(rng_state=gen)

    if stddev == 0:

        def zero_noise_fn(grads, _state):
            return grads

        return zero_noise_fn, state

    def noise_fn(grads: Any, st: GaussianStatefulState):
        """Add Gaussian noise to gradients."""
        g = st.rng_state

        def add_noise_to_tensor(tensor: torch.Tensor) -> torch.Tensor:
            noise = torch.randn(
                tensor.shape,
                dtype=tensor.dtype,
                generator=g,
            ).to(device=tensor.device)
            return tensor + noise * stddev

        return tree_map(add_noise_to_tensor, grads)

    return noise_fn, state


__all__ = ["gaussian_stateful", "GaussianStatefulState"]
