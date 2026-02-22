"""Gaussian noise generation for differential privacy.

This module provides a higher-order function for adding calibrated Gaussian noise
to gradients in DP-SGD (Differentially Private Stochastic Gradient Descent).

The API returns ``(noise_fn, state)`` where state is always immutable:

    >>> from opaque.random import key
    >>> noise_fn, state = gaussian_noise(stddev=1.0, key=key(42))
    >>> noisy_grads, state = noise_fn(grads, state)

The noise function is **purely local** — it uses exactly the key you provide.
For synchronized noise in distributed training, pass the same key on every rank.
For independent noise, derive a per-rank key with ``fold_in(key, rank)``.
"""

import dataclasses
from collections.abc import Callable
from typing import Any

import torch

from opaque.random import RngKey, generator_from_key
from opaque.random import (
    fold_in as rng_fold_in,
)
from opaque.utils.pytree import tree_map


@dataclasses.dataclass(frozen=True)
class GaussianNoiseState:
    """Immutable state for Gaussian noise generation.

    Holds an immutable RNG key for deterministic per-step derivation.
    Noise for step ``t`` is generated from ``fold_in(rng_key, t)``.

    Attributes:
        step_counter: Number of noise_fn calls made.
        rng_key: Immutable RNG key for deterministic derivation.
    """

    step_counter: int
    rng_key: RngKey


def gaussian_noise(
    stddev: float,
    *,
    key: RngKey,
) -> tuple[
    Callable[[Any, GaussianNoiseState], tuple[Any, GaussianNoiseState]],
    GaussianNoiseState,
]:
    """Create a Gaussian noise function with immutable state.

    Returns ``(noise_fn, state)`` where ``noise_fn`` adds calibrated Gaussian
    noise N(0, stddev²) to gradients and returns updated state.

    The noise function uses exactly the ``key`` you provide — no auto-detection
    of distributed state. For synchronized noise in DDP, pass the same key on
    every rank. For independent noise, derive a per-rank key::

        from opaque.random import key, fold_in
        my_key = fold_in(key(42), rank)  # different noise per rank
        noise_fn, state = gaussian_noise(stddev=1.1, key=my_key)

    Args:
        stddev: Standard deviation of Gaussian noise
            (usually ``noise_multiplier * clip_norm``).
        key: Explicit RNG key for deterministic, functional randomness.
            Same key on all ranks → same noise (synchronized).
            ``fold_in(key, rank)`` → independent noise per rank.

    Returns:
        A tuple ``(noise_fn, state)`` where:

        - ``noise_fn(grads, state) -> (noisy_grads, new_state)``
        - ``state`` is a :class:`GaussianNoiseState`

    Example:
        >>> import torch
        >>> from opaque.noise import gaussian_noise
        >>> from opaque.random import key
        >>>
        >>> noise_fn, state = gaussian_noise(stddev=1.1, key=key(42))
        >>> grads = torch.zeros(10)
        >>> noisy_grads, state = noise_fn(grads, state)

    Example (distributed — synchronized noise on all ranks):
        >>> # All ranks pass the same key → identical noise → models stay in sync
        >>> noise_fn, state = gaussian_noise(stddev=1.1, key=key(42))

    Example (distributed — independent noise per rank):
        >>> from opaque.random import key, fold_in
        >>> rank = torch.distributed.get_rank()
        >>> noise_fn, state = gaussian_noise(stddev=1.1, key=fold_in(key(42), rank))
    """
    if stddev < 0:
        raise ValueError(f"stddev must be non-negative, got {stddev}")

    if not isinstance(key, RngKey):
        raise TypeError(f"key must be RngKey, got {type(key)}")

    state = GaussianNoiseState(
        step_counter=0,
        rng_key=key,
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
            step_counter=st.step_counter + 1,
            rng_key=st.rng_key,
        )

    return noise_fn, state


__all__ = ["gaussian_noise", "GaussianNoiseState"]
