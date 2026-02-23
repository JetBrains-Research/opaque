"""Rectified Gaussian noise generation for differential privacy.

This module implements the rectified Gaussian mechanism: sample from a
standard Gaussian N(0, σ²) and clamp the result to [−R·σ, R·σ].  This is
simpler than the truncated Gaussian (which resamples to stay in-bounds) and
has tighter privacy accounting than unbounded Gaussian when R ≥ 3.

The API returns ``(noise_fn, state)`` where state is always immutable:

    >>> from opaque.random import key
    >>> noise_fn, state = rectified_gaussian_noise(stddev=1.0, radius=5.0, key=key(42))
    >>> noisy_grads, state = noise_fn(grads, state)

The noise function is **purely local** — it uses exactly the key you provide.
For synchronized noise in distributed training, pass the same key on every rank.
For independent noise, derive a per-rank key with ``fold_in(key, rank)``.
"""

from collections.abc import Callable
from typing import Any

import torch

from opaque.noise.gaussian_noise import GaussianNoiseState
from opaque.random import RngKey, generator_from_key
from opaque.random import fold_in as rng_fold_in
from opaque.utils.pytree import tree_map


def rectified_gaussian_noise(
    stddev: float,
    radius: float,
    *,
    key: RngKey,
) -> tuple[
    Callable[[Any, GaussianNoiseState], tuple[Any, GaussianNoiseState]],
    GaussianNoiseState,
]:
    """Create a rectified Gaussian noise function with immutable state.

    Returns ``(noise_fn, state)`` where ``noise_fn`` adds noise drawn from
    a standard Gaussian N(0, stddev²) and clamped to [−radius·stddev, radius·stddev].

    This is the **rectified** Gaussian mechanism — simpler than truncated
    Gaussian (no rejection sampling or inverse-CDF), just a hard clamp.
    Use :func:`~opaque.accounting.mechanisms.rectified_gaussian` for matching
    privacy accounting.

    The noise function uses exactly the ``key`` you provide — no auto-detection
    of distributed state. For synchronized noise in DDP, pass the same key on
    every rank. For independent noise, derive a per-rank key::

        from opaque.random import key, fold_in
        my_key = fold_in(key(42), rank)  # different noise per rank
        noise_fn, state = rectified_gaussian_noise(stddev=1.1, radius=5.0, key=my_key)

    Args:
        stddev: Standard deviation of the underlying Gaussian noise
            (usually ``noise_multiplier * clip_norm``).
        radius: Clamping radius in units of standard deviations.
            Noise is clamped to [−radius·stddev, radius·stddev].
            Must be positive. Typical values: 3–10.
        key: Explicit RNG key for deterministic, functional randomness.
            Same key on all ranks → same noise (synchronized).
            ``fold_in(key, rank)`` → independent noise per rank.

    Returns:
        A tuple ``(noise_fn, state)`` where:

        - ``noise_fn(grads, state) -> (noisy_grads, new_state)``
        - ``state`` is a :class:`~opaque.noise.gaussian_noise.GaussianNoiseState`

    Raises:
        ValueError: If ``stddev`` is negative or ``radius`` is not positive.

    Example:
        >>> import torch
        >>> from opaque.noise import rectified_gaussian_noise
        >>> from opaque.random import key
        >>>
        >>> noise_fn, state = rectified_gaussian_noise(
        ...     stddev=1.0, radius=5.0, key=key(42),
        ... )
        >>> grads = torch.zeros(100)
        >>> noisy, state = noise_fn(grads, state)
        >>> bound = 1.0 * 5.0
        >>> assert noisy.min() >= -bound and noisy.max() <= bound
    """
    if stddev < 0:
        raise ValueError(f"stddev must be non-negative, got {stddev}")

    if radius <= 0:
        raise ValueError(f"radius must be positive, got {radius}")

    if not isinstance(key, RngKey):
        raise TypeError(f"key must be RngKey, got {type(key)}")

    state = GaussianNoiseState(
        step_counter=0,
        rng_key=key,
    )

    bound = stddev * radius

    if stddev == 0:

        def zero_noise_fn(grads, st):
            return grads, st

        return zero_noise_fn, state

    def noise_fn(grads, st):
        """Add rectified Gaussian noise to gradients."""
        step_key = rng_fold_in(st.rng_key, st.step_counter)
        g = generator_from_key(step_key)

        def add_noise_to_tensor(tensor: torch.Tensor) -> torch.Tensor:
            # torch.Generator is CPU-only; generate on CPU and move if needed
            noise = torch.randn(
                tensor.shape,
                dtype=tensor.dtype,
                generator=g,
            )
            # Rectification: clamp noise to [-bound, bound]
            noise = torch.clamp(noise * stddev, min=-bound, max=bound)
            return tensor + noise.to(device=tensor.device)

        noisy = tree_map(add_noise_to_tensor, grads)

        return noisy, GaussianNoiseState(
            step_counter=st.step_counter + 1,
            rng_key=st.rng_key,
        )

    return noise_fn, state


__all__ = ["rectified_gaussian_noise"]
