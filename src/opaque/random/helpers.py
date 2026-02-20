"""Convenience helpers for common RNG patterns.

This module provides ergonomic wrappers around the core RNG primitives for
typical use cases in DP training:

- ``random_key()`` - Non-deterministic keys for prototyping
- ``training_key()`` - Deterministic training loop keys with proper derivation order
"""

from __future__ import annotations

import secrets
from typing import Literal

from .engine import RngKey, fold_in, key


def random_key() -> RngKey:
    """Create a non-deterministic key using system entropy.

    Useful for prototyping when reproducibility is not critical. For production
    training, prefer ``training_key()`` with an explicit base_seed.

    Returns:
        A randomly initialized RngKey.

    Example:
        >>> from opaque.random import random_key
        >>> from opaque.noise import gaussian_noise
        >>> k = random_key()
        >>> noise_fn, state = gaussian_noise(l2_norm_clip=1.0, noise_multiplier=1.1, key=k)
    """
    random_seed = secrets.randbits(64)
    return key(random_seed)


def training_key(
    base_seed: int,
    step: int,
    rank: int | None = None,
    worker_id: int | None = None,
    synchronized: bool | Literal["auto"] | None = None,
) -> RngKey:
    """Create a deterministic key for training loops with proper derivation order.

    Follows the canonical derivation chain: ``step → rank → worker_id``.

    The ``synchronized`` parameter controls whether noise is identical across ranks:

    - ``True``: Same key for all ranks (centralized DP-SGD with synchronized noise)
    - ``False``: Different keys per rank via ``fold_in(rank)``
    - ``"auto"``: Synchronized if ``rank is None``, unsynchronized otherwise
    - ``None`` (default): Must not pass ``rank`` (raises ValueError)

    Args:
        base_seed: Reproducible seed for the entire training run.
        step: Training step counter (folded first).
        rank: Distributed rank (folded after step if unsynchronized).
        worker_id: DataLoader worker ID (folded last).
        synchronized: Noise synchronization policy for distributed training.

    Returns:
        Derived RngKey following step → rank → worker_id order.

    Raises:
        ValueError: If ``rank`` is passed without specifying ``synchronized``.
        ValueError: If ``synchronized`` has an invalid value.

    Example:
        >>> from opaque.random import training_key
        >>> from opaque.noise import gaussian_noise
        >>>
        >>> # Reproducible training loop
        >>> for step in range(epochs):
        ...     k = training_key(base_seed=42, step=step)
        ...     noise_fn, state = gaussian_noise(l2_norm_clip=1.0, noise_multiplier=1.1, key=k)
        ...     # ... train ...
        >>>
        >>> # Distributed training with per-rank noise
        >>> k = training_key(base_seed=42, step=0, rank=local_rank, synchronized=False)
        >>>
        >>> # Distributed training with synchronized noise
        >>> k = training_key(base_seed=42, step=0, rank=local_rank, synchronized=True)
        >>>
        >>> # Auto mode: synchronized if no rank, unsynchronized if rank provided
        >>> k = training_key(base_seed=42, step=0, rank=local_rank, synchronized="auto")
    """
    # Validate synchronized parameter
    if rank is not None and synchronized is None:
        raise ValueError(
            "Must specify synchronized parameter when passing rank. "
            'Use synchronized="auto" for automatic behavior, '
            "synchronized=True for centralized DP-SGD, or "
            "synchronized=False for per-rank noise."
        )

    if synchronized not in {True, False, "auto", None}:
        raise ValueError(
            f"synchronized must be True, False, 'auto', or None, got {synchronized!r}"
        )

    # Resolve auto mode
    if synchronized == "auto":
        synchronized = rank is None

    # Start derivation chain: base_seed -> step
    k = fold_in(key(base_seed), step)

    # Fold in rank if present and unsynchronized
    if rank is not None and not synchronized:
        k = fold_in(k, rank)

    # Fold in worker_id if present
    if worker_id is not None:
        k = fold_in(k, worker_id)

    return k


__all__ = ["random_key", "training_key"]
