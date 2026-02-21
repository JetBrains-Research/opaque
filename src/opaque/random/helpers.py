"""Convenience helpers for common RNG patterns.

This module provides ergonomic wrappers around the core RNG primitives for
typical use cases in DP training:

- ``random_key()`` - Non-deterministic keys for prototyping
- ``training_key()`` - Deterministic training loop keys with proper derivation order
- ``set_reproducible_pytorch_seed()`` - Configure PyTorch/CUDNN reproducibility
"""

from __future__ import annotations

import os
import secrets
from typing import TYPE_CHECKING, Literal

from .engine import RngKey, fold_in, key, split

if TYPE_CHECKING:
    pass


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


def set_reproducible_pytorch_seed(key_val: RngKey) -> None:
    """Set all PyTorch/CUDNN seeds from a single RngKey for reproducible training.

    Configures PyTorch and CUDNN for deterministic behavior by setting:

    - ``torch.manual_seed()`` for CPU operations
    - ``torch.cuda.manual_seed_all()`` for all GPU operations
    - ``torch.backends.cudnn`` flags for deterministic convolutions
    - Environment variables for cuBLAS library determinism

    This function has side effects (modifies global state). Call it once at
    the start of your training script before creating any tensors or models.

    Args:
        key_val: RngKey from opaque.random module (typically from ``key()``
            or ``random_key()``).

    Example:
        Setup framework reproducibility once at startup, then use
        ``training_key()`` for per-step DP operations:

        >>> from opaque.random import key, training_key, set_reproducible_pytorch_seed
        >>> from opaque.noise import gaussian_noise
        >>>
        >>> # At start of training - configure all PyTorch/CUDNN RNG sources
        >>> set_reproducible_pytorch_seed(key(42))
        >>>
        >>> # Then use training_key for per-step DP randomness
        >>> for step in range(num_steps):
        ...     step_key = training_key(base_seed=42, step=step)
        ...     noise_fn, state = gaussian_noise(l2_norm_clip=1.0, noise_multiplier=1.1, key=step_key)
        ...     # ... training step ...

    Note:
        Setting determinism flags has a performance cost (typically 10-30% slower).
        See PyTorch documentation on ``torch.use_deterministic_algorithms()``
        and ``torch.backends.cudnn`` for details.

    See Also:
        - ``key()``: Create RngKey from integer seed
        - ``training_key()``: Deterministic per-step key derivation
        - ``random_key()``: Non-deterministic key from system entropy
        - torch.manual_seed: PyTorch CPU RNG seed setter
        - torch.use_deterministic_algorithms: PyTorch determinism flag
    """
    # Import here to avoid requiring torch as hard dependency at module load
    import numpy as np
    import torch

    # Derive independent seeds for different RNG sources
    seeds = split(key_val, num=3)
    torch_cpu_seed = seeds[0].seed % (2**31 - 1)  # Keep within C++ int32 range
    torch_gpu_seed = seeds[1].seed % (2**31 - 1)
    numpy_seed = seeds[2].seed % (2**32)

    # Set CPU and GPU seeds
    torch.manual_seed(int(torch_cpu_seed))
    torch.cuda.manual_seed_all(int(torch_gpu_seed))

    # Set numpy seed if available
    try:
        np.random.seed(int(numpy_seed))
    except (ImportError, RuntimeError):
        pass  # numpy not available or already seeded

    # Configure CUDNN for deterministic behavior
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Set PyTorch deterministic algorithms (PyTorch >= 1.11)
    # May reduce performance and may raise errors if operations lack deterministic implementations
    try:
        torch.use_deterministic_algorithms(True)
    except (AttributeError, RuntimeError):
        # Older PyTorch or flag already set - silently ignore
        pass

    # Configure cuBLAS for deterministic behavior
    # Format: ":size" or ":size:seed" (we use ":16:8" - 16MB workspace with seed)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"


__all__ = ["random_key", "training_key", "set_reproducible_pytorch_seed"]
