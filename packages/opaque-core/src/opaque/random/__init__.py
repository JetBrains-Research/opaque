"""Generic RNG primitives for Opaque.

Immutable, JAX-style key semantics via :func:`key`, :func:`split`, and
:func:`fold_in` with a PyTorch generator bridge. Plus convenience helpers
like :func:`random_key` for prototyping and
:func:`set_reproducible_pytorch_seed` for PyTorch/CUDNN reproducibility.

The :class:`RngKey` data class is reachable via :mod:`opaque.random.types`.
"""

from opaque.random._engine import fold_in, generator_from_key, key, split
from opaque.random._helpers import random_key, set_reproducible_pytorch_seed

__all__ = [
    "key",
    "split",
    "fold_in",
    "generator_from_key",
    "random_key",
    "set_reproducible_pytorch_seed",
]
