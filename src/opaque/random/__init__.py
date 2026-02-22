"""Generic RNG primitives for Opaque.

Provides immutable, JAX-style key semantics via ``key``, ``split``, and
``fold_in`` with a PyTorch generator bridge, plus convenience helpers
like ``random_key()`` for prototyping.
Also provides ``set_reproducible_pytorch_seed()`` for PyTorch/CUDNN
reproducibility configuration.
"""

from .engine import RngKey, fold_in, generator_from_key, key, split
from .helpers import random_key, set_reproducible_pytorch_seed

__all__ = [
    "RngKey",
    "key",
    "split",
    "fold_in",
    "generator_from_key",
    "random_key",
    "set_reproducible_pytorch_seed",
]
