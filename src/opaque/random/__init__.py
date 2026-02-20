"""Generic RNG primitives for Opaque.

Provides immutable, JAX-style key semantics via ``key``, ``split``, and
``fold_in`` with a PyTorch generator bridge.
"""

from .engine import RngKey, fold_in, generator_from_key, key, split

__all__ = ["RngKey", "key", "split", "fold_in", "generator_from_key"]
