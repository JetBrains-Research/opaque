"""Generic RNG primitives for Opaque.

Immutable, JAX-style key semantics via :func:`key`, :func:`split`, and
:func:`fold_in`, plus :func:`random_key` for prototyping.

The :class:`RngKey` data class is reachable via :mod:`opaque.random.types`.
"""

from opaque.api.engine.random._engine import (
    fold_in,
    key,
    normal,
    split,
)
from opaque.api.engine.random._helpers import random_key

__all__ = [
    "fold_in",
    "key",
    "normal",
    "random_key",
    "split",
]
