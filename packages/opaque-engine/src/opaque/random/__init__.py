"""Immutable RNG keys and backend-dispatched sampling helpers."""

from opaque.api.engine.random import (
    fold_in,
    generator_from_key,
    key,
    normal,
    random_key,
    set_reproducible_pytorch_seed,
    split,
)

__all__ = [
    "fold_in",
    "generator_from_key",
    "key",
    "normal",
    "random_key",
    "set_reproducible_pytorch_seed",
    "split",
]
