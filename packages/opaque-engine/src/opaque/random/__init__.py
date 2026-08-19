"""Immutable RNG keys and backend-dispatched sampling helpers."""

from opaque.api.engine.random import (
    fold_in,
    key,
    normal,
    random_key,
    split,
)

__all__ = [
    "fold_in",
    "key",
    "normal",
    "random_key",
    "split",
]


def __getattr__(name: str):
    # Transitional re-exports while downstream packages migrate to the
    # provider wheels; scheduled for removal once the migration completes.
    if name in ("generator_from_key", "set_reproducible_pytorch_seed"):
        import opaque.torch.random as _torch_random

        return getattr(_torch_random, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
