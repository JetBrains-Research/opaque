"""RNG keys + helpers — torch.Generator-backed deterministic RNG."""

from opaque.api.engine.random import (
    fold_in,
    generator_from_key,
    key,
    random_key,
    set_reproducible_pytorch_seed,
    split,
)

__all__ = [
    "key",
    "split",
    "fold_in",
    "generator_from_key",
    "random_key",
    "set_reproducible_pytorch_seed",
]
