"""Torch-specific random bridges."""

from opaque.api.torch.random._helpers import (
    generator_from_key,
    set_reproducible_pytorch_seed,
)

__all__ = ["generator_from_key", "set_reproducible_pytorch_seed"]
