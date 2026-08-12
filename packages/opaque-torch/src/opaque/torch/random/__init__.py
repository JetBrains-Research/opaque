"""Torch RNG bridge façade."""

from opaque.api.torch.random import generator_from_key, set_reproducible_pytorch_seed

__all__ = ["generator_from_key", "set_reproducible_pytorch_seed"]
