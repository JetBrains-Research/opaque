"""Torch in-place collectives facade."""

from opaque.api.torch.distributed import all_reduce_, reduce_pytree_, sum_gradients_

__all__ = ["all_reduce_", "reduce_pytree_", "sum_gradients_"]
