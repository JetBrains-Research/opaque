"""Federated per-client clipping."""

from opaque.api.federated.clipping._callbacks import make_clipping_aggregate
from opaque.api.federated.clipping._clipped_grad import clipped_grad

__all__ = ["clipped_grad", "make_clipping_aggregate"]
