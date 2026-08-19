"""Torch provider implementation package."""

from opaque.api.torch.backend import torch_backend
from opaque.api.torch.backend._serialization import register_torch_serialization

# Import-time side effect: any code that imports the Torch provider package
# can serialize torch tensors/Parameters immediately, matching the pre-split
# engine behavior. Backend activation registers the same handlers again
# idempotently for flows that never import this package directly (the
# serialization fallback resolver activates the provider on first contact
# with an unknown native leaf).
register_torch_serialization()

__all__ = ["torch_backend"]
