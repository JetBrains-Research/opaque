"""Torch provider implementation package."""

import torch.distributed as _dist
from opaque.api.engine.distributed.collectives import register_distributed_probe
from opaque.api.torch.backend import torch_backend
from opaque.api.torch.backend._serialization import register_torch_serialization

# Import-time side effect: any code that imports the Torch provider package
# can serialize torch tensors/Parameters immediately, matching the pre-split
# engine behavior. Backend activation registers the same handlers again
# idempotently for flows that never import this package directly (the
# serialization fallback resolver activates the provider on first contact
# with an unknown native leaf).
register_torch_serialization()


def _torch_process_group() -> tuple[int, int] | None:
    """Return ``(rank, world_size)`` when a Torch process group is live."""
    if not (_dist.is_available() and _dist.is_initialized()):
        return None
    return _dist.get_rank(), _dist.get_world_size()


# Backend selection is context-local and value-driven, so a rank query can
# reach the engine before any tensor has activated this provider. Registering
# the probe lets the engine answer from the live group instead of the
# single-process defaults, and hand the follow-on collectives — which carry
# Python scalars, not tensors — the backend that owns that group.
register_distributed_probe(_torch_process_group, backend_factory=torch_backend)

__all__ = ["torch_backend"]
