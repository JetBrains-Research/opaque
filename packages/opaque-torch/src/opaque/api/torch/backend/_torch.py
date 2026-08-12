"""Torch backend identity."""

from opaque.api.engine.backend import KnownBackend


class TorchBackend:
    """Stable identity for the Torch provider."""

    name = KnownBackend.TORCH.value


__all__ = ["TorchBackend"]
