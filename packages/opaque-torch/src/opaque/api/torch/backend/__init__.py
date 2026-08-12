"""Factory for the optional Torch compute backend."""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from opaque.api.engine.backend import Backend

__all__ = ["torch_backend"]


def torch_backend() -> Backend:
    """Construct the Torch backend and register its integrations."""
    try:
        importlib.import_module("torch")
    except ModuleNotFoundError as exc:
        if exc.name != "torch" and not (exc.name or "").startswith("torch."):
            raise
        raise ImportError(
            "Torch support requires the 'torch' dependency. Install opaque-torch "
            "with `pip install opaque-torch`."
        ) from exc

    from opaque.api.torch.backend import _runtime  # noqa: F401
    from opaque.api.torch.backend._core import TorchBackend
    from opaque.api.torch.backend._serialization import register_torch_serialization

    register_torch_serialization()
    return TorchBackend()
