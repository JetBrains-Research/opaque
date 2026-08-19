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
        torch = importlib.import_module("torch")
    except ModuleNotFoundError as exc:
        if exc.name != "torch" and not (exc.name or "").startswith("torch."):
            raise
        raise ImportError(
            "Torch support requires the 'torch' dependency. Install opaque-torch "
            "with `pip install opaque-torch`."
        ) from exc

    # Side-effect imports: optimizer-state sync auditors register with the
    # engine's type-dispatched sync() when the provider activates.
    import opaque.api.torch.optimizers.distributed  # noqa: F401
    from opaque.api.torch.backend import _execution, _runtime  # noqa: F401
    from opaque.api.torch.backend._core import TorchBackend
    from opaque.api.torch.backend._serialization import register_torch_serialization

    register_torch_serialization()

    # Dispatch consults the context-local ContextVar in eager execution and
    # may only trust the module-global mirror inside a traced graph, where
    # ContextVar reads are untraceable. Install Torch's tracing probe so
    # the engine can tell the two apart.
    from opaque.api.engine.backend import _registry

    _registry._set_compiling_detector(torch.compiler.is_compiling)
    return TorchBackend()
