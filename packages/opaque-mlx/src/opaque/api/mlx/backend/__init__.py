"""MLX backend factory and primitive registrations."""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING, cast

if TYPE_CHECKING:
    from opaque.api.engine.backend import Backend


def mlx_backend() -> Backend:
    """Return the stable MLX backend identity after registering primitives."""
    try:
        importlib.import_module("mlx")
    except ModuleNotFoundError as exc:
        if exc.name != "mlx" and not (exc.name or "").startswith("mlx."):
            raise
        raise ImportError(
            "MLX support requires the 'mlx' dependency on Apple Silicon. "
            "Install opaque-mlx on a supported macOS arm64 system."
        ) from exc

    from opaque.api.mlx.backend import _core, _execution, _runtime, _serialization

    _serialization.register_mlx_serialization()
    del _execution, _runtime
    return cast("Backend", _core.MlxBackend())


__all__ = ["mlx_backend"]
