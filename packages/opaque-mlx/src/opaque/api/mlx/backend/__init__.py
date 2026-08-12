"""Factory for the optional MLX compute backend."""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from opaque.api.engine.backend import Backend

__all__ = ["mlx_backend"]


def mlx_backend() -> Backend:
    """Construct the MLX backend and register its integrations.

    Raises:
        ImportError: If MLX is unavailable in the current environment.
    """
    try:
        importlib.import_module("mlx.core")
    except ModuleNotFoundError as exc:
        if exc.name != "mlx" and not (exc.name or "").startswith("mlx."):
            raise
        raise ImportError(
            "MLX support requires the 'mlx' dependency. Install opaque-mlx on "
            "Apple Silicon or install the Opaque bundle with `pip install "
            "'opaque[mlx]'`."
        ) from exc

    from opaque.api.mlx.backend import _runtime  # noqa: F401
    from opaque.api.mlx.backend._core import MlxBackend
    from opaque.api.mlx.backend._serialization import register_mlx_serialization

    register_mlx_serialization()
    return MlxBackend()
