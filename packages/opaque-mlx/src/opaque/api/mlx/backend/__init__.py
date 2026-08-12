"""Factory for the optional MLX compute backend."""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from opaque.api.engine.backend import Backend

__all__ = ["mlx_backend"]


def mlx_backend() -> Backend:
    """Construct the MLX backend.

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

    from opaque.api.mlx.backend._mlx import MlxBackend, register_core_primitives

    register_core_primitives()
    return MlxBackend()
