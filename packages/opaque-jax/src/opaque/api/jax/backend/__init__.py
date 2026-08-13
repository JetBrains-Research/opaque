"""Factory for the optional JAX compute backend."""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from opaque.api.engine.backend import Backend

__all__ = ["jax_backend"]


def jax_backend() -> Backend:
    """Construct the JAX backend and register its integrations.

    Raises:
        ImportError: If JAX is unavailable in the current environment.
    """
    try:
        importlib.import_module("jax")
    except ModuleNotFoundError as exc:
        if exc.name != "jax" and not (exc.name or "").startswith("jax."):
            raise
        raise ImportError(
            "JAX support requires the 'jax' dependency. Install opaque-jax or "
            "install the Opaque bundle with `pip install 'opaque[jax]'`."
        ) from exc

    from opaque.api.jax.backend import (
        _execution,  # noqa: F401
        _runtime,  # noqa: F401
    )
    from opaque.api.jax.backend._core import JaxBackend
    from opaque.api.jax.backend._serialization import register_jax_serialization

    register_jax_serialization()
    return JaxBackend()
