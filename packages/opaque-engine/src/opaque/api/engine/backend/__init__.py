"""Backend seam for the DP clipping compute.

Declares the five-primitive :class:`Backend` protocol (autodiff,
vectorization, pytree ops, array math, RNG), ships a :class:`TorchBackend`
implementation, and exposes the process-wide resolver
(:func:`active_backend` / :func:`set_backend` / :func:`use_backend`).

Importing this package registers :class:`TorchBackend` as the import-time
default active backend, so :func:`active_backend` works with no setup.
"""

from opaque.api.engine.backend._protocol import Backend
from opaque.api.engine.backend._registry import (
    active_backend,
    set_backend,
    use_backend,
)
from opaque.api.engine.backend._torch import TorchBackend

__all__ = [
    "Backend",
    "TorchBackend",
    "active_backend",
    "set_backend",
    "use_backend",
]
