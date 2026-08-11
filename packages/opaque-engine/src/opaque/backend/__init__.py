"""Compute-backend seam — protocol + process-wide active-backend resolver.

Exposes the :class:`Backend` protocol and the ``active_backend`` /
``set_backend`` / ``use_backend`` resolver used by the DP clipping compute.
A PyTorch backend is registered as the default, so ``active_backend()`` works
with no setup.
"""

from opaque.api.engine.backend import (
    Backend,
    active_backend,
    set_backend,
    use_backend,
)

__all__ = [
    "Backend",
    "active_backend",
    "set_backend",
    "use_backend",
]
