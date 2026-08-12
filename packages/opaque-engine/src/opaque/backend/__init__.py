"""Backend identity and context-local active-backend resolver.

Primitive dispatch resolves through the active backend's stable name. A
PyTorch backend is active by default, so ``active_backend()`` needs no setup.
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
