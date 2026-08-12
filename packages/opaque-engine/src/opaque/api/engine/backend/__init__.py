"""Backend identity and context-local active-backend resolver.

``Backend`` provides the stable name used by primitive dispatch. The bundled
``TorchBackend`` remains the zero-configuration default and retains its
compatibility methods while primitive registrations become the extension seam.
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
