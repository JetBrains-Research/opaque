"""Backend identity, argument inference, and context-local lifecycle.

The lifecycle functions and the error hierarchy a caller catches live
here; the ``Backend`` protocol and the ``KnownBackend`` identity enum
live in :mod:`opaque.backend.types`.
"""

from opaque.api.engine.backend import (
    BackendError,
    BackendMismatchError,
    BackendNotSelectedError,
    BackendProviderError,
    MixedBackendError,
    active_backend,
    clear_backend,
    ensure_backend,
    set_backend,
    use_backend,
)
from opaque.backend import types

__all__ = [
    "BackendError",
    "BackendMismatchError",
    "BackendNotSelectedError",
    "BackendProviderError",
    "MixedBackendError",
    "active_backend",
    "clear_backend",
    "ensure_backend",
    "set_backend",
    "types",
    "use_backend",
]
