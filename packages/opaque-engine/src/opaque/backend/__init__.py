"""Backend identity, argument inference, and context-local lifecycle."""

from opaque.api.engine.backend import (
    Backend,
    BackendError,
    BackendMismatchError,
    BackendNotSelectedError,
    BackendProviderError,
    KnownBackend,
    MixedBackendError,
    active_backend,
    clear_backend,
    ensure_backend,
    set_backend,
    use_backend,
)

__all__ = [
    "Backend",
    "BackendError",
    "BackendMismatchError",
    "BackendNotSelectedError",
    "BackendProviderError",
    "KnownBackend",
    "MixedBackendError",
    "active_backend",
    "clear_backend",
    "ensure_backend",
    "set_backend",
    "use_backend",
]
