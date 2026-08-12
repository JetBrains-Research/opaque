"""Context-local active-backend resolver."""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from typing import TYPE_CHECKING

from opaque.api.engine.backend._torch import TorchBackend
from opaque.api.engine.primitive import validate_core_primitives

if TYPE_CHECKING:
    from collections.abc import Iterator

    from opaque.api.engine.backend._protocol import Backend

_DEFAULT_BACKEND: Backend = TorchBackend()
validate_core_primitives(_DEFAULT_BACKEND)
_ACTIVE: ContextVar[Backend | None] = ContextVar("opaque_active_backend", default=None)


def active_backend() -> Backend:
    """Return the backend active in the current execution context."""
    return _ACTIVE.get() or _DEFAULT_BACKEND


def set_backend(backend: Backend) -> None:
    """Activate ``backend`` in the current execution context."""
    validate_core_primitives(backend)
    _ACTIVE.set(backend)


@contextmanager
def use_backend(backend: Backend) -> Iterator[Backend]:
    """Temporarily activate ``backend`` and restore the context token on exit."""
    validate_core_primitives(backend)
    token = _ACTIVE.set(backend)
    try:
        yield backend
    finally:
        _ACTIVE.reset(token)


__all__ = ["active_backend", "set_backend", "use_backend"]
