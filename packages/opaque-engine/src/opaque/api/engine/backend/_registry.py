"""Process-wide active-backend resolver.

A single module-global singleton (:data:`_ACTIVE`) holds the backend that the
DP clipping compute resolves through.  :class:`~opaque.api.engine.backend._torch.TorchBackend`
is registered as the import-time default, so :func:`active_backend` works with
zero configuration and every call site stays parameter-free.

:func:`set_backend` swaps the active backend permanently; :func:`use_backend`
is a context manager that swaps it for the duration of a block and restores the
previous backend on exit (mainly for tests and future backends).
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import TYPE_CHECKING

from opaque.api.engine.backend._torch import TorchBackend

if TYPE_CHECKING:
    from collections.abc import Iterator

    from opaque.api.engine.backend._protocol import Backend

# Module-global singleton, resolved once and cached.  ``TorchBackend`` is the
# import-time default so ``active_backend()`` needs no configuration.
_ACTIVE: Backend = TorchBackend()


def active_backend() -> Backend:
    """Return the process-wide active backend (a cheap cached global read)."""
    return _ACTIVE


def set_backend(backend: Backend) -> None:
    """Set the process-wide active backend, replacing the current one."""
    global _ACTIVE
    _ACTIVE = backend


@contextmanager
def use_backend(backend: Backend) -> Iterator[Backend]:
    """Temporarily activate ``backend``, restoring the previous one on exit."""
    global _ACTIVE
    previous = _ACTIVE
    _ACTIVE = backend
    try:
        yield backend
    finally:
        _ACTIVE = previous


__all__ = ["active_backend", "set_backend", "use_backend"]
