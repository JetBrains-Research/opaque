"""Backend types — the provider protocol and the known-backend identity enum.

:func:`opaque.backend.set_backend` and :func:`opaque.backend.use_backend`
accept either; the types live here for ``isinstance`` checks and type
annotations, matching :mod:`opaque.scheduling.types`.
"""

from opaque.api.engine.backend import Backend, KnownBackend

__all__ = ["Backend", "KnownBackend"]
