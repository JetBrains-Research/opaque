"""Primitive types — the dispatch object, its tier enum, and the provider protocol.

:func:`opaque.primitive.primitive` returns a ``Primitive``; a provider
satisfies ``BackendProvider``. They live here for ``isinstance`` checks
and type annotations, matching :mod:`opaque.backend.types`.
"""

from opaque.api.engine.primitive import BackendProvider, Primitive, PrimitiveTier

__all__ = ["BackendProvider", "Primitive", "PrimitiveTier"]
