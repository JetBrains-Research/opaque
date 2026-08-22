"""Public surface for declaring and dispatching primitives.

An extension declares a primitive here and registers an implementation per
backend; the dispatch machinery resolves it from the active backend.
Declarations made through this façade are ``PrimitiveTier.OPTIONAL``, the
default and the only tier valid outside the engine — guard a call with
``.supports(...)`` and provide a fallback where an implementation is missing.

The ``Primitive`` object, the ``PrimitiveTier`` enum, and the
``BackendProvider`` protocol a registration is written against live in
:mod:`opaque.primitive.types`; the errors a caller catches stay here.

The portable-core machinery — ``CORE_PRIMITIVES``, ``core_profile``,
``declare_core_primitives``, ``validate_core_primitives`` — governs the contract
every provider must satisfy before it may activate. It stays at
``opaque.api.engine.primitive``: a ``CORE`` declaration made outside the engine
appends to that global profile and makes every shipped provider incomplete.
"""

from opaque.api.engine.primitive import (
    DuplicatePrimitiveRegistrationError,
    IncompleteBackendError,
    InvalidPrimitiveRegistrationError,
    PrimitiveError,
    UnsupportedPrimitiveError,
    primitive,
    registered_backends,
    supports,
)
from opaque.primitive import types

__all__ = [
    "DuplicatePrimitiveRegistrationError",
    "IncompleteBackendError",
    "InvalidPrimitiveRegistrationError",
    "PrimitiveError",
    "UnsupportedPrimitiveError",
    "primitive",
    "registered_backends",
    "supports",
    "types",
]
