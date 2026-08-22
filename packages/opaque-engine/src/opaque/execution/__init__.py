"""Portable backend-dispatched execution transforms.

:func:`compile`, :func:`checkpoint`, and :func:`optimize_saved_activations`
are the transforms; :func:`supports_profile` and :func:`profile_primitives`
answer what the active provider implements. The ``ExecutionProfile`` enum,
the ``ExecutionProfileSnapshot`` record, and ``EXECUTION_PROFILE_VERSION``
live in :mod:`opaque.execution.types`.
"""

from opaque.api.engine.execution import (
    checkpoint,
    compile,
    optimize_saved_activations,
    profile_primitives,
    supports_profile,
)
from opaque.execution import types

__all__ = [
    "checkpoint",
    "compile",
    "optimize_saved_activations",
    "profile_primitives",
    "supports_profile",
    "types",
]
