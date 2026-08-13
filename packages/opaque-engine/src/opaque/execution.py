"""Portable backend-dispatched execution transforms."""

from opaque.api.engine.execution import (
    EXECUTION_PROFILE_VERSION,
    ExecutionProfile,
    ExecutionProfileSnapshot,
    checkpoint,
    compile,
    optimize_saved_activations,
    profile_primitives,
    supports_profile,
)

__all__ = [
    "EXECUTION_PROFILE_VERSION",
    "ExecutionProfile",
    "ExecutionProfileSnapshot",
    "checkpoint",
    "compile",
    "optimize_saved_activations",
    "profile_primitives",
    "supports_profile",
]
