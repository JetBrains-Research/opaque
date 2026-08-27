"""Optional, backend-dispatched execution transforms.

These transforms are not part of the portable core profile; a provider may
register them individually.  Callers discover support through
:class:`ExecutionProfile` before relying on a transform.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import TYPE_CHECKING, Any

from opaque.api.engine.autodiff import _deferred_transform
from opaque.api.engine.primitive import Primitive, primitive

if TYPE_CHECKING:
    from collections.abc import Callable


@primitive(tier="optional")
def _compile_transform(fn: Callable[..., Any]) -> Callable[..., Any]:
    """Provider-native compilation/JIT for ``fn``."""
    raise NotImplementedError


@primitive(tier="optional")
def _checkpoint_transform(fn: Callable[..., Any]) -> Callable[..., Any]:
    """Selective activation checkpointing for ``fn``."""
    raise NotImplementedError


@primitive(tier="optional")
def _optimize_saved_activations_transform(fn: Callable[..., Any]) -> Callable[..., Any]:
    """Reduce separate accelerator-memory pressure for ``fn``."""
    raise NotImplementedError


def compile(fn: Callable[..., Any]) -> Callable[..., Any]:
    """Return a callable that compiles ``fn`` with its invocation backend."""
    return _deferred_transform(_compile_transform, fn)


def checkpoint(fn: Callable[..., Any]) -> Callable[..., Any]:
    """Return a checkpointed version of ``fn`` bound to its invocation backend."""
    return _deferred_transform(_checkpoint_transform, fn)


def optimize_saved_activations(fn: Callable[..., Any]) -> Callable[..., Any]:
    """Return a version of ``fn`` that optimizes saved-activation placement."""
    return _deferred_transform(_optimize_saved_activations_transform, fn)


class ExecutionProfile(StrEnum):
    """Named optional execution-transform profiles."""

    COMPILATION = "compilation"
    CHECKPOINTING = "checkpointing"
    SAVED_ACTIVATIONS = "saved_activations"

    @property
    def primitives(self) -> tuple[Primitive, ...]:
        """Return the primitive declarations required by this profile."""
        return profile_primitives(self)

    def supports(self, backend: object | str | None = None) -> bool:
        """Return whether ``backend`` implements this complete profile."""
        return supports_profile(self, backend)


EXECUTION_PROFILE_VERSION = 1
"""Version of the named optional execution profile contract."""


_EXECUTION_PROFILES: dict[ExecutionProfile, tuple[Primitive, ...]] = {
    ExecutionProfile.COMPILATION: (_compile_transform,),
    ExecutionProfile.CHECKPOINTING: (_checkpoint_transform,),
    ExecutionProfile.SAVED_ACTIVATIONS: (_optimize_saved_activations_transform,),
}


@dataclass(frozen=True)
class ExecutionProfileSnapshot:
    """Versioned execution profile requirements."""

    version: int
    primitives: tuple[Primitive, ...]


def profile_primitives(profile: ExecutionProfile | str) -> tuple[Primitive, ...]:
    """Return the declarations required by a named execution profile."""
    return _EXECUTION_PROFILES[ExecutionProfile(profile)]


def supports_profile(
    profile: ExecutionProfile | str,
    backend: object | str | None = None,
) -> bool:
    """Return whether ``backend`` registered every primitive in ``profile``."""
    return all(operation.supports(backend) for operation in profile_primitives(profile))


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
