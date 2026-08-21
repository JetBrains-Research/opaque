"""Backend-name primitive dispatch for Opaque providers and extensions."""

from __future__ import annotations

import importlib
import os
import threading
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from enum import StrEnum
from functools import update_wrapper
from typing import TYPE_CHECKING, Any, TypeVar, overload

_VALIDATE_DISPATCH_ARGS = os.environ.get("OPAQUE_VALIDATE_BACKEND_ARGS", "") == "1"

if TYPE_CHECKING:
    from opaque.api.engine.backend._protocol import Backend

Implementation = Callable[..., Any]
Declaration = TypeVar("Declaration", bound=Callable[..., Any])


class PrimitiveError(RuntimeError):
    """Base class for primitive registry failures."""


class InvalidPrimitiveRegistrationError(PrimitiveError, ValueError):
    """Raised when a primitive identity or implementation is invalid."""


class DuplicatePrimitiveRegistrationError(PrimitiveError):
    """Raised when an implementation is registered without replacement."""


class UnsupportedPrimitiveError(PrimitiveError):
    """Raised when the active backend does not implement a primitive."""

    def __init__(self, primitive_name: str, backend_name: str) -> None:
        self.primitive_name = primitive_name
        self.backend_name = backend_name
        super().__init__(
            f"Primitive {primitive_name!r} is not supported by backend {backend_name!r}."
        )


class IncompleteBackendError(PrimitiveError):
    """Raised when a backend lacks required portable-core primitives."""

    def __init__(
        self,
        backend_name: str,
        missing_primitives: tuple[str, ...],
        profile_version: int,
    ) -> None:
        self.backend_name = backend_name
        self.missing_primitives = missing_primitives
        self.profile_version = profile_version
        missing = ", ".join(missing_primitives)
        # A CORE declaration appends to the global profile every provider must
        # satisfy, so declaring one outside Opaque makes every shipped provider
        # incomplete forever.  Name that cause instead of blaming the provider,
        # which is what the missing primitive's own module says.
        foreign = tuple(
            name for name in missing_primitives if not name.startswith("opaque.")
        )
        if foreign:
            super().__init__(
                f"Backend {backend_name!r} cannot activate because "
                f"{', '.join(foreign)} was declared as a CORE primitive outside "
                "Opaque. CORE is the profile every provider must implement in "
                "full, so adding to it makes the shipped providers incomplete. "
                "Declare extension primitives with "
                "`@primitive(tier=PrimitiveTier.OPTIONAL)` — the default — and "
                "guard their use with `.supports(...)`."
            )
            return
        super().__init__(
            f"Backend {backend_name!r} does not satisfy portable core profile "
            f"v{profile_version}; missing: {missing}."
        )


class PrimitiveTier(StrEnum):
    """Completeness tier for a primitive declaration."""

    CORE = "core"
    OPTIONAL = "optional"


class LazyImplementation:
    """An implementation target imported only when the primitive is called."""

    def __init__(self, target: str) -> None:
        module_name, separator, attribute = target.partition(":")
        if not separator or not module_name or not attribute or ":" in attribute:
            raise InvalidPrimitiveRegistrationError(
                "A lazy implementation target must be 'module:attribute'."
            )
        self.target = target
        self._module_name = module_name
        self._attribute = attribute
        self._resolved: Implementation | None = None
        self._lock = threading.Lock()

    @property
    def is_resolved(self) -> bool:
        """Whether the target has already been imported."""
        return self._resolved is not None

    def resolve(self) -> Implementation:
        """Import and cache the callable target."""
        if self._resolved is None:
            with self._lock:
                if self._resolved is None:
                    implementation = getattr(
                        importlib.import_module(self._module_name), self._attribute
                    )
                    if not callable(implementation):
                        raise InvalidPrimitiveRegistrationError(
                            f"Lazy target {self.target!r} is not callable."
                        )
                    self._resolved = implementation
        return self._resolved


def lazy_implementation(target: str) -> LazyImplementation:
    """Describe a callable target to resolve lazily during dispatch."""
    return LazyImplementation(target)


_PRIMITIVES: dict[str, Primitive] = {}
_PRIMITIVES_LOCK = threading.RLock()


class Primitive:
    """A canonical operation resolved from the active backend's stable name.

    Primitive declarations are callable descriptor objects. When created with
    :func:`primitive`, they retain the wrapped function's metadata and
    signature while dispatching to a provider implementation at runtime.
    """

    def __new__(
        cls,
        name: str,
        *,
        tier: PrimitiveTier | str = PrimitiveTier.OPTIONAL,
    ) -> Primitive:
        _validate_primitive_name(name)
        normalized_tier = _normalize_tier(tier)
        with _PRIMITIVES_LOCK:
            existing = _PRIMITIVES.get(name)
            if existing is not None:
                if existing.tier is not normalized_tier:
                    raise InvalidPrimitiveRegistrationError(
                        f"Primitive {name!r} is already declared with tier "
                        f"{existing.tier.value!r}."
                    )
                return existing
            primitive = super().__new__(cls)
            _PRIMITIVES[name] = primitive
            return primitive

    def __init__(
        self,
        name: str,
        *,
        tier: PrimitiveTier | str = PrimitiveTier.OPTIONAL,
    ) -> None:
        with _PRIMITIVES_LOCK:
            if getattr(self, "_initialized", False):
                return
            self.name = name
            self.tier = _normalize_tier(tier)
            self._implementations: dict[str, Implementation | LazyImplementation] = {}
            self._declaration: Implementation | None = None
            self._lock = threading.RLock()
            self._initialized = True
        if self.tier is PrimitiveTier.CORE:
            declare_core_primitives(self)

    def __repr__(self) -> str:
        return f"Primitive({self.name!r}, tier={self.tier.value!r})"

    def bind(self, declaration: Declaration) -> Primitive:
        """Attach a real function declaration and preserve its metadata."""
        if not callable(declaration):
            raise InvalidPrimitiveRegistrationError(
                f"Declaration for primitive {self.name!r} must be callable."
            )
        with self._lock:
            self._declaration = declaration
            update_wrapper(self, declaration)
        return self

    def register(
        self,
        backend_name: str,
        implementation: Implementation | LazyImplementation,
        *,
        replace: bool = False,
    ) -> None:
        """Register an eager or lazy implementation for ``backend_name``."""
        _validate_backend_name(backend_name)
        if not isinstance(implementation, LazyImplementation) and not callable(
            implementation
        ):
            raise InvalidPrimitiveRegistrationError(
                f"Implementation for primitive {self.name!r} must be callable or lazy."
            )
        with self._lock:
            if backend_name in self._implementations and not replace:
                raise DuplicatePrimitiveRegistrationError(
                    f"Primitive {self.name!r} already has an implementation for "
                    f"backend {backend_name!r}. Pass replace=True to replace it."
                )
            self._implementations[backend_name] = implementation

    def register_many(
        self,
        implementations: Mapping[str, Implementation | LazyImplementation],
        *,
        replace: bool = False,
    ) -> None:
        """Register implementations for multiple backend names."""
        for backend_name, implementation in implementations.items():
            self.register(backend_name, implementation, replace=replace)

    def supports(self, backend: Backend | str | None = None) -> bool:
        """Return whether ``backend`` has a registered implementation.

        Args:
            backend: A backend object, a backend name, or ``None`` for the
                active backend. Naming a first-party backend answers about the
                installation rather than about what this process happens to
                have imported — the provider is discovered first if it has not
                been already, so the answer does not depend on import order.

        Returns:
            Whether that backend implements this primitive.

        With ``None`` and no backend active the answer is ``False`` — nothing
        is selected, so nothing supports it — rather than an error. Name the
        backend when you mean "is this available at all"; leave it out when you
        mean "can the context I am in do this right now".
        """
        backend_name = _active_backend_name() if backend is None else None
        if backend is None and backend_name is None:
            return False
        if backend_name is None:
            backend_name = _backend_name(backend)
        with self._lock:
            if backend_name in self._implementations:
                return True
        _ensure_provider_discovered(backend_name)
        with self._lock:
            return backend_name in self._implementations

    def registered_backends(self) -> tuple[str, ...]:
        """Return registered backend names without resolving lazy targets."""
        with self._lock:
            return tuple(sorted(self._implementations))

    def resolve(self, backend: Backend | str | None = None) -> Implementation:
        """Resolve the callable implementation for ``backend``."""
        backend_name = _backend_name(backend)
        with self._lock:
            implementation = self._implementations.get(backend_name)
        if implementation is None:
            raise UnsupportedPrimitiveError(self.name, backend_name)
        if isinstance(implementation, LazyImplementation):
            return implementation.resolve()
        return implementation

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        """Infer or validate a backend, then dispatch its implementation.

        With a sticky active backend the per-call argument walk is skipped;
        set ``OPAQUE_VALIDATE_BACKEND_ARGS=1`` to re-enable full inference
        (mixed-backend and mismatch validation) on every dispatched call.
        """
        from opaque.api.engine.backend import _registry

        if _VALIDATE_DISPATCH_ARGS:
            backend = _registry.ensure_backend(args, kwargs)
        else:
            # Eager dispatch always consults the context-local ContextVar,
            # preserving the documented lifecycle (an unselected context
            # fails closed even when another context holds an active
            # backend). Inside a traced graph (torch.compile) ContextVar
            # reads are untraceable, so the traced branch trusts the
            # module-global mirror instead — sound because tracing runs
            # under the active backend and the mirror is only trusted
            # while a single backend name has ever been active in this
            # process (both are plain globals compiled graphs can guard).
            if _registry._SINGLE_BACKEND and _registry._IS_COMPILING():
                backend = _registry._ACTIVE_HINT
            else:
                backend = _registry._ACTIVE.get()
            if backend is None:
                backend = _registry.ensure_backend(args, kwargs)
        # Lock-free fast path: registrations are append-only, and dict reads
        # are atomic, so a hit needs no lock; misses — and instances whose
        # ``resolve`` was overridden (test stubs) — go through resolve().
        implementation = self._implementations.get(backend.name)
        if (
            implementation is None
            or isinstance(implementation, LazyImplementation)
            or "resolve" in self.__dict__
        ):
            return self.resolve(backend)(*args, **kwargs)
        return implementation(*args, **kwargs)


class BackendProvider:
    """Stable backend identity with primitive-bound registration decorators."""

    def __init__(self, backend: Backend | str) -> None:
        self.name = _backend_name(backend)

    def __repr__(self) -> str:
        return f"BackendProvider({self.name!r})"

    def implements(
        self,
        operation: Primitive,
        *,
        replace: bool = False,
    ) -> Callable[[Declaration], Declaration]:
        """Decorate and register an implementation for ``operation``."""
        if not isinstance(operation, Primitive):
            raise InvalidPrimitiveRegistrationError(
                "BackendProvider.implements() requires a Primitive object."
            )

        def register(implementation: Declaration) -> Declaration:
            operation.register(self.name, implementation, replace=replace)
            return implementation

        return register


@overload
def primitive(
    declaration: Declaration,
    *,
    tier: PrimitiveTier | str = PrimitiveTier.OPTIONAL,
    name: str | None = None,
) -> Primitive: ...


@overload
def primitive(
    declaration: None = None,
    *,
    tier: PrimitiveTier | str = PrimitiveTier.OPTIONAL,
    name: str | None = None,
) -> Callable[[Declaration], Primitive]: ...


def primitive(
    declaration: Declaration | None = None,
    *,
    tier: PrimitiveTier | str = PrimitiveTier.OPTIONAL,
    name: str | None = None,
) -> Primitive | Callable[[Declaration], Primitive]:
    """Declare a backend-dispatched primitive from a real function."""

    def decorate(function: Declaration) -> Primitive:
        operation = Primitive(name or _declaration_name(function), tier=tier)
        return operation.bind(function)

    if declaration is None:
        return decorate
    return decorate(declaration)


@dataclass(frozen=True)
class CoreProfile:
    """Versioned portable-core primitive requirements."""

    version: int
    primitives: tuple[Primitive, ...]


CORE_PROFILE_VERSION = 6
"""Version of the portable core contract declared by this engine release."""

CORE_PRIMITIVES: list[Primitive] = []
"""Portable primitives required before a backend may be activated."""


def declare_core_primitives(*primitives: Primitive) -> None:
    """Add canonical primitives to the portable activation profile.

    This is intended for the engine's core primitive declarations. Providers
    register implementations separately, and optional primitives never appear
    in this profile.
    """
    with _PRIMITIVES_LOCK:
        for primitive in primitives:
            if primitive not in CORE_PRIMITIVES:
                CORE_PRIMITIVES.append(primitive)


def core_profile() -> CoreProfile:
    """Return a snapshot of the versioned portable-core requirements."""
    with _PRIMITIVES_LOCK:
        return CoreProfile(CORE_PROFILE_VERSION, tuple(CORE_PRIMITIVES))


def validate_core_primitives(backend: Backend | str) -> None:
    """Raise when ``backend`` lacks a required portable primitive.

    Validation uses registration presence only, so activating a provider does
    not import optional lazy implementation modules.
    """
    backend_name = _backend_name(backend)
    profile = core_profile()
    missing = tuple(
        primitive.name
        for primitive in profile.primitives
        if not primitive.supports(backend_name)
    )
    if missing:
        raise IncompleteBackendError(backend_name, missing, profile.version)


def supports(primitive: Primitive, backend: Backend | str | None = None) -> bool:
    """Return whether a primitive is supported by ``backend``."""
    return primitive.supports(backend)


def registered_backends(primitive: Primitive) -> tuple[str, ...]:
    """Return the backends that registered ``primitive``."""
    return primitive.registered_backends()


def registered_primitives() -> tuple[Primitive, ...]:
    """Return all canonical primitive declarations in deterministic order."""
    with _PRIMITIVES_LOCK:
        return tuple(_PRIMITIVES[name] for name in sorted(_PRIMITIVES))


def _validate_primitive_name(name: str) -> None:
    if (
        not isinstance(name, str)
        or not name
        or name.startswith(".")
        or name.endswith(".")
    ):
        raise InvalidPrimitiveRegistrationError(
            "A primitive name must be a non-empty qualified identity."
        )
    if len(name.split(".")) < 2 or any(not part for part in name.split(".")):
        raise InvalidPrimitiveRegistrationError(
            "A primitive name must contain at least two non-empty dot-separated parts."
        )


def _declaration_name(declaration: Implementation) -> str:
    module = declaration.__module__
    engine_prefix = "opaque.api.engine."
    if module.startswith(engine_prefix):
        module = f"opaque.{module.removeprefix(engine_prefix)}"
    if module.endswith("._engine"):
        module = module.removesuffix("._engine")
    return f"{module}.{declaration.__name__}"


def _normalize_tier(tier: PrimitiveTier | str) -> PrimitiveTier:
    try:
        return PrimitiveTier(tier)
    except ValueError as exc:
        raise InvalidPrimitiveRegistrationError(
            f"Unknown primitive tier {tier!r}; expected 'core' or 'optional'."
        ) from exc


def _validate_backend_name(backend_name: str) -> None:
    if (
        not isinstance(backend_name, str)
        or not backend_name
        or backend_name.strip() != backend_name
    ):
        raise InvalidPrimitiveRegistrationError(
            "A backend name must be a non-empty string without surrounding whitespace."
        )


def _backend_name(backend: Backend | str | None) -> str:
    if backend is None:
        from opaque.api.engine.backend._registry import (
            BackendNotSelectedError,
            active_backend,
        )

        backend = active_backend()
        if backend is None:
            raise BackendNotSelectedError(
                "No backend is active, so there is nothing for the default "
                "`backend=None` to refer to. Name one explicitly — "
                "`supports(primitive, 'torch')` — or activate one with "
                "`set_backend(...)` / `ensure_backend(array)` first."
            )
    backend_name = (
        backend if isinstance(backend, str) else getattr(backend, "name", None)
    )
    _validate_backend_name(backend_name)
    return backend_name


def _active_backend_name() -> str | None:
    from opaque.api.engine.backend._registry import active_backend

    backend = active_backend()
    return None if backend is None else getattr(backend, "name", None)


_DISCOVERY_ATTEMPTED: set[object] = set()
"""First-party backends already offered to load, so a miss is tried once."""


def _ensure_provider_discovered(backend_name: str) -> None:
    """Load a named first-party provider so support checks see its registry.

    A provider registers its implementations when its backend is constructed,
    so a support check made before that answers about this process's import
    history rather than about the installation — ``supports(p, "torch")`` would
    be False at module scope and True after the first tensor touched the
    engine. A caller who names a backend is asking the second question ("is
    this usable here?"), so the provider is loaded before answering.

    Loading constructs the backend; it does not *activate* it, so a support
    check never changes which backend subsequent dispatch selects. A provider
    that is not installed, or a name that is not a first-party backend, leaves
    the answer at False.
    """
    from opaque.api.engine.backend import _registry

    try:
        kind = _registry.KnownBackend(backend_name)
    except ValueError:
        return
    with _PRIMITIVES_LOCK:
        if kind in _DISCOVERY_ATTEMPTED:
            return
        _DISCOVERY_ATTEMPTED.add(kind)
    try:
        _registry._load_backend(kind)
    except _registry.BackendProviderError:
        return


__all__ = [
    "BackendProvider",
    "CORE_PROFILE_VERSION",
    "CORE_PRIMITIVES",
    "CoreProfile",
    "DuplicatePrimitiveRegistrationError",
    "IncompleteBackendError",
    "InvalidPrimitiveRegistrationError",
    "LazyImplementation",
    "Primitive",
    "PrimitiveError",
    "PrimitiveTier",
    "UnsupportedPrimitiveError",
    "core_profile",
    "declare_core_primitives",
    "lazy_implementation",
    "primitive",
    "registered_backends",
    "registered_primitives",
    "supports",
    "validate_core_primitives",
]
