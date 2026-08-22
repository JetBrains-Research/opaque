"""Context-local backend inference and lifecycle management."""

from __future__ import annotations

import importlib
import sys
from collections.abc import Mapping
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import fields, is_dataclass
from enum import StrEnum
from typing import TYPE_CHECKING

from opaque.api.engine.primitive import validate_core_primitives

if TYPE_CHECKING:
    from collections.abc import Iterator

    from opaque.api.engine.backend._protocol import Backend


class KnownBackend(StrEnum):
    """First-party backends that can be inferred from argument types."""

    TORCH = "torch"
    JAX = "jax"
    MLX = "mlx"


class BackendError(RuntimeError):
    """Base class for backend selection failures."""


class BackendNotSelectedError(BackendError):
    """Raised when neither arguments nor context identify a backend."""


class MixedBackendError(BackendError):
    """Raised when one call contains values from multiple backends."""


class BackendMismatchError(BackendError):
    """Raised when arguments conflict with the sticky active backend."""


class BackendProviderError(BackendError, ImportError):
    """Raised when an inferred first-party backend provider cannot be loaded."""


_BACKEND_FACTORY_TARGETS: Mapping[KnownBackend, str] = {
    KnownBackend.TORCH: "opaque.api.torch.backend:torch_backend",
    KnownBackend.JAX: "opaque.api.jax.backend:jax_backend",
    KnownBackend.MLX: "opaque.api.mlx.backend:mlx_backend",
}

_BACKEND_PROVIDER_PACKAGES: Mapping[KnownBackend, str] = {
    KnownBackend.TORCH: "opaque.api.torch",
    KnownBackend.JAX: "opaque.api.jax",
    KnownBackend.MLX: "opaque.api.mlx",
}

# Top-level runtime modules a provider is built on. A framework absent from
# ``sys.modules`` cannot own any live object in this process, so the engine
# can rule its provider out with a dict lookup instead of an import.
_BACKEND_RUNTIME_ROOTS: Mapping[KnownBackend, tuple[str, ...]] = {
    KnownBackend.TORCH: ("torch",),
    KnownBackend.JAX: ("jax", "jaxlib"),
    KnownBackend.MLX: ("mlx",),
}

_INSTALL_GUIDANCE: Mapping[KnownBackend, str] = {
    KnownBackend.TORCH: "Install opaque-torch with `pip install opaque-torch`.",
    KnownBackend.JAX: "Install opaque-jax with `pip install opaque-jax`.",
    KnownBackend.MLX: "Install opaque-mlx with `pip install opaque-mlx`.",
}

_ACTIVE: ContextVar[Backend | None] = ContextVar("opaque_active_backend", default=None)

# Process-wide record of every backend name that has been activated, plus
# a plain-global mirror of the most recent activation. While at most one
# backend name has ever been active, per-context divergence is impossible,
# so dispatch layers may trust the mirror on their hot path — a module
# global read that compiled graphs (torch.compile fullgraph) can trace,
# unlike ``ContextVar.get``. Multi-backend processes fall back to the
# context-local source of truth.
_LOADED_BACKEND_NAMES: set[str] = set()
_SINGLE_BACKEND: bool = True
_ACTIVE_HINT: Backend | None = None


def _never_compiling() -> bool:
    return False


# Graph-tracing detector for the dispatch fast path. Providers with a graph
# compiler install their own detector at activation (Torch installs
# ``torch.compiler.is_compiling``). Inside a traced graph, ContextVar reads
# are untraceable, so dispatch trusts the module-global mirror there; in
# eager execution dispatch always consults the context-local ContextVar.
_IS_COMPILING = _never_compiling


def _set_compiling_detector(detector) -> None:
    """Install ``detector`` as the process-wide graph-tracing probe."""
    global _IS_COMPILING
    _IS_COMPILING = detector


def _note_backend_loaded(name: str) -> None:
    global _SINGLE_BACKEND
    _LOADED_BACKEND_NAMES.add(name)
    _SINGLE_BACKEND = len(_LOADED_BACKEND_NAMES) <= 1


def _reset_loaded_backends() -> None:
    """Testing hook: forget which backends this process has activated.

    Production activation history is deliberately monotonic; test suites
    that register throwaway backends call this between tests so the
    single-backend fast path stays representative of real processes.
    """
    global _SINGLE_BACKEND, _ACTIVE_HINT
    _LOADED_BACKEND_NAMES.clear()
    active = _ACTIVE.get()
    if active is not None:
        _LOADED_BACKEND_NAMES.add(active.name)
    _SINGLE_BACKEND = len(_LOADED_BACKEND_NAMES) <= 1
    _ACTIVE_HINT = active


def active_backend() -> Backend | None:
    """Return the active backend, or ``None`` when the context is unselected."""
    return _ACTIVE.get()


def ensure_backend(*values: object) -> Backend:
    """Infer, activate, or validate the backend for ``values``.

    Framework labels are collected from all values before any provider is
    loaded. Once selected, a backend remains active in the current context
    until :func:`clear_backend` is called or :func:`use_backend` overrides it.
    """
    inferred = _infer_backends(values)
    if len(inferred) > 1:
        names = ", ".join(backend.value for backend in inferred)
        raise MixedBackendError(
            f"Backend-bearing arguments use mixed backends: {names}. "
            "Pass values from one backend only."
        )

    active = _ACTIVE.get()
    if inferred:
        detected = inferred[0]
        if active is not None:
            known_active = next(
                (kind for kind in KnownBackend if kind.value == active.name), None
            )
            if known_active is not None and known_active is not detected:
                raise BackendMismatchError(
                    f"Arguments belong to {detected.value.upper()}, but the active "
                    f"backend is {active.name!r}. Call clear_backend() before "
                    "selecting a new sticky backend, or use use_backend(...) for "
                    "a temporary override."
                )
            return active
        backend = _load_backend(detected)
        set_backend(backend)
        return backend

    if active is None:
        raise BackendNotSelectedError(
            "No backend is active and the arguments do not identify one. Pass a "
            "Torch, JAX, or MLX array or model, call set_backend(...), or use "
            "use_backend(...)."
        )
    return active


def _coerce_backend(backend: Backend | str) -> Backend:
    """Resolve a known backend name to a loaded provider instance.

    Strings must name a :class:`KnownBackend`; the provider factory runs
    (importing and registering its implementations) before activation.
    Custom providers pass their :class:`Backend` instance directly.
    """
    if not isinstance(backend, str):
        return backend
    try:
        kind = KnownBackend(backend)
    except ValueError:
        known = ", ".join(repr(kind.value) for kind in KnownBackend)
        raise BackendProviderError(
            f"Unknown backend name {backend!r}; known names are {known}. "
            "Pass a Backend instance to activate a custom provider."
        ) from None
    return _load_backend(kind)


def set_backend(backend: Backend | str) -> None:
    """Activate ``backend`` — an instance or a known name — in this context."""
    global _ACTIVE_HINT
    backend = _coerce_backend(backend)
    validate_core_primitives(backend)
    _note_backend_loaded(backend.name)
    _ACTIVE.set(backend)
    _ACTIVE_HINT = backend


def clear_backend() -> None:
    """Return the current execution context to its unselected state."""
    global _ACTIVE_HINT
    _ACTIVE.set(None)
    _ACTIVE_HINT = None


@contextmanager
def use_backend(backend: Backend | str) -> Iterator[Backend]:
    """Temporarily activate ``backend`` and restore the context token on exit."""
    global _ACTIVE_HINT
    backend = _coerce_backend(backend)
    validate_core_primitives(backend)
    _note_backend_loaded(backend.name)
    token = _ACTIVE.set(backend)
    _ACTIVE_HINT = backend
    try:
        yield backend
    finally:
        _ACTIVE.reset(token)
        _ACTIVE_HINT = _ACTIVE.get()


def loaded_runtime_backends() -> tuple[KnownBackend, ...]:
    """Return the first-party backends whose runtime is already imported.

    A backend absent from this tuple owns nothing live in this process:
    no code has imported the framework that would create it. The check is
    a ``sys.modules`` lookup and never triggers an import, so a
    provider-free install — or any process that simply has not touched a
    framework — pays nothing and discovers nothing.
    """
    return tuple(
        kind
        for kind in KnownBackend
        if any(root in sys.modules for root in _BACKEND_RUNTIME_ROOTS[kind])
    )


def import_provider_package(kind: KnownBackend) -> bool:
    """Import ``kind``'s provider package; return whether it is installed.

    Importing the package runs the provider's registration side effects —
    serialization handlers, process-group probes — without constructing
    the backend or touching its implementation modules. Providers that are
    not installed return ``False``; the framework itself is never imported
    by this call.
    """
    try:
        importlib.import_module(_BACKEND_PROVIDER_PACKAGES[kind])
    except ImportError:
        return False
    return True


def _infer_backends(values: object) -> tuple[KnownBackend, ...]:
    inferred: set[KnownBackend] = set()
    seen: set[int] = set()
    _collect_backends(values, inferred, seen)
    return tuple(backend for backend in KnownBackend if backend in inferred)


def _collect_backends(
    value: object,
    inferred: set[KnownBackend],
    seen: set[int],
) -> None:
    value_id = id(value)
    if value_id in seen:
        return
    seen.add(value_id)

    classes = value.__mro__ if isinstance(value, type) else type(value).__mro__
    for cls in classes:
        backend = _backend_from_module(getattr(cls, "__module__", ""))
        if backend is not None:
            inferred.add(backend)

    if isinstance(value, Mapping):
        for key, item in value.items():
            _collect_backends(key, inferred, seen)
            _collect_backends(item, inferred, seen)
    elif isinstance(value, (list, tuple, set, frozenset)):
        for item in value:
            _collect_backends(item, inferred, seen)
    elif not isinstance(value, type) and is_dataclass(value):
        for field in fields(value):
            _collect_backends(getattr(value, field.name), inferred, seen)


def _backend_from_module(module_name: str) -> KnownBackend | None:
    root = module_name.partition(".")[0]
    if root == "torch":
        return KnownBackend.TORCH
    if root in {"jax", "jaxlib"}:
        return KnownBackend.JAX
    if root == "mlx":
        return KnownBackend.MLX
    return None


def _load_backend(kind: KnownBackend) -> Backend:
    target = _BACKEND_FACTORY_TARGETS[kind]
    module_name, separator, attribute = target.partition(":")
    if not separator or not module_name or not attribute:
        raise BackendProviderError(
            f"Invalid factory target {target!r} for {kind.value.upper()}."
        )
    try:
        module = importlib.import_module(module_name)
        factory = getattr(module, attribute)
        backend = factory()
    except (ImportError, AttributeError) as exc:
        raise BackendProviderError(
            f"{kind.value.upper()} backend provider is unavailable. "
            f"{_INSTALL_GUIDANCE[kind]}"
        ) from exc
    if getattr(backend, "name", None) != kind.value:
        raise BackendProviderError(
            f"Factory {target!r} returned backend "
            f"{getattr(backend, 'name', None)!r}; expected {kind.value!r}."
        )
    return backend


__all__ = [
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
