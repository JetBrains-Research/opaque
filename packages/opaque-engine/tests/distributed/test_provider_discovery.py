"""Process-group queries answer truthfully before any value selects a backend.

The engine reports the single-process defaults when nothing can tell it a
group is live. These tests pin the two halves of that contract: discovery
is inert — not merely harmless — when no framework is loaded, and a live
group is reported together with the provider that owns it.
"""

from __future__ import annotations

import sys
from typing import TYPE_CHECKING

import pytest

from opaque.api.engine import runtime
from opaque.api.engine.backend import (
    KnownBackend,
    _registry,
    active_backend,
    clear_backend,
)
from opaque.api.engine.distributed import collectives
from opaque.api.engine.primitive import BackendProvider

if TYPE_CHECKING:
    from collections.abc import Iterator


class _Backend:
    """Stand-in provider: a distinct name, with the runtime profile it needs."""

    name = "probe-provider"


_PROVIDER = BackendProvider(_Backend.name)
# What the stand-in provider's process group currently looks like. The probe
# and the dispatched primitives read the same holder, so the answer stays
# consistent once a query has activated the provider.
_GROUP: list[tuple[int, int] | None] = [None]


@_PROVIDER.implements(runtime.distributed_initialized)
def _initialized() -> bool:
    return _GROUP[0] is not None


@_PROVIDER.implements(runtime.distributed_rank)
def _rank() -> int:
    return 0 if _GROUP[0] is None else _GROUP[0][0]


@_PROVIDER.implements(runtime.distributed_world_size)
def _world_size() -> int:
    return 1 if _GROUP[0] is None else _GROUP[0][1]


def _probe() -> tuple[int, int] | None:
    return _GROUP[0]


def _probe_backend() -> _Backend:
    return _Backend()


@pytest.fixture(autouse=True)
def _isolated(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Start from an unselected context, an empty probe list, and no framework."""
    clear_backend()
    monkeypatch.setattr(_registry, "validate_core_primitives", lambda backend: None)
    monkeypatch.setattr(collectives, "_DISTRIBUTED_PROBES", [])
    monkeypatch.setattr(collectives, "_DISCOVERY_ATTEMPTED", set())
    _GROUP[0] = None
    for roots in _registry._BACKEND_RUNTIME_ROOTS.values():
        for root in roots:
            monkeypatch.delitem(sys.modules, root, raising=False)
    yield
    _GROUP[0] = None
    clear_backend()


def _record_imports(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """Record provider-package imports; report every provider as absent."""
    attempted: list[str] = []

    def _absent(name: str) -> None:
        attempted.append(name)
        raise ModuleNotFoundError(name)

    monkeypatch.setattr(_registry.importlib, "import_module", _absent)
    return attempted


def _register_on_import(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """Stand in for a provider package that registers its probe when imported."""
    attempted: list[str] = []

    def _import(name: str) -> None:
        attempted.append(name)
        collectives.register_distributed_probe(_probe, backend_factory=_probe_backend)

    monkeypatch.setattr(_registry.importlib, "import_module", _import)
    return attempted


def test_discovery_is_inert_when_no_framework_is_loaded(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """With no framework in ``sys.modules`` nothing is imported at all."""
    attempted = _record_imports(monkeypatch)

    assert _registry.loaded_runtime_backends() == ()
    assert collectives.is_distributed() is False
    assert collectives.get_rank() == 0
    assert collectives.get_world_size() == 1
    assert collectives.is_main_process() is True
    collectives.barrier()

    assert attempted == []
    assert active_backend() is None


def test_discovery_skips_a_framework_whose_provider_is_absent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A loaded framework with no installed provider keeps the defaults."""
    monkeypatch.setitem(sys.modules, "jax", object())
    attempted = _record_imports(monkeypatch)

    assert _registry.loaded_runtime_backends() == (KnownBackend.JAX,)
    assert collectives.is_distributed() is False
    assert collectives.get_world_size() == 1

    assert attempted == ["opaque.api.jax"]
    assert active_backend() is None


def test_an_absent_provider_is_not_re_imported_on_every_query(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setitem(sys.modules, "jax", object())
    attempted = _record_imports(monkeypatch)

    for _ in range(5):
        assert collectives.is_distributed() is False

    assert attempted == ["opaque.api.jax"]


def test_discovery_imports_the_provider_of_a_loaded_framework(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A framework this process loaded gets its provider package imported."""
    monkeypatch.setitem(sys.modules, "torch", object())
    attempted = _register_on_import(monkeypatch)
    _GROUP[0] = (3, 8)

    assert collectives.is_distributed() is True
    assert collectives.get_rank() == 3
    assert collectives.get_world_size() == 8
    assert collectives.is_main_process() is False
    assert attempted == ["opaque.api.torch"]


def test_a_reported_group_activates_the_provider_that_owns_it() -> None:
    """Scalar collectives after a positive answer have a backend to dispatch on."""
    collectives.register_distributed_probe(_probe, backend_factory=_probe_backend)
    _GROUP[0] = (1, 2)

    assert active_backend() is None
    assert collectives.is_distributed() is True
    assert active_backend() is not None
    assert active_backend().name == _Backend.name


def test_a_quiet_probe_does_not_activate_a_backend() -> None:
    """No live group means no inference: the context stays unselected."""
    collectives.register_distributed_probe(_probe, backend_factory=_probe_backend)

    assert collectives.is_distributed() is False
    assert collectives.get_world_size() == 1
    assert active_backend() is None


def test_probe_registration_requires_a_backend_factory() -> None:
    with pytest.raises(TypeError):
        collectives.register_distributed_probe(_probe)  # type: ignore[call-arg]


def test_a_group_started_after_a_negative_answer_is_still_found(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The discovery memo caches the import attempt, never the answer."""
    monkeypatch.setitem(sys.modules, "torch", object())
    attempted = _register_on_import(monkeypatch)

    assert collectives.is_distributed() is False
    _GROUP[0] = (0, 4)
    assert collectives.is_distributed() is True
    assert collectives.get_world_size() == 4
    assert attempted == ["opaque.api.torch"]
