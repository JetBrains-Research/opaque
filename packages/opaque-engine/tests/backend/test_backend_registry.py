"""Backend-neutral tests for inference and backend lifecycle management."""

from __future__ import annotations

from dataclasses import dataclass

import pytest

import opaque.backend as facade
from opaque.api.engine.backend import (
    BackendMismatchError,
    BackendNotSelectedError,
    BackendProviderError,
    KnownBackend,
    MixedBackendError,
    _registry,
    active_backend,
    clear_backend,
    ensure_backend,
    set_backend,
    use_backend,
)


class _Backend:
    def __init__(self, name: str) -> None:
        self.name = name


_factory_calls: list[KnownBackend] = []
_validation_calls: list[tuple[str, str | None]] = []


def _backend_factory(kind: KnownBackend) -> _Backend:
    _factory_calls.append(kind)
    return _Backend(kind.value)


def _torch_backend() -> _Backend:
    return _backend_factory(KnownBackend.TORCH)


def _jax_backend() -> _Backend:
    return _backend_factory(KnownBackend.JAX)


def _mlx_backend() -> _Backend:
    return _backend_factory(KnownBackend.MLX)


def _validate_backend(backend: _Backend) -> None:
    active = active_backend()
    _validation_calls.append(
        (backend.name, active.name if active is not None else None)
    )


@pytest.fixture(autouse=True)
def _isolated_registry(monkeypatch: pytest.MonkeyPatch):
    clear_backend()
    _factory_calls.clear()
    _validation_calls.clear()
    module = __name__
    monkeypatch.setattr(_registry, "validate_core_primitives", _validate_backend)
    monkeypatch.setattr(
        _registry,
        "_BACKEND_FACTORY_TARGETS",
        {
            KnownBackend.TORCH: f"{module}:_torch_backend",
            KnownBackend.JAX: f"{module}:_jax_backend",
            KnownBackend.MLX: f"{module}:_mlx_backend",
        },
    )
    yield
    clear_backend()


TorchTensor = type("Tensor", (), {"__module__": "torch"})
JaxArray = type("ArrayImpl", (), {"__module__": "jaxlib._jax"})
MlxArray = type("array", (), {"__module__": "mlx.core"})


def test_registry_starts_unselected_and_requires_backend_bearing_values() -> None:
    assert active_backend() is None

    with pytest.raises(BackendNotSelectedError, match="Pass a Torch, JAX, or MLX"):
        ensure_backend(1, "neutral", object())


def test_first_use_loads_provider_once_and_sticks() -> None:
    backend = ensure_backend(TorchTensor())

    assert backend.name == "torch"
    assert active_backend() is backend
    assert ensure_backend(TorchTensor()) is backend
    assert ensure_backend(1, None) is backend
    assert _factory_calls == [KnownBackend.TORCH]
    assert _validation_calls == [("torch", None)]


def test_detection_recurses_through_containers_dataclasses_and_model_mro() -> None:
    torch_module = type("Module", (), {"__module__": "torch.nn.modules.module"})
    user_model = type("UserModel", (torch_module,), {"__module__": "user.models"})

    @dataclass
    class Payload:
        arrays: object

    backend = ensure_backend({"payload": Payload(arrays=[(user_model(),)])})

    assert backend.name == "torch"


def test_clear_backend_allows_a_different_backend_to_be_inferred() -> None:
    torch_backend = ensure_backend(TorchTensor())
    clear_backend()
    assert active_backend() is None

    jax_backend = ensure_backend(JaxArray())

    assert jax_backend.name == "jax"
    assert jax_backend is not torch_backend
    assert _factory_calls == [KnownBackend.TORCH, KnownBackend.JAX]


def test_use_backend_temporarily_overrides_and_restores_sticky_backend() -> None:
    sticky = ensure_backend(TorchTensor())
    temporary = _Backend("custom")

    with use_backend(temporary) as yielded:
        assert yielded is temporary
        assert active_backend() is temporary
        assert ensure_backend(object()) is temporary

    assert active_backend() is sticky


def test_explicit_custom_backend_can_accept_first_party_framework_values() -> None:
    custom = _Backend("custom")

    with use_backend(custom):
        assert ensure_backend(TorchTensor()) is custom


def test_use_backend_restores_unselected_state_after_exception() -> None:
    temporary = _Backend("custom")

    with pytest.raises(RuntimeError, match="boom"), use_backend(temporary):
        raise RuntimeError("boom")

    assert active_backend() is None


def test_mixed_backend_values_are_rejected_before_loading_a_provider() -> None:
    values = {"left": TorchTensor(), "right": [JaxArray(), MlxArray()]}

    with pytest.raises(MixedBackendError, match=r"torch, jax, mlx"):
        ensure_backend(values)

    assert active_backend() is None
    assert not _factory_calls


def test_active_backend_mismatch_is_actionable_and_does_not_replace_it() -> None:
    sticky = ensure_backend(TorchTensor())

    with pytest.raises(
        BackendMismatchError,
        match=r"JAX.*active backend is 'torch'.*clear_backend\(\).*use_backend",
    ):
        ensure_backend(JaxArray())

    assert active_backend() is sticky
    assert _factory_calls == [KnownBackend.TORCH]


def test_missing_provider_reports_backend_specific_install_guidance(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    targets = dict(_registry._BACKEND_FACTORY_TARGETS)
    targets[KnownBackend.MLX] = "opaque.api.missing_mlx.backend:mlx_backend"
    monkeypatch.setattr(_registry, "_BACKEND_FACTORY_TARGETS", targets)

    with pytest.raises(
        BackendProviderError,
        match=r"MLX.*opaque-mlx.*pip install opaque-mlx",
    ):
        ensure_backend(MlxArray())

    assert active_backend() is None


def test_set_backend_validates_and_persists_without_inference() -> None:
    backend = _Backend("custom")

    set_backend(backend)

    assert active_backend() is backend
    assert ensure_backend(object()) is backend
    assert _validation_calls == [("custom", None)]


def test_unselected_context_fails_closed_while_another_context_is_active() -> None:
    """Eager dispatch is context-local: a context that never selected a
    backend must fail closed even when the process has a single active
    backend elsewhere (the compile-only global mirror must not leak)."""
    import contextvars

    from opaque.api.engine.primitive import PrimitiveTier, primitive

    @primitive(tier=PrimitiveTier.OPTIONAL, name="opaque.test.context_isolation_probe")
    def probe() -> object:
        raise NotImplementedError

    backend = _Backend("custom")
    probe.register(backend.name, lambda: "custom-result")

    pristine = contextvars.copy_context()
    _registry._reset_loaded_backends()
    set_backend(backend)
    assert probe() == "custom-result"

    with pytest.raises(BackendNotSelectedError):
        pristine.run(probe)


def test_set_backend_accepts_known_backend_name() -> None:
    set_backend("torch")

    active = active_backend()
    assert active is not None
    assert active.name == "torch"
    assert _factory_calls == [KnownBackend.TORCH]
    assert _validation_calls == [("torch", None)]


def test_use_backend_accepts_known_backend_name() -> None:
    with use_backend("jax") as backend:
        assert backend.name == "jax"
        assert active_backend() is backend

    assert active_backend() is None
    assert _factory_calls == [KnownBackend.JAX]


def test_unknown_backend_name_is_rejected_without_activation() -> None:
    with pytest.raises(
        BackendProviderError, match=r"Unknown backend name 'numpy'.*'torch'"
    ):
        set_backend("numpy")

    assert active_backend() is None
    assert _factory_calls == []


def test_facade_reexports_complete_lifecycle_surface() -> None:
    assert facade.KnownBackend is KnownBackend
    assert facade.ensure_backend is ensure_backend
    assert facade.clear_backend is clear_backend
    assert facade.active_backend is active_backend
    assert facade.set_backend is set_backend
    assert facade.use_backend is use_backend
