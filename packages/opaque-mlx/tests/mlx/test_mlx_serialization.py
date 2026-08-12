"""Native MLX array serialization coverage."""

from __future__ import annotations

import pytest

from opaque import ops
from opaque.api.base.serialization import _registry as serializer_registry
from opaque.api.base.serialization import lookup_serializer
from opaque.api.engine.backend import clear_backend
from opaque.mlx import mlx_backend
from opaque.serialization import from_state_dict, state_dict

mx = pytest.importorskip("mlx.core")


@pytest.fixture(autouse=True)
def _reset_backend() -> None:
    clear_backend()
    yield
    clear_backend()


def test_explicit_activation_registers_nested_array_round_trip() -> None:
    mlx_backend()
    value = {
        "scalar": mx.array(3, dtype=mx.int32),
        "tree": [mx.arange(6, dtype=mx.float32).reshape(2, 3)],
    }
    template = {
        "scalar": mx.array(0, dtype=mx.int32),
        "tree": [mx.zeros((2, 3), dtype=mx.float32)],
    }

    restored = from_state_dict(template, state_dict(value))

    for expected, actual in zip(
        (value["scalar"], value["tree"][0]),
        (restored["scalar"], restored["tree"][0]),
        strict=True,
    ):
        mx.eval(actual, expected)
        assert isinstance(actual, mx.array)
        assert actual.dtype == expected.dtype
        assert actual.shape == expected.shape
        assert actual.tolist() == expected.tolist()


def test_automatic_activation_registers_array_serializer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    value = mx.array([1.0, 2.0])
    monkeypatch.delitem(serializer_registry._REGISTRY, mx.array, raising=False)
    assert lookup_serializer(mx.array) is None

    ops.square(value)

    assert lookup_serializer(mx.array) is not None


def test_array_loader_rejects_non_mlx_state() -> None:
    mlx_backend()

    with pytest.raises(TypeError, match="MLX array"):
        from_state_dict(mx.zeros(2), {"": [0.0, 0.0]})


def test_provider_factory_registration_is_idempotent() -> None:
    mlx_backend()
    first = lookup_serializer(mx.array)
    mlx_backend()

    assert first is not None
    assert lookup_serializer(mx.array) == first
