"""Native JAX array serialization coverage."""

from __future__ import annotations

import numpy as np
import pytest

from opaque import ops
from opaque.api.base.serialization import _registry as serializer_registry
from opaque.api.base.serialization import lookup_serializer
from opaque.api.engine.backend import clear_backend
from opaque.jax import jax_backend
from opaque.serialization import from_state_dict, state_dict

jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")


@pytest.fixture(autouse=True)
def _reset_backend() -> None:
    clear_backend()
    yield
    clear_backend()


def test_explicit_activation_registers_nested_array_round_trip() -> None:
    jax_backend()
    value = {
        "scalar": jnp.array(3, dtype=jnp.int32),
        "tree": [jnp.arange(6, dtype=jnp.float32).reshape(2, 3)],
    }
    template = {
        "scalar": jnp.array(0, dtype=jnp.int32),
        "tree": [jnp.zeros((2, 3), dtype=jnp.float32)],
    }

    restored = from_state_dict(template, state_dict(value))

    for expected, actual in zip(
        (value["scalar"], value["tree"][0]),
        (restored["scalar"], restored["tree"][0]),
        strict=True,
    ):
        assert isinstance(actual, jax.Array)
        assert actual.dtype == expected.dtype
        assert actual.shape == expected.shape
        np.testing.assert_array_equal(actual, expected)


def test_automatic_activation_registers_array_serializer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    value = jnp.array([1.0, 2.0])
    monkeypatch.delitem(serializer_registry._REGISTRY, jax.Array, raising=False)
    assert lookup_serializer(jax.Array) is None

    ops.square(value)

    assert lookup_serializer(jax.Array) is not None


def test_array_loader_rejects_non_jax_state() -> None:
    jax_backend()

    with pytest.raises(TypeError, match="JAX array"):
        from_state_dict(jnp.zeros(2), {"": np.zeros(2)})


def test_provider_factory_registration_is_idempotent() -> None:
    jax_backend()
    first = lookup_serializer(jax.Array)
    jax_backend()

    assert first is not None
    assert lookup_serializer(jax.Array) == first
