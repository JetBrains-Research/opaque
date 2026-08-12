"""Cross-provider capability and integration conformance matrix."""

from __future__ import annotations

from typing import Any

import pytest
from tests.integration.backend._providers import provider_case

from opaque import ops
from opaque.api.engine import runtime
from opaque.api.engine.backend import (
    active_backend,
    clear_backend,
    ensure_backend,
    use_backend,
)
from opaque.api.engine.primitive import core_profile
from opaque.serialization import from_state_dict, state_dict

_OPTIONAL_CAPABILITIES = {
    "torch": (True, True, True),
    "jax": (False, False, True),
    "mlx": (True, True, False),
}


@pytest.fixture(autouse=True)
def _reset_backend() -> None:
    clear_backend()
    yield
    clear_backend()


@pytest.mark.parametrize("provider_name", ["torch", "jax", "mlx"])
def test_first_party_provider_capability_matrix(provider_name: str) -> None:
    case = provider_case(provider_name)
    clear_cache, reset_peak, trace = _OPTIONAL_CAPABILITIES[provider_name]

    assert all(
        primitive.supports(case.backend) for primitive in core_profile().primitives
    )
    assert runtime.RuntimeProfile.DISTRIBUTED.supports(case.backend)
    assert runtime.RuntimeProfile.OBSERVABILITY.supports(case.backend)
    assert runtime.clear_memory_cache.supports(case.backend) is clear_cache
    assert runtime.reset_peak_memory.supports(case.backend) is reset_peak
    assert runtime.trace_scope.supports(case.backend) is trace

    with use_backend(case.backend):
        assert runtime.synchronize() is None
        assert isinstance(runtime.memory_stats(), runtime.MemoryStats)


@pytest.mark.parametrize("provider_name", ["torch", "jax", "mlx"])
def test_nested_native_value_activates_matching_provider(provider_name: str) -> None:
    case = provider_case(provider_name)

    backend = ensure_backend({"params": [case.value]})

    assert backend.name == provider_name
    assert active_backend() is backend
    result = ops.square(case.value)
    assert isinstance(result, case.array_type)
    assert _tolist(result) == [1.0, 4.0]


@pytest.mark.parametrize("provider_name", ["torch", "jax", "mlx"])
def test_first_party_native_arrays_round_trip_nested_state(provider_name: str) -> None:
    case = provider_case(provider_name)
    value = {"params": [case.value]}
    with use_backend(case.backend):
        template = {"params": [ops.zeros_like(case.value)]}

    restored = from_state_dict(template, state_dict(value))["params"][0]

    assert isinstance(restored, case.array_type)
    assert restored.dtype == case.value.dtype
    assert restored.shape == case.value.shape
    assert _tolist(restored) == _tolist(case.value)


def _tolist(value: Any) -> Any:
    return value.tolist()
