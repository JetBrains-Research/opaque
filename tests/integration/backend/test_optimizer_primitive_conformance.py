"""Cross-provider conformance tests for optimizer-oriented engine primitives."""

from __future__ import annotations

import numpy as np
import pytest
from tests.integration.backend._providers import provider_case

from opaque import ops
from opaque.api.engine.backend import clear_backend, use_backend


@pytest.fixture(autouse=True)
def _reset_backend() -> None:
    clear_backend()
    yield
    clear_backend()


@pytest.mark.parametrize("provider_name", ["torch", "jax", "mlx"])
def test_optimizer_primitives_match_numpy_reference(provider_name: str) -> None:
    case = provider_case(provider_name)
    value = case.array([1.0, 4.0, 16.0])

    with use_backend(case.backend):
        rsqrt_result = ops.rsqrt(value)
        pow_result = ops.pow(value, 3.0)
        reciprocal_result = ops.reciprocal(value)

    case.evaluate(rsqrt_result)
    case.evaluate(pow_result)
    case.evaluate(reciprocal_result)

    assert case.to_numpy(rsqrt_result) == pytest.approx(
        1.0 / np.sqrt(np.array([1.0, 4.0, 16.0]))
    )
    assert case.to_numpy(pow_result) == pytest.approx(np.array([1.0, 64.0, 4096.0]))
    assert case.to_numpy(reciprocal_result) == pytest.approx(
        1.0 / np.array([1.0, 4.0, 16.0])
    )


@pytest.mark.parametrize("provider_name", ["torch", "jax", "mlx"])
def test_mean_reduces_consistently_across_providers(provider_name: str) -> None:
    case = provider_case(provider_name)
    value = case.array([[1.0, 2.0], [3.0, 4.0]])

    with use_backend(case.backend):
        global_mean = ops.mean(value)
        axis_mean = ops.mean(value, axis=1)

    case.evaluate(global_mean)
    case.evaluate(axis_mean)

    assert float(case.to_numpy(global_mean)) == pytest.approx(2.5)
    assert case.to_numpy(axis_mean).tolist() == pytest.approx([1.5, 3.5])


@pytest.mark.parametrize("provider_name", ["torch", "jax", "mlx"])
def test_accumulator_dtype_respects_provider_conventions(provider_name: str) -> None:
    case = provider_case(provider_name)

    with use_backend(case.backend):
        low_precision = (case.dtype("float16"), case.dtype("bfloat16"))
        for dtype in low_precision:
            assert ops.accumulator_dtype(dtype) == case.dtype("float32")

        if provider_name == "mlx":
            assert ops.accumulator_dtype(case.dtype("float32")) == case.dtype("float32")
        elif provider_name == "jax":
            import jax

            expected = (
                case.dtype("float64")
                if jax.config.x64_enabled
                else case.dtype("float32")
            )
            assert ops.accumulator_dtype(case.dtype("float32")) == expected
        else:
            assert ops.accumulator_dtype(case.dtype("float32")) == case.dtype("float64")
