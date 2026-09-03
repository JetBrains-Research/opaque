"""MLX-native optional execution-transform coverage."""

from __future__ import annotations

import mlx.core as mx
import numpy as np

from opaque.execution import (
    ExecutionProfile,
    checkpoint,
    compile,
    optimize_saved_activations,
)


def test_mlx_execution_profiles_are_available() -> None:
    assert ExecutionProfile.COMPILATION.supports("mlx")
    assert ExecutionProfile.CHECKPOINTING.supports("mlx")
    assert ExecutionProfile.SAVED_ACTIVATIONS.supports("mlx")


def test_mlx_compile_preserves_lazy_array_values() -> None:
    compiled = compile(lambda value: (value * value) + 1)

    result = compiled(mx.array([1.0, 2.0], dtype=mx.float32))
    mx.eval(result)

    np.testing.assert_array_equal(np.array(result), [2.0, 5.0])


def test_mlx_checkpoint_preserves_gradients() -> None:
    checkpointed = checkpoint(lambda value: mx.sum(value * value))
    gradient = mx.grad(checkpointed)(mx.array([3.0, 4.0], dtype=mx.float32))
    mx.eval(gradient)

    np.testing.assert_array_equal(np.array(gradient), [6.0, 8.0])


def test_mlx_saved_activation_transform_preserves_execution() -> None:
    optimized = optimize_saved_activations(lambda value: value + 1)
    result = optimized(mx.array([1.0, 2.0], dtype=mx.float32))
    mx.eval(result)

    np.testing.assert_array_equal(np.array(result), [2.0, 3.0])
