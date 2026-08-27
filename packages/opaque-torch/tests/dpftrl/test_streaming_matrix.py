"""Torch-native coverage for portable DP-FTRL streaming matrices."""

import numpy as np
import torch

from opaque.api.dpftrl.noise._blt_math import (
    BufferedToeplitz,
    as_streaming_matrix,
)
from opaque.api.dpftrl.noise._blt_math import (
    inverse_as_streaming_matrix as blt_inverse_as_streaming_matrix,
)
from opaque.api.dpftrl.noise._streaming_matrix import (
    diagonal,
    identity,
    momentum_sgd_matrix,
    multiply_array,
    multiply_streaming_matrices,
    prefix_sum,
)
from opaque.api.dpftrl.noise._toeplitz import (
    inverse_as_streaming_matrix as toeplitz_inverse_as_streaming_matrix,
)
from opaque.api.dpftrl.noise._toeplitz import (
    materialize_lower_triangular,
)


def test_native_core_matrices_preserve_torch_execution():
    values = torch.tensor([[1.0], [2.0], [3.0]], dtype=torch.float32)
    prefix = prefix_sum()
    composed = multiply_streaming_matrices(identity(), prefix)

    actual = multiply_array(composed, values)
    diagonal_actual = multiply_array(diagonal(np.array([2.0, 3.0, 4.0])), values)

    assert isinstance(actual, torch.Tensor)
    assert actual.dtype == values.dtype
    assert actual.device == values.device
    torch.testing.assert_close(
        actual, torch.tensor([[1.0], [3.0], [6.0]], dtype=torch.float32)
    )
    torch.testing.assert_close(
        diagonal_actual,
        torch.tensor([[2.0], [6.0], [12.0]], dtype=torch.float32),
    )


def test_prefix_sum_updates_pytree_state_functionally():
    matrix = prefix_sum()
    first = {
        "weight": torch.tensor([1.0, 2.0]),
        "nested": (torch.tensor([3.0]),),
    }
    second = {
        "weight": torch.tensor([4.0, 5.0]),
        "nested": (torch.tensor([6.0]),),
    }

    state = matrix.init_multiply(first)
    first_output, first_state = matrix.multiply_next(first, state)
    second_output, _ = matrix.multiply_next(second, first_state)

    torch.testing.assert_close(first_output["weight"], first["weight"])
    torch.testing.assert_close(first_state.leaf_states[1], first["weight"])
    torch.testing.assert_close(second_output["weight"], torch.tensor([5.0, 7.0]))
    torch.testing.assert_close(second_output["nested"][0], torch.tensor([9.0]))
    torch.testing.assert_close(first_state.leaf_states[1], first["weight"])


def test_momentum_uses_native_accumulation_dtype():
    matrix = momentum_sgd_matrix(
        momentum=0.5,
        learning_rates=np.array([1.0, 2.0, 3.0]),
    )
    values = torch.ones((3, 1), dtype=torch.float16)

    actual = multiply_array(matrix, values)

    assert actual.dtype == torch.float32
    assert actual.device == values.device
    torch.testing.assert_close(
        actual, torch.tensor([[1.0], [4.0], [9.25]], dtype=torch.float32)
    )


def test_toeplitz_inverse_matches_dense_and_keeps_old_state_unchanged():
    coef = np.array([1.0, 0.5, 0.25], dtype=np.float64)
    matrix = toeplitz_inverse_as_streaming_matrix(coef)
    values = torch.eye(3, dtype=torch.float64)

    actual = multiply_array(matrix, values)
    expected = torch.from_numpy(np.linalg.inv(materialize_lower_triangular(coef)))

    torch.testing.assert_close(actual, expected)

    state = matrix.init_multiply(values[0])
    _, first_state = matrix.multiply_next(values[0], state)
    snapshot = tuple(value.clone() for value in first_state)
    matrix.multiply_next(values[1], first_state)
    for value, original in zip(first_state, snapshot, strict=True):
        torch.testing.assert_close(value, original)


def test_blt_and_inverse_match_dense_with_native_low_precision_state():
    blt = BufferedToeplitz.build(
        buf_decay=[0.7, 0.3],
        output_scale=[0.4, 0.2],
    )
    values = torch.eye(5, dtype=torch.float16)

    matrix = as_streaming_matrix(blt)
    inverse_matrix = blt_inverse_as_streaming_matrix(blt)
    dense = multiply_array(matrix, values)
    inverse_dense = multiply_array(inverse_matrix, values)

    assert dense.dtype == torch.float32
    assert dense.device == values.device
    torch.testing.assert_close(
        dense @ inverse_dense,
        torch.eye(5, dtype=torch.float32),
        atol=2e-5,
        rtol=2e-5,
    )
