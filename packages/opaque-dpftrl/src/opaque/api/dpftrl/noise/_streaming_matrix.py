"""Memory-efficient operations with lower-triangular streaming matrices."""

from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING, Any

import numpy as np

from opaque.api.engine import ops, pytree

if TYPE_CHECKING:
    from collections.abc import Callable

    from numpy.typing import ArrayLike, NDArray


@dataclasses.dataclass(frozen=True)
class _PytreeState:
    treedef: Any
    leaf_states: tuple[Any, ...]


def _is_host_array(value: object) -> bool:
    return isinstance(value, np.ndarray)


def _zeros_like(value: Any, *, dtype: Any = None) -> Any:
    if _is_host_array(value):
        return np.zeros_like(value, dtype=dtype)
    zero = ops.zeros_like(value)
    return zero if dtype is None else ops.astype(zero, dtype)


def _add(left: Any, right: Any) -> Any:
    if _is_host_array(left) or _is_host_array(right):
        return left + right
    return ops.add(left, right)


def _multiply(left: Any, right: Any) -> Any:
    if _is_host_array(left) or _is_host_array(right):
        return left * right
    return ops.multiply(left, right)


def _scalar_like(value: Any, like: Any, *, dtype: Any = None) -> Any:
    if _is_host_array(like):
        return np.asarray(value, dtype=dtype or like.dtype)
    return ops.scalar(value, dtype=dtype or ops.dtype(like), like=like)


@dataclasses.dataclass(frozen=True)
class StreamingMatrix:
    """A lower-triangular linear mapping that is evaluated one row at a time."""

    init_multiply: Callable[[Any], Any]
    multiply_next: Callable[[Any, Any], tuple[Any, Any]]

    @classmethod
    def from_array_implementation(
        cls,
        init_multiply_fn: Callable[[Any], Any],
        multiply_next_fn: Callable[[Any, Any], tuple[Any, Any]],
    ) -> StreamingMatrix:
        """Lift single-array init/next functions to native-array PyTrees."""

        def lifted_init(abstract_value):
            if _is_host_array(abstract_value):
                return init_multiply_fn(abstract_value)

            if ops.is_array(abstract_value):
                return init_multiply_fn(abstract_value)

            flat_values, treedef = pytree.tree_flatten(abstract_value)
            return _PytreeState(
                treedef=treedef,
                leaf_states=tuple(init_multiply_fn(value) for value in flat_values),
            )

        def lifted_next(value, state):
            if _is_host_array(value):
                return multiply_next_fn(value, state)

            if ops.is_array(value):
                return multiply_next_fn(value, state)

            if not isinstance(state, _PytreeState):
                raise TypeError("PyTree streaming state does not match the input value")
            flat_values, _ = pytree.tree_flatten(value)
            outputs = []
            next_states = []
            for leaf, leaf_state in zip(flat_values, state.leaf_states, strict=True):
                output, next_state = multiply_next_fn(leaf, leaf_state)
                outputs.append(output)
                next_states.append(next_state)
            return (
                pytree.tree_unflatten(state.treedef, outputs),
                _PytreeState(state.treedef, tuple(next_states)),
            )

        return cls(lifted_init, lifted_next)

    def materialize(self, n: int) -> NDArray[np.float64]:
        """Materialize the leading ``n x n`` block as a host NumPy array."""
        return multiply_array(self, np.eye(n, dtype=np.float64))

    def row_norms_squared(self, n: int) -> NDArray[np.float64]:
        """Compute squared L2 norms of the first ``n`` rows on the host."""
        zero = np.zeros(n, dtype=np.float64)
        state = self.init_multiply(zero)
        norms = []
        for row_index in range(n):
            basis_row = np.eye(1, n, row_index, dtype=np.float64)[0]
            row, state = self.multiply_next(basis_row, state)
            norms.append(np.dot(row, row))
        return np.asarray(norms, dtype=np.float64)

    def __matmul__(self, other):
        """Multiply by another StreamingMatrix or a native/host array."""
        if isinstance(other, StreamingMatrix):
            return multiply_streaming_matrices(self, other)
        if _is_host_array(other) or ops.is_array(other):
            return multiply_array(self, other)
        return NotImplemented

    def __mul__(self, other: float) -> StreamingMatrix:
        """Multiply by a scalar."""
        return multiply_streaming_matrices(
            self, diagonal(np.asarray([other], dtype=np.float64))
        )


def scale_rows_and_columns(
    matrix: StreamingMatrix,
    row_scale: Any = None,
    col_scale: Any = None,
) -> StreamingMatrix:
    """Return a matrix with optional row and column scaling."""
    result = matrix
    if row_scale is not None:
        result = multiply_streaming_matrices(diagonal(row_scale), result)
    if col_scale is not None:
        result = multiply_streaming_matrices(result, diagonal(col_scale))
    return result


def multiply_array(A: StreamingMatrix, x: Any) -> Any:
    """Compute ``A @ x`` along the leading sequence axis."""
    if _is_host_array(x):
        n = x.shape[0]
        item = x.__getitem__
    else:
        n = ops.shape(x)[0]

        def item(index):
            return ops.slice_array(x, index)

    if n == 0:
        raise ValueError("StreamingMatrix input must have at least one element")

    state = A.init_multiply(item(0))
    results = []
    for index in range(n):
        result_slice, state = A.multiply_next(item(index), state)
        results.append(result_slice)

    if _is_host_array(x):
        return np.stack(results)
    return ops.stack(results, axis=0)


def multiply_streaming_matrices(
    A: StreamingMatrix,
    B: StreamingMatrix,
) -> StreamingMatrix:
    """Compose two streaming matrices and return ``A @ B``."""

    def init_multiply(abstract_value):
        return A.init_multiply(abstract_value), B.init_multiply(abstract_value)

    def multiply_next(value, state):
        A_state, B_state = state
        inner, next_B_state = B.multiply_next(value, B_state)
        outer, next_A_state = A.multiply_next(inner, A_state)
        return outer, (next_A_state, next_B_state)

    return StreamingMatrix(init_multiply, multiply_next)


def identity() -> StreamingMatrix:
    """Create an identity StreamingMatrix."""
    return StreamingMatrix(lambda _: (), lambda value, _: (value, ()))


def prefix_sum() -> StreamingMatrix:
    """Create the lower-triangular all-ones prefix-sum matrix."""

    def init_multiply(abstract_value):
        return _zeros_like(abstract_value)

    def multiply_next(value, state):
        result = _add(state, value)
        return result, result

    return StreamingMatrix.from_array_implementation(init_multiply, multiply_next)


def diagonal(diag: Any) -> StreamingMatrix:
    """Create an infinite diagonal matrix, repeating the final coefficient."""
    native_diag = diag if ops.is_array(diag) else None
    host_diag = None if native_diag is not None else np.asarray(diag)
    length = ops.shape(native_diag)[0] if native_diag is not None else len(host_diag)
    if length == 0:
        raise ValueError("diag must contain at least one element")

    def init_fn(_):
        return 0

    def next_fn(value, index):
        coefficient_index = min(index, length - 1)
        if native_diag is None:
            coefficient = _scalar_like(host_diag[coefficient_index], value)
        else:
            coefficient = ops.slice_array(native_diag, coefficient_index)
        return _multiply(value, coefficient), index + 1

    return StreamingMatrix.from_array_implementation(init_fn, next_fn)


def momentum_sgd_matrix(
    momentum: float = 0,
    learning_rates: ArrayLike | None = None,
) -> StreamingMatrix:
    """Create the workload matrix for momentum SGD with step learning rates."""
    lr_schedule = np.asarray(
        [1.0] if learning_rates is None else learning_rates, dtype=np.float64
    )
    if lr_schedule.ndim != 1 or len(lr_schedule) == 0:
        raise ValueError("Learning rates must be a non-empty one-dimensional array")
    if np.min(lr_schedule) <= 0.0:
        raise ValueError(f"Learning rates must be positive. Found {learning_rates}")

    def init_multiply(abstract_value):
        if _is_host_array(abstract_value):
            dtype = np.result_type(abstract_value.dtype, lr_schedule.dtype)
        else:
            dtype = ops.accumulator_dtype(abstract_value)
        zero = _zeros_like(abstract_value, dtype=dtype)
        return 0, zero, zero

    def multiply_next(value, state):
        index, momentum_buffer, result = state
        momentum_value = _scalar_like(momentum, momentum_buffer)
        momentum_buffer = _add(_multiply(momentum_value, momentum_buffer), value)
        learning_rate = _scalar_like(
            lr_schedule[min(index, len(lr_schedule) - 1)], result
        )
        result = _add(result, _multiply(learning_rate, momentum_buffer))
        return result, (index + 1, momentum_buffer, result)

    return StreamingMatrix.from_array_implementation(init_multiply, multiply_next)


__all__ = [
    "StreamingMatrix",
    "diagonal",
    "identity",
    "momentum_sgd_matrix",
    "multiply_array",
    "multiply_streaming_matrices",
    "prefix_sum",
    "scale_rows_and_columns",
]
