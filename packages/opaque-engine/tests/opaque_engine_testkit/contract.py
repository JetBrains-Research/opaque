"""Inherited executable conformance contract for production providers."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, ClassVar

import numpy as np
import pytest

from opaque import autodiff, ops, pytree, random
from opaque.api.engine import pytree as engine_pytree
from opaque.api.engine.backend import active_backend
from opaque.api.engine.primitive import CORE_PRIMITIVES, CORE_PROFILE_VERSION, supports
from opaque_engine_testkit.validation import covers, validate_backend_contract

if TYPE_CHECKING:
    from collections.abc import Mapping

    from opaque_engine_testkit.matrix import BackendCase


class BackendContractTests:
    """Non-collected portable primitive contract inherited by provider tests."""

    __test__ = False
    provider_name: ClassVar[str]
    core_profile_version: ClassVar[int] = CORE_PROFILE_VERSION

    def setup_method(self) -> None:
        validate_backend_contract(type(self))

    def array(self, case: BackendCase, value: Any, *, dtype: Any | None = None) -> Any:
        """Construct a native array; providers may narrow this adapter hook."""
        return case.array(value, dtype=dtype)

    def dtype(self, case: BackendCase, name: str) -> Any:
        """Look up a native dtype; providers may narrow this adapter hook."""
        return case.dtype(name)

    def to_host(self, case: BackendCase, value: Any) -> np.ndarray:
        """Observe a native value on the host without exposing provider APIs."""
        return case.to_host(value)

    def tolerances(self) -> tuple[float, float]:
        """Return portable numerical comparison tolerances."""
        return (1e-6, 1e-6)

    def capabilities(self) -> Mapping[str, bool]:
        """Declare documented optional capabilities without changing core tests."""
        return {}

    def assert_allclose(self, case: BackendCase, actual: Any, expected: Any) -> None:
        """Compare a portable result using this provider's declared tolerance."""
        rtol, atol = self.tolerances()
        case.assert_allclose(actual, expected, rtol=rtol, atol=atol)

    def _assert_provider(self, case: BackendCase) -> None:
        assert case.name == self.provider_name
        assert active_backend().name == self.provider_name

    @covers()
    def test_provider_registers_complete_core_profile(
        self, provider_case: BackendCase
    ) -> None:
        self._assert_provider(provider_case)
        assert all(
            supports(primitive, self.provider_name) for primitive in CORE_PRIMITIVES
        )

    @covers("opaque.ops.is_array", "opaque.ops.dtype", "opaque.ops.shape")
    def test_array_identity(self, provider_case: BackendCase) -> None:
        self._assert_provider(provider_case)
        value = self.array(provider_case, [1.0, 4.0], dtype=ops.float32())
        assert ops.is_array(value)
        assert not ops.is_array([1.0])
        assert ops.dtype(value) == ops.float32()
        assert ops.shape(value) == (2,)

    @covers(
        "opaque.ops.is_floating",
        "opaque.ops.is_low_precision",
        "opaque.ops.is_complex",
        "opaque.ops.float32",
        "opaque.ops.boolean",
        "opaque.ops.real_dtype",
        "opaque.ops.scalar",
        "opaque.ops.scalar_item",
    )
    def test_scalar_and_dtype_rules(self, provider_case: BackendCase) -> None:
        value = self.array(provider_case, [1.0, 4.0], dtype=ops.float32())
        integer = self.array(
            provider_case, [1], dtype=self.dtype(provider_case, "int32")
        )
        low_precision = self.array(
            provider_case, [1.0], dtype=self.dtype(provider_case, "float16")
        )
        complex_dtype = self.dtype(provider_case, "complex64")
        assert ops.shape(ops.scalar(1.0, like=value)) == ()
        assert ops.is_floating(value)
        assert ops.is_floating(ops.float32())
        assert not ops.is_floating(integer)
        assert ops.is_low_precision(low_precision)
        assert not ops.is_complex(value)
        assert ops.is_complex(complex_dtype)
        assert ops.real_dtype(complex_dtype) == ops.float32()
        assert ops.dtype(ops.scalar(1.0, like=low_precision)) == self.dtype(
            provider_case, "float16"
        )
        assert ops.dtype(ops.scalar(True, dtype=ops.boolean())) == ops.boolean()
        assert ops.scalar_item(ops.scalar(3.0)) == 3.0

    @covers("opaque.ops.zeros", "opaque.ops.zeros_like", "opaque.ops.ones_like")
    def test_array_creation(self, provider_case: BackendCase) -> None:
        value = self.array(provider_case, [1.0, 4.0], dtype=ops.float32())
        assert ops.shape(ops.zeros((2,), like=value)) == (2,)
        assert ops.shape(ops.zeros(())) == ()
        self.assert_allclose(provider_case, ops.zeros_like(value), [0.0, 0.0])
        self.assert_allclose(provider_case, ops.ones_like(value), [1.0, 1.0])

    @covers(
        "opaque.ops.astype",
        "opaque.ops.clone",
        "opaque.ops.detach",
        "opaque.ops.transfer",
    )
    def test_copy_and_transfer(self, provider_case: BackendCase) -> None:
        value = self.array(provider_case, [1.0, 4.0], dtype=ops.float32())
        float16 = self.dtype(provider_case, "float16")
        assert ops.dtype(ops.astype(value, float16)) == float16
        self.assert_allclose(provider_case, ops.clone(value), [1.0, 4.0])
        self.assert_allclose(provider_case, ops.detach(value), [1.0, 4.0])
        assert ops.dtype(ops.transfer(value, dtype=float16)) == float16

    @covers(
        "opaque.ops.sqrt",
        "opaque.ops.exp",
        "opaque.ops.erf",
        "opaque.ops.erfinv",
        "opaque.ops.rsqrt",
        "opaque.ops.square",
        "opaque.ops.abs",
    )
    def test_unary_and_complex_math(self, provider_case: BackendCase) -> None:
        value = self.array(provider_case, [0.0, 1.0, 4.0], dtype=ops.float32())
        self.assert_allclose(provider_case, ops.sqrt(value), [0.0, 1.0, 2.0])
        self.assert_allclose(provider_case, ops.exp(ops.scalar(0.0, like=value)), 1.0)
        self.assert_allclose(
            provider_case,
            ops.erfinv(ops.erf(self.array(provider_case, [-0.5, 0.5]))),
            [-0.5, 0.5],
        )
        self.assert_allclose(provider_case, ops.rsqrt(value[1:]), [1.0, 0.5])
        self.assert_allclose(provider_case, ops.square(value), [0.0, 1.0, 16.0])
        complex_value = self.array(
            provider_case, [3.0 + 4.0j], dtype=self.dtype(provider_case, "complex64")
        )
        self.assert_allclose(provider_case, ops.abs(complex_value), [5.0])

    @covers(
        "opaque.ops.finfo_eps",
        "opaque.ops.finfo_smallest_normal",
        "opaque.ops.to_host",
    )
    def test_host_copy_and_numeric_edges(self, provider_case: BackendCase) -> None:
        value = self.array(provider_case, [0.0, 1.0, 4.0], dtype=ops.float32())
        assert ops.finfo_eps(value) == ops.finfo_eps(ops.float32()) > 0.0
        assert (
            ops.finfo_smallest_normal(value)
            == ops.finfo_smallest_normal(ops.float32())
            > 0.0
        )
        host = ops.to_host(value)
        host[0] = 99.0
        self.assert_allclose(provider_case, value, [0.0, 1.0, 4.0])
        assert np.isnan(
            self.to_host(provider_case, ops.sqrt(self.array(provider_case, [-1.0])))
        ).item()

    @covers(
        "opaque.ops.add",
        "opaque.ops.subtract",
        "opaque.ops.multiply",
        "opaque.ops.divide",
        "opaque.ops.pow",
        "opaque.ops.reciprocal",
    )
    def test_broadcasting_and_arithmetic(self, provider_case: BackendCase) -> None:
        left = self.array(provider_case, [1.0, 4.0], dtype=ops.float32())
        right = self.array(provider_case, [2.0, 3.0], dtype=ops.float32())
        matrix = self.array(
            provider_case, [[1.0, 2.0], [3.0, 4.0]], dtype=ops.float32()
        )
        column = self.array(provider_case, [[10.0], [20.0]], dtype=ops.float32())
        self.assert_allclose(provider_case, ops.add(left, right), [3.0, 7.0])
        self.assert_allclose(provider_case, ops.subtract(left, right), [-1.0, 1.0])
        self.assert_allclose(provider_case, ops.multiply(left, right), [2.0, 12.0])
        self.assert_allclose(provider_case, ops.divide(right, left), [2.0, 0.75])
        self.assert_allclose(provider_case, ops.pow(left, 2), [1.0, 16.0])
        self.assert_allclose(provider_case, ops.reciprocal(left), [1.0, 0.25])
        self.assert_allclose(
            provider_case, ops.add(matrix, column), [[11.0, 12.0], [23.0, 24.0]]
        )

    @covers(
        "opaque.ops.mean",
        "opaque.ops.sum",
        "opaque.ops.accumulator_dtype",
        "opaque.ops.amin",
        "opaque.ops.amax",
    )
    def test_reductions_and_nan_semantics(self, provider_case: BackendCase) -> None:
        value = self.array(provider_case, [1.0, 4.0], dtype=ops.float32())
        matrix = self.array(
            provider_case, [[1.0, 2.0], [3.0, 4.0]], dtype=ops.float32()
        )
        assert ops.scalar_item(ops.mean(value)) == 2.5
        self.assert_allclose(provider_case, ops.mean(matrix, axis=1), [1.5, 3.5])
        assert ops.scalar_item(ops.sum(value)) == 5.0
        accumulator_dtype = ops.accumulator_dtype(value)
        assert ops.dtype(ops.sum(value, dtype=accumulator_dtype)) == accumulator_dtype
        assert ops.scalar_item(ops.amin(value)) == 1.0
        assert ops.scalar_item(ops.amax(value)) == 4.0
        with_nan = self.array(provider_case, [[1.0, float("nan")], [3.0, 4.0]])
        assert np.isnan(self.to_host(provider_case, ops.amin(with_nan, axis=1))[0])
        assert np.isnan(self.to_host(provider_case, ops.amax(with_nan, axis=1))[0])

    @covers(
        "opaque.ops.greater",
        "opaque.ops.minimum",
        "opaque.ops.maximum",
        "opaque.ops.where",
    )
    def test_predicates(self, provider_case: BackendCase) -> None:
        left = self.array(provider_case, [1.0, 4.0])
        right = self.array(provider_case, [2.0, 3.0])
        predicate = ops.greater(left, right)
        self.assert_allclose(provider_case, ops.minimum(left, right), [1.0, 3.0])
        self.assert_allclose(provider_case, ops.maximum(left, right), [2.0, 4.0])
        self.assert_allclose(
            provider_case, ops.where(predicate, left, right), [2.0, 4.0]
        )

    @covers(
        "opaque.ops.isfinite",
        "opaque.ops.all",
        "opaque.ops.nan_to_num",
        "opaque.ops.clamp",
    )
    def test_nan_rules(self, provider_case: BackendCase) -> None:
        finite = ops.isfinite(
            self.array(provider_case, [0.0, float("nan"), float("inf")])
        )
        self.assert_allclose(provider_case, finite, [True, False, False])
        assert ops.scalar_item(ops.all(finite)) is False
        self.assert_allclose(
            provider_case,
            ops.nan_to_num(
                self.array(provider_case, [float("nan"), float("inf")]),
                nan=1.0,
                posinf=2.0,
            ),
            [1.0, 2.0],
        )
        self.assert_allclose(
            provider_case,
            ops.clamp(self.array(provider_case, [-1.0, 0.5, 2.0]), 0.0, 1.0),
            [0.0, 0.5, 1.0],
        )

    @covers("opaque.ops.concatenate", "opaque.ops.stack", "opaque.ops.slice_array")
    def test_array_assembly_and_slicing(self, provider_case: BackendCase) -> None:
        left = self.array(provider_case, [1.0, 4.0])
        right = self.array(provider_case, [2.0, 3.0])
        assert ops.shape(ops.concatenate([left, right])) == (4,)
        matrix = ops.stack([left, right])
        assert ops.shape(matrix) == (2, 2)
        assert ops.shape(ops.slice_array(matrix, 0)) == (2,)

    @covers("opaque.ops.expand_dims", "opaque.ops.squeeze", "opaque.ops.promote_dtype")
    def test_shape_and_dtype_promotion(self, provider_case: BackendCase) -> None:
        left = self.array(provider_case, [1.0, 4.0])
        assert ops.shape(ops.expand_dims(left, -1)) == (2, 1)
        assert ops.shape(ops.squeeze(ops.expand_dims(left, 0))) == (2,)
        assert (
            ops.promote_dtype(self.dtype(provider_case, "float16"), ops.float32())
            == ops.float32()
        )

    @covers("opaque.autodiff.grad_and_value")
    def test_grad_and_value(self, provider_case: BackendCase) -> None:
        value = self.array(provider_case, [3.0, 4.0], dtype=ops.float32())
        grads, result = autodiff.grad_and_value(lambda item: ops.sum(ops.square(item)))(
            value
        )
        self.assert_allclose(provider_case, grads, [6.0, 8.0])
        assert ops.scalar_item(result) == 25.0

    @covers("opaque.autodiff.vmap")
    def test_vmap(self, provider_case: BackendCase) -> None:
        value = self.array(provider_case, [[1.0, 2.0], [3.0, 4.0]], dtype=ops.float32())
        self.assert_allclose(
            provider_case,
            autodiff.vmap(lambda item: ops.multiply(item, 2))(value),
            [[2.0, 4.0], [6.0, 8.0]],
        )

    @covers()
    def test_nested_transforms(self, provider_case: BackendCase) -> None:
        value = self.array(provider_case, [[1.0, 2.0], [3.0, 4.0]], dtype=ops.float32())
        self.assert_allclose(
            provider_case,
            autodiff.vmap(autodiff.vmap(lambda item: ops.multiply(item, 2)))(value),
            [[2.0, 4.0], [6.0, 8.0]],
        )

    @covers()
    def test_transform_errors(self, provider_case: BackendCase) -> None:
        with pytest.raises((RuntimeError, ValueError)):
            autodiff.vmap(lambda item: item, randomness="unknown")

    @covers(
        "opaque.pytree.tree_flatten_with_paths",
        "opaque.pytree.tree_flatten",
        "opaque.pytree.tree_unflatten",
        "opaque.pytree.tree_structure",
        "opaque.pytree.tree_leaves",
        "opaque.pytree.tree_map",
    )
    def test_pytree_structure(self, provider_case: BackendCase) -> None:
        tree = {
            "flat.key": self.array(provider_case, [3.0, 4.0]),
            "nested": [self.array(provider_case, [12.0]), 1],
        }
        paths, leaves, treedef = pytree.tree_flatten_with_paths(tree)
        flattened, flattened_treedef = pytree.tree_flatten(tree)
        assert set(paths) == {("flat.key",), ("nested", 0), ("nested", 1)}
        assert len(flattened) == len(leaves)
        assert pytree.tree_structure(tree) == flattened_treedef
        self.assert_allclose(
            provider_case,
            pytree.tree_unflatten(treedef, leaves)["flat.key"],
            [3.0, 4.0],
        )
        self.assert_allclose(
            provider_case, pytree.tree_map(lambda leaf: leaf, tree)["nested"][0], [12.0]
        )

    @covers(
        "opaque.pytree._squared_l2_norms", "opaque.pytree._squared_l2_norm_roundoff"
    )
    def test_pytree_norms(self, provider_case: BackendCase) -> None:
        tree = {
            "flat.key": self.array(provider_case, [3.0, 4.0]),
            "nested": [self.array(provider_case, [12.0]), 1],
        }
        native_leaves = pytree.tree_leaves(tree)
        squared, grouped = engine_pytree._squared_l2_norms(
            native_leaves, ["first", "second"], dtype=ops.float32()
        )
        assert ops.scalar_item(squared) == 169.0
        assert ops.scalar_item(grouped["first"]) == 25.0
        assert (
            engine_pytree._squared_l2_norm_roundoff(native_leaves, dtype=ops.float32())
            > 0.0
        )

    @covers("opaque.random.normal")
    def test_keyed_randomness_is_deterministic(
        self, provider_case: BackendCase
    ) -> None:
        key = random.key(7)
        first = random.normal(key, (2, 3), dtype=ops.float32())
        second = random.normal(key, (2, 3), dtype=ops.float32())
        np.testing.assert_array_equal(
            self.to_host(provider_case, first), self.to_host(provider_case, second)
        )
        assert ops.shape(first) == (2, 3)
        assert ops.dtype(first) == ops.float32()

    @covers()
    def test_keyed_randomness_separates_domains(
        self, provider_case: BackendCase
    ) -> None:
        del provider_case
        key = random.key(7)
        child_one, child_two = random.split(random.fold_in(key, "contract"))
        assert child_one != child_two
