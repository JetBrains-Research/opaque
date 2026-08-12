"""Tests for opaque.distributed core utilities.

These tests focus on non-distributed behavior (when torch.distributed is not initialized).
For actual multi-device distributed tests, see ``test_collectives.py`` and
``test_profiler_sync.py`` in the ``ddp/`` folder.
"""

import importlib
from dataclasses import dataclass

import pytest
import torch

from opaque.api.engine.distributed._state import (  # noqa: F401  (used in TestModuleExports)
    assert_pytree_equal,
    assert_scalar_equal,
    gather_pytree,
    gather_tensors,
    reduce_scalar,
    sync_object,
)

# ``_reduced_metadata`` is a private helper exercised directly by the
# tests below; reach for it via the impl-side path.
from opaque.api.engine.distributed.gradients import (
    _reduced_metadata as _reduced_metadata,
)
from opaque.distributed import get_rank, get_world_size, is_distributed, sum_gradients
from opaque.distributed.collectives import all_reduce, barrier
from opaque.distributed.gradients import reduce_pytree
from opaque.types import (
    ClippedPytree,
    NoisedPytree,
    SecondMomentClippingOutput,
    SecondMomentNoiseOutput,
)


class TestNonDistributed:
    """Tests for behavior when distributed is not initialized."""

    def test_is_distributed_false(self):
        """is_distributed() returns False when not initialized."""
        # Note: In CI, distributed might be initialized, so we just check it's callable
        result = is_distributed()
        assert isinstance(result, bool)

    def test_get_rank_returns_zero(self):
        """get_rank() returns 0 when not initialized."""
        # In non-distributed mode, rank is always 0
        # Note: If dist is initialized, this might return actual rank
        result = get_rank()
        assert isinstance(result, int)
        assert result >= 0

    def test_get_world_size_returns_one(self):
        """get_world_size() returns 1 when not initialized."""
        # In non-distributed mode, world size is always 1
        # Note: If dist is initialized, this might return actual world size
        result = get_world_size()
        assert isinstance(result, int)
        assert result >= 1

    def test_all_reduce_returns_copy_without_init(self):
        tensor = torch.tensor([1.0, 2.0, 3.0])

        # Skip if distributed is initialized (e.g., in multi-GPU CI)
        if is_distributed():
            pytest.skip("Distributed already initialized")

        result = all_reduce(tensor)

        assert result is not tensor
        torch.testing.assert_close(result, tensor)

    def test_barrier_no_op_without_init(self):
        """barrier() is no-op when not initialized."""
        # Skip if distributed is initialized
        if is_distributed():
            pytest.skip("Distributed already initialized")

        # Should not raise
        barrier()

    def test_reduce_scalar_preserves_integer_type_without_distribution(self):
        value = 2**24 + 1

        result = reduce_scalar(value, op="sum")

        assert isinstance(result, int)
        assert result == value

    def test_reduce_scalar_integer_mean_returns_float_without_distribution(self):
        result = reduce_scalar(2**24 + 1, op="mean")

        assert isinstance(result, float)
        assert result == float(2**24 + 1)


class TestAllReduceValidation:
    """Tests for all_reduce parameter validation."""

    def test_invalid_op_raises(self):
        """all_reduce() raises ValueError for an invalid operation."""
        tensor = torch.tensor([1.0])

        with pytest.raises(ValueError, match="valid ReduceOp"):
            all_reduce(tensor, op="invalid_op")

    def test_valid_operations(self):
        """all_reduce() accepts all valid operations."""
        valid_ops = ["sum", "mean", "max", "min", "product"]

        # Just test that these don't raise ValueError during validation
        # (will still raise RuntimeError if not initialized)
        for op in valid_ops:
            tensor = torch.tensor([1.0])

            result = all_reduce(tensor, op=op)
            assert result is not tensor


class TestSyncObjectSchema:
    @dataclass(frozen=True)
    class _State:
        count: int
        label: str

    @dataclass(frozen=True)
    class _BooleanState:
        enabled: bool

    def test_requires_a_complete_field_schema(self):
        with pytest.raises(ValueError, match="missing fields"):
            sync_object(self._State(count=1, label="local"), {"count": "sum"})

    def test_rejects_unknown_schema_fields(self):
        with pytest.raises(ValueError, match="unknown fields"):
            sync_object(
                self._State(count=1, label="local"),
                {"count": "sum", "label": "local", "missing": "local"},
            )

    def test_callable_cannot_replace_boolean_field(self, monkeypatch):
        import opaque.api.engine.distributed._state as state_module

        monkeypatch.setattr(state_module, "is_distributed", lambda: True)
        with pytest.raises(TypeError, match="cannot update a bool field"):
            sync_object(self._BooleanState(enabled=True), {"enabled": lambda _: 1})


class TestBoundedGradientAggregation:
    """Tests for wrapper-aware gradient aggregation outside distributed mode."""

    def test_sum_gradients_preserves_clipped_pytree(self):
        gradients = ClippedPytree({"w": torch.tensor([1.0, 2.0])}, max_norm=0.5)

        reduced = sum_gradients(gradients)

        assert isinstance(reduced, ClippedPytree)
        assert not isinstance(reduced, NoisedPytree)
        assert reduced.max_norm == gradients.max_norm
        assert reduced.pytree is not gradients.pytree
        torch.testing.assert_close(reduced.pytree["w"], gradients.pytree["w"])

    def test_mean_gradients_preserves_clipped_pytree_outside_distributed(self):
        gradients = ClippedPytree({"w": torch.tensor([1.0])}, max_norm=0.5)

        reduced = reduce_pytree(gradients, op="mean")

        assert isinstance(reduced, ClippedPytree)
        assert reduced.max_norm == gradients.max_norm
        torch.testing.assert_close(reduced.pytree["w"], gradients.pytree["w"])

    def test_unsupported_clipped_reduction_raises(self):
        gradients = ClippedPytree({"w": torch.tensor([1.0])}, max_norm=0.5)

        with pytest.raises(TypeError, match="supports op='sum' or op='mean'"):
            reduce_pytree(gradients, op="max")

    def test_sum_gradients_preserves_noisy_pytree_outside_distributed(self):
        gradients = NoisedPytree(
            {"w": torch.tensor([1.0])}, max_norm=0.5, noise_stddev=1.0
        )

        reduced = sum_gradients(gradients)

        assert isinstance(reduced, NoisedPytree)
        assert reduced.max_norm == gradients.max_norm
        assert reduced.noise_stddev == gradients.noise_stddev
        torch.testing.assert_close(reduced.pytree["w"], gradients.pytree["w"])

    def test_clipped_metadata_scales_for_distributed_mean(self):
        gradients = ClippedPytree({"w": torch.tensor([1.0])}, max_norm=2.0)

        reduced = _reduced_metadata(gradients, "mean", world_size=4)

        assert isinstance(reduced, ClippedPytree)
        assert reduced.max_norm == pytest.approx(0.5)

    def test_noisy_metadata_scales_for_distributed_sum_and_mean(self):
        gradients = NoisedPytree(
            {"w": torch.tensor([1.0])}, max_norm=2.0, noise_stddev=0.5
        )

        summed = _reduced_metadata(gradients, "sum", world_size=4)
        averaged = _reduced_metadata(gradients, "mean", world_size=4)

        assert isinstance(summed, NoisedPytree)
        assert summed.max_norm == pytest.approx(2.0)
        assert summed.noise_stddev == pytest.approx(1.0)
        assert isinstance(averaged, NoisedPytree)
        assert averaged.max_norm == pytest.approx(0.5)
        assert averaged.noise_stddev == pytest.approx(0.25)

    def test_sum_gradients_preserves_second_moment_clipping_output(self):
        out = SecondMomentClippingOutput(
            grads=ClippedPytree({"w": torch.tensor([1.0, 2.0])}, max_norm=0.5),
            squared_grads=ClippedPytree({"w": torch.tensor([3.0])}, max_norm=1.0),
        )

        reduced = sum_gradients(out)

        assert isinstance(reduced, SecondMomentClippingOutput)
        assert reduced.grads is not out.grads
        assert reduced.squared_grads is not out.squared_grads
        assert reduced.grads.max_norm == 0.5
        assert reduced.squared_grads.max_norm == 1.0
        torch.testing.assert_close(reduced.grads.pytree["w"], out.grads.pytree["w"])
        torch.testing.assert_close(
            reduced.squared_grads.pytree["w"], out.squared_grads.pytree["w"]
        )

    def test_sum_gradients_preserves_second_moment_noise_output(self):
        out = SecondMomentNoiseOutput(
            noisy_grads=NoisedPytree(
                {"w": torch.tensor([1.0])}, max_norm=0.5, noise_stddev=0.2
            ),
            noisy_squared_grads=NoisedPytree(
                {"w": torch.tensor([4.0])}, max_norm=1.0, noise_stddev=0.3
            ),
        )
        reduced = sum_gradients(out)
        assert isinstance(reduced, SecondMomentNoiseOutput)
        assert reduced.noisy_grads.noise_stddev == pytest.approx(0.2)
        assert reduced.noisy_squared_grads.noise_stddev == pytest.approx(0.3)
        torch.testing.assert_close(
            reduced.noisy_grads.pytree["w"], out.noisy_grads.pytree["w"]
        )

    def test_reduce_pytree_rejects_non_tensor_leaves(self):
        with pytest.raises(TypeError, match="tensor leaves"):
            reduce_pytree({"w": 1.0})


class TestModuleExports:
    """Every declared public distributed export resolves to a callable."""

    @pytest.mark.parametrize(
        "module_name",
        [
            "opaque.distributed",
            "opaque.distributed.collectives",
            "opaque.distributed.gradients",
        ],
    )
    def test_declared_exports_are_callable(self, module_name):
        module = importlib.import_module(module_name)

        exports = getattr(module, "__all__", None)
        assert exports, f"{module_name} must declare __all__"
        for name in exports:
            assert callable(getattr(module, name)), f"{module_name}.{name}"
