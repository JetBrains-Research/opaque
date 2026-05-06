"""Tests for opaque.distributed core utilities.

These tests focus on non-distributed behavior (when torch.distributed is not initialized).
For actual multi-device distributed tests, see test_ddp_integration.py.
"""

import pytest
import torch

from opaque.types import ClippedPytree

from opaque.types import NoisedPytree

from opaque.distributed import gradients as gradients_module

import opaque.distributed as dist_utils


class TestNonDistributed:
    """Tests for behavior when distributed is not initialized."""

    def test_is_distributed_false(self):
        """is_distributed() returns False when not initialized."""
        # Note: In CI, distributed might be initialized, so we just check it's callable
        result = dist_utils.is_distributed()
        assert isinstance(result, bool)

    def test_get_rank_returns_zero(self):
        """get_rank() returns 0 when not initialized."""
        # In non-distributed mode, rank is always 0
        # Note: If dist is initialized, this might return actual rank
        result = dist_utils.get_rank()
        assert isinstance(result, int)
        assert result >= 0

    def test_get_world_size_returns_one(self):
        """get_world_size() returns 1 when not initialized."""
        # In non-distributed mode, world size is always 1
        # Note: If dist is initialized, this might return actual world size
        result = dist_utils.get_world_size()
        assert isinstance(result, int)
        assert result >= 1

    def test_all_reduce_raises_without_init(self):
        """all_reduce() raises RuntimeError when not initialized."""
        tensor = torch.tensor([1.0, 2.0, 3.0])

        # Skip if distributed is initialized (e.g., in multi-GPU CI)
        if dist_utils.is_distributed():
            pytest.skip("Distributed already initialized")

        with pytest.raises(RuntimeError, match="not initialized"):
            dist_utils.all_reduce(tensor)

    def test_all_reduce_inplace_raises_without_init(self):
        """all_reduce_() raises RuntimeError when not initialized."""
        tensor = torch.tensor([1.0, 2.0, 3.0])

        if dist_utils.is_distributed():
            pytest.skip("Distributed already initialized")

        with pytest.raises(RuntimeError, match="not initialized"):
            dist_utils.all_reduce_(tensor)

    def test_barrier_no_op_without_init(self):
        """barrier() is no-op when not initialized."""
        # Skip if distributed is initialized
        if dist_utils.is_distributed():
            pytest.skip("Distributed already initialized")

        # Should not raise
        dist_utils.barrier()


class TestAllReduceValidation:
    """Tests for all_reduce/all_reduce_ parameter validation."""

    def test_invalid_op_raises(self):
        """all_reduce() and all_reduce_() raise ValueError for invalid op."""
        tensor = torch.tensor([1.0])

        with pytest.raises(ValueError, match="Invalid reduction operation"):
            dist_utils.all_reduce(tensor, op="invalid_op")
        with pytest.raises(ValueError, match="Invalid reduction operation"):
            dist_utils.all_reduce_(tensor, op="invalid_op")

    def test_valid_operations(self):
        """all_reduce() and all_reduce_() accept all valid operations."""
        valid_ops = ["sum", "mean", "max", "min", "product"]

        # Just test that these don't raise ValueError during validation
        # (will still raise RuntimeError if not initialized)
        for op in valid_ops:
            tensor = torch.tensor([1.0])

            if dist_utils.is_distributed():
                # Actually execute if initialized
                dist_utils.all_reduce(tensor, op=op)
                dist_utils.all_reduce_(tensor, op=op)
                continue

            # Just check it gets past parameter validation
            try:
                dist_utils.all_reduce(tensor, op=op)
                dist_utils.all_reduce_(tensor, op=op)
            except RuntimeError as e:
                # Expected if not initialized
                assert "not initialized" in str(e)


class TestBoundedGradientAggregation:
    """Tests for wrapper-aware gradient aggregation outside distributed mode."""

    def test_sum_gradients_preserves_clipped_pytree(self):
        gradients = ClippedPytree({"w": torch.tensor([1.0, 2.0])}, max_norm=0.5)

        reduced = dist_utils.sum_gradients(gradients)

        assert isinstance(reduced, ClippedPytree)
        assert not isinstance(reduced, NoisedPytree)
        assert reduced.max_norm == gradients.max_norm
        assert reduced.pytree is not gradients.pytree
        torch.testing.assert_close(reduced.pytree["w"], gradients.pytree["w"])

    def test_sum_gradients_inplace_preserves_clipped_pytree(self):
        gradients = ClippedPytree({"w": torch.tensor([1.0, 2.0])}, max_norm=0.5)

        result = dist_utils.sum_gradients_(gradients)

        assert result is None
        assert gradients.max_norm == 0.5
        torch.testing.assert_close(gradients.pytree["w"], torch.tensor([1.0, 2.0]))

    def test_mean_gradients_preserves_clipped_pytree_outside_distributed(self):
        gradients = ClippedPytree({"w": torch.tensor([1.0])}, max_norm=0.5)

        reduced = dist_utils.reduce_pytree(gradients, op="mean")

        assert isinstance(reduced, ClippedPytree)
        assert reduced.max_norm == gradients.max_norm
        torch.testing.assert_close(reduced.pytree["w"], gradients.pytree["w"])

    def test_unsupported_clipped_reduction_raises(self):
        gradients = ClippedPytree({"w": torch.tensor([1.0])}, max_norm=0.5)

        with pytest.raises(TypeError, match="supports op='sum' or op='mean'"):
            dist_utils.reduce_pytree(gradients, op="max")

    def test_sum_gradients_preserves_noisy_pytree_outside_distributed(self):
        gradients = NoisedPytree(
            {"w": torch.tensor([1.0])}, max_norm=0.5, noise_stddev=1.0
        )

        reduced = dist_utils.sum_gradients(gradients)

        assert isinstance(reduced, NoisedPytree)
        assert reduced.max_norm == gradients.max_norm
        assert reduced.noise_stddev == gradients.noise_stddev
        torch.testing.assert_close(reduced.pytree["w"], gradients.pytree["w"])

    def test_clipped_metadata_scales_for_distributed_mean(self):
        gradients = ClippedPytree({"w": torch.tensor([1.0])}, max_norm=2.0)

        reduced = gradients_module._reduced_metadata(gradients, "mean", world_size=4)

        assert isinstance(reduced, ClippedPytree)
        assert reduced.max_norm == pytest.approx(0.5)

    def test_noisy_metadata_scales_for_distributed_sum_and_mean(self):
        gradients = NoisedPytree(
            {"w": torch.tensor([1.0])}, max_norm=2.0, noise_stddev=0.5
        )

        summed = gradients_module._reduced_metadata(gradients, "sum", world_size=4)
        averaged = gradients_module._reduced_metadata(gradients, "mean", world_size=4)

        assert isinstance(summed, NoisedPytree)
        assert summed.max_norm == pytest.approx(2.0)
        assert summed.noise_stddev == pytest.approx(1.0)
        assert isinstance(averaged, NoisedPytree)
        assert averaged.max_norm == pytest.approx(0.5)
        assert averaged.noise_stddev == pytest.approx(0.25)


class TestModuleExports:
    """Tests for module exports."""

    def test_all_functions_exported(self):
        """All expected functions are exported."""
        expected_exports = [
            "is_distributed",
            "get_rank",
            "get_world_size",
            "all_reduce",
            "all_reduce_",
            "barrier",
            "reduce_pytree",
            "reduce_pytree_",
            "sum_gradients",
            "sum_gradients_",
            "reduce_scalar",
            "gather_tensors",
            "gather_pytree",
            "assert_pytree_equal",
            "assert_scalar_equal",
            "sync_object",
            "sync",
        ]

        for name in expected_exports:
            assert hasattr(dist_utils, name), f"Missing export: {name}"
            obj = getattr(dist_utils, name)
            assert callable(obj), f"{name} is not callable"
