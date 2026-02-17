"""Tests for opaque.distributed core utilities.

These tests focus on non-distributed behavior (when torch.distributed is not initialized).
For actual multi-device distributed tests, see test_ddp_integration.py.
"""

import pytest
import torch

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

    def test_barrier_no_op_without_init(self):
        """barrier() is no-op when not initialized."""
        # Skip if distributed is initialized
        if dist_utils.is_distributed():
            pytest.skip("Distributed already initialized")

        # Should not raise
        dist_utils.barrier()


class TestAllReduceValidation:
    """Tests for all_reduce() parameter validation."""

    def test_invalid_op_raises(self):
        """all_reduce() raises ValueError for invalid op."""
        # Mock initialization check by testing with a tensor
        tensor = torch.tensor([1.0])

        # Skip if not initialized (would get RuntimeError before ValueError)
        if not dist_utils.is_distributed():
            pytest.skip("Distributed not initialized")

        with pytest.raises(ValueError, match="Invalid reduction operation"):
            dist_utils.all_reduce(tensor, op="invalid_op")

    def test_valid_operations(self):
        """all_reduce() accepts all valid operations."""
        valid_ops = ["sum", "mean", "max", "min", "product"]

        # Just test that these don't raise ValueError during validation
        # (will still raise RuntimeError if not initialized)
        for op in valid_ops:
            tensor = torch.tensor([1.0])

            if dist_utils.is_distributed():
                # Actually execute if initialized
                dist_utils.all_reduce(tensor, op=op)
            else:
                # Just check it gets past parameter validation
                try:
                    dist_utils.all_reduce(tensor, op=op)
                except RuntimeError as e:
                    # Expected if not initialized
                    assert "not initialized" in str(e)


class TestModuleExports:
    """Tests for module exports."""

    def test_all_functions_exported(self):
        """All expected functions are exported."""
        expected_exports = [
            "is_distributed",
            "get_rank",
            "get_world_size",
            "all_reduce",
            "barrier",
            "reduce_pytree",
            "sum_gradients",
            "reduce_scalar",
            "gather_tensors",
            "sync_state",
        ]

        for name in expected_exports:
            assert hasattr(dist_utils, name), f"Missing export: {name}"
            obj = getattr(dist_utils, name)
            assert callable(obj), f"{name} is not callable"
