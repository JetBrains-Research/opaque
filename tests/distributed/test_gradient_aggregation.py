"""Tests for gradient aggregation in distributed training.

These tests focus on non-distributed behavior (when torch.distributed is not initialized).
For actual multi-device distributed tests, see test_ddp_integration.py.
"""

import pytest
import torch

import opaque.distributed as dist_utils


class TestNonDistributedGradientAggregation:
    """Tests for gradient aggregation when distributed is not initialized."""

    def test_all_reduce_gradients_returns_unchanged(self):
        """all_reduce_gradients() returns input unchanged when not initialized."""
        if dist_utils.is_initialized():
            pytest.skip("Distributed already initialized")

        grads = {
            "weight": torch.tensor([[1.0, 2.0], [3.0, 4.0]]),
            "bias": torch.tensor([0.5, 1.0]),
        }
        grads_copy = {k: v.clone() for k, v in grads.items()}

        result, work_handles = dist_utils.all_reduce_gradients(grads, op="sum")

        # Should return same gradients unchanged
        assert result is grads
        assert work_handles is None
        for key in grads:
            assert torch.allclose(grads[key], grads_copy[key])

    def test_average_gradients_returns_unchanged(self):
        """average_gradients() returns input unchanged when not initialized."""
        if dist_utils.is_initialized():
            pytest.skip("Distributed already initialized")

        grads = {
            "weight": torch.tensor([[1.0, 2.0], [3.0, 4.0]]),
            "bias": torch.tensor([0.5, 1.0]),
        }
        grads_copy = {k: v.clone() for k, v in grads.items()}

        result = dist_utils.average_gradients(grads)

        # Should return same gradients unchanged
        assert result is grads
        for key in grads:
            assert torch.allclose(grads[key], grads_copy[key])

    def test_all_reduce_gradients_with_nested_pytree(self):
        """all_reduce_gradients() handles nested PyTrees."""
        if dist_utils.is_initialized():
            pytest.skip("Distributed already initialized")

        grads = {
            "layer1": {
                "weight": torch.tensor([[1.0, 2.0]]),
                "bias": torch.tensor([0.5]),
            },
            "layer2": {"weight": torch.tensor([[3.0]]), "bias": torch.tensor([1.0])},
        }

        result, work_handles = dist_utils.all_reduce_gradients(grads, op="sum")

        # Should handle nested structure
        assert "layer1" in result
        assert "layer2" in result
        assert "weight" in result["layer1"]
        assert "bias" in result["layer1"]
        assert work_handles is None

    def test_all_reduce_gradients_preserves_dtype(self):
        """all_reduce_gradients() preserves tensor dtypes."""
        if dist_utils.is_initialized():
            pytest.skip("Distributed already initialized")

        grads = {
            "float32": torch.tensor([1.0, 2.0], dtype=torch.float32),
            "float64": torch.tensor([1.0, 2.0], dtype=torch.float64),
            "int32": torch.tensor([1, 2], dtype=torch.int32),
        }

        result, _ = dist_utils.all_reduce_gradients(grads, op="sum")

        assert result["float32"].dtype == torch.float32
        assert result["float64"].dtype == torch.float64
        assert result["int32"].dtype == torch.int32

    def test_all_reduce_gradients_preserves_device(self, device):
        """all_reduce_gradients() preserves tensor devices."""
        if dist_utils.is_initialized():
            pytest.skip("Distributed already initialized")

        grads = {
            "weight": torch.tensor([[1.0, 2.0]], device=device),
            "bias": torch.tensor([0.5], device=device),
        }

        result, _ = dist_utils.all_reduce_gradients(grads, op="sum")

        assert result["weight"].device.type == device.type
        assert result["bias"].device.type == device.type

    def test_average_gradients_with_explicit_world_size(self):
        """average_gradients() uses explicit world_size if provided."""
        if dist_utils.is_initialized():
            pytest.skip("Distributed already initialized")

        grads = {"weight": torch.tensor([10.0, 20.0])}

        # With world_size=1, should be unchanged (10/1 = 10)
        result = dist_utils.average_gradients(grads, world_size=1)
        assert torch.allclose(result["weight"], torch.tensor([10.0, 20.0]))

    def test_all_reduce_gradients_valid_operations(self):
        """all_reduce_gradients() accepts all valid operations."""
        if dist_utils.is_initialized():
            pytest.skip("Distributed already initialized")

        grads = {"weight": torch.tensor([1.0, 2.0])}
        valid_ops = ["sum", "mean", "max", "min", "product"]

        for op in valid_ops:
            result, _ = dist_utils.all_reduce_gradients(grads, op=op)
            # Should not raise
            assert result is grads


class TestGradientAggregationEdgeCases:
    """Tests for edge cases in gradient aggregation."""

    def test_empty_pytree(self):
        """all_reduce_gradients() handles empty PyTree."""
        if dist_utils.is_initialized():
            pytest.skip("Distributed already initialized")

        grads = {}  # Empty dict

        result, _ = dist_utils.all_reduce_gradients(grads, op="sum")
        assert result == {}

    def test_single_tensor(self):
        """all_reduce_gradients() handles single tensor (not dict)."""
        if dist_utils.is_initialized():
            pytest.skip("Distributed already initialized")

        grad = torch.tensor([1.0, 2.0, 3.0])

        result, _ = dist_utils.all_reduce_gradients(grad, op="sum")
        # Should work with single tensor
        assert torch.allclose(result, grad)

    def test_list_of_tensors(self):
        """all_reduce_gradients() handles list of tensors."""
        if dist_utils.is_initialized():
            pytest.skip("Distributed already initialized")

        grads = [
            torch.tensor([1.0, 2.0]),
            torch.tensor([3.0, 4.0]),
        ]

        result, _ = dist_utils.all_reduce_gradients(grads, op="sum")
        # Should work with list
        assert len(result) == 2
        assert torch.allclose(result[0], grads[0])
        assert torch.allclose(result[1], grads[1])

    def test_tuple_of_tensors(self):
        """all_reduce_gradients() handles tuple of tensors."""
        if dist_utils.is_initialized():
            pytest.skip("Distributed already initialized")

        grads = (
            torch.tensor([1.0, 2.0]),
            torch.tensor([3.0, 4.0]),
        )

        result, _ = dist_utils.all_reduce_gradients(grads, op="sum")
        # Should work with tuple
        assert len(result) == 2
        assert torch.allclose(result[0], grads[0])
        assert torch.allclose(result[1], grads[1])
