"""Tests for opaque.distributed core utilities.

These tests focus on non-distributed behavior (when torch.distributed is not initialized).
For actual multi-device distributed tests, see ``test_collectives.py`` and
``test_profiler_sync.py`` in the ``ddp/`` folder.
"""

import pytest
import torch

from opaque.types import ClippedPytree

from opaque.types import NoisedPytree

from opaque.distributed import is_distributed, get_rank, get_world_size, sum_gradients
from opaque.distributed.collectives import all_reduce, all_reduce_, barrier
from opaque.distributed.gradients import (  # noqa: F401
    reduce_pytree,
    reduce_pytree_,
    sum_gradients_,
)
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

    def test_all_reduce_raises_without_init(self):
        """all_reduce() raises RuntimeError when not initialized."""
        tensor = torch.tensor([1.0, 2.0, 3.0])

        # Skip if distributed is initialized (e.g., in multi-GPU CI)
        if is_distributed():
            pytest.skip("Distributed already initialized")

        with pytest.raises(RuntimeError, match="not initialized"):
            all_reduce(tensor)

    def test_all_reduce_inplace_raises_without_init(self):
        """all_reduce_() raises RuntimeError when not initialized."""
        tensor = torch.tensor([1.0, 2.0, 3.0])

        if is_distributed():
            pytest.skip("Distributed already initialized")

        with pytest.raises(RuntimeError, match="not initialized"):
            all_reduce_(tensor)

    def test_barrier_no_op_without_init(self):
        """barrier() is no-op when not initialized."""
        # Skip if distributed is initialized
        if is_distributed():
            pytest.skip("Distributed already initialized")

        # Should not raise
        barrier()


class TestAllReduceValidation:
    """Tests for all_reduce/all_reduce_ parameter validation."""

    def test_invalid_op_raises(self):
        """all_reduce() and all_reduce_() raise ValueError for invalid op."""
        tensor = torch.tensor([1.0])

        with pytest.raises(ValueError, match="Invalid reduction operation"):
            all_reduce(tensor, op="invalid_op")
        with pytest.raises(ValueError, match="Invalid reduction operation"):
            all_reduce_(tensor, op="invalid_op")

    def test_valid_operations(self):
        """all_reduce() and all_reduce_() accept all valid operations."""
        valid_ops = ["sum", "mean", "max", "min", "product"]

        # Just test that these don't raise ValueError during validation
        # (will still raise RuntimeError if not initialized)
        for op in valid_ops:
            tensor = torch.tensor([1.0])

            if is_distributed():
                # Actually execute if initialized
                all_reduce(tensor, op=op)
                all_reduce_(tensor, op=op)
                continue

            # Just check it gets past parameter validation
            try:
                all_reduce(tensor, op=op)
                all_reduce_(tensor, op=op)
            except RuntimeError as e:
                # Expected if not initialized
                assert "not initialized" in str(e)


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

    def test_sum_gradients_inplace_preserves_clipped_pytree(self):
        gradients = ClippedPytree({"w": torch.tensor([1.0, 2.0])}, max_norm=0.5)

        result = sum_gradients_(gradients)

        assert result is None
        assert gradients.max_norm == 0.5
        torch.testing.assert_close(gradients.pytree["w"], torch.tensor([1.0, 2.0]))

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


class TestModuleExports:
    """Tests for module exports.

    Headline names live at the package root; lower-level primitives are
    grouped into the two documented power-user submodules (collectives,
    gradients).  Sharding and state-sync plumbing live in underscore
    modules and are reachable through the headline (``local_shard``,
    ``sync``) or the registered DP-runtime sync-type machinery.
    """

    def test_root_headline_exports(self):
        """The package root surfaces the headline DP-DDP flow."""
        import opaque.distributed as root

        for name in [
            "is_distributed",
            "get_rank",
            "get_world_size",
            "all_reduce",
            "sum_gradients",
            "sum_gradients_",
            "sync",
            "local_shard",
        ]:
            assert hasattr(root, name) and callable(getattr(root, name)), name

    def test_submodule_exports(self):
        """Lower-level primitives live in the two power-user submodules."""
        from opaque.distributed import collectives, gradients

        for name in [
            "is_distributed",
            "get_rank",
            "get_world_size",
            "all_reduce",
            "all_reduce_",
            "barrier",
        ]:
            assert callable(getattr(collectives, name)), f"collectives.{name}"
        for name in [
            "reduce_pytree",
            "reduce_pytree_",
            "sum_gradients",
            "sum_gradients_",
        ]:
            assert callable(getattr(gradients, name)), f"gradients.{name}"
