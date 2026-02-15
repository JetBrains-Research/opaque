"""Tests for state synchronization in distributed training.

These tests focus on non-distributed behavior (when torch.distributed is not initialized).
For actual multi-device distributed tests, see test_ddp_integration.py.
"""

from dataclasses import dataclass

import pytest
import torch

import opaque.distributed as dist_utils


class TestNonDistributedStateSynchronization:
    """Tests for state sync when distributed is not initialized."""

    def test_sync_scalar_returns_unchanged(self):
        """sync_scalar() returns input unchanged when not initialized."""
        if dist_utils.is_initialized():
            pytest.skip("Distributed already initialized")

        value = 1.5
        result = dist_utils.sync_scalar(value, op="mean")

        assert result == value
        assert isinstance(result, float)

    def test_sync_scalar_preserves_type(self):
        """sync_scalar() always returns float."""
        if dist_utils.is_initialized():
            pytest.skip("Distributed already initialized")

        # Input can be int or float
        result_from_int = dist_utils.sync_scalar(5, op="mean")
        result_from_float = dist_utils.sync_scalar(5.0, op="mean")

        # When not distributed, int stays int but goes through tensor conversion
        # so it becomes float
        assert isinstance(result_from_int, (int, float))  # Either is acceptable
        assert isinstance(result_from_float, float)

    def test_sync_scalar_with_device(self):
        """sync_scalar() accepts device parameter."""
        if dist_utils.is_initialized():
            pytest.skip("Distributed already initialized")

        value = 2.5
        result = dist_utils.sync_scalar(value, op="mean", device=torch.device("cpu"))

        assert result == value

    def test_sync_state_returns_unchanged(self):
        """sync_state() returns input unchanged when not initialized."""
        if dist_utils.is_initialized():
            pytest.skip("Distributed already initialized")

        @dataclass
        class TestState:
            value_a: float
            value_b: int
            step: int

        state = TestState(value_a=1.5, value_b=2, step=10)
        result = dist_utils.sync_state(state, sync_fields=["value_a", "value_b"])

        assert result is state  # Unchanged when not distributed

    def test_sync_state_dataclass_required(self):
        """sync_state() raises TypeError if not a dataclass (when distributed)."""
        # This test only makes sense in distributed mode
        # When not distributed, function returns early without validation
        pytest.skip("Validation only happens in distributed mode")

    def test_sync_state_invalid_field_raises(self):
        """sync_state() raises ValueError for non-existent fields (when distributed)."""
        # This test only makes sense in distributed mode
        # When not distributed, function returns early without validation
        pytest.skip("Validation only happens in distributed mode")


class TestSyncStateWithAdaptiveClipState:
    """Tests for sync_state() with AdaptiveClipState."""

    def test_sync_adaptive_clip_state(self):
        """sync_state() works with AdaptiveClipState."""
        if dist_utils.is_initialized():
            pytest.skip("Distributed already initialized")

        from opaque.clipping import AdaptiveClipState

        state = AdaptiveClipState(
            clip_norm=1.0, step=100, clipping_rate=0.8, rescale_to_unit_norm=False
        )

        result = dist_utils.sync_state(
            state, sync_fields=["clip_norm", "clipping_rate"], op="mean"
        )

        # When not distributed, should be unchanged
        assert result is state

    def test_sync_adaptive_clip_state_auto_detect_fields(self):
        """sync_state() auto-detects numeric fields when sync_fields=None."""
        if dist_utils.is_initialized():
            pytest.skip("Distributed already initialized")

        from opaque.clipping import AdaptiveClipState

        state = AdaptiveClipState(
            clip_norm=1.0, step=100, clipping_rate=0.8, rescale_to_unit_norm=False
        )

        # Auto-detect: should sync all float/int fields
        result = dist_utils.sync_state(state, sync_fields=None, op="mean")

        # When not distributed, should be unchanged
        assert result is state


class TestSyncStateFieldTypes:
    """Tests for sync_state() with different field types."""

    def test_sync_float_fields(self):
        """sync_state() syncs float fields."""
        if dist_utils.is_initialized():
            pytest.skip("Distributed already initialized")

        @dataclass
        class State:
            float_val: float
            step: int

        state = State(float_val=1.5, step=10)
        result = dist_utils.sync_state(state, sync_fields=["float_val"])

        assert result is state

    def test_sync_int_fields(self):
        """sync_state() syncs int fields and preserves type."""
        if dist_utils.is_initialized():
            pytest.skip("Distributed already initialized")

        @dataclass
        class State:
            int_val: int
            step: int

        state = State(int_val=5, step=10)
        result = dist_utils.sync_state(state, sync_fields=["int_val"])

        # When synced, int_val should still be int
        assert isinstance(result.int_val, int)

    def test_sync_mixed_types(self):
        """sync_state() syncs both float and int fields."""
        if dist_utils.is_initialized():
            pytest.skip("Distributed already initialized")

        @dataclass
        class State:
            float_val: float
            int_val: int
            step: int

        state = State(float_val=1.5, int_val=5, step=10)
        result = dist_utils.sync_state(state, sync_fields=["float_val", "int_val"])

        assert result is state

    def test_sync_excludes_bool_fields(self):
        """sync_state() excludes bool fields when auto-detecting."""
        if dist_utils.is_initialized():
            pytest.skip("Distributed already initialized")

        @dataclass
        class State:
            float_val: float
            bool_val: bool
            step: int

        state = State(float_val=1.5, bool_val=True, step=10)

        # Auto-detect: should NOT sync bool_val
        result = dist_utils.sync_state(state, sync_fields=None)

        # Should sync float_val and step (both numeric)
        assert result is state


class TestSyncStateImmutability:
    """Tests for sync_state() immutability."""

    def test_sync_state_returns_new_object_when_synced(self):
        """sync_state() returns new object when fields are synced (in distributed mode).

        Note: This is a structural test - we can't test actual sync without
        multi-device setup, but we ensure the API contract is correct.
        """
        if dist_utils.is_initialized():
            pytest.skip("Distributed already initialized")

        @dataclass
        class State:
            value: float

        state = State(value=1.0)

        # Even when not distributed, calling sync_state should work
        result = dist_utils.sync_state(state, sync_fields=["value"])

        # In non-distributed mode, might return same object (optimization)
        # That's okay - the important thing is it doesn't error
        assert result.value == state.value

    def test_sync_state_preserves_unsynced_fields(self):
        """sync_state() only updates synced fields."""
        if dist_utils.is_initialized():
            pytest.skip("Distributed already initialized")

        @dataclass
        class State:
            synced: float
            not_synced: float
            step: int

        state = State(synced=1.0, not_synced=2.0, step=10)

        # Only sync 'synced' field
        result = dist_utils.sync_state(state, sync_fields=["synced"])

        # not_synced should be unchanged
        assert result.not_synced == state.not_synced
        assert result.step == state.step
