"""Tests for adaptive gradient clipping with explicit state-passing."""

import pytest
import torch

from opaque.clipping import (
    AdaptiveClipState,
    adaptive_clipped_grad,
)
from opaque.clipping.types import NeighboringRelation


class TestAdaptiveClipState:
    """Tests for AdaptiveClipState dataclass."""

    def test_init_valid(self):
        """Test creating valid state."""
        state = AdaptiveClipState(clip_norm=1.0, step=0, clipping_rate=0.5)
        assert state.clip_norm == 1.0
        assert state.step == 0
        assert state.clipping_rate == 0.5

    def test_frozen(self):
        """Test that state is immutable."""
        state = AdaptiveClipState(clip_norm=1.0, step=0)
        with pytest.raises(Exception):  # FrozenInstanceError
            state.clip_norm = 2.0

    def test_invalid_clip_norm(self):
        """Test that negative clip_norm raises error."""
        with pytest.raises(ValueError, match="clip_norm must be positive"):
            AdaptiveClipState(clip_norm=-1.0, step=0)

    def test_invalid_step(self):
        """Test that negative step raises error."""
        with pytest.raises(ValueError, match="step must be non-negative"):
            AdaptiveClipState(clip_norm=1.0, step=-1)

    def test_invalid_clipping_rate(self):
        """Test that out-of-range clipping_rate raises error."""
        with pytest.raises(ValueError, match="clipping_rate must be in"):
            AdaptiveClipState(clip_norm=1.0, step=0, clipping_rate=1.5)


class TestAdaptiveClippedGrad:
    """Tests for adaptive_clipped_grad with explicit state-passing."""

    def test_basic_usage(self):
        """Test basic adaptive clipping with state return."""

        def loss_fn(params, x, y):
            pred = x @ params
            return ((pred - y) ** 2).mean()

        # Create adaptive clipping function
        grad_fn, state = adaptive_clipped_grad(
            loss_fn,
            initial_clip_norm=1.0,
            target_quantile=0.5,
            batch_argnums=(1, 2),
        )

        # Check initial state
        assert isinstance(state, AdaptiveClipState)
        assert state.clip_norm == 1.0
        assert state.step == 0
        assert state.clipping_rate == 0.0

        # Generate data
        params = torch.randn(5, requires_grad=False)
        batch_x = torch.randn(10, 5)
        batch_y = torch.randn(10)

        # Compute gradients
        grad, new_state = grad_fn(params, batch_x, batch_y, state=state)

        # Check outputs
        assert grad.shape == params.shape
        assert isinstance(new_state, AdaptiveClipState)
        assert new_state.step == state.step + 1
        assert new_state.clip_norm != state.clip_norm  # Should have adapted

    def test_state_independence(self):
        """Test that old state is not mutated."""

        def loss_fn(params, x):
            return (params * x).sum()

        grad_fn, state1 = adaptive_clipped_grad(
            loss_fn, initial_clip_norm=1.0, batch_argnums=1
        )

        params = torch.tensor([1.0, 2.0], requires_grad=False)
        batch_x = torch.randn(5, 2)

        # Call twice
        _, state2 = grad_fn(params, batch_x, state=state1)
        _, state3 = grad_fn(params, batch_x, state=state2)

        # Old states should be unchanged
        assert state1.step == 0
        assert state2.step == 1
        assert state3.step == 2
        assert state1.clip_norm == 1.0  # Original unchanged

    def test_clip_norm_adaptation(self):
        """Test that clip_norm adapts based on clipping rate."""

        def loss_fn(params, x):
            return (params * x).sum()

        grad_fn, state = adaptive_clipped_grad(
            loss_fn,
            initial_clip_norm=0.1,  # Very low → many clipped
            target_quantile=0.5,
            learning_rate=0.2,
            batch_argnums=1,
        )

        params = torch.tensor([1.0, 2.0], requires_grad=False)
        batch_x = torch.randn(100, 2) * 10  # Large gradients

        # Run multiple steps
        for _ in range(10):
            _, state = grad_fn(params, batch_x, state=state)

        # Clip norm should have increased (many gradients were clipped)
        assert state.clip_norm > 0.1

    def test_sensitivity_computation(self):
        """Test sensitivity method on state."""
        # Without rescaling
        state_no_rescale = AdaptiveClipState(
            clip_norm=1.5, step=10, clipping_rate=0.3, rescale_to_unit_norm=False
        )
        sens = state_no_rescale.sensitivity()
        assert sens == 1.5

        # With rescaling
        state_rescale = AdaptiveClipState(
            clip_norm=1.5, step=10, clipping_rate=0.3, rescale_to_unit_norm=True
        )
        sens = state_rescale.sensitivity()
        assert sens == 1.0

        # Different neighboring relations
        sens_add = state_no_rescale.sensitivity(
            neighboring_relation=NeighboringRelation.ADD_OR_REMOVE_ONE,
        )
        assert sens_add == 1.5

        sens_replace = state_no_rescale.sensitivity(
            neighboring_relation=NeighboringRelation.REPLACE_ONE,
        )
        assert sens_replace == 3.0  # 2 * clip_norm

    def test_with_aux(self):
        """Test adaptive clipping with auxiliary outputs."""

        def loss_fn(params, x):
            return (params * x).sum(), {"extra": x.mean()}

        grad_fn, state = adaptive_clipped_grad(
            loss_fn, has_aux=True, batch_argnums=1
        )

        params = torch.tensor([1.0, 2.0], requires_grad=False)
        batch_x = torch.randn(5, 2)

        (grad, aux), new_state = grad_fn(params, batch_x, state=state)

        assert grad.shape == params.shape
        assert aux.aux is not None
        assert "extra" in aux.aux
        assert isinstance(new_state, AdaptiveClipState)

    def test_parameter_validation(self):
        """Test that invalid parameters raise errors."""

        def loss_fn(params, x):
            return params.sum()

        # Invalid initial_clip_norm
        with pytest.raises(ValueError, match="initial_clip_norm must be positive"):
            adaptive_clipped_grad(loss_fn, initial_clip_norm=-1.0)

        # Invalid target_quantile
        with pytest.raises(ValueError, match="target_quantile must be in"):
            adaptive_clipped_grad(loss_fn, target_quantile=1.5)

        # Invalid learning_rate
        with pytest.raises(ValueError, match="learning_rate must be positive"):
            adaptive_clipped_grad(loss_fn, learning_rate=-0.1)

        # Invalid clip_norm_max < clip_norm_min
        with pytest.raises(ValueError, match="clip_norm_max"):
            adaptive_clipped_grad(
                loss_fn, clip_norm_min=10.0, clip_norm_max=5.0
            )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
