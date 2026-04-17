"""Tests for data-dependent AUTO-S clipping (Phase 3)."""

import pytest
import torch

from opaque.clipping.auto_data import (
    DataDependentAutoClipState,
    data_dependent_auto_clipped_grad,
)
from opaque.clipping.auto import AutoClippedGradAux


class TestDataDependentAutoClipState:
    def test_sensitivity_equals_safety_clip(self):
        s = DataDependentAutoClipState(
            clipping_norm=2.0,
            normalize_by=10.0,
            gamma=0.01,
            threshold_scale=1.0,
            last_threshold=1.5,
        )
        assert s.sensitivity == 2.0 / 10.0

    def test_validation(self):
        with pytest.raises(ValueError, match="clipping_norm"):
            DataDependentAutoClipState(
                clipping_norm=-1.0,
                normalize_by=1.0,
                gamma=0.01,
                threshold_scale=1.0,
                last_threshold=1.0,
            )
        with pytest.raises(ValueError, match="gamma"):
            DataDependentAutoClipState(
                clipping_norm=1.0,
                normalize_by=1.0,
                gamma=-0.01,
                threshold_scale=1.0,
                last_threshold=1.0,
            )
        with pytest.raises(ValueError, match="threshold_scale"):
            DataDependentAutoClipState(
                clipping_norm=1.0,
                normalize_by=1.0,
                gamma=0.01,
                threshold_scale=-1.0,
                last_threshold=1.0,
            )


class TestDataDependentAutoClippedGrad:
    @staticmethod
    def _simple_loss(param, data):
        return 0.5 * ((data - param) ** 2).mean()

    def test_basic(self):
        grad_fn, clip_state = data_dependent_auto_clipped_grad(
            self._simple_loss,
            safety_clip_norm=10.0,
            threshold_scale=1.0,
            gamma=0.01,
        )
        param = torch.tensor(3.0)
        data = torch.tensor([0.0, 7.0, -2.0])
        grad, new_state = grad_fn(param, data, state=clip_state)
        assert isinstance(grad, torch.Tensor)
        assert isinstance(new_state, DataDependentAutoClipState)
        assert new_state.last_threshold > 0
        assert new_state.last_threshold <= 10.0

    def test_threshold_adapts(self):
        """Different data should produce different thresholds."""
        grad_fn, clip_state = data_dependent_auto_clipped_grad(
            self._simple_loss,
            safety_clip_norm=100.0,
            threshold_scale=1.0,
            gamma=0.01,
        )
        param = torch.tensor(0.0)

        data_small = torch.tensor([0.1, 0.2, 0.3])
        _, state_small = grad_fn(param, data_small, state=clip_state)

        data_large = torch.tensor([10.0, 20.0, 30.0])
        _, state_large = grad_fn(param, data_large, state=clip_state)

        assert state_large.last_threshold > state_small.last_threshold

    def test_threshold_capped_at_safety_clip(self):
        """C_t should never exceed safety_clip_norm."""
        grad_fn, clip_state = data_dependent_auto_clipped_grad(
            self._simple_loss,
            safety_clip_norm=0.5,
            threshold_scale=100.0,
            gamma=0.01,
        )
        param = torch.tensor(0.0)
        data = torch.tensor([10.0, 20.0, 30.0])
        _, new_state = grad_fn(param, data, state=clip_state)
        assert new_state.last_threshold <= 0.5

    def test_sensitivity_uses_safety_clip(self):
        """sensitivity should use safety_clip_norm, not C_t."""
        _, clip_state = data_dependent_auto_clipped_grad(
            self._simple_loss,
            safety_clip_norm=5.0,
            normalize_by=10.0,
        )
        assert clip_state.sensitivity == 5.0 / 10.0

    def test_return_aux(self):
        grad_fn, clip_state = data_dependent_auto_clipped_grad(
            self._simple_loss,
            safety_clip_norm=10.0,
            gamma=0.01,
            return_aux=True,
        )
        param = torch.tensor(3.0)
        data = torch.tensor([0.0, 7.0, -2.0])
        (grad, aux), new_state = grad_fn(param, data, state=clip_state)
        assert isinstance(aux, AutoClippedGradAux)
        assert aux.batch_size == 3
        assert aux.grad_norms is not None
        assert aux.grad_norms.shape == (3,)

    def test_empty_batch(self):
        grad_fn, clip_state = data_dependent_auto_clipped_grad(
            self._simple_loss,
            safety_clip_norm=1.0,
        )
        param = torch.tensor(3.0)
        data = torch.tensor([])
        result, state = grad_fn(param, data, state=clip_state)
        assert result.item() == 0.0
        assert state.last_threshold == 0.0

    def test_pytree_params(self):
        def loss(params, data):
            pred = params["w"] * data + params["b"]
            return ((pred - data) ** 2).mean()

        grad_fn, clip_state = data_dependent_auto_clipped_grad(
            loss,
            safety_clip_norm=5.0,
            gamma=0.01,
        )
        params = {"w": torch.tensor(2.0), "b": torch.tensor(0.5)}
        data = torch.tensor([1.0, 2.0, 3.0])
        grad, _ = grad_fn(params, data, state=clip_state)
        assert isinstance(grad, dict)
        assert "w" in grad and "b" in grad

    def test_validation_params(self):
        with pytest.raises(ValueError, match="safety_clip_norm"):
            data_dependent_auto_clipped_grad(self._simple_loss, safety_clip_norm=-1.0)
        with pytest.raises(ValueError, match="gamma"):
            data_dependent_auto_clipped_grad(
                self._simple_loss, safety_clip_norm=1.0, gamma=0.0
            )
        with pytest.raises(ValueError, match="threshold_scale"):
            data_dependent_auto_clipped_grad(
                self._simple_loss, safety_clip_norm=1.0, threshold_scale=-1.0
            )


class TestDataDependentWithAccounting:
    """Integration with auto_clip_gaussian accounting."""

    def test_accounting_roundtrip(self):
        from opaque_accounting import auto_clip_gaussian

        grad_fn, clip_state = data_dependent_auto_clipped_grad(
            lambda p, d: 0.5 * ((d - p) ** 2).mean(),
            safety_clip_norm=2.0,
            gamma=0.01,
            normalize_by=4.0,
        )
        param = torch.tensor(3.0)
        data = torch.tensor([0.0, 7.0, -2.0, 1.0])
        _, new_state = grad_fn(param, data, state=clip_state)

        nm = 1.1

        # Compute worst-case accounting parameters
        sensitivity = 1.0 / nm
        noise_ratio = 1.02  # small ratio change typical for large batches

        proc = auto_clip_gaussian(
            sensitivity=sensitivity,
            noise_ratio=noise_ratio,
            dimension=1,
        )
        eps = proc.pld().epsilon_at(1e-5)
        assert eps > 0
        assert isinstance(eps, float)
