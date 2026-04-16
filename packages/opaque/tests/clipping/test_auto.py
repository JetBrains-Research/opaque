"""Tests for AUTO-S automatic gradient clipping.

Tests the scale_pytree_auto_s primitive and the auto_clipped_grad process.
"""

import math

import pytest
import torch

from opaque.clipping import auto_clipped_grad
from opaque.clipping.auto import AutoClipState, AutoClippedGradAux
from opaque.clipping.pytree import scale_pytree_auto_s, ClipPytreeAux
from opaque.utils.per_group import PerGroup


# ─── scale_pytree_auto_s ────────────────────────────────────────────────────


class TestScalePytreeAutoS:
    """Unit tests for the AUTO-S scaling primitive."""

    def test_closed_form_scalar(self):
        """Scaled norm should equal R * ||g|| / (||g|| + gamma)."""
        g = {"w": torch.tensor([3.0, 4.0])}  # norm = 5
        R, gamma = 1.0, 0.01
        scaled, aux = scale_pytree_auto_s(g, clipping_norm=R, gamma=gamma)
        expected_norm = R * 5.0 / (5.0 + gamma)
        actual_norm = torch.linalg.norm(scaled["w"]).item()
        assert abs(actual_norm - expected_norm) < 1e-5

    def test_output_norm_less_than_R(self):
        """For any finite input, ||output|| < R (strictly)."""
        for norm_val in [0.001, 0.1, 1.0, 10.0, 1000.0]:
            g = {"w": torch.randn(100) * norm_val}
            R = 2.0
            scaled, _ = scale_pytree_auto_s(g, clipping_norm=R, gamma=0.01)
            out_norm = torch.linalg.norm(scaled["w"]).item()
            assert out_norm < R, (
                f"Output norm {out_norm} >= R={R} for input norm {norm_val}"
            )

    def test_zero_grad_stays_zero(self):
        """Zero gradients produce zero output (gamma prevents division by zero)."""
        g = {"w": torch.zeros(5)}
        scaled, aux = scale_pytree_auto_s(g, clipping_norm=1.0, gamma=0.01)
        torch.testing.assert_close(scaled["w"], torch.zeros(5))
        assert aux.norm.item() == 0.0

    def test_nan_sanitized_to_zero(self):
        """NaN in input is sanitized to zero before scaling."""
        g = {"w": torch.tensor([float("nan"), 1.0, float("nan")])}
        scaled, _ = scale_pytree_auto_s(g, clipping_norm=1.0, gamma=0.01)
        assert torch.isfinite(scaled["w"]).all()

    def test_inf_sanitized_to_zero(self):
        """Inf in input is sanitized to zero before scaling."""
        g = {"w": torch.tensor([float("inf"), 1.0, float("-inf")])}
        scaled, _ = scale_pytree_auto_s(g, clipping_norm=1.0, gamma=0.01)
        assert torch.isfinite(scaled["w"]).all()

    def test_aux_returns_original_norm(self):
        """aux.norm should be the L2 norm of the sanitized input."""
        g = {"a": torch.tensor([3.0]), "b": torch.tensor([4.0])}
        _, aux = scale_pytree_auto_s(g, clipping_norm=1.0, gamma=0.01)
        assert isinstance(aux, ClipPytreeAux)
        torch.testing.assert_close(aux.norm, torch.tensor(5.0))

    def test_direction_preserved(self):
        """Scaling should not change gradient direction."""
        g = {"w": torch.tensor([3.0, 4.0])}
        scaled, _ = scale_pytree_auto_s(g, clipping_norm=2.0, gamma=0.01)
        original_dir = g["w"] / torch.linalg.norm(g["w"])
        scaled_dir = scaled["w"] / torch.linalg.norm(scaled["w"])
        torch.testing.assert_close(original_dir, scaled_dir, atol=1e-6, rtol=1e-6)

    def test_larger_norms_get_larger_scaled_output(self):
        """AUTO-S preserves relative magnitude: larger inputs → larger outputs."""
        R, gamma = 1.0, 0.01
        g_small = {"w": torch.tensor([0.5])}
        g_large = {"w": torch.tensor([5.0])}
        scaled_small, _ = scale_pytree_auto_s(g_small, clipping_norm=R, gamma=gamma)
        scaled_large, _ = scale_pytree_auto_s(g_large, clipping_norm=R, gamma=gamma)
        assert abs(scaled_large["w"].item()) > abs(scaled_small["w"].item())

    def test_gamma_zero_raises(self):
        """gamma=0 would remove the stability constant — not supported via auto_clipped_grad."""
        # scale_pytree_auto_s itself doesn't validate gamma (caller does),
        # but we verify the behavior at boundary: gamma=0 means pure normalization.
        g = {"w": torch.tensor([3.0, 4.0])}
        # With gamma=0, this is just g * R / ||g||, norm exactly R.
        scaled, _ = scale_pytree_auto_s(g, clipping_norm=1.0, gamma=0.0)
        out_norm = torch.linalg.norm(scaled["w"]).item()
        assert abs(out_norm - 1.0) < 1e-5

    def test_multitensor_pytree(self):
        """Works with dict of multiple tensors of different shapes."""
        g = {
            "w1": torch.randn(10, 5),
            "b1": torch.randn(5),
            "w2": torch.randn(5, 3),
        }
        R = 0.5
        scaled, aux = scale_pytree_auto_s(g, clipping_norm=R, gamma=0.1)
        total_norm_sq = sum((scaled[k] ** 2).sum().item() for k in g)
        assert math.sqrt(total_norm_sq) < R


class TestScalePytreeAutoSPerGroup:
    """Tests for AUTO-S with PerGroup clipping norms."""

    def test_per_group_independent_scaling(self):
        """Each group should be scaled by its own R_k / (||g||_k + gamma)."""
        pytree = {
            "attn.q": torch.tensor([3.0]),
            "attn.k": torch.tensor([4.0]),
            "mlp.w": torch.tensor([2.0]),
        }
        pg = PerGroup(
            groups={"attn.q": "attn", "attn.k": "attn", "mlp.w": "mlp"},
            values={"attn": 1.0, "mlp": 0.5},
        )
        gamma = 0.01
        scaled, aux = scale_pytree_auto_s(pytree, clipping_norm=pg, gamma=gamma)

        # attn group: norm=5, scale = 1.0/(5.0+0.01)
        attn_scale = 1.0 / (5.0 + gamma)
        torch.testing.assert_close(scaled["attn.q"], torch.tensor([3.0 * attn_scale]))
        torch.testing.assert_close(scaled["attn.k"], torch.tensor([4.0 * attn_scale]))

        # mlp group: norm=2, scale = 0.5/(2.0+0.01)
        mlp_scale = 0.5 / (2.0 + gamma)
        torch.testing.assert_close(scaled["mlp.w"], torch.tensor([2.0 * mlp_scale]))

    def test_per_group_output_bounded(self):
        """Each group's output norm should be < R_k."""
        pytree = {
            "a": torch.randn(20),
            "b": torch.randn(15),
        }
        pg = PerGroup(
            groups={"a": "g1", "b": "g2"},
            values={"g1": 2.0, "g2": 0.5},
        )
        scaled, aux = scale_pytree_auto_s(pytree, clipping_norm=pg, gamma=0.01)
        assert torch.linalg.norm(scaled["a"]).item() < 2.0
        assert torch.linalg.norm(scaled["b"]).item() < 0.5

    def test_per_group_aux_has_group_norms(self):
        """aux.group_norms should contain per-group original norms."""
        pytree = {"x": torch.tensor([3.0]), "y": torch.tensor([4.0])}
        pg = PerGroup(
            groups={"x": "g1", "y": "g2"},
            values={"g1": 1.0, "g2": 1.0},
        )
        _, aux = scale_pytree_auto_s(pytree, clipping_norm=pg, gamma=0.01)
        assert aux.group_norms is not None
        torch.testing.assert_close(aux.group_norms["g1"], torch.tensor(3.0))
        torch.testing.assert_close(aux.group_norms["g2"], torch.tensor(4.0))


# ─── AutoClipState ───────────────────────────────────────────────────────────


class TestAutoClipState:
    """Tests for AutoClipState validation and properties."""

    def test_sensitivity_scalar(self):
        s = AutoClipState(clipping_norm=2.0, normalize_by=4.0, gamma=0.01)
        assert s.sensitivity == 0.5

    def test_sensitivity_per_group(self):
        pg = PerGroup(
            groups={"a": "g1", "b": "g2"},
            values={"g1": 3.0, "g2": 4.0},
        )
        s = AutoClipState(clipping_norm=pg, normalize_by=1.0, gamma=0.01)
        assert abs(s.sensitivity - 5.0) < 1e-10  # sqrt(9+16) = 5

    def test_negative_clipping_norm_raises(self):
        with pytest.raises(ValueError, match="clipping_norm must be positive"):
            AutoClipState(clipping_norm=-1.0, normalize_by=1.0, gamma=0.01)

    def test_zero_normalize_by_raises(self):
        with pytest.raises(ValueError, match="normalize_by must be positive"):
            AutoClipState(clipping_norm=1.0, normalize_by=0.0, gamma=0.01)

    def test_negative_gamma_raises(self):
        with pytest.raises(ValueError, match="gamma must be positive"):
            AutoClipState(clipping_norm=1.0, normalize_by=1.0, gamma=-0.01)


# ─── auto_clipped_grad ──────────────────────────────────────────────────────


class TestAutoClippedGrad:
    """Tests for the auto_clipped_grad clipping process."""

    @staticmethod
    def _simple_loss(param, data):
        return 0.5 * ((data - param) ** 2).mean()

    def test_basic_returns_gradient(self):
        grad_fn, clip_state = auto_clipped_grad(
            self._simple_loss,
            clipping_norm=10.0,
            gamma=0.01,
        )
        param = torch.tensor(3.0)
        data = torch.tensor([0.0, 7.0, -2.0])
        grad, new_state = grad_fn(param, data, state=clip_state)
        assert isinstance(grad, torch.Tensor)
        assert grad.shape == param.shape
        assert isinstance(new_state, AutoClipState)

    def test_sensitivity_unchanged(self):
        """ClipState.sensitivity must be R / normalize_by regardless of input."""
        R, nb = 2.0, 10.0
        _, clip_state = auto_clipped_grad(
            self._simple_loss,
            clipping_norm=R,
            gamma=0.01,
            normalize_by=nb,
        )
        assert clip_state.sensitivity == R / nb

    def test_output_norm_bounded(self):
        """Sum of per-example scaled norms, divided by normalize_by,
        should produce output with norm <= R * batch_size / normalize_by."""
        R = 1.0
        nb = 3.0  # equal to batch size
        grad_fn, clip_state = auto_clipped_grad(
            self._simple_loss,
            clipping_norm=R,
            gamma=0.01,
            normalize_by=nb,
        )
        param = torch.tensor(3.0)
        data = torch.tensor([0.0, 7.0, -2.0])
        grad, _ = grad_fn(param, data, state=clip_state)
        # Each per-example scaled grad has norm < R; sum / nb has norm < R*3/3 = R
        assert abs(grad.item()) < R + 0.01

    def test_return_aux(self):
        grad_fn, clip_state = auto_clipped_grad(
            self._simple_loss,
            clipping_norm=10.0,
            gamma=0.01,
            return_aux=True,
        )
        param = torch.tensor(3.0)
        data = torch.tensor([0.0, 7.0, -2.0])
        (grad, aux), _ = grad_fn(param, data, state=clip_state)
        assert isinstance(aux, AutoClippedGradAux)
        assert aux.batch_size == 3
        assert aux.grad_norms is not None
        assert aux.grad_norms.shape == (3,)
        assert aux.clipped_grad_norms is not None
        assert aux.clipped_grad_norms.shape == (3,)
        assert isinstance(aux.clipping_rate, float)
        assert 0.0 <= aux.clipping_rate <= 1.0
        assert aux.loss_values is not None
        assert aux.loss_values.shape == (3,)

    def test_aux_scaled_norms_less_than_R(self):
        """Each per-example scaled grad norm should be < R."""
        R = 0.5
        grad_fn, clip_state = auto_clipped_grad(
            self._simple_loss,
            clipping_norm=R,
            gamma=0.01,
            return_aux=True,
        )
        param = torch.tensor(10.0)
        data = torch.linspace(-5, 5, 20)
        (_, aux), _ = grad_fn(param, data, state=clip_state)
        assert (aux.clipped_grad_norms < R).all()

    def test_pytree_params(self):
        def loss(params, data):
            pred = params["w"] * data + params["b"]
            return ((pred - data) ** 2).mean()

        grad_fn, clip_state = auto_clipped_grad(
            loss,
            clipping_norm=5.0,
            gamma=0.01,
        )
        params = {"w": torch.tensor(2.0), "b": torch.tensor(0.5)}
        data = torch.tensor([1.0, 2.0, 3.0])
        grad, _ = grad_fn(params, data, state=clip_state)
        assert isinstance(grad, dict)
        assert "w" in grad and "b" in grad

    def test_has_aux(self):
        def loss_with_aux(param, data):
            loss = 0.5 * ((data - param) ** 2).mean()
            return loss, {"pred": data - param}

        grad_fn, clip_state = auto_clipped_grad(
            loss_with_aux,
            has_aux=True,
            clipping_norm=10.0,
            gamma=0.01,
            return_aux=True,
        )
        param = torch.tensor(3.0)
        data = torch.tensor([0.0, 7.0, -2.0])
        (grad, aux), _ = grad_fn(param, data, state=clip_state)
        assert isinstance(grad, torch.Tensor)
        assert aux.loss_aux is not None

    def test_microbatch_matches_full_batch(self):
        """Microbatch and full-batch should produce identical results."""
        R = 1.0
        param = torch.tensor(3.0)
        data = torch.linspace(-5, 5, 12)

        grad_fn_full, cs_full = auto_clipped_grad(
            self._simple_loss,
            clipping_norm=R,
            gamma=0.01,
            return_aux=True,
        )
        grad_fn_mb, cs_mb = auto_clipped_grad(
            self._simple_loss,
            clipping_norm=R,
            gamma=0.01,
            return_aux=True,
            microbatch_size=4,
        )
        (grad_full, aux_full), _ = grad_fn_full(param, data, state=cs_full)
        (grad_mb, aux_mb), _ = grad_fn_mb(param, data, state=cs_mb)
        torch.testing.assert_close(grad_full, grad_mb, atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(
            aux_full.grad_norms, aux_mb.grad_norms, atol=1e-5, rtol=1e-5
        )

    def test_empty_batch(self):
        grad_fn, clip_state = auto_clipped_grad(
            self._simple_loss,
            clipping_norm=1.0,
            gamma=0.01,
        )
        param = torch.tensor(3.0)
        data = torch.tensor([])
        result, state = grad_fn(param, data, state=clip_state)
        assert isinstance(result, torch.Tensor)
        assert result.item() == 0.0

    def test_empty_batch_with_aux(self):
        grad_fn, clip_state = auto_clipped_grad(
            self._simple_loss,
            clipping_norm=1.0,
            gamma=0.01,
            return_aux=True,
        )
        param = torch.tensor(3.0)
        data = torch.tensor([])
        (result, aux), state = grad_fn(param, data, state=clip_state)
        assert result.item() == 0.0
        assert aux.batch_size == 0

    def test_state_immutable(self):
        """State should not mutate between calls (functional pattern)."""
        grad_fn, clip_state = auto_clipped_grad(
            self._simple_loss,
            clipping_norm=1.0,
            gamma=0.01,
        )
        param = torch.tensor(3.0)
        data = torch.tensor([1.0, 2.0])
        _, state_1 = grad_fn(param, data, state=clip_state)
        _, state_2 = grad_fn(param, data, state=state_1)
        assert state_1.clipping_norm == state_2.clipping_norm
        assert state_1.gamma == state_2.gamma

    def test_validation_negative_clipping_norm(self):
        with pytest.raises(ValueError, match="clipping_norm must be positive"):
            auto_clipped_grad(self._simple_loss, clipping_norm=-1.0)

    def test_validation_negative_gamma(self):
        with pytest.raises(ValueError, match="gamma must be positive"):
            auto_clipped_grad(self._simple_loss, clipping_norm=1.0, gamma=-0.01)

    def test_validation_zero_gamma(self):
        with pytest.raises(ValueError, match="gamma must be positive"):
            auto_clipped_grad(self._simple_loss, clipping_norm=1.0, gamma=0.0)

    def test_validation_negative_normalize_by(self):
        with pytest.raises(ValueError, match="normalize_by must be > 0"):
            auto_clipped_grad(self._simple_loss, clipping_norm=1.0, normalize_by=-1.0)


class TestAutoClippedGradWithNoise:
    """Integration tests: auto_clipped_grad + gaussian_noise."""

    def test_noise_calibration_uses_sensitivity(self):
        """gaussian_noise(nm * clip_state.sensitivity) should work unchanged."""
        from opaque.noise import gaussian_noise
        from opaque.random import key

        R, nb = 1.0, 4.0
        nm = 1.1
        grad_fn, clip_state = auto_clipped_grad(
            lambda p, d: 0.5 * ((d - p) ** 2).mean(),
            clipping_norm=R,
            gamma=0.01,
            normalize_by=nb,
        )
        stddev = nm * clip_state.sensitivity
        assert abs(stddev - nm * R / nb) < 1e-10

        noise_fn, noise_state = gaussian_noise(stddev=stddev, key=key(42))
        param = torch.tensor(3.0)
        data = torch.tensor([0.0, 7.0, -2.0, 1.0])
        grad, clip_state = grad_fn(param, data, state=clip_state)
        noisy_grad, _ = noise_fn(grad, noise_state)
        assert isinstance(noisy_grad, torch.Tensor)
        assert torch.isfinite(noisy_grad).all()

    def test_per_group_with_noise(self):
        """auto_clipped_grad with PerGroup + isotropic noise roundtrip."""
        from opaque.noise import gaussian_noise
        from opaque.random import key

        def loss(params, data):
            return ((params["w"] * data + params["b"] - data) ** 2).mean()

        pg = PerGroup(
            groups={"w": "main", "b": "bias"},
            values={"main": 1.0, "bias": 0.5},
        )
        grad_fn, clip_state = auto_clipped_grad(
            loss,
            clipping_norm=pg,
            gamma=0.01,
            normalize_by=3.0,
            return_aux=True,
        )
        assert abs(clip_state.sensitivity - pg.effective / 3.0) < 1e-10

        noise_fn, noise_state = gaussian_noise(
            stddev=1.1 * clip_state.sensitivity, key=key(0)
        )
        params = {"w": torch.tensor(2.0), "b": torch.tensor(0.5)}
        data = torch.tensor([1.0, 2.0, 3.0])
        (grad, aux), _ = grad_fn(params, data, state=clip_state)
        noisy, _ = noise_fn(grad, noise_state)
        assert "w" in noisy and "b" in noisy
        assert torch.isfinite(noisy["w"]).all()
        assert torch.isfinite(noisy["b"]).all()
