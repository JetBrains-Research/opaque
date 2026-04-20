"""Tests for gaussian_noise with PerGroup stddev."""

import pytest
import torch

from opaque.noise.gaussian import gaussian_noise
from opaque.noise.gaussian import GaussianNoiseState
from opaque.random import key
from opaque.utils.per_group import PerGroup


class TestGaussianNoisePerGroup:
    """Tests for gaussian_noise() with PerGroup stddev."""

    def _make_pg(self, param_keys, group_values):
        """Helper: one group per param, value = group_values[key]."""
        groups = {k: k for k in param_keys}
        values = {k: v for k, v in zip(param_keys, group_values)}
        return PerGroup(groups=groups, values=values)

    def test_returns_tuple(self):
        pg = self._make_pg(["w", "b"], [1.0, 2.0])
        noise_fn, state = gaussian_noise(stddev=pg, key=key(0))
        assert callable(noise_fn)
        assert isinstance(state, GaussianNoiseState)

    def test_adds_per_group_noise(self):
        """Each parameter should receive noise scaled by its group's stddev."""
        pg = PerGroup(
            groups={"weight": "attn", "bias": "mlp"},
            values={"attn": 1.0, "mlp": 5.0},
        )
        noise_fn, state = gaussian_noise(stddev=pg, key=key(42))
        grads = {
            "weight": torch.zeros(100),
            "bias": torch.zeros(100),
        }
        noisy, state = noise_fn(grads, state)

        # Both should have noise
        assert not torch.allclose(noisy["weight"], grads["weight"])
        assert not torch.allclose(noisy["bias"], grads["bias"])

        # mlp (stddev=5) should have larger noise than attn (stddev=1)
        attn_var = noisy["weight"].var().item()
        mlp_var = noisy["bias"].var().item()
        assert mlp_var > attn_var * 5  # rough check: 25x variance ratio

    def test_zero_stddev_group_returns_original(self):
        """A group with stddev=0 should return the original tensor."""
        pg = PerGroup(
            groups={"noisy": "g1", "clean": "g2"},
            values={"g1": 1.0, "g2": 0.0},
        )
        # Note: all groups zero → early return. Test mixed case.
        noise_fn, state = gaussian_noise(stddev=pg, key=key(0))
        grads = {
            "noisy": torch.ones(5),
            "clean": torch.ones(5),
        }
        noisy, state = noise_fn(grads, state)
        # "clean" group with stddev=0 should get noise * 0 = original
        torch.testing.assert_close(noisy["clean"], grads["clean"])
        # "noisy" group should differ
        assert not torch.allclose(noisy["noisy"], grads["noisy"])

    def test_all_zero_stddev_returns_original(self):
        """All groups with stddev=0 should return original gradients."""
        pg = PerGroup(
            groups={"a": "g1", "b": "g2"},
            values={"g1": 0.0, "g2": 0.0},
        )
        noise_fn, state = gaussian_noise(stddev=pg, key=key(0))
        grads = {"a": torch.randn(3), "b": torch.randn(3)}
        noisy, state = noise_fn(grads, state)
        torch.testing.assert_close(noisy["a"], grads["a"])
        torch.testing.assert_close(noisy["b"], grads["b"])

    def test_per_call_override(self):
        """Per-call stddev override should work with PerGroup."""
        pg_default = PerGroup(groups={"w": "g1"}, values={"g1": 1.0})
        pg_override = PerGroup(groups={"w": "g1"}, values={"g1": 100.0})
        noise_fn, state = gaussian_noise(stddev=pg_default, key=key(0))
        grads = {"w": torch.zeros(100)}

        # Default stddev=1
        noisy_default, state = noise_fn(grads, state)
        # Override stddev=100
        noisy_override, state = noise_fn(grads, state, stddev=pg_override)

        var_default = noisy_default["w"].var().item()
        var_override = noisy_override["w"].var().item()
        assert var_override > var_default * 100  # Much larger variance

    def test_dtype_preservation(self):
        pg = PerGroup(groups={"w": "g"}, values={"g": 1.0})
        noise_fn, state = gaussian_noise(stddev=pg, key=key(0))

        grads_f32 = {"w": torch.randn(5, dtype=torch.float32)}
        noisy, state = noise_fn(grads_f32, state)
        assert noisy["w"].dtype == torch.float32

        grads_f64 = {"w": torch.randn(5, dtype=torch.float64)}
        noisy, state = noise_fn(grads_f64, state)
        assert noisy["w"].dtype == torch.float64

    def test_step_counter_advances(self):
        pg = PerGroup(groups={"w": "g"}, values={"g": 1.0})
        noise_fn, state = gaussian_noise(stddev=pg, key=key(0))
        assert state._step_counter == 0

        grads = {"w": torch.zeros(3)}
        _, state = noise_fn(grads, state)
        assert state._step_counter == 1
        _, state = noise_fn(grads, state)
        assert state._step_counter == 2

    def test_deterministic_noise(self):
        """Same key + step should produce same noise."""
        pg = PerGroup(groups={"w": "g"}, values={"g": 1.0})
        grads = {"w": torch.zeros(10)}

        noise_fn1, state1 = gaussian_noise(stddev=pg, key=key(42))
        noisy1, _ = noise_fn1(grads, state1)

        noise_fn2, state2 = gaussian_noise(stddev=pg, key=key(42))
        noisy2, _ = noise_fn2(grads, state2)

        torch.testing.assert_close(noisy1["w"], noisy2["w"])

    def test_negative_stddev_raises(self):
        pg = PerGroup(groups={"w": "g"}, values={"g": -1.0})
        with pytest.raises(ValueError, match="non-negative"):
            gaussian_noise(stddev=pg, key=key(0))

    def test_scalar_override_with_per_group_default(self):
        """Scalar override should work even when default is PerGroup."""
        pg = PerGroup(groups={"w": "g"}, values={"g": 1.0})
        noise_fn, state = gaussian_noise(stddev=pg, key=key(0))

        grads = {"w": torch.zeros(5)}
        # Override with scalar
        noisy, state = noise_fn(grads, state, stddev=0.0)
        torch.testing.assert_close(noisy["w"], grads["w"])


class TestEndToEndPerGroup:
    """Integration test: clipped_grad + gaussian_noise with PerGroup."""

    def test_full_pipeline_isotropic(self):
        """Per-group clipping with isotropic noise (scalar sensitivity)."""
        from opaque.clipping import clipped_grad
        from opaque.utils.per_group import per_group

        def loss(params, data):
            pred = params["attn_w"] * data + params["mlp_w"] * data
            return (pred**2).mean()

        params = {
            "attn_w": torch.tensor(1.0),
            "mlp_w": torch.tensor(2.0),
        }

        pg = per_group(params, attn=1.0, mlp=2.0)

        grad_fn, clip_state = clipped_grad(
            loss,
            argnums=0,
            batch_argnums=1,
            clipping_norm=pg,
            normalize_by=10.0,
        )

        noise_multiplier = 1.1
        # sensitivity is scalar for PerGroup: sqrt(1^2 + 2^2) / 10
        stddev = noise_multiplier * clip_state.sensitivity
        assert isinstance(stddev, float)

        noise_fn, noise_state = gaussian_noise(stddev=stddev, key=key(42))

        # Run one step
        data = torch.randn(10)
        grads, clip_state = grad_fn(params, data, state=clip_state)
        noisy_grads, noise_state = noise_fn(grads, noise_state, stddev=stddev)

        assert isinstance(noisy_grads, dict)
        assert "attn_w" in noisy_grads
        assert "mlp_w" in noisy_grads

    def test_full_pipeline_per_group_noise(self):
        """Per-group clipping with per-group noise via per_group_noise_stddev."""
        from opaque.clipping import clipped_grad
        from opaque.noise.per_group_noise import per_group_noise_stddev
        from opaque.utils.per_group import per_group

        def loss(params, data):
            pred = params["attn_w"] * data + params["mlp_w"] * data
            return (pred**2).mean()

        params = {
            "attn_w": torch.tensor(1.0),
            "mlp_w": torch.tensor(2.0),
        }

        pg = per_group(params, attn=1.0, mlp=2.0)

        grad_fn, clip_state = clipped_grad(
            loss,
            argnums=0,
            batch_argnums=1,
            clipping_norm=pg,
            normalize_by=10.0,
        )

        noise_multiplier = 1.1
        stddev = per_group_noise_stddev(clip_state, noise_multiplier)
        assert isinstance(stddev, PerGroup)

        noise_fn, noise_state = gaussian_noise(stddev=stddev, key=key(42))

        data = torch.randn(10)
        grads, clip_state = grad_fn(params, data, state=clip_state)
        noisy_grads, noise_state = noise_fn(grads, noise_state, stddev=stddev)

        assert isinstance(noisy_grads, dict)
        assert "attn_w" in noisy_grads
        assert "mlp_w" in noisy_grads
