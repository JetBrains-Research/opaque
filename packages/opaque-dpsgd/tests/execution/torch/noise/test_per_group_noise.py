"""Torch-only float64 behavior for per-group Gaussian noise."""

import torch

from opaque.dpsgd.noise import gaussian_noise
from opaque.random import key
from opaque.types import PerGroup, clipped


class TestGaussianNoisePerGroup:
    def test_dtype_preservation(self):
        max_norm = PerGroup(groups={"w": "g"}, values={"g": 1.0})
        noise_fn, state = gaussian_noise(noise_multiplier=1.0, key=key(0))

        grads_f32 = {"w": torch.randn(5, dtype=torch.float32)}
        out_f32, state = noise_fn(clipped(grads_f32, max_norm=max_norm), state)
        assert out_f32.pytree["w"].dtype == torch.float32

        grads_f64 = {"w": torch.randn(5, dtype=torch.float64)}
        out_f64, state = noise_fn(clipped(grads_f64, max_norm=max_norm), state)
        assert out_f64.pytree["w"].dtype == torch.float64
