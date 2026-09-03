"""Torch-only behavior for bounded Gaussian noise (``gaussian_noise(bound=...)``)."""

import torch

from opaque.dpsgd.noise import gaussian_noise
from opaque.random import key
from opaque.types import clipped


class TestBoundedGaussianTorchNative:
    """Behavior specific to the Torch runtime, not shared across providers."""

    def test_dtype_preservation_float64(self):
        noise_fn, state = gaussian_noise(noise_multiplier=1.0, bound=5.0, key=key(0))
        grad_f64 = torch.randn(5, 3, dtype=torch.float64)
        out_f64, state = noise_fn(clipped(grad_f64, max_norm=1.0), state)
        assert out_f64.pytree.dtype == torch.float64

    def test_device_preservation(self):
        noise_fn, state = gaussian_noise(noise_multiplier=1.0, bound=5.0, key=key(0))

        grad_cpu = torch.randn(5, 3)
        out_cpu, state = noise_fn(clipped(grad_cpu, max_norm=1.0), state)
        assert out_cpu.pytree.device == torch.device("cpu")

        if torch.backends.mps.is_available():
            grad_mps = torch.randn(5, 3, device="mps")
            out_mps, state = noise_fn(clipped(grad_mps, max_norm=1.0), state)
            assert out_mps.pytree.device.type == "mps"
