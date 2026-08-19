"""Golden-value pins for the keyed RNG streams on the Torch provider.

These values are the contract for DP noise reproducibility: an Opaque or
PyTorch upgrade that silently changes any of them breaks checkpoint-resume
determinism and changes the realized noise of in-flight training runs.

A failure here must be treated as a deliberate breaking change — bump
``DP_STATE_BUNDLE_VERSION``, note it in the changelog, and only then
re-pin. Never "fix" these numbers to make CI pass.
"""

from __future__ import annotations

import pytest
import torch

from opaque.backend import clear_backend, set_backend
from opaque.dpsgd.noise import gaussian_noise
from opaque.random import fold_in, key, normal
from opaque.types import clipped


@pytest.fixture(autouse=True)
def _torch_backend():
    set_backend("torch")
    yield
    clear_backend()


class TestNormalStreamPins:
    def test_normal_vector_golden(self):
        values = normal(key(42), (5,))
        expected = torch.tensor(
            [
                3.3669036627e-01,
                1.2880940735e-01,
                2.3446236551e-01,
                2.3033303022e-01,
                -1.1228563786e00,
            ],
            dtype=torch.float32,
        )
        assert values.dtype == torch.float32
        torch.testing.assert_close(values, expected, rtol=0.0, atol=0.0)

    def test_normal_scalar_golden(self):
        value = normal(key(123), ())
        assert value.shape == ()
        assert value.dtype == torch.float32
        expected = torch.tensor(-1.1146711558e-01, dtype=torch.float32)
        assert value.item() == expected.item()

    def test_leaf_fold_produces_distinct_stream(self):
        base = normal(key(42), (5,))
        folded = normal(fold_in(key(42), "gaussian_noise_leaf", 0), (5,))
        assert not torch.equal(base, folded)


class TestGaussianNoiseStreamPins:
    def test_two_step_two_leaf_stream_golden(self):
        """Pins the per-leaf key folding and the per-step key schedule.

        Zero gradients isolate the raw noise: the output IS the stream.
        """
        noise_fn, state = gaussian_noise(noise_multiplier=1.0, key=key(7))
        grads = {"a": torch.zeros(3), "b": torch.zeros(2)}

        out1, state = noise_fn(clipped(grads, max_norm=1.0), state)
        out2, state = noise_fn(clipped(grads, max_norm=1.0), state)

        expected = {
            "step1_a": [-5.1886880398e-01, -7.7415519953e-01, 1.4632741213e00],
            "step1_b": [-1.3098659515e00, 7.1730536222e-01],
            "step2_a": [-1.1790795326e00, -1.2313234806e00, -2.1688857079e00],
            "step2_b": [-5.0522208214e-01, 3.4721356630e-01],
        }
        for got, want in (
            (out1.pytree["a"], expected["step1_a"]),
            (out1.pytree["b"], expected["step1_b"]),
            (out2.pytree["a"], expected["step2_a"]),
            (out2.pytree["b"], expected["step2_b"]),
        ):
            torch.testing.assert_close(
                got,
                torch.tensor(want, dtype=torch.float32),
                rtol=0.0,
                atol=0.0,
            )
        assert out1.noise_stddev == 1.0
        assert out2.noise_stddev == 1.0

    def test_leaf_order_not_shape_determines_stream(self):
        """Each leaf's noise comes from its flatten-order fold, so a leaf's
        stream is stable under changes to *other* leaves' shapes."""
        noise_fn, state = gaussian_noise(noise_multiplier=1.0, key=key(7))
        out_small, _ = noise_fn(
            clipped({"a": torch.zeros(3), "b": torch.zeros(2)}, max_norm=1.0), state
        )
        noise_fn2, state2 = gaussian_noise(noise_multiplier=1.0, key=key(7))
        out_big, _ = noise_fn2(
            clipped({"a": torch.zeros(3), "b": torch.zeros(50)}, max_norm=1.0), state2
        )
        torch.testing.assert_close(
            out_small.pytree["a"], out_big.pytree["a"], rtol=0.0, atol=0.0
        )
