"""Unit tests for bounded truncated Gaussian noise."""

import math

import pytest
import scipy.stats
import torch

from opaque.bounded import NoisyPytree, bounded, noisy
from opaque.clipping.per_group import PerGroup
from opaque.dpsgd.noise.gaussian import GaussianNoiseState
from opaque.dpsgd.noise.truncated_gaussian import truncated_gaussian_noise
from opaque.random import key


class TestBoundedGaussian:
    """Tests for truncated_gaussian_noise()."""

    def test_returns_tuple(self):
        noise_fn, state = truncated_gaussian_noise(
            noise_multiplier=1.0, radius=3.0, key=key(0)
        )
        assert callable(noise_fn)
        assert isinstance(state, GaussianNoiseState)

    def test_adds_noise_to_tensor(self):
        noise_fn, state = truncated_gaussian_noise(
            noise_multiplier=1.0, radius=5.0, key=key(0)
        )
        grad = torch.zeros(10, 5)
        output, state = noise_fn(bounded(grad, bound=1.0), state)

        assert isinstance(output, NoisyPytree)
        assert output.pytree.shape == grad.shape
        assert output.pytree.dtype == grad.dtype
        assert output.bound == pytest.approx(1.0)
        assert output.noise_stddev == pytest.approx(1.0)
        assert not torch.allclose(output.pytree, grad)

    def test_adds_noise_to_pytree(self):
        noise_fn, state = truncated_gaussian_noise(
            noise_multiplier=1.0, radius=5.0, key=key(0)
        )
        grads = {
            "weight": torch.zeros(10, 5),
            "bias": torch.zeros(10),
        }
        output, state = noise_fn(bounded(grads, bound=1.0), state)

        assert set(output.pytree.keys()) == set(grads.keys())
        assert output.pytree["weight"].shape == grads["weight"].shape
        assert output.pytree["bias"].shape == grads["bias"].shape
        assert not torch.allclose(output.pytree["weight"], grads["weight"])

    def test_requires_bounded_input(self):
        noise_fn, state = truncated_gaussian_noise(
            noise_multiplier=1.0, radius=5.0, key=key(0)
        )

        with pytest.raises(TypeError, match="expects BoundedPytree"):
            noise_fn(torch.zeros(8), state)

    def test_rejects_already_noisy_input(self):
        noise_fn, state = truncated_gaussian_noise(
            noise_multiplier=1.0, radius=5.0, key=key(0)
        )

        with pytest.raises(TypeError, match="not NoisyPytree"):
            noise_fn(noisy(torch.zeros(8), bound=1.0, noise_stddev=1.0), state)

    def test_output_within_bounds(self):
        noise_multiplier, contribution_bound, radius = 1.0, 2.0, 2.0
        stddev = noise_multiplier * contribution_bound
        output_bound = stddev * radius
        noise_fn, state = truncated_gaussian_noise(
            noise_multiplier=noise_multiplier, radius=radius, key=key(0)
        )
        grad = torch.zeros(10000)
        output, state = noise_fn(bounded(grad, bound=contribution_bound), state)

        assert output.pytree.min().item() >= -output_bound
        assert output.pytree.max().item() <= output_bound
        assert output.noise_stddev == pytest.approx(stddev)

    def test_output_within_bounds_nonzero_center(self):
        noise_fn, state = truncated_gaussian_noise(
            noise_multiplier=1.0, radius=3.0, key=key(0)
        )
        grad = torch.tensor([2.5, -2.5, 0.0, 1.0, -1.0]).repeat(2000)
        output, state = noise_fn(bounded(grad, bound=1.0), state)

        assert output.pytree.min().item() >= -3.0
        assert output.pytree.max().item() <= 3.0

    def test_zero_noise_multiplier_clamps_to_zero(self):
        noise_fn, state = truncated_gaussian_noise(
            noise_multiplier=0.0, radius=5.0, key=key(0)
        )
        grad = torch.tensor([2.0, -2.0, 0.0])
        output, state = noise_fn(bounded(grad, bound=1.0), state)
        assert torch.equal(output.pytree, torch.zeros_like(grad))
        assert output.noise_stddev == pytest.approx(0.0)

    def test_negative_noise_multiplier_raises(self):
        with pytest.raises(ValueError, match="noise_multiplier must be non-negative"):
            truncated_gaussian_noise(noise_multiplier=-1.0, radius=3.0, key=key(0))

    def test_negative_bound_raises_at_call(self):
        noise_fn, state = truncated_gaussian_noise(
            noise_multiplier=1.0, radius=3.0, key=key(0)
        )
        with pytest.raises(ValueError, match="non-negative"):
            noise_fn(bounded(torch.zeros(3), bound=-1.0), state)

    def test_nonpositive_radius_raises(self):
        with pytest.raises(ValueError, match="radius must be positive"):
            truncated_gaussian_noise(noise_multiplier=1.0, radius=0.0, key=key(0))
        with pytest.raises(ValueError, match="radius must be positive"):
            truncated_gaussian_noise(noise_multiplier=1.0, radius=-1.0, key=key(0))

    def test_dtype_preservation(self):
        noise_fn, state = truncated_gaussian_noise(
            noise_multiplier=1.0, radius=5.0, key=key(0)
        )

        grad_f32 = torch.randn(5, 3, dtype=torch.float32)
        out_f32, state = noise_fn(bounded(grad_f32, bound=1.0), state)
        assert out_f32.pytree.dtype == torch.float32

        grad_f64 = torch.randn(5, 3, dtype=torch.float64)
        out_f64, state = noise_fn(bounded(grad_f64, bound=1.0), state)
        assert out_f64.pytree.dtype == torch.float64

    def test_device_preservation(self):
        noise_fn, state = truncated_gaussian_noise(
            noise_multiplier=1.0, radius=5.0, key=key(0)
        )

        grad_cpu = torch.randn(5, 3)
        out_cpu, state = noise_fn(bounded(grad_cpu, bound=1.0), state)
        assert out_cpu.pytree.device == torch.device("cpu")

        if torch.backends.mps.is_available():
            grad_mps = torch.randn(5, 3, device="mps")
            out_mps, state = noise_fn(bounded(grad_mps, bound=1.0), state)
            assert out_mps.pytree.device.type == "mps"

    def test_noise_distribution_truncated_normal(self):
        stddev = 1.0
        radius = 2.0
        noise_fn, state = truncated_gaussian_noise(
            noise_multiplier=1.0, radius=radius, key=key(42)
        )
        zeros = torch.zeros(50000)
        output, state = noise_fn(bounded(zeros, bound=1.0), state)

        _, p_value = scipy.stats.kstest(
            output.pytree.numpy(),
            "truncnorm",
            args=(-radius, radius, 0.0, stddev),
        )
        assert p_value > 0.01, f"KS test failed with p={p_value}"

    def test_variance_less_than_unbounded(self):
        noise_fn, state = truncated_gaussian_noise(
            noise_multiplier=1.0, radius=2.0, key=key(0)
        )
        zeros = torch.zeros(50000)
        output, state = noise_fn(bounded(zeros, bound=1.0), state)

        assert output.pytree.var().item() < 1.0

    def test_uniqueness(self):
        noise_fn, state = truncated_gaussian_noise(
            noise_multiplier=1.0, radius=5.0, key=key(0)
        )
        grad = bounded(torch.zeros(100), bound=1.0)

        output1, state = noise_fn(grad, state)
        output2, state = noise_fn(grad, state)

        assert not torch.allclose(output1.pytree, output2.pytree)

    def test_nested_pytree(self):
        noise_fn, state = truncated_gaussian_noise(
            noise_multiplier=1.0, radius=5.0, key=key(0)
        )
        grads = {
            "layer1": {"w": torch.zeros(10, 5), "b": torch.zeros(10)},
            "layer2": {"w": torch.zeros(5, 3), "b": torch.zeros(3)},
        }
        output, state = noise_fn(bounded(grads, bound=1.0), state)

        assert set(output.pytree.keys()) == {"layer1", "layer2"}
        assert not torch.allclose(output.pytree["layer1"]["w"], grads["layer1"]["w"])

    def test_tuple_pytree(self):
        noise_fn, state = truncated_gaussian_noise(
            noise_multiplier=1.0, radius=5.0, key=key(0)
        )
        grads = (torch.zeros(10, 5), torch.zeros(10))
        output, state = noise_fn(bounded(grads, bound=1.0), state)

        assert len(output.pytree) == 2
        assert not torch.allclose(output.pytree[0], grads[0])


class TestBoundedGaussianKey:
    """Tests for key parameter."""

    def test_reproducibility(self):
        noise_fn1, state1 = truncated_gaussian_noise(
            noise_multiplier=1.0, radius=3.0, key=key(42)
        )
        noise_fn2, state2 = truncated_gaussian_noise(
            noise_multiplier=1.0, radius=3.0, key=key(42)
        )

        grad = bounded(torch.zeros(10, 10), bound=1.0)
        output1, state1 = noise_fn1(grad, state1)
        output2, state2 = noise_fn2(grad, state2)

        assert torch.allclose(output1.pytree, output2.pytree)

    def test_different_seeds(self):
        noise_fn1, state1 = truncated_gaussian_noise(
            noise_multiplier=1.0, radius=3.0, key=key(42)
        )
        noise_fn2, state2 = truncated_gaussian_noise(
            noise_multiplier=1.0, radius=3.0, key=key(43)
        )

        grad = bounded(torch.zeros(10, 10), bound=1.0)
        output1, _ = noise_fn1(grad, state1)
        output2, _ = noise_fn2(grad, state2)

        assert not torch.allclose(output1.pytree, output2.pytree)

    def test_invalid_key_type_raises(self):
        with pytest.raises(TypeError, match="key must be"):
            truncated_gaussian_noise(noise_multiplier=1.0, radius=3.0, key="bad")


class TestTruncatedGaussianPerGroup:
    """Tests for truncated_gaussian_noise() with PerGroup bounds."""

    def test_adds_per_group_noise(self):
        bound = PerGroup(
            groups={"weight": "attn", "bias": "mlp"},
            values={"attn": 1.0, "mlp": 5.0},
        )
        noise_fn, state = truncated_gaussian_noise(
            noise_multiplier=1.0, radius=5.0, key=key(42)
        )
        grads = {
            "weight": torch.zeros(1000),
            "bias": torch.zeros(1000),
        }
        output, state = noise_fn(bounded(grads, bound=bound), state)

        assert isinstance(output.noise_stddev, PerGroup)
        assert output.noise_stddev.values == {
            "attn": pytest.approx(math.sqrt(6.0)),
            "mlp": pytest.approx(math.sqrt(30.0)),
        }
        assert not torch.allclose(output.pytree["weight"], grads["weight"])
        assert not torch.allclose(output.pytree["bias"], grads["bias"])

        attn_var = output.pytree["weight"].var().item()
        mlp_var = output.pytree["bias"].var().item()
        assert mlp_var > attn_var * 2.5

    def test_per_group_bounds_respected(self):
        bound = PerGroup(
            groups={"small": "lo", "large": "hi"},
            values={"lo": 0.5, "hi": 2.0},
        )
        radius = 3.0
        noise_fn, state = truncated_gaussian_noise(
            noise_multiplier=1.0, radius=radius, key=key(0)
        )
        grads = {
            "small": torch.zeros(10000),
            "large": torch.zeros(10000),
        }
        output, state = noise_fn(bounded(grads, bound=bound), state)

        small_std = output.noise_stddev.values["lo"]
        large_std = output.noise_stddev.values["hi"]
        assert output.pytree["small"].min().item() >= -small_std * radius
        assert output.pytree["small"].max().item() <= small_std * radius
        assert output.pytree["large"].min().item() >= -large_std * radius
        assert output.pytree["large"].max().item() <= large_std * radius

    def test_all_zero_bound_returns_original(self):
        bound = PerGroup(
            groups={"a": "g1", "b": "g2"},
            values={"g1": 0.0, "g2": 0.0},
        )
        noise_fn, state = truncated_gaussian_noise(
            noise_multiplier=1.0, radius=3.0, key=key(0)
        )
        grads = {"a": torch.randn(3), "b": torch.randn(3)}
        output, state = noise_fn(bounded(grads, bound=bound), state)
        torch.testing.assert_close(output.pytree["a"], grads["a"])
        torch.testing.assert_close(output.pytree["b"], grads["b"])

    def test_negative_group_bound_raises(self):
        bound = PerGroup(groups={"w": "g"}, values={"g": -1.0})
        noise_fn, state = truncated_gaussian_noise(
            noise_multiplier=1.0, radius=3.0, key=key(0)
        )
        with pytest.raises(ValueError, match="non-negative"):
            noise_fn(bounded({"w": torch.zeros(3)}, bound=bound), state)

    def test_deterministic_noise(self):
        bound = PerGroup(groups={"w": "g"}, values={"g": 1.0})
        grads = bounded({"w": torch.zeros(10)}, bound=bound)

        noise_fn1, state1 = truncated_gaussian_noise(
            noise_multiplier=1.0, radius=3.0, key=key(42)
        )
        output1, _ = noise_fn1(grads, state1)

        noise_fn2, state2 = truncated_gaussian_noise(
            noise_multiplier=1.0, radius=3.0, key=key(42)
        )
        output2, _ = noise_fn2(grads, state2)

        torch.testing.assert_close(output1.pytree["w"], output2.pytree["w"])

    def test_step_counter_advances(self):
        bound = PerGroup(groups={"w": "g"}, values={"g": 1.0})
        noise_fn, state = truncated_gaussian_noise(
            noise_multiplier=1.0, radius=3.0, key=key(0)
        )
        assert state._step_counter == 0

        grads = bounded({"w": torch.zeros(3)}, bound=bound)
        _, state = noise_fn(grads, state)
        assert state._step_counter == 1
        _, state = noise_fn(grads, state)
        assert state._step_counter == 2
