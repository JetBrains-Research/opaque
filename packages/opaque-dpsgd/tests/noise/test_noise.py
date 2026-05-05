"""Unit tests for Gaussian noise over bounded DP query pytrees."""

import pytest
import scipy.stats
import torch

from opaque.bounded import BoundedPytree, NoisyPytree, bounded, noisy
from opaque.clipping.per_group import PerGroup
from opaque.dpsgd.noise.gaussian import GaussianNoiseState, gaussian_noise
from opaque.random import key


class TestGaussian:
    """Tests for gaussian_noise() function."""

    def test_returns_tuple(self):
        noise_fn, state = gaussian_noise(noise_multiplier=1.0, key=key(0))
        assert callable(noise_fn)
        assert isinstance(state, GaussianNoiseState)

    def test_adds_noise_to_tensor(self):
        noise_fn, state = gaussian_noise(noise_multiplier=1.0, key=key(0))
        grad = torch.zeros(10, 5)
        output, state = noise_fn(bounded(grad, bound=1.0), state)

        assert isinstance(output, NoisyPytree)
        assert output.pytree.shape == grad.shape
        assert output.pytree.dtype == grad.dtype
        assert output.bound == pytest.approx(1.0)
        assert output.noise_stddev == pytest.approx(1.0)
        assert not torch.allclose(output.pytree, grad)

    def test_adds_noise_to_pytree(self):
        noise_fn, state = gaussian_noise(noise_multiplier=1.0, key=key(0))
        grads = {
            "weight": torch.zeros(10, 5),
            "bias": torch.zeros(10),
        }
        output, state = noise_fn(bounded(grads, bound=1.0), state)

        assert isinstance(output, NoisyPytree)
        assert set(output.pytree.keys()) == set(grads.keys())
        assert output.pytree["weight"].shape == grads["weight"].shape
        assert output.pytree["bias"].shape == grads["bias"].shape
        assert not torch.allclose(output.pytree["weight"], grads["weight"])

    def test_zero_noise_multiplier(self):
        noise_fn, state = gaussian_noise(noise_multiplier=0.0, key=key(0))
        grad = torch.randn(5, 3)
        output, state = noise_fn(bounded(grad, bound=1.0), state)
        assert torch.equal(output.pytree, grad)
        assert output.noise_stddev == pytest.approx(0.0)

    def test_negative_noise_multiplier_raises(self):
        with pytest.raises(ValueError, match="noise_multiplier must be non-negative"):
            gaussian_noise(noise_multiplier=-1.0, key=key(0))

    def test_negative_bound_raises_at_call(self):
        noise_fn, state = gaussian_noise(noise_multiplier=1.0, key=key(0))
        with pytest.raises(ValueError, match="non-negative"):
            noise_fn(bounded(torch.zeros(3), bound=-1.0), state)

    def test_dtype_preservation(self):
        noise_fn, state = gaussian_noise(noise_multiplier=1.0, key=key(0))

        grad_f32 = torch.randn(5, 3, dtype=torch.float32)
        out_f32, state = noise_fn(bounded(grad_f32, bound=1.0), state)
        assert out_f32.pytree.dtype == torch.float32

        grad_f64 = torch.randn(5, 3, dtype=torch.float64)
        out_f64, state = noise_fn(bounded(grad_f64, bound=1.0), state)
        assert out_f64.pytree.dtype == torch.float64

    def test_device_preservation(self):
        noise_fn, state = gaussian_noise(noise_multiplier=1.0, key=key(0))

        grad_cpu = torch.randn(5, 3)
        out_cpu, state = noise_fn(bounded(grad_cpu, bound=1.0), state)
        assert out_cpu.pytree.device == torch.device("cpu")

        if torch.backends.mps.is_available():
            grad_mps = torch.randn(5, 3, device="mps")
            out_mps, state = noise_fn(bounded(grad_mps, bound=1.0), state)
            assert out_mps.pytree.device.type == "mps"

    def test_noise_normality(self):
        noise_fn, state = gaussian_noise(noise_multiplier=1.0, key=key(42))
        zeros = torch.zeros(10000)
        output, state = noise_fn(bounded(zeros, bound=1.0), state)

        _, p_value = scipy.stats.kstest(output.pytree.numpy(), "norm", args=(0, 1))
        assert p_value > 0.01

    def test_noise_stddev(self):
        target_stddev = 2.5
        noise_fn, state = gaussian_noise(noise_multiplier=target_stddev, key=key(0))
        zeros = torch.zeros(10000)
        output, state = noise_fn(bounded(zeros, bound=1.0), state)

        measured_stddev = output.pytree.std().item()
        assert output.noise_stddev == pytest.approx(target_stddev)
        assert abs(measured_stddev - target_stddev) < 0.1

    def test_uniqueness(self):
        noise_fn, state = gaussian_noise(noise_multiplier=1.0, key=key(0))
        grad = bounded(torch.zeros(100), bound=1.0)

        noisy1, state = noise_fn(grad, state)
        noisy2, state = noise_fn(grad, state)

        assert not torch.allclose(noisy1.pytree, noisy2.pytree)

    def test_nested_pytree(self):
        noise_fn, state = gaussian_noise(noise_multiplier=1.0, key=key(0))
        grads = {
            "layer1": {"w": torch.zeros(10, 5), "b": torch.zeros(10)},
            "layer2": {"w": torch.zeros(5, 3), "b": torch.zeros(3)},
        }
        output, state = noise_fn(bounded(grads, bound=1.0), state)

        assert set(output.pytree.keys()) == {"layer1", "layer2"}
        assert not torch.allclose(output.pytree["layer1"]["w"], grads["layer1"]["w"])

    def test_tuple_pytree(self):
        noise_fn, state = gaussian_noise(noise_multiplier=1.0, key=key(0))
        grads = (torch.zeros(10, 5), torch.zeros(10))
        output, state = noise_fn(bounded(grads, bound=1.0), state)

        assert len(output.pytree) == 2
        assert not torch.allclose(output.pytree[0], grads[0])

    def test_requires_bounded_input(self):
        noise_fn, state = gaussian_noise(noise_multiplier=1.0, key=key(0))

        with pytest.raises(TypeError, match="expects BoundedPytree"):
            noise_fn(torch.zeros(8), state)

    def test_rejects_already_noisy_input(self):
        noise_fn, state = gaussian_noise(noise_multiplier=1.0, key=key(0))

        with pytest.raises(TypeError, match="not NoisyPytree"):
            noise_fn(noisy(torch.zeros(8), bound=2.0, noise_stddev=2.0), state)

    def test_bounded_input_returns_noisy_pytree(self):
        noise_fn, state = gaussian_noise(noise_multiplier=1.5, key=key(0))
        bounded_value = BoundedPytree(torch.zeros(10000), bound=2.0)

        output, state = noise_fn(bounded_value, state)

        assert isinstance(output, NoisyPytree)
        assert output.bound == pytest.approx(2.0)
        assert output.noise_stddev == pytest.approx(3.0)
        assert output.pytree.shape == bounded_value.pytree.shape
        assert abs(output.pytree.std().item() - 3.0) < 0.1

    def test_bounded_zero_multiplier_preserves_pytree(self):
        noise_fn, state = gaussian_noise(noise_multiplier=0.0, key=key(0))
        bounded_value = BoundedPytree(torch.ones(8), bound=2.0)

        output, state = noise_fn(bounded_value, state)

        assert isinstance(output, NoisyPytree)
        assert output.bound == pytest.approx(2.0)
        assert output.noise_stddev == pytest.approx(0.0)
        assert torch.equal(output.pytree, bounded_value.pytree)

    def test_bounded_per_group_noise_multiplier(self):
        noise_fn, state = gaussian_noise(noise_multiplier=2.0, key=key(0))
        bound = PerGroup(
            groups={"w": "small", "b": "large"},
            values={"small": 1.0, "large": 3.0},
        )
        bounded_value = BoundedPytree(
            {"w": torch.zeros(4), "b": torch.zeros(4)},
            bound=bound,
        )

        output, state = noise_fn(bounded_value, state)

        assert isinstance(output, NoisyPytree)
        assert isinstance(output.noise_stddev, PerGroup)
        assert output.noise_stddev.groups == bound.groups
        assert output.noise_stddev.values == {
            "small": pytest.approx(4.0),
            "large": pytest.approx(6.928203230275509),
        }
        assert output.pytree["w"].shape == bounded_value.pytree["w"].shape
        assert output.pytree["b"].shape == bounded_value.pytree["b"].shape


class TestGaussianKey:
    """Tests for required key parameter."""

    def test_noise_multiplier_required(self):
        with pytest.raises(TypeError, match="missing 1 required keyword-only argument"):
            gaussian_noise(key=key(0))

    def test_key_required(self):
        with pytest.raises(TypeError, match="missing 1 required keyword-only argument"):
            gaussian_noise(noise_multiplier=1.0)

    def test_generator_int_reproducible(self):
        noise_fn1, state1 = gaussian_noise(noise_multiplier=1.0, key=key(42))
        noise_fn2, state2 = gaussian_noise(noise_multiplier=1.0, key=key(42))

        grad = bounded(torch.zeros(10, 10), bound=1.0)
        noisy1, state1 = noise_fn1(grad, state1)
        noisy2, state2 = noise_fn2(grad, state2)

        assert torch.allclose(noisy1.pytree, noisy2.pytree)

    def test_generator_different_seeds(self):
        noise_fn1, state1 = gaussian_noise(noise_multiplier=1.0, key=key(42))
        noise_fn2, state2 = gaussian_noise(noise_multiplier=1.0, key=key(43))

        grad = bounded(torch.zeros(10, 10), bound=1.0)
        noisy1, _ = noise_fn1(grad, state1)
        noisy2, _ = noise_fn2(grad, state2)

        assert not torch.allclose(noisy1.pytree, noisy2.pytree)

    def test_seed_int_produces_reproducible_state(self):
        noise_fn1, state1 = gaussian_noise(noise_multiplier=1.0, key=key(42))
        noise_fn2, state2 = gaussian_noise(noise_multiplier=1.0, key=key(42))

        grad = bounded(torch.zeros(10), bound=1.0)
        noisy1, _ = noise_fn1(grad, state1)
        noisy2, _ = noise_fn2(grad, state2)
        assert torch.allclose(noisy1.pytree, noisy2.pytree)

    def test_state_evolution(self):
        noise_fn, state = gaussian_noise(noise_multiplier=1.0, key=key(42))

        grad = bounded(torch.zeros(10), bound=1.0)
        noisy1, state = noise_fn(grad, state)
        noisy2, state = noise_fn(grad, state)

        assert not torch.allclose(noisy1.pytree, noisy2.pytree)

    def test_saved_state_replay_is_deterministic(self):
        noise_fn, state = gaussian_noise(noise_multiplier=1.0, key=key(42))
        grad = bounded(torch.zeros(10), bound=1.0)

        noisy1, _ = noise_fn(grad, state)
        noisy2, _ = noise_fn(grad, state)

        assert torch.allclose(noisy1.pytree, noisy2.pytree)

    def test_zero_multiplier_with_seed(self):
        noise_fn, state = gaussian_noise(noise_multiplier=0.0, key=key(42))
        grad = torch.randn(5, 3)
        output, state = noise_fn(bounded(grad, bound=1.0), state)
        assert torch.equal(output.pytree, grad)

    def test_invalid_key_type_raises(self):
        with pytest.raises(TypeError, match="key must be"):
            gaussian_noise(noise_multiplier=1.0, key="bad")
