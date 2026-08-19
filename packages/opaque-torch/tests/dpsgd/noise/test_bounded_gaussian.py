"""Unit tests for bounded Gaussian noise (``gaussian_noise(bound=...)``)."""

import math

import pytest
import scipy.stats
import torch

from opaque.dpsgd.noise import gaussian_noise
from opaque.dpsgd.noise.types import GaussianNoiseState
from opaque.random import key
from opaque.types import NoisedPytree, PerGroup, clipped, noised


class TestBoundedGaussian:
    """Tests for ``gaussian_noise(bound=...)``."""

    def test_returns_tuple(self):
        noise_fn, state = gaussian_noise(noise_multiplier=1.0, bound=3.0, key=key(0))
        assert callable(noise_fn)
        assert isinstance(state, GaussianNoiseState)

    def test_adds_noise_to_tensor(self):
        noise_fn, state = gaussian_noise(noise_multiplier=1.0, bound=5.0, key=key(0))
        grad = torch.zeros(10, 5)
        output, state = noise_fn(clipped(grad, max_norm=1.0), state)

        assert isinstance(output, NoisedPytree)
        assert output.pytree.shape == grad.shape
        assert output.pytree.dtype == grad.dtype
        assert output.max_norm == pytest.approx(1.0)
        assert output.noise_stddev == pytest.approx(1.0)
        assert not torch.allclose(output.pytree, grad)

    def test_adds_noise_to_pytree(self):
        noise_fn, state = gaussian_noise(noise_multiplier=1.0, bound=5.0, key=key(0))
        grads = {
            "weight": torch.zeros(10, 5),
            "bias": torch.zeros(10),
        }
        output, state = noise_fn(clipped(grads, max_norm=1.0), state)

        assert set(output.pytree.keys()) == set(grads.keys())
        assert output.pytree["weight"].shape == grads["weight"].shape
        assert output.pytree["bias"].shape == grads["bias"].shape
        assert not torch.allclose(output.pytree["weight"], grads["weight"])

    def test_requires_clipped_input(self):
        noise_fn, state = gaussian_noise(noise_multiplier=1.0, bound=5.0, key=key(0))

        with pytest.raises(TypeError, match="expects ClippedPytree"):
            noise_fn(torch.zeros(8), state)

    def test_rejects_already_noisy_input(self):
        noise_fn, state = gaussian_noise(noise_multiplier=1.0, bound=5.0, key=key(0))

        with pytest.raises(TypeError, match="not NoisedPytree"):
            noise_fn(noised(torch.zeros(8), max_norm=1.0, noise_stddev=1.0), state)

    def test_output_within_symmetric_bounds(self):
        bound = 2.0
        noise_fn, state = gaussian_noise(noise_multiplier=1.0, bound=bound, key=key(0))
        grad = torch.zeros(10000)
        output, state = noise_fn(clipped(grad, max_norm=2.0), state)

        assert output.pytree.min().item() >= -bound
        assert output.pytree.max().item() <= bound

    def test_output_within_asymmetric_bounds(self):
        low, high = -1.0, 4.0
        noise_fn, state = gaussian_noise(
            noise_multiplier=1.0, bound=(low, high), key=key(0)
        )
        grad = torch.zeros(10000)
        output, state = noise_fn(clipped(grad, max_norm=1.0), state)

        assert output.pytree.min().item() >= low
        assert output.pytree.max().item() <= high

    def test_bound_accepts_list(self):
        noise_fn, state = gaussian_noise(
            noise_multiplier=1.0, bound=[-0.5, 2.5], key=key(0)
        )
        grad = torch.zeros(1000)
        output, _ = noise_fn(clipped(grad, max_norm=1.0), state)
        assert output.pytree.min().item() >= -0.5
        assert output.pytree.max().item() <= 2.5

    def test_output_within_bounds_nonzero_center(self):
        noise_fn, state = gaussian_noise(noise_multiplier=1.0, bound=3.0, key=key(0))
        grad = torch.tensor([2.5, -2.5, 0.0, 1.0, -1.0]).repeat(2000)
        output, state = noise_fn(clipped(grad, max_norm=1.0), state)

        assert output.pytree.min().item() >= -3.0
        assert output.pytree.max().item() <= 3.0

    def test_zero_noise_multiplier_clamps_to_bound(self):
        # σ=0 + bound=(low, high) → clamp(input, low, high).
        noise_fn, state = gaussian_noise(noise_multiplier=0.0, bound=1.5, key=key(0))
        grad = torch.tensor([2.0, -2.0, 0.5, -0.5])
        output, _ = noise_fn(clipped(grad, max_norm=1.0), state)
        expected = torch.tensor([1.5, -1.5, 0.5, -0.5])
        torch.testing.assert_close(output.pytree, expected)
        assert output.noise_stddev == pytest.approx(0.0)

    def test_zero_noise_multiplier_unbounded_is_identity(self):
        # σ=0 + bound=None → input unchanged.
        noise_fn, state = gaussian_noise(noise_multiplier=0.0, key=key(0))
        grad = torch.tensor([2.0, -2.0, 0.5, -0.5])
        output, _ = noise_fn(clipped(grad, max_norm=1.0), state)
        torch.testing.assert_close(output.pytree, grad)
        assert output.noise_stddev == pytest.approx(0.0)

    def test_negative_noise_multiplier_raises(self):
        with pytest.raises(ValueError, match="noise_multiplier must be non-negative"):
            gaussian_noise(noise_multiplier=-1.0, bound=3.0, key=key(0))

    def test_negative_bound_raises_at_call(self):
        noise_fn, state = gaussian_noise(noise_multiplier=1.0, bound=3.0, key=key(0))
        with pytest.raises(ValueError, match="non-negative"):
            noise_fn(clipped(torch.zeros(3), max_norm=-1.0), state)

    def test_nonpositive_scalar_bound_raises(self):
        with pytest.raises(ValueError, match="scalar bound must be positive"):
            gaussian_noise(noise_multiplier=1.0, bound=0.0, key=key(0))
        with pytest.raises(ValueError, match="scalar bound must be positive"):
            gaussian_noise(noise_multiplier=1.0, bound=-1.0, key=key(0))

    def test_inverted_tuple_bound_raises(self):
        with pytest.raises(ValueError, match="low < high"):
            gaussian_noise(noise_multiplier=1.0, bound=(2.0, 1.0), key=key(0))

    def test_bound_not_straddling_zero_raises(self):
        with pytest.raises(ValueError, match="straddle zero"):
            gaussian_noise(noise_multiplier=1.0, bound=(1.0, 2.0), key=key(0))
        with pytest.raises(ValueError, match="straddle zero"):
            gaussian_noise(noise_multiplier=1.0, bound=(-2.0, -1.0), key=key(0))

    def test_wrong_length_tuple_raises(self):
        with pytest.raises(ValueError, match="2-tuple"):
            gaussian_noise(noise_multiplier=1.0, bound=(1.0, 2.0, 3.0), key=key(0))

    def test_dtype_preservation(self):
        noise_fn, state = gaussian_noise(noise_multiplier=1.0, bound=5.0, key=key(0))

        grad_f32 = torch.randn(5, 3, dtype=torch.float32)
        out_f32, state = noise_fn(clipped(grad_f32, max_norm=1.0), state)
        assert out_f32.pytree.dtype == torch.float32

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

    def test_noise_distribution_truncnorm(self):
        # gaussian_noise(bound=B) at max_norm=1 reduces to a univariate
        # truncated normal of stddev nm·1 on [-B, B] centred at zero.
        bound = 2.0
        sigma = 1.0
        noise_fn, state = gaussian_noise(
            noise_multiplier=sigma, bound=bound, key=key(42)
        )
        zeros = torch.zeros(50000)
        output, state = noise_fn(clipped(zeros, max_norm=1.0), state)

        _, p_value = scipy.stats.kstest(
            output.pytree.numpy(),
            scipy.stats.truncnorm.cdf,
            args=(-bound / sigma, bound / sigma, 0.0, sigma),
        )
        assert p_value > 0.01, f"KS test failed with p={p_value}"

    def test_variance_less_than_unclipped(self):
        noise_fn, state = gaussian_noise(noise_multiplier=1.0, bound=2.0, key=key(0))
        zeros = torch.zeros(50000)
        output, state = noise_fn(clipped(zeros, max_norm=1.0), state)

        assert output.pytree.var().item() < 1.0

    def test_uniqueness(self):
        noise_fn, state = gaussian_noise(noise_multiplier=1.0, bound=5.0, key=key(0))
        grad = clipped(torch.zeros(100), max_norm=1.0)

        output1, state = noise_fn(grad, state)
        output2, state = noise_fn(grad, state)

        assert not torch.allclose(output1.pytree, output2.pytree)

    def test_nested_pytree(self):
        noise_fn, state = gaussian_noise(noise_multiplier=1.0, bound=5.0, key=key(0))
        grads = {
            "layer1": {"w": torch.zeros(10, 5), "b": torch.zeros(10)},
            "layer2": {"w": torch.zeros(5, 3), "b": torch.zeros(3)},
        }
        output, state = noise_fn(clipped(grads, max_norm=1.0), state)

        assert set(output.pytree.keys()) == {"layer1", "layer2"}
        assert not torch.allclose(output.pytree["layer1"]["w"], grads["layer1"]["w"])

    def test_tuple_pytree(self):
        noise_fn, state = gaussian_noise(noise_multiplier=1.0, bound=5.0, key=key(0))
        grads = (torch.zeros(10, 5), torch.zeros(10))
        output, state = noise_fn(clipped(grads, max_norm=1.0), state)

        assert len(output.pytree) == 2
        assert not torch.allclose(output.pytree[0], grads[0])


class TestBoundedGaussianKey:
    """Tests for key parameter."""

    def test_reproducibility(self):
        noise_fn1, state1 = gaussian_noise(noise_multiplier=1.0, bound=3.0, key=key(42))
        noise_fn2, state2 = gaussian_noise(noise_multiplier=1.0, bound=3.0, key=key(42))

        grad = clipped(torch.zeros(10, 10), max_norm=1.0)
        output1, state1 = noise_fn1(grad, state1)
        output2, state2 = noise_fn2(grad, state2)

        assert torch.allclose(output1.pytree, output2.pytree)

    def test_different_seeds(self):
        noise_fn1, state1 = gaussian_noise(noise_multiplier=1.0, bound=3.0, key=key(42))
        noise_fn2, state2 = gaussian_noise(noise_multiplier=1.0, bound=3.0, key=key(43))

        grad = clipped(torch.zeros(10, 10), max_norm=1.0)
        output1, _ = noise_fn1(grad, state1)
        output2, _ = noise_fn2(grad, state2)

        assert not torch.allclose(output1.pytree, output2.pytree)

    def test_invalid_key_type_raises(self):
        with pytest.raises(TypeError, match="key must be"):
            gaussian_noise(noise_multiplier=1.0, bound=3.0, key="bad")


class TestBoundedGaussianPerGroup:
    """Tests for ``gaussian_noise(bound=...)`` with PerGroup bounds."""

    def test_adds_per_group_noise(self):
        max_norm = PerGroup(
            groups={"weight": "attn", "bias": "mlp"},
            values={"attn": 1.0, "mlp": 5.0},
        )
        noise_fn, state = gaussian_noise(noise_multiplier=1.0, bound=20.0, key=key(42))
        grads = {
            "weight": torch.zeros(1000),
            "bias": torch.zeros(1000),
        }
        output, state = noise_fn(clipped(grads, max_norm=max_norm), state)

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

    def test_nested_per_group_noise(self):
        """PerGroup σ follows nested ParamPaths, not only flat named_parameters."""
        from opaque.api.engine.clipping._per_group import per_group

        nested = {
            "layer1": {
                "attn": torch.zeros(2000),
                "mlp": torch.zeros(2000),
            },
            "layer2": {
                "attn": torch.zeros(2000),
                "mlp": torch.zeros(2000),
            },
        }
        pg = per_group(nested, attn=1.0, mlp=5.0)
        noise_fn, state = gaussian_noise(noise_multiplier=1.0, bound=50.0, key=key(7))
        output, _ = noise_fn(clipped(nested, max_norm=pg), state)

        assert isinstance(output.noise_stddev, PerGroup)
        assert ("layer1", "attn") in output.noise_stddev.groups
        assert ("layer2", "mlp") in output.noise_stddev.groups
        attn_var = output.pytree["layer1"]["attn"].var().item()
        mlp_var = output.pytree["layer1"]["mlp"].var().item()
        assert mlp_var > attn_var * 2.5
        torch.testing.assert_close(
            output.pytree["layer1"]["attn"].var(),
            output.pytree["layer2"]["attn"].var(),
            rtol=0.3,
            atol=0.05,
        )

    def test_per_group_bounds_respected(self):
        # Bound is absolute and shared across groups.
        max_norm = PerGroup(
            groups={"small": "lo", "large": "hi"},
            values={"lo": 0.5, "hi": 2.0},
        )
        bound = 3.0
        noise_fn, state = gaussian_noise(noise_multiplier=1.0, bound=bound, key=key(0))
        grads = {
            "small": torch.zeros(10000),
            "large": torch.zeros(10000),
        }
        output, state = noise_fn(clipped(grads, max_norm=max_norm), state)

        for tensor in output.pytree.values():
            assert tensor.min().item() >= -bound
            assert tensor.max().item() <= bound

    def test_per_group_stddev_path_mismatch_raises(self):
        max_norm = PerGroup(groups={"w": "g"}, values={"g": 1.0})
        noise_fn, state = gaussian_noise(noise_multiplier=1.0, bound=3.0, key=key(0))
        # List grads use path (0,), not ("w",).
        with pytest.raises(KeyError):
            noise_fn(clipped([torch.zeros(3)], max_norm=max_norm), state)

    def test_all_zero_bound_clamps_to_bound(self):
        # σ_g = 0 + absolute bound → clamp(input, low, high) per group.
        max_norm = PerGroup(
            groups={"a": "g1", "b": "g2"},
            values={"g1": 0.0, "g2": 0.0},
        )
        bound = 0.5
        noise_fn, state = gaussian_noise(noise_multiplier=1.0, bound=bound, key=key(0))
        grads = {
            "a": torch.tensor([2.0, -2.0, 0.1]),
            "b": torch.tensor([-1.0, 1.0, 0.3]),
        }
        output, state = noise_fn(clipped(grads, max_norm=max_norm), state)
        torch.testing.assert_close(
            output.pytree["a"], torch.tensor([bound, -bound, 0.1])
        )
        torch.testing.assert_close(
            output.pytree["b"], torch.tensor([-bound, bound, 0.3])
        )

    def test_negative_group_bound_raises(self):
        max_norm = PerGroup(groups={"w": "g"}, values={"g": -1.0})
        noise_fn, state = gaussian_noise(noise_multiplier=1.0, bound=3.0, key=key(0))
        with pytest.raises(ValueError, match="non-negative"):
            noise_fn(clipped({"w": torch.zeros(3)}, max_norm=max_norm), state)

    def test_deterministic_noise(self):
        max_norm = PerGroup(groups={"w": "g"}, values={"g": 1.0})
        grads = clipped({"w": torch.zeros(10)}, max_norm=max_norm)

        noise_fn1, state1 = gaussian_noise(noise_multiplier=1.0, bound=3.0, key=key(42))
        output1, _ = noise_fn1(grads, state1)

        noise_fn2, state2 = gaussian_noise(noise_multiplier=1.0, bound=3.0, key=key(42))
        output2, _ = noise_fn2(grads, state2)

        torch.testing.assert_close(output1.pytree["w"], output2.pytree["w"])

    def test_step_counter_advances(self):
        max_norm = PerGroup(groups={"w": "g"}, values={"g": 1.0})
        noise_fn, state = gaussian_noise(noise_multiplier=1.0, bound=3.0, key=key(0))
        assert state._step_counter == 0

        grads = clipped({"w": torch.zeros(3)}, max_norm=max_norm)
        _, state = noise_fn(grads, state)
        assert state._step_counter == 1
        _, state = noise_fn(grads, state)
        assert state._step_counter == 2
