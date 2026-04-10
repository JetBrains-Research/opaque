"""Tests for DP-AdamW optimizer (Algorithms 1 and 2)."""

import pytest
import torch

torchopt = pytest.importorskip(
    "torchopt", reason="torchopt required for optimizer tests"
)

from opaque.optimizers import DPAdamWState, dp_adamw  # noqa: E402


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def params():
    """Simple flat parameter dict."""
    return {"weight": torch.randn(4, 3), "bias": torch.randn(3)}


@pytest.fixture
def grads(params):
    """Random gradients matching params."""
    return {k: torch.randn_like(v) for k, v in params.items()}


@pytest.fixture
def nested_params():
    """Nested parameter pytree."""
    return {
        "layer1": {"weight": torch.randn(4, 3), "bias": torch.randn(3)},
        "layer2": {"weight": torch.randn(3, 2)},
    }


# ---------------------------------------------------------------------------
# Algorithm 1: standard mode (noise_variance=0) — delegates to torchopt
# ---------------------------------------------------------------------------


class TestStandardMode:
    """When noise_variance=0, dp_adamw delegates to torchopt.adamw."""

    def test_returns_gradient_transformation(self):
        opt = dp_adamw(lr=1e-3)
        assert hasattr(opt, "init")
        assert hasattr(opt, "update")

    def test_init_and_update(self, params, grads):
        opt = dp_adamw(lr=1e-3, weight_decay=0.01)
        state = opt.init(params)

        updates, new_state = opt.update(grads, state, params=params)

        # Updates have same structure as params.
        assert set(updates.keys()) == set(params.keys())
        for k in params:
            assert updates[k].shape == params[k].shape

    def test_params_change_after_apply(self, params, grads):
        opt = dp_adamw(lr=1e-2)
        state = opt.init(params)
        # Save original values — apply_updates is in-place.
        orig = {k: v.clone() for k, v in params.items()}

        updates, _ = opt.update(grads, state, params=params)
        torchopt.apply_updates(params, updates)

        changed = any(not torch.equal(params[k], orig[k]) for k in params)
        assert changed

    def test_matches_torchopt_adamw(self, params, grads):
        """dp_adamw(noise_variance=0) should produce identical results to
        torchopt.adamw with the same hyperparameters."""
        kwargs = dict(lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01)

        opt_dp = dp_adamw(**kwargs)
        opt_ref = torchopt.adamw(**kwargs)

        state_dp = opt_dp.init(params)
        state_ref = opt_ref.init(params)

        updates_dp, _ = opt_dp.update(grads, state_dp, params=params)
        updates_ref, _ = opt_ref.update(grads, state_ref, params=params)

        for k in params:
            torch.testing.assert_close(updates_dp[k], updates_ref[k])


# ---------------------------------------------------------------------------
# Algorithm 2: BC mode (noise_variance > 0)
# ---------------------------------------------------------------------------


def _bc_state(chain_state):
    """Extract DPAdamWState from the chain state tuple."""
    return chain_state[0]


class TestBCMode:
    """DP-AdamW-BC: bias-corrected second moment."""

    def test_returns_gradient_transformation(self):
        opt = dp_adamw(lr=1e-3, noise_variance=0.5)
        assert hasattr(opt, "init")
        assert hasattr(opt, "update")

    def test_init_returns_dp_state(self, params):
        opt = dp_adamw(lr=1e-3, noise_variance=0.5)
        state = opt.init(params)
        bc = _bc_state(state)

        assert isinstance(bc, DPAdamWState)
        assert bc.step == 0
        # Moments initialised to zeros.
        for k in params:
            assert torch.equal(bc.mu[k], torch.zeros_like(params[k]))
            assert torch.equal(bc.nu[k], torch.zeros_like(params[k]))

    def test_update_advances_step(self, params, grads):
        opt = dp_adamw(lr=1e-3, noise_variance=0.5)
        state = opt.init(params)
        _, state2 = opt.update(grads, state, params=params)
        assert _bc_state(state2).step == 1
        _, state3 = opt.update(grads, state2, params=params)
        assert _bc_state(state3).step == 2

    def test_moments_updated(self, params, grads):
        opt = dp_adamw(lr=1e-3, noise_variance=0.5)
        state = opt.init(params)
        _, state2 = opt.update(grads, state, params=params)

        bc = _bc_state(state2)
        b1, b2 = 0.9, 0.999
        for k in grads:
            expected_mu = (1 - b1) * grads[k]
            expected_nu = (1 - b2) * grads[k] * grads[k]
            torch.testing.assert_close(bc.mu[k], expected_mu)
            torch.testing.assert_close(bc.nu[k], expected_nu)

    def test_bc_differs_from_standard(self, params, grads):
        """BC mode must produce different updates than standard mode."""
        opt_std = dp_adamw(lr=1e-3)
        opt_bc = dp_adamw(lr=1e-3, noise_variance=0.5)

        state_std = opt_std.init(params)
        state_bc = opt_bc.init(params)

        # Run a few steps so moments build up (BC effect is negligible at t=1
        # when v_hat is tiny compared to Phi).
        for _ in range(5):
            upd_std, state_std = opt_std.update(grads, state_std, params=params)
            upd_bc, state_bc = opt_bc.update(grads, state_bc, params=params)

        differs = any(not torch.equal(upd_std[k], upd_bc[k]) for k in params)
        assert differs

    def test_bc_increases_effective_lr(self, params, grads):
        """Subtracting Phi from v_hat shrinks the denominator, so BC updates
        should have larger magnitude than standard ones (same hyperparams)."""
        # Use large grads so v_hat >> Phi and the effect is measurable.
        big_grads = {k: v * 10 for k, v in grads.items()}

        opt_std = dp_adamw(lr=1e-3, noise_variance=0.0)
        opt_bc = dp_adamw(lr=1e-3, noise_variance=0.01)

        s_std = opt_std.init(params)
        s_bc = opt_bc.init(params)

        for _ in range(10):
            u_std, s_std = opt_std.update(big_grads, s_std, params=params)
            u_bc, s_bc = opt_bc.update(big_grads, s_bc, params=params)

        # BC updates should be at least as large (componentwise, on average).
        norm_std = sum(u.norm() for u in u_std.values())
        norm_bc = sum(u.norm() for u in u_bc.values())
        assert norm_bc >= norm_std

    def test_bc_floor_prevents_zero_denominator(self):
        """With huge noise_variance, v_hat - Phi goes negative;
        bc_floor must keep the denominator positive."""
        params = {"w": torch.ones(3)}
        grads = {"w": torch.ones(3) * 0.01}

        opt = dp_adamw(lr=1e-3, noise_variance=1e6, bc_floor=1e-8)
        state = opt.init(params)
        updates, _ = opt.update(grads, state, params=params)

        assert torch.isfinite(updates["w"]).all()

    def test_nested_pytree(self, nested_params):
        grads = {
            k: (
                {kk: torch.randn_like(vv) for kk, vv in v.items()}
                if isinstance(v, dict)
                else torch.randn_like(v)
            )
            for k, v in nested_params.items()
        }

        opt = dp_adamw(lr=1e-3, noise_variance=0.1)
        state = opt.init(nested_params)
        updates, state2 = opt.update(grads, state, params=nested_params)

        assert _bc_state(state2).step == 1
        assert "layer1" in updates and "layer2" in updates
        assert (
            updates["layer1"]["weight"].shape == nested_params["layer1"]["weight"].shape
        )


# ---------------------------------------------------------------------------
# Weight decay (BC mode — standard mode reuses torchopt.adamw which is
# already tested upstream)
# ---------------------------------------------------------------------------


class TestWeightDecay:
    def test_decoupled_weight_decay(self):
        """With zero gradients, updates should only reflect weight decay."""
        params = {"w": torch.ones(4) * 2.0}
        grads = {"w": torch.zeros(4)}

        opt = dp_adamw(lr=0.1, weight_decay=0.1, noise_variance=0.01, bc_floor=1e-30)
        state = opt.init(params)
        updates, _ = opt.update(grads, state, params=params)

        # With zero grads, m_hat=0 so the adam-scaled part is zero.
        # Only weight decay contributes:  update = -lr * wd * params.
        expected = -0.1 * 0.1 * params["w"]
        torch.testing.assert_close(updates["w"], expected)

    def test_no_weight_decay_no_params_needed(self):
        params = {"w": torch.randn(3)}
        grads = {"w": torch.randn(3)}

        opt = dp_adamw(lr=1e-3, weight_decay=0.0, noise_variance=0.1)
        state = opt.init(params)
        # Should work without passing params.
        updates, _ = opt.update(grads, state)
        assert updates["w"].shape == params["w"].shape


# ---------------------------------------------------------------------------
# Convergence
# ---------------------------------------------------------------------------


class TestConvergence:
    def test_converges_on_quadratic(self):
        """Minimise f(x) = ||x - target||^2 and verify convergence."""
        target = torch.tensor([1.0, 2.0, 3.0])
        params = {"x": torch.zeros(3)}

        opt = dp_adamw(lr=0.05, weight_decay=0.0, noise_variance=0.0)
        state = opt.init(params)

        for _ in range(200):
            grads = {"x": 2.0 * (params["x"] - target)}
            updates, state = opt.update(grads, state, params=params)
            params = torchopt.apply_updates(params, updates)

        torch.testing.assert_close(params["x"], target, atol=0.05, rtol=0)

    def test_bc_converges_on_quadratic(self):
        """BC variant should also converge (possibly faster/slower)."""
        target = torch.tensor([1.0, 2.0, 3.0])
        params = {"x": torch.zeros(3)}

        opt = dp_adamw(lr=0.05, weight_decay=0.0, noise_variance=0.01)
        state = opt.init(params)

        for _ in range(200):
            grads = {"x": 2.0 * (params["x"] - target)}
            updates, state = opt.update(grads, state, params=params)
            params = torchopt.apply_updates(params, updates)

        torch.testing.assert_close(params["x"], target, atol=0.1, rtol=0)


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


class TestValidation:
    def test_negative_noise_variance_raises(self):
        with pytest.raises(ValueError, match="non-negative"):
            dp_adamw(noise_variance=-1.0)

    def test_apply_updates_compatible(self, params, grads):
        """Updates from BC mode work with torchopt.apply_updates."""
        opt = dp_adamw(lr=1e-3, noise_variance=0.1)
        state = opt.init(params)
        updates, _ = opt.update(grads, state, params=params)

        new_params = torchopt.apply_updates(params, updates)
        for k in params:
            assert new_params[k].shape == params[k].shape
            assert torch.isfinite(new_params[k]).all()


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------


class TestDeterminism:
    def test_same_input_same_output(self, params, grads):
        """Identical inputs must produce identical outputs (no hidden RNG)."""
        opt = dp_adamw(lr=1e-3, noise_variance=0.1)

        state_a = opt.init(params)
        state_b = opt.init(params)

        upd_a, _ = opt.update(grads, state_a, params=params)
        upd_b, _ = opt.update(grads, state_b, params=params)

        for k in params:
            torch.testing.assert_close(upd_a[k], upd_b[k])
