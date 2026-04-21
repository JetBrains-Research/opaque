"""Tests for AdamW-BC optimizer (Algorithms 1 and 2)."""

import pytest
import torch

torchopt = pytest.importorskip(
    "torchopt", reason="torchopt required for optimizer tests"
)

from opaque.optimizers.adamw_bc import AdamWBCState, adamw_bc  # noqa: E402
from opaque.core.utils.per_group import PerGroup  # noqa: E402


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
# Algorithm 1: standard mode (noise_stddev=0) — same math as torchopt.adamw
# ---------------------------------------------------------------------------


class TestStandardMode:
    """When noise_stddev=0, adamw_bc is numerically identical to torchopt.adamw."""

    def test_returns_gradient_transformation(self):
        opt = adamw_bc(lr=1e-3)
        assert hasattr(opt, "init")
        assert hasattr(opt, "update")

    def test_init_and_update(self, params, grads):
        opt = adamw_bc(lr=1e-3, weight_decay=0.01)
        state = opt.init(params)

        updates, new_state = opt.update(grads, state, params=params)

        # Updates have same structure as params.
        assert set(updates.keys()) == set(params.keys())
        for k in params:
            assert updates[k].shape == params[k].shape

    def test_state_is_chain_tuple(self, params):
        opt = adamw_bc(lr=1e-3)
        state = opt.init(params)
        # Chain state: (AdamWBCState, wd_state, lr_state).
        assert isinstance(state, tuple)
        assert isinstance(state[0], AdamWBCState)

    def test_params_change_after_apply(self, params, grads):
        opt = adamw_bc(lr=1e-2)
        state = opt.init(params)
        # Save original values — apply_updates is in-place.
        orig = {k: v.clone() for k, v in params.items()}

        updates, _ = opt.update(grads, state, params=params)
        params = torchopt.apply_updates(params, updates)

        changed = any(not torch.equal(params[k], orig[k]) for k in params)
        assert changed

    @pytest.mark.parametrize(
        "kwargs",
        [
            dict(lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01),
            dict(lr=0.1, betas=(0.85, 0.99), eps=1e-6, weight_decay=0.0),
            dict(lr=5e-4, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.1),
        ],
        ids=["default", "high_lr_no_wd", "heavy_wd"],
    )
    def test_matches_torchopt_adamw(self, params, kwargs):
        """adamw_bc(noise_stddev=0) must produce identical updates to
        torchopt.adamw at every step, across hyperparameter configs."""
        opt_dp = adamw_bc(**kwargs)
        opt_ref = torchopt.adamw(**kwargs)

        state_dp = opt_dp.init(params)
        state_ref = opt_ref.init(params)

        torch.manual_seed(42)
        for step in range(10):
            # Varying gradients expose moment accumulation differences.
            grads = {k: torch.randn_like(v) for k, v in params.items()}

            updates_dp, state_dp = opt_dp.update(grads, state_dp, params=params)
            updates_ref, state_ref = opt_ref.update(grads, state_ref, params=params)

            for k in params:
                torch.testing.assert_close(
                    updates_dp[k],
                    updates_ref[k],
                    msg=f"Mismatch at step {step}, key '{k}', config {kwargs}",
                )


# ---------------------------------------------------------------------------
# Algorithm 2: BC mode (noise_stddev > 0)
# ---------------------------------------------------------------------------


def _bc_state(chain_state):
    """Extract AdamWBCState from the chain state tuple."""
    return chain_state[0]


class TestBCMode:
    """AdamW-BC-BC: bias-corrected second moment."""

    def test_returns_gradient_transformation(self):
        opt = adamw_bc(lr=1e-3, noise_stddev=0.5)
        assert hasattr(opt, "init")
        assert hasattr(opt, "update")

    def test_init_returns_dp_state(self, params):
        opt = adamw_bc(lr=1e-3, noise_stddev=0.5)
        state = opt.init(params)
        bc = _bc_state(state)

        assert isinstance(bc, AdamWBCState)
        assert bc.step == 0
        assert bc.phi == 0.0
        # Moments initialised to zeros.
        for k in params:
            assert torch.equal(bc.mu[k], torch.zeros_like(params[k]))
            assert torch.equal(bc.nu[k], torch.zeros_like(params[k]))

    def test_update_advances_step(self, params, grads):
        opt = adamw_bc(lr=1e-3, noise_stddev=0.5)
        state = opt.init(params)
        _, state2 = opt.update(grads, state, params=params)
        assert _bc_state(state2).step == 1
        _, state3 = opt.update(grads, state2, params=params)
        assert _bc_state(state3).step == 2

    def test_moments_updated(self, params, grads):
        opt = adamw_bc(lr=1e-3, noise_stddev=0.5)
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
        opt_std = adamw_bc(lr=1e-3)
        opt_bc = adamw_bc(lr=1e-3, noise_stddev=0.5)

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

        opt_std = adamw_bc(lr=1e-3, noise_stddev=0.0)
        opt_bc = adamw_bc(lr=1e-3, noise_stddev=0.01)

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
        """With huge noise_stddev, v_hat - Phi goes negative;
        bc_floor must keep the denominator positive."""
        params = {"w": torch.ones(3)}
        grads = {"w": torch.ones(3) * 0.01}

        opt = adamw_bc(lr=1e-3, noise_stddev=1e6, bc_floor=1e-8)
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

        opt = adamw_bc(lr=1e-3, noise_stddev=0.1)
        state = opt.init(nested_params)
        updates, state2 = opt.update(grads, state, params=nested_params)

        assert _bc_state(state2).step == 1
        assert "layer1" in updates and "layer2" in updates
        assert (
            updates["layer1"]["weight"].shape == nested_params["layer1"]["weight"].shape
        )


# ---------------------------------------------------------------------------
# Noise variance EMA tracking (adaptive clipping support)
# ---------------------------------------------------------------------------


class TestNoiseVarianceEMA:
    """The phi EMA must track per-step noise variance correctly."""

    def test_phi_ema_matches_manual_computation(self, params, grads):
        """After several steps with constant noise_stddev, phi EMA
        should equal the manual recurrence: phi_t = b2*phi_{t-1} + (1-b2)*nv."""
        b2 = 0.999
        noise_stddev = 0.5
        opt = adamw_bc(lr=1e-3, betas=(0.9, b2), noise_stddev=noise_stddev)
        state = opt.init(params)

        expected_phi = 0.0
        for _ in range(10):
            _, state = opt.update(grads, state, params=params)
            expected_phi = b2 * expected_phi + (1 - b2) * (noise_stddev**2)

        assert _bc_state(state).phi == pytest.approx(expected_phi)

    def test_constant_phi_converges_to_nv(self, params, grads):
        """With constant noise_stddev, phi_hat = phi/(1-b2^t) → sigma^2."""
        b2 = 0.999
        noise_stddev = 0.25
        opt = adamw_bc(lr=1e-3, betas=(0.9, b2), noise_stddev=noise_stddev)
        state = opt.init(params)

        for _ in range(5000):
            _, state = opt.update(grads, state, params=params)

        bc = _bc_state(state)
        phi_hat = bc.phi / (1 - b2**bc.step)
        assert phi_hat == pytest.approx(noise_stddev**2, rel=1e-3)

    def test_per_step_override_tracks_varying_stddev(self, params, grads):
        """Per-step noise_stddev override should produce different phi
        than using the constructor default."""
        b2 = 0.999
        opt = adamw_bc(lr=1e-3, betas=(0.9, b2), noise_stddev=0.1)
        state_default = opt.init(params)
        state_override = opt.init(params)

        for step in range(10):
            _, state_default = opt.update(grads, state_default, params=params)
            # Override with increasing noise stddev.
            varying_stddev = 0.1 + step * 0.05
            _, state_override = opt.update(
                grads, state_override, params=params, noise_stddev=varying_stddev
            )

        # Default uses constant 0.1; override uses increasing values -> different phi.
        assert _bc_state(state_default).phi != pytest.approx(
            _bc_state(state_override).phi
        )
        # Override phi should be larger (noise_stddev was always >= 0.1).
        assert _bc_state(state_override).phi > _bc_state(state_default).phi

    def test_per_step_override_ema_manual(self, params, grads):
        """Verify the EMA recurrence with per-step overrides."""
        b2 = 0.999
        opt = adamw_bc(lr=1e-3, betas=(0.9, b2), noise_stddev=0.0)
        state = opt.init(params)

        expected_phi = 0.0
        noise_stddev_schedule = [0.1, 0.2, 0.3, 0.2, 0.1]
        for noise_stddev in noise_stddev_schedule:
            _, state = opt.update(
                grads, state, params=params, noise_stddev=noise_stddev
            )
            expected_phi = b2 * expected_phi + (1 - b2) * (noise_stddev**2)

        assert _bc_state(state).phi == pytest.approx(expected_phi)

    def test_zero_override_disables_bc_for_step(self, params, grads):
        """Passing noise_stddev=0 per-step should contribute 0 to the EMA."""
        b2 = 0.999
        opt = adamw_bc(lr=1e-3, betas=(0.9, b2), noise_stddev=0.5)
        state = opt.init(params)

        # 5 steps with default nv=0.5, then 5 with override nv=0.
        for _ in range(5):
            _, state = opt.update(grads, state, params=params)
        phi_after_5 = _bc_state(state).phi

        for _ in range(5):
            _, state = opt.update(grads, state, params=params, noise_stddev=0.0)
        phi_after_10 = _bc_state(state).phi

        # phi should decay toward 0 when we feed in 0.
        assert phi_after_10 < phi_after_5

    def test_standard_mode_phi_stays_zero(self, params, grads):
        """With noise_stddev=0 and no override, phi remains 0."""
        opt = adamw_bc(lr=1e-3, noise_stddev=0.0)
        state = opt.init(params)

        for _ in range(10):
            _, state = opt.update(grads, state, params=params)

        assert _bc_state(state).phi == 0.0


# ---------------------------------------------------------------------------
# Algorithm 2: per-group BC (PerGroup noise_stddev)
# ---------------------------------------------------------------------------


class TestPerGroupBC:
    """AdamW-BC-BC with per-group noise variance (MSE-optimal allocation)."""

    @pytest.fixture
    def pg_params(self):
        return {"q_proj.weight": torch.randn(4, 3), "mlp.weight": torch.randn(3, 2)}

    @pytest.fixture
    def pg_grads(self, pg_params):
        return {k: torch.randn_like(v) for k, v in pg_params.items()}

    @pytest.fixture
    def pg_stddev(self):
        """PerGroup of noise stddevs (like per_group_noise_stddev returns)."""
        return PerGroup(
            groups={"q_proj.weight": "attn", "mlp.weight": "mlp"},
            values={"attn": 0.3, "mlp": 0.8},
        )

    def test_per_group_produces_different_correction_per_key(
        self, pg_params, pg_grads, pg_stddev
    ):
        """Each param group should get its own BC correction."""
        opt_pg = adamw_bc(lr=1e-3, noise_stddev=pg_stddev)
        # Use scalar variance matching the "attn" group — updates differ
        # for the "mlp" key because its variance is different.
        opt_scalar = adamw_bc(lr=1e-3, noise_stddev=0.3)

        s_pg = opt_pg.init(pg_params)
        s_sc = opt_scalar.init(pg_params)

        for _ in range(5):
            u_pg, s_pg = opt_pg.update(pg_grads, s_pg, params=pg_params)
            u_sc, s_sc = opt_scalar.update(pg_grads, s_sc, params=pg_params)

        # "attn" group has stddev=0.3 in both → should match.
        torch.testing.assert_close(u_pg["q_proj.weight"], u_sc["q_proj.weight"])
        # "mlp" group has stddev=0.8 vs 0.3 → must differ.
        assert not torch.equal(u_pg["mlp.weight"], u_sc["mlp.weight"])

    def test_per_group_bc_increases_effective_lr(self, pg_params, pg_grads, pg_stddev):
        """Per-group BC should produce larger updates than standard mode."""
        big_grads = {k: v * 10 for k, v in pg_grads.items()}

        opt_std = adamw_bc(lr=1e-3, noise_stddev=0.0)
        opt_pg = adamw_bc(lr=1e-3, noise_stddev=pg_stddev)

        s_std = opt_std.init(pg_params)
        s_pg = opt_pg.init(pg_params)

        for _ in range(10):
            u_std, s_std = opt_std.update(big_grads, s_std, params=pg_params)
            u_pg, s_pg = opt_pg.update(big_grads, s_pg, params=pg_params)

        norm_std = sum(u.norm() for u in u_std.values())
        norm_pg = sum(u.norm() for u in u_pg.values())
        assert norm_pg >= norm_std

    def test_per_group_with_zero_group(self, pg_params, pg_grads):
        """A group with stddev=0 should match standard AdamW for that key."""
        pg_mixed = PerGroup(
            groups={"q_proj.weight": "attn", "mlp.weight": "mlp"},
            values={"attn": 0.0, "mlp": 0.5},
        )
        opt = adamw_bc(lr=1e-3, noise_stddev=pg_mixed)
        opt_std = adamw_bc(lr=1e-3, noise_stddev=0.0)

        s = opt.init(pg_params)
        s_std = opt_std.init(pg_params)

        for _ in range(5):
            u, s = opt.update(pg_grads, s, params=pg_params)
            u_std, s_std = opt_std.update(pg_grads, s_std, params=pg_params)

        # attn group has stddev=0 → matches standard.
        torch.testing.assert_close(u["q_proj.weight"], u_std["q_proj.weight"])
        # mlp group has stddev=0.5 → differs.
        assert not torch.equal(u["mlp.weight"], u_std["mlp.weight"])

    def test_per_group_floor_prevents_zero_denominator(self, pg_params, pg_grads):
        """Huge per-group stddev should be clamped by bc_floor."""
        huge = PerGroup(
            groups={"q_proj.weight": "attn", "mlp.weight": "mlp"},
            values={"attn": 1000.0, "mlp": 1000.0},
        )
        opt = adamw_bc(lr=1e-3, noise_stddev=huge, bc_floor=1e-8)
        state = opt.init(pg_params)
        updates, _ = opt.update(pg_grads, state, params=pg_params)

        for k in pg_params:
            assert torch.isfinite(updates[k]).all()


# ---------------------------------------------------------------------------
# Weight decay (BC mode — standard mode reuses torchopt.adamw which is
# already tested upstream)
# ---------------------------------------------------------------------------


class TestWeightDecay:
    def test_decoupled_weight_decay(self):
        """With zero gradients, updates should only reflect weight decay."""
        params = {"w": torch.ones(4) * 2.0}
        grads = {"w": torch.zeros(4)}

        opt = adamw_bc(lr=0.1, weight_decay=0.1, noise_stddev=0.01, bc_floor=1e-30)
        state = opt.init(params)
        updates, _ = opt.update(grads, state, params=params)

        # With zero grads, m_hat=0 so the adam-scaled part is zero.
        # Only weight decay contributes:  update = -lr * wd * params.
        expected = -0.1 * 0.1 * params["w"]
        torch.testing.assert_close(updates["w"], expected)

    def test_no_weight_decay_no_params_needed(self):
        params = {"w": torch.randn(3)}
        grads = {"w": torch.randn(3)}

        opt = adamw_bc(lr=1e-3, weight_decay=0.0, noise_stddev=0.1)
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

        opt = adamw_bc(lr=0.05, weight_decay=0.0, noise_stddev=0.0)
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

        opt = adamw_bc(lr=0.05, weight_decay=0.0, noise_stddev=0.01)
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
    def test_negative_noise_stddev_raises(self):
        with pytest.raises(ValueError, match="non-negative"):
            adamw_bc(noise_stddev=-1.0)

    def test_apply_updates_compatible(self, params, grads):
        """Updates from BC mode work with torchopt.apply_updates."""
        opt = adamw_bc(lr=1e-3, noise_stddev=0.1)
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
        opt = adamw_bc(lr=1e-3, noise_stddev=0.1)

        state_a = opt.init(params)
        state_b = opt.init(params)

        upd_a, _ = opt.update(grads, state_a, params=params)
        upd_b, _ = opt.update(grads, state_b, params=params)

        for k in params:
            torch.testing.assert_close(upd_a[k], upd_b[k])
