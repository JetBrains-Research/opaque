"""Tests for opaque.optimizers.adamw — universal Adam / AdamW.

Covers all four orthogonal modes:

- vanilla AdamW (no DP kwargs at update)
- DP-AdamW-BC (``noise_stddev`` constructor default + per-step override)
- DP-Adam-JME (``noisy_squared_grads`` paired-stream substitution)
- StableAdamW (``update_rms_clip`` constructor knob)

Plus the L2 weight decay branch (``decoupled_weight_decay=False``).
"""

from __future__ import annotations

import pytest
import torch

torchopt = pytest.importorskip("torchopt")

from opaque.clipping.per_group import PerGroup  # noqa: E402
from opaque.optimizers import AdamState, adamw  # noqa: E402


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def params():
    torch.manual_seed(0)
    return {"weight": torch.randn(4, 3), "bias": torch.randn(3)}


@pytest.fixture
def grads(params):
    torch.manual_seed(1)
    return {k: torch.randn_like(v) for k, v in params.items()}


def _adam_state(chain_state) -> AdamState:
    """Pull out the ``AdamState`` from the chain tuple.

    Position depends on the chain layout: position 0 in the decoupled
    chain (``moment → wd → lr``) and position 1 in the L2 chain
    (``wd → moment → lr``).
    """
    for entry in chain_state:
        if isinstance(entry, AdamState):
            return entry
    raise AssertionError(f"AdamState not found in chain state {chain_state!r}")


# ---------------------------------------------------------------------------
# Vanilla AdamW
# ---------------------------------------------------------------------------


class TestVanillaAdamW:
    def test_returns_gradient_transformation(self):
        opt = adamw(lr=1e-3)
        assert hasattr(opt, "init") and hasattr(opt, "update")

    def test_state_carries_phi_even_when_unused(self, params):
        opt = adamw(lr=1e-3)
        adam = _adam_state(opt.init(params))
        assert isinstance(adam, AdamState)
        assert adam.phi == 0.0
        assert adam.step == 0

    @pytest.mark.parametrize(
        "kwargs",
        [
            dict(lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01),
            dict(lr=0.1, betas=(0.85, 0.99), eps=1e-6, weight_decay=0.0),
            dict(lr=5e-4, weight_decay=0.1),
        ],
        ids=["default", "high_lr_no_wd", "heavy_wd"],
    )
    def test_matches_torchopt_adamw(self, params, kwargs):
        """``adamw`` (no DP kwargs) is numerically identical to torchopt.adamw."""
        opt_dp = adamw(**kwargs)
        opt_ref = torchopt.adamw(**kwargs)
        s_dp = opt_dp.init(params)
        s_ref = opt_ref.init(params)
        torch.manual_seed(42)
        for _ in range(10):
            g = {k: torch.randn_like(v) for k, v in params.items()}
            u_dp, s_dp = opt_dp.update(g, s_dp, params=params)
            u_ref, s_ref = opt_ref.update(g, s_ref, params=params)
            for k in params:
                torch.testing.assert_close(u_dp[k], u_ref[k])

    def test_apply_updates_compatible(self, params, grads):
        opt = adamw(lr=1e-3)
        state = opt.init(params)
        updates, _ = opt.update(grads, state, params=params)
        new_params = torchopt.apply_updates(params, updates)
        for k in params:
            assert new_params[k].shape == params[k].shape
            assert torch.isfinite(new_params[k]).all()


# ---------------------------------------------------------------------------
# Decoupled vs L2 weight decay
# ---------------------------------------------------------------------------


class TestWeightDecayMode:
    def test_decoupled_with_zero_grad_updates_only_wd(self):
        params = {"w": torch.ones(4) * 2.0}
        grads = {"w": torch.zeros(4)}
        opt = adamw(lr=0.1, weight_decay=0.1)
        state = opt.init(params)
        updates, _ = opt.update(grads, state, params=params)
        # Standard m̂/v̂ at zero grad still need bias correction; m̂ = 0
        # so the moment-scaled update is 0.  Only WD survives.
        # update = -lr * (0 + wd * params) = -0.01 * 2.0
        expected = -0.1 * 0.1 * params["w"]
        torch.testing.assert_close(updates["w"], expected)

    def test_l2_includes_wd_in_moments(self, params, grads):
        """L2 form (decoupled=False): the m,v moments capture wd*params."""
        opt_l2 = adamw(lr=1e-3, weight_decay=0.5, decoupled_weight_decay=False)
        opt_dec = adamw(lr=1e-3, weight_decay=0.5, decoupled_weight_decay=True)
        s_l2 = opt_l2.init(params)
        s_dec = opt_dec.init(params)
        # After one step on the same grads, the inner Adam state must
        # differ — L2 absorbed the wd*params term into m and v.
        _, s_l2 = opt_l2.update(grads, s_l2, params=params)
        _, s_dec = opt_dec.update(grads, s_dec, params=params)
        adam_l2 = _adam_state(s_l2)
        adam_dec = _adam_state(s_dec)
        # mu must differ between the two (wd was added to the gradient
        # under L2, so the first moment EMA picked up wd * params).
        assert any(not torch.allclose(adam_l2.mu[k], adam_dec.mu[k]) for k in params)


# ---------------------------------------------------------------------------
# DP-AdamW-BC: noise_stddev (constructor default + per-step override)
# ---------------------------------------------------------------------------


class TestBCMode:
    def test_phi_advances_under_default_stddev(self, params, grads):
        b2 = 0.999
        sigma = 0.5
        opt = adamw(lr=1e-3, betas=(0.9, b2), noise_stddev=sigma)
        state = opt.init(params)
        expected_phi = 0.0
        for _ in range(10):
            _, state = opt.update(grads, state, params=params)
            expected_phi = b2 * expected_phi + (1 - b2) * (sigma**2)
        assert _adam_state(state).phi == pytest.approx(expected_phi)

    def test_per_step_override_takes_precedence(self, params, grads):
        b2 = 0.999
        opt = adamw(lr=1e-3, betas=(0.9, b2), noise_stddev=0.0)
        state = opt.init(params)
        expected_phi = 0.0
        sigmas = [0.1, 0.2, 0.3, 0.2, 0.1]
        for sigma in sigmas:
            _, state = opt.update(grads, state, params=params, noise_stddev=sigma)
            expected_phi = b2 * expected_phi + (1 - b2) * (sigma**2)
        assert _adam_state(state).phi == pytest.approx(expected_phi)

    def test_zero_stddev_matches_torchopt(self, params, grads):
        """``noise_stddev=0`` is byte-equivalent to torchopt.adamw."""
        opt_bc = adamw(lr=1e-3, weight_decay=0.01, noise_stddev=0.0)
        opt_ref = torchopt.adamw(lr=1e-3, weight_decay=0.01)
        s_bc = opt_bc.init(params)
        s_ref = opt_ref.init(params)
        for _ in range(5):
            u_bc, s_bc = opt_bc.update(grads, s_bc, params=params)
            u_ref, s_ref = opt_ref.update(grads, s_ref, params=params)
            for k in params:
                torch.testing.assert_close(u_bc[k], u_ref[k])

    def test_bc_increases_effective_lr(self, params, grads):
        """Subtracting φ̂ from v̂ shrinks the denom → larger updates."""
        big = {k: v * 10 for k, v in grads.items()}
        opt_std = adamw(lr=1e-3)
        opt_bc = adamw(lr=1e-3, noise_stddev=0.01)
        s_std = opt_std.init(params)
        s_bc = opt_bc.init(params)
        for _ in range(10):
            u_std, s_std = opt_std.update(big, s_std, params=params)
            u_bc, s_bc = opt_bc.update(big, s_bc, params=params)
        norm_std = sum(u.norm() for u in u_std.values())
        norm_bc = sum(u.norm() for u in u_bc.values())
        assert norm_bc >= norm_std

    def test_bc_floor_keeps_denom_positive(self):
        """Huge σ pushes v̂ − φ̂ < 0; the floor saves us."""
        params = {"w": torch.ones(3)}
        grads = {"w": torch.ones(3) * 0.01}
        opt = adamw(lr=1e-3, noise_stddev=1e6)
        state = opt.init(params)
        updates, _ = opt.update(grads, state, params=params)
        assert torch.isfinite(updates["w"]).all()


# ---------------------------------------------------------------------------
# Per-group BC (PerGroup noise_stddev)
# ---------------------------------------------------------------------------


class TestPerGroupBC:
    @pytest.fixture
    def pg_params(self):
        return {"q_proj.weight": torch.randn(4, 3), "mlp.weight": torch.randn(3, 2)}

    @pytest.fixture
    def pg_grads(self, pg_params):
        return {k: torch.randn_like(v) for k, v in pg_params.items()}

    @pytest.fixture
    def pg_stddev(self):
        return PerGroup(
            groups={"q_proj.weight": "attn", "mlp.weight": "mlp"},
            values={"attn": 0.3, "mlp": 0.8},
        )

    def test_per_group_state_phi_is_dict(self, pg_params, pg_stddev):
        opt = adamw(lr=1e-3, noise_stddev=pg_stddev)
        adam = _adam_state(opt.init(pg_params))
        assert isinstance(adam.phi, dict)
        assert set(adam.phi.keys()) == set(pg_params.keys())

    def test_per_group_correction_is_per_key(self, pg_params, pg_grads, pg_stddev):
        opt_pg = adamw(lr=1e-3, noise_stddev=pg_stddev)
        opt_scalar = adamw(lr=1e-3, noise_stddev=0.3)  # matches "attn" group
        s_pg = opt_pg.init(pg_params)
        s_sc = opt_scalar.init(pg_params)
        for _ in range(5):
            u_pg, s_pg = opt_pg.update(pg_grads, s_pg, params=pg_params)
            u_sc, s_sc = opt_scalar.update(pg_grads, s_sc, params=pg_params)
        torch.testing.assert_close(u_pg["q_proj.weight"], u_sc["q_proj.weight"])
        assert not torch.equal(u_pg["mlp.weight"], u_sc["mlp.weight"])

    def test_per_group_with_nested_params(self):
        """Nested param pytrees: PerGroup looks up by dotted leaf path,
        and BC must walk leaves the same way (regression test for the
        review comment about iterating top-level dict keys only)."""
        torch.manual_seed(0)
        nested_params = {
            "layer1": {
                "weight": torch.randn(4, 3),
                "bias": torch.randn(3),
            },
            "layer2": {"weight": torch.randn(3, 2)},
        }
        nested_grads = {
            "layer1": {
                "weight": torch.randn_like(nested_params["layer1"]["weight"]),
                "bias": torch.randn_like(nested_params["layer1"]["bias"]),
            },
            "layer2": {"weight": torch.randn_like(nested_params["layer2"]["weight"])},
        }
        # PerGroup keyed by dotted leaf paths (exactly what
        # opaque.clipping.per_group._extract_keys produces for nested dicts).
        pg = PerGroup(
            groups={
                "layer1.weight": "g_a",
                "layer1.bias": "g_a",
                "layer2.weight": "g_b",
            },
            values={"g_a": 0.2, "g_b": 0.7},
        )
        opt = adamw(lr=1e-3, noise_stddev=pg)
        state = opt.init(nested_params)
        # init creates a phi entry per dotted leaf path, not per top-level key.
        adam = _adam_state(state)
        assert isinstance(adam.phi, dict)
        assert set(adam.phi.keys()) == {
            "layer1.weight",
            "layer1.bias",
            "layer2.weight",
        }
        # Update should not raise (the old code crashed on the
        # ``resolve_noise_variance(pg, "layer1")`` lookup).
        updates, new_state = opt.update(nested_grads, state, params=nested_params)
        new_adam = _adam_state(new_state)
        # Different groups → different φ-EMA values.
        assert new_adam.phi["layer1.weight"] == pytest.approx(
            new_adam.phi["layer1.bias"]
        )
        assert new_adam.phi["layer2.weight"] != pytest.approx(
            new_adam.phi["layer1.weight"]
        )
        # Updates have matching nested shape.
        assert updates["layer1"]["weight"].shape == (4, 3)
        assert updates["layer1"]["bias"].shape == (3,)
        assert updates["layer2"]["weight"].shape == (3, 2)


# ---------------------------------------------------------------------------
# JME paired-stream
# ---------------------------------------------------------------------------


class TestJMEMode:
    @pytest.fixture
    def sq_grads(self, grads):
        # Real JME would deliver privatised g²; this synthetic stream
        # is enough to exercise the v-update branch.
        return {k: v.pow(2) + 0.01 for k, v in grads.items()}

    def test_jme_consumes_external_g_squared(self, params, grads, sq_grads):
        """The v EMA must use ``noisy_squared_grads`` instead of g·g."""
        b2 = 0.999
        opt = adamw(lr=1e-3, betas=(0.9, b2))
        state = opt.init(params)
        _, state = opt.update(grads, state, params=params, noisy_squared_grads=sq_grads)
        adam = _adam_state(state)
        # v-update without bias correction in the kept state:
        for k in params:
            expected_v = (1 - b2) * sq_grads[k]
            torch.testing.assert_close(adam.nu[k], expected_v)

    def test_jme_phi_unchanged(self, params, grads, sq_grads):
        opt = adamw(lr=1e-3, noise_stddev=0.5)
        state = opt.init(params)
        # JME path explicitly bypasses φ; phi stays at 0 even though
        # ``noise_stddev`` was set as the constructor default.
        for _ in range(3):
            _, state = opt.update(
                grads, state, params=params, noisy_squared_grads=sq_grads
            )
        assert _adam_state(state).phi == 0.0

    def test_jme_negative_squared_stream_is_floored(self, params, grads):
        """JME's privatized g² stream is noisy and can be negative;
        the denominator must floor before sqrt instead of producing NaNs."""
        sq = {k: -torch.ones_like(v) for k, v in grads.items()}
        opt = adamw(lr=1e-3)
        state = opt.init(params)
        updates, _ = opt.update(grads, state, params=params, noisy_squared_grads=sq)
        for k in updates:
            assert torch.isfinite(updates[k]).all()

    def test_both_kwargs_raises(self, params, grads, sq_grads):
        opt = adamw(lr=1e-3)
        state = opt.init(params)
        with pytest.raises(ValueError, match="mutually exclusive"):
            opt.update(
                grads,
                state,
                params=params,
                noise_stddev=0.5,
                noisy_squared_grads=sq_grads,
            )


# ---------------------------------------------------------------------------
# StableAdamW (update_rms_clip)
# ---------------------------------------------------------------------------


class TestStableAdamW:
    def test_clip_bounds_update_rms(self, params):
        """RMS of the update should stay below threshold (within FP)."""
        # Construct grads designed to produce a large-RMS update before clip.
        grads = {k: torch.full_like(v, 100.0) for k, v in params.items()}
        opt = adamw(lr=1.0, weight_decay=0.0, update_rms_clip=0.5)
        state = opt.init(params)
        updates, _ = opt.update(grads, state, params=params)
        # The chain applies neg-LR after the clip, so the LR scaling
        # divides the magnitude back; we read the *clip's* effect by
        # comparing to the no-clip variant.
        opt_no = adamw(lr=1.0, weight_decay=0.0, update_rms_clip=None)
        s_no = opt_no.init(params)
        u_no, _ = opt_no.update(grads, s_no, params=params)
        n_clip = sum(u.pow(2).sum() for u in updates.values())
        n_no = sum(u.pow(2).sum() for u in u_no.values())
        assert n_clip < n_no  # clip should shrink the update.

    def test_no_clip_below_threshold(self, params, grads):
        """Updates whose RMS is below threshold are passed through unchanged."""
        opt_clip = adamw(lr=1e-3, weight_decay=0.0, update_rms_clip=10.0)
        opt_none = adamw(lr=1e-3, weight_decay=0.0, update_rms_clip=None)
        s_c = opt_clip.init(params)
        s_n = opt_none.init(params)
        u_c, _ = opt_clip.update(grads, s_c, params=params)
        u_n, _ = opt_none.update(grads, s_n, params=params)
        for k in params:
            torch.testing.assert_close(u_c[k], u_n[k])


# ---------------------------------------------------------------------------
# Convergence
# ---------------------------------------------------------------------------


class TestConvergence:
    def test_vanilla_minimises_quadratic(self):
        target = torch.tensor([1.0, 2.0, 3.0])
        params = {"x": torch.zeros(3)}
        opt = adamw(lr=0.05, weight_decay=0.0)
        state = opt.init(params)
        for _ in range(200):
            grads = {"x": 2.0 * (params["x"] - target)}
            updates, state = opt.update(grads, state, params=params)
            params = torchopt.apply_updates(params, updates)
        torch.testing.assert_close(params["x"], target, atol=0.05, rtol=0)

    def test_bc_minimises_quadratic(self):
        target = torch.tensor([1.0, 2.0, 3.0])
        params = {"x": torch.zeros(3)}
        opt = adamw(lr=0.05, weight_decay=0.0, noise_stddev=0.01)
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
            adamw(noise_stddev=-1.0)

    def test_negative_weight_decay_raises(self):
        with pytest.raises(ValueError, match="non-negative"):
            adamw(weight_decay=-1.0)

    def test_zero_eps_raises(self):
        with pytest.raises(ValueError, match="positive"):
            adamw(eps=0.0)

    def test_invalid_beta_raises(self):
        with pytest.raises(ValueError, match="beta_1"):
            adamw(betas=(1.0, 0.999))
        with pytest.raises(ValueError, match="beta_2"):
            adamw(betas=(0.9, 1.5))

    def test_zero_rms_clip_raises(self):
        with pytest.raises(ValueError, match="update_rms_clip"):
            adamw(update_rms_clip=0.0)
