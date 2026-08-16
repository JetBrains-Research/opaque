"""Tests for opaque.optimizers.adamw — universal Adam / AdamW."""

from __future__ import annotations

import pytest
import torch

from opaque.optimizers import adam, adamw, apply_updates
from opaque.optimizers.types import AdamState
from opaque.serialization import from_state_dict, state_dict
from opaque.types import (
    PerGroup,
    SecondMomentNoiseOutput,
    clipped,
    noised,
)

torchopt = pytest.importorskip("torchopt")


@pytest.fixture
def params():
    torch.manual_seed(0)
    return {"weight": torch.randn(4, 3), "bias": torch.randn(3)}


@pytest.fixture
def grads(params):
    torch.manual_seed(1)
    return {k: torch.randn_like(v) for k, v in params.items()}


# ---------------------------------------------------------------------------
# Vanilla AdamW
# ---------------------------------------------------------------------------


class TestVanillaAdamW:
    def test_returns_step_and_state(self, params):
        step, state = adamw(params, lr=1e-3)
        assert callable(step)
        assert isinstance(state, AdamState)

    def test_state_carries_phi_even_when_unused(self, params):
        _step, state = adamw(params, lr=1e-3)
        assert isinstance(state, AdamState)
        assert state.phi == 0.0
        assert state.step == 0

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"lr": 1e-3, "betas": (0.9, 0.999), "eps": 1e-8, "weight_decay": 0.01},
            {"lr": 0.1, "betas": (0.85, 0.99), "eps": 1e-6, "weight_decay": 0.0},
            {"lr": 5e-4, "weight_decay": 0.1},
        ],
        ids=["default", "high_lr_no_wd", "heavy_wd"],
    )
    def test_matches_torchopt_adamw(self, params, kwargs):
        step, state = adamw(params, **kwargs)
        ref = torchopt.adamw(**kwargs)
        ref_state = ref.init(params)

        torch.manual_seed(42)
        for _ in range(10):
            g = {k: torch.randn_like(v) for k, v in params.items()}
            updates, state = step(g, state, params=params)
            ref_updates, ref_state = ref.update(g, ref_state, params=params)
            for k in params:
                torch.testing.assert_close(updates[k], ref_updates[k])

    def test_apply_updates_compatible(self, params, grads):
        step, state = adamw(params, lr=1e-3)
        updates, _ = step(grads, state, params=params)
        new_params = apply_updates(params, updates)
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
        step, state = adamw(params, lr=0.1, weight_decay=0.1)
        updates, _ = step(grads, state, params=params)
        expected = -0.1 * 0.1 * params["w"]
        torch.testing.assert_close(updates["w"], expected)


class TestAdamAlias:
    def test_adam_uses_l2_weight_decay_branch(self, params, grads):
        step, state = adam(params, lr=1e-3, weight_decay=0.01)
        ref_step, ref_state = adamw(
            params, lr=1e-3, weight_decay=0.01, decoupled_weight_decay=False
        )

        updates, _ = step(grads, state, params=params)
        ref_updates, _ = ref_step(grads, ref_state, params=params)
        for name in updates:
            torch.testing.assert_close(updates[name], ref_updates[name])

    def test_adam_consumes_noisy_metadata(self, params, grads):
        step, state = adam(params, lr=1e-3, noise_bias_correction=True)
        _, state = step(
            noised(grads, max_norm=1.0, noise_stddev=0.5), state, params=params
        )
        assert state.phi != 0.0

    def test_l2_includes_wd_in_moments(self, params, grads):
        step_l2, state_l2 = adamw(
            params, lr=1e-3, weight_decay=0.5, decoupled_weight_decay=False
        )
        step_dec, state_dec = adamw(
            params, lr=1e-3, weight_decay=0.5, decoupled_weight_decay=True
        )
        _, state_l2 = step_l2(grads, state_l2, params=params)
        _, state_dec = step_dec(grads, state_dec, params=params)
        assert any(not torch.allclose(state_l2.mu[k], state_dec.mu[k]) for k in params)


# ---------------------------------------------------------------------------
# DP-AdamW-BC
# ---------------------------------------------------------------------------


class TestBCMode:
    def test_raw_pytree_keeps_phi_zero(self, params, grads):
        step, state = adamw(params, lr=1e-3)
        for _ in range(10):
            _, state = step(grads, state, params=params)
        assert state.phi == 0.0

    def test_bc_init_accepts_leaf_tensor(self):
        params = torch.randn(4, 3)
        step, state = adamw(params, lr=1e-3, noise_bias_correction=True)
        assert isinstance(state.phi, dict)
        assert set(state.phi) == {()}
        assert state.phi[()] == 0.0
        grads = torch.randn_like(params)
        updates, new_state = step(
            noised(grads, max_norm=1.0, noise_stddev=0.5),
            state,
            params=params,
        )
        assert updates.shape == params.shape
        assert new_state.phi[()] > 0.0

    def test_phi_advances_under_noisy_metadata(self, params, grads):
        b2 = 0.999
        sigma = 0.5
        step, state = adamw(
            params, lr=1e-3, betas=(0.9, b2), noise_bias_correction=True
        )
        expected_phi = 0.0
        for _ in range(10):
            _, state = step(
                noised(grads, max_norm=1.0, noise_stddev=sigma),
                state,
                params=params,
            )
            expected_phi = b2 * expected_phi + (1 - b2) * (sigma**2)
        assert isinstance(state.phi, dict)
        assert all(v == pytest.approx(expected_phi) for v in state.phi.values())

    def test_noisy_updates_take_per_step_metadata(self, params, grads):
        b2 = 0.999
        step, state = adamw(
            params, lr=1e-3, betas=(0.9, b2), noise_bias_correction=True
        )
        expected_phi = 0.0
        sigmas = [0.1, 0.2, 0.3, 0.2, 0.1]
        for sigma in sigmas:
            _, state = step(
                noised(grads, max_norm=1.0, noise_stddev=sigma),
                state,
                params=params,
            )
            expected_phi = b2 * expected_phi + (1 - b2) * (sigma**2)
        assert isinstance(state.phi, dict)
        assert all(v == pytest.approx(expected_phi) for v in state.phi.values())

    def test_explicit_noise_stddev_kwarg_rejected(self, params, grads):
        step, state = adamw(params, lr=1e-3)
        with pytest.raises(TypeError, match="noise_stddev"):
            step(grads, state, params=params, noise_stddev=0.5)

    def test_clipped_updates_are_rejected(self, params, grads):
        step, state = adamw(params, lr=1e-3)
        with pytest.raises(
            TypeError, match="have not passed through a noise mechanism"
        ):
            step(clipped(grads, max_norm=1.0), state, params=params)

    def test_raw_pytree_matches_torchopt(self, params, grads):
        step, state = adamw(params, lr=1e-3, weight_decay=0.01)
        ref = torchopt.adamw(lr=1e-3, weight_decay=0.01)
        ref_state = ref.init(params)
        for _ in range(5):
            g = {k: v.clone() for k, v in grads.items()}
            updates, state = step(g, state, params=params)
            ref_updates, ref_state = ref.update(g, ref_state, params=params)
            for k in params:
                torch.testing.assert_close(updates[k], ref_updates[k])

    def test_bc_increases_effective_lr(self, params, grads):
        big = {k: v * 10 for k, v in grads.items()}
        step_std, state_std = adamw(params, lr=1e-3)
        step_bc, state_bc = adamw(params, lr=1e-3, noise_bias_correction=True)
        for _ in range(10):
            u_std, state_std = step_std(big, state_std, params=params)
            u_bc, state_bc = step_bc(
                noised(big, max_norm=1.0, noise_stddev=0.01),
                state_bc,
                params=params,
            )
        norm_std = sum(u.norm() for u in u_std.values())
        norm_bc = sum(u.norm() for u in u_bc.values())
        assert norm_bc >= norm_std

    def test_bc_floor_keeps_denom_positive(self):
        params = {"w": torch.ones(3)}
        grads = {"w": torch.ones(3) * 0.01}
        step, state = adamw(params, lr=1e-3, noise_bias_correction=True)
        updates, _ = step(
            noised(grads, max_norm=1.0, noise_stddev=1e6),
            state,
            params=params,
        )
        assert torch.isfinite(updates["w"]).all()

    def test_bc_flag_disables_noisy_metadata_correction(self, params, grads):
        step, state = adamw(params, lr=1e-3, noise_bias_correction=False)
        _, state = step(
            noised(grads, max_norm=1.0, noise_stddev=0.5),
            state,
            params=params,
        )
        assert state.phi == 0.0


# ---------------------------------------------------------------------------
# Per-group BC
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
        step, state = adamw(pg_params, lr=1e-3, noise_bias_correction=True)
        _, state = step(
            noised(pg_params, max_norm=1.0, noise_stddev=pg_stddev),
            state,
            params=pg_params,
        )
        assert isinstance(state.phi, dict)
        assert set(state.phi.keys()) == {(k,) for k in pg_params}

    def test_per_group_correction_is_per_key(self, pg_params, pg_grads, pg_stddev):
        step_pg, state_pg = adamw(pg_params, lr=1e-3, noise_bias_correction=True)
        step_sc, state_sc = adamw(pg_params, lr=1e-3, noise_bias_correction=True)
        for _ in range(5):
            u_pg, state_pg = step_pg(
                noised(
                    {k: v.clone() for k, v in pg_grads.items()},
                    max_norm=1.0,
                    noise_stddev=pg_stddev,
                ),
                state_pg,
                params=pg_params,
            )
            u_sc, state_sc = step_sc(
                noised(
                    {k: v.clone() for k, v in pg_grads.items()},
                    max_norm=1.0,
                    noise_stddev=0.3,
                ),
                state_sc,
                params=pg_params,
            )
        torch.testing.assert_close(u_pg["q_proj.weight"], u_sc["q_proj.weight"])
        assert not torch.equal(u_pg["mlp.weight"], u_sc["mlp.weight"])

    def test_per_group_with_nested_params(self):
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
        pg = PerGroup(
            groups={
                ("layer1", "weight"): "g_a",
                ("layer1", "bias"): "g_a",
                ("layer2", "weight"): "g_b",
            },
            values={"g_a": 0.2, "g_b": 0.7},
        )
        step, state = adamw(nested_params, lr=1e-3, noise_bias_correction=True)
        init_phi = state.phi
        assert isinstance(init_phi, dict)
        assert set(init_phi) == {
            ("layer1", "weight"),
            ("layer1", "bias"),
            ("layer2", "weight"),
        }
        assert all(v == 0.0 for v in init_phi.values())

        updates, new_state = step(
            noised(nested_grads, max_norm=1.0, noise_stddev=pg),
            state,
            params=nested_params,
        )
        new_adam = new_state
        assert set(new_adam.phi.keys()) == {
            ("layer1", "weight"),
            ("layer1", "bias"),
            ("layer2", "weight"),
        }
        assert new_adam.phi[("layer1", "weight")] == pytest.approx(
            new_adam.phi[("layer1", "bias")]
        )
        assert new_adam.phi[("layer2", "weight")] != pytest.approx(
            new_adam.phi[("layer1", "weight")]
        )
        assert updates["layer1"]["weight"].shape == (4, 3)
        assert updates["layer1"]["bias"].shape == (3,)
        assert updates["layer2"]["weight"].shape == (3, 2)


# ---------------------------------------------------------------------------
# Private second-moment stream
# ---------------------------------------------------------------------------


class TestSecondMomentMode:
    @pytest.fixture
    def sq_grads(self, grads):
        return {k: v.pow(2) + 0.01 for k, v in grads.items()}

    def test_second_moment_consumes_external_g_squared(self, params, grads, sq_grads):
        b2 = 0.999
        step, state = adamw(params, lr=1e-3, betas=(0.9, b2))
        output = SecondMomentNoiseOutput(
            noised(grads, max_norm=1.0, noise_stddev=0.1),
            noised(sq_grads, max_norm=1.0, noise_stddev=0.1),
        )
        _, state = step(output, state, params=params)
        for k in params:
            expected_v = (1 - b2) * sq_grads[k]
            torch.testing.assert_close(state.nu[k], expected_v)

    def test_second_moment_output_unwraps_noisy_streams(self, params, grads, sq_grads):
        b2 = 0.999
        step, state = adamw(params, lr=1e-3, betas=(0.9, b2))
        output = SecondMomentNoiseOutput(
            noised(grads, max_norm=1.0, noise_stddev=0.5),
            noised(sq_grads, max_norm=1.0, noise_stddev=0.1),
        )
        _, state = step(output, state, params=params)
        assert state.phi == 0.0
        for k in params:
            expected_v = (1 - b2) * sq_grads[k]
            torch.testing.assert_close(state.nu[k], expected_v)

    def test_second_moment_output_rejects_clipped_stream(self, params, grads, sq_grads):
        step, state = adamw(params, lr=1e-3)
        output = SecondMomentNoiseOutput(
            noised(grads, max_norm=1.0, noise_stddev=0.5),
            clipped(sq_grads, max_norm=1.0),
        )
        with pytest.raises(
            TypeError, match=r"SecondMomentNoiseOutput.noisy_squared_grads"
        ):
            step(output, state, params=params)

    def test_second_moment_phi_unchanged(self, params, grads, sq_grads):
        step, state = adamw(params, lr=1e-3)
        for _ in range(3):
            output = SecondMomentNoiseOutput(
                noised(
                    {k: v.clone() for k, v in grads.items()},
                    max_norm=1.0,
                    noise_stddev=0.1,
                ),
                noised(
                    {k: v.clone() for k, v in sq_grads.items()},
                    max_norm=1.0,
                    noise_stddev=0.1,
                ),
            )
            _, state = step(output, state, params=params)
        assert state.phi == 0.0

    def test_second_moment_negative_squared_stream_bounded(self, params, grads):
        sq = {k: -torch.ones_like(v) for k, v in grads.items()}
        step, state = adamw(params, lr=1e-3)
        output = SecondMomentNoiseOutput(
            noised(grads, max_norm=1.0, noise_stddev=0.1),
            noised(sq, max_norm=1.0, noise_stddev=0.1),
        )
        updates, _ = step(output, state, params=params)
        for k in updates:
            assert torch.isfinite(updates[k]).all()
            assert updates[k].abs().max().item() < 10.0

    def test_explicit_second_moment_kwarg_rejected(self, params, grads, sq_grads):
        step, state = adamw(params, lr=1e-3)
        with pytest.raises(TypeError, match="noisy_squared_grads"):
            step(grads, state, params=params, noisy_squared_grads=sq_grads)


# ---------------------------------------------------------------------------
# StableAdamW
# ---------------------------------------------------------------------------


class TestStableAdamW:
    def test_clip_bounds_update_rms(self, params):
        grads = {k: torch.full_like(v, 100.0) for k, v in params.items()}
        step, state = adamw(params, lr=1.0, weight_decay=0.0, update_rms_clip=0.5)
        updates, _ = step(grads, state, params=params)

        step_no, state_no = adamw(params, lr=1.0, weight_decay=0.0)
        u_no, _ = step_no(grads, state_no, params=params)
        n_clip = sum(u.pow(2).sum() for u in updates.values())
        n_no = sum(u.pow(2).sum() for u in u_no.values())
        assert n_clip < n_no

    def test_no_clip_below_threshold(self, params, grads):
        step_clip, state_clip = adamw(
            params, lr=1e-3, weight_decay=0.0, update_rms_clip=10.0
        )
        step_none, state_none = adamw(params, lr=1e-3, weight_decay=0.0)
        u_c, _ = step_clip(grads, state_clip, params=params)
        u_n, _ = step_none(grads, state_none, params=params)
        for k in params:
            torch.testing.assert_close(u_c[k], u_n[k])

    def test_clip_uses_single_global_scale_not_per_leaf(self):
        params = {
            "big": torch.zeros(16),
            "small": torch.zeros(16),
        }
        grads = {
            "big": torch.full((16,), 10.0),
            "small": torch.full((16,), 1.0),
        }
        threshold = 0.85
        step, state = adamw(params, lr=1.0, weight_decay=0.0, update_rms_clip=threshold)
        step_no, state_no = adamw(params, lr=1.0, weight_decay=0.0)
        updates, _ = step(grads, state, params=params)
        updates_no, _ = step_no(grads, state_no, params=params)

        global_rms = torch.sqrt(
            (updates_no["big"].pow(2).sum() + updates_no["small"].pow(2).sum())
            / (updates_no["big"].numel() + updates_no["small"].numel())
        )
        expected_scale = torch.clamp(global_rms / threshold, min=1.0).item()
        torch.testing.assert_close(
            updates["big"], updates_no["big"] / expected_scale, atol=1e-6, rtol=0
        )
        torch.testing.assert_close(
            updates["small"], updates_no["small"] / expected_scale, atol=1e-6, rtol=0
        )
        assert expected_scale > 1.0
        assert not torch.allclose(updates["small"], updates_no["small"])


# ---------------------------------------------------------------------------
# Convergence
# ---------------------------------------------------------------------------


class TestConvergence:
    def test_vanilla_minimises_quadratic(self):
        target = torch.tensor([1.0, 2.0, 3.0])
        params = {"x": torch.zeros(3)}
        step, state = adamw(params, lr=0.05, weight_decay=0.0)
        for _ in range(200):
            grads = {"x": 2.0 * (params["x"] - target)}
            updates, state = step(grads, state, params=params)
            params = apply_updates(params, updates)
        torch.testing.assert_close(params["x"], target, atol=0.05, rtol=0)

    def test_bc_minimises_quadratic(self):
        target = torch.tensor([1.0, 2.0, 3.0])
        params = {"x": torch.zeros(3)}
        step, state = adamw(
            params, lr=0.05, weight_decay=0.0, noise_bias_correction=True
        )
        for _ in range(200):
            grads = {"x": 2.0 * (params["x"] - target)}
            updates, state = step(
                noised(grads, max_norm=1.0, noise_stddev=0.01),
                state,
                params=params,
            )
            params = apply_updates(params, updates)
        torch.testing.assert_close(params["x"], target, atol=0.1, rtol=0)


# ---------------------------------------------------------------------------
# Callable schedules
# ---------------------------------------------------------------------------


class TestSchedule:
    def test_callable_schedule_receives_zero_indexed_steps(self, params, grads):
        calls = []

        def schedule(step):
            calls.append(step)
            return 1e-2 / (step + 1)

        step, state = adamw(params, lr=schedule, weight_decay=0.0)
        for _ in range(3):
            _, state = step(
                {k: v.clone() for k, v in grads.items()}, state, params=params
            )

        assert calls == [0, 1, 2]

    def test_callable_schedule_lr_is_applied(self):
        params = {"w": torch.ones(2)}
        grads = {"w": torch.zeros(2)}
        calls = []

        def schedule(step):
            calls.append(step)
            return 0.1 * (step + 1)

        step, state = adamw(params, lr=schedule, weight_decay=0.1)
        for i in range(3):
            updates, state = step(grads, state, params=params)
            expected = -schedule(i) * 0.1 * params["w"]
            torch.testing.assert_close(updates["w"], expected)

    def test_adam_alias_uses_same_schedule_indexing(self, params, grads):
        calls = []

        def schedule(step):
            calls.append(step)
            return 1e-3

        step, state = adam(params, lr=schedule, weight_decay=0.0)
        for _ in range(3):
            _, state = step(
                {k: v.clone() for k, v in grads.items()}, state, params=params
            )

        assert calls == [0, 1, 2]


# ---------------------------------------------------------------------------
# Serialization
# ---------------------------------------------------------------------------


class TestSerialization:
    def test_adam_state_round_trips(self, params, grads):
        step, state = adamw(
            params, lr=1e-3, weight_decay=0.01, noise_bias_correction=True
        )
        _, state = step(
            noised(
                {k: v.clone() for k, v in grads.items()}, max_norm=1.0, noise_stddev=0.5
            ),
            state,
            params=params,
        )
        sd = state_dict(state)
        restored = from_state_dict(state, sd)
        assert isinstance(restored, AdamState)
        assert restored.step == state.step
        assert restored.phi == state.phi
        for k in params:
            torch.testing.assert_close(restored.mu[k], state.mu[k])
            torch.testing.assert_close(restored.nu[k], state.nu[k])


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


class TestValidation:
    def test_negative_weight_decay_raises(self, params):
        with pytest.raises(ValueError, match="non-negative"):
            adamw(params, weight_decay=-1.0)

    def test_zero_eps_raises(self, params):
        with pytest.raises(ValueError, match="positive"):
            adamw(params, eps=0.0)

    def test_invalid_beta_raises(self, params):
        with pytest.raises(ValueError, match="beta_1"):
            adamw(params, betas=(1.0, 0.999))
        with pytest.raises(ValueError, match="beta_2"):
            adamw(params, betas=(0.9, 1.5))

    def test_zero_rms_clip_raises(self, params):
        with pytest.raises(ValueError, match="update_rms_clip"):
            adamw(params, update_rms_clip=0.0)
