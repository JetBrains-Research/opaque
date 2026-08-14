"""Tests for opaque.optimizers.ademamix."""

from __future__ import annotations

import pytest
import torch

from opaque.optimizers import adamw, ademamix
from opaque.optimizers.types import AdEMAMixState
from opaque.types import (
    SecondMomentNoiseOutput,
    noised,
)


@pytest.fixture
def params():
    torch.manual_seed(0)
    return {"weight": torch.randn(4, 3), "bias": torch.randn(3)}


@pytest.fixture
def grads(params):
    torch.manual_seed(1)
    return {k: torch.randn_like(v) for k, v in params.items()}


class TestVanillaAdEMAMix:
    def test_state_has_two_first_moments(self, params):
        _step, st = ademamix(params, lr=1e-3)
        assert isinstance(st, AdEMAMixState)
        for k in params:
            assert torch.equal(st.m_fast[k], torch.zeros_like(params[k]))
            assert torch.equal(st.m_slow[k], torch.zeros_like(params[k]))
            assert torch.equal(st.nu[k], torch.zeros_like(params[k]))

    def test_emas_advance_with_distinct_betas(self, params, grads):
        b1, b2, b3 = 0.9, 0.999, 0.9999
        step, state = ademamix(
            params, lr=1e-3, betas=(b1, b2, b3), alpha=5.0, weight_decay=0.0
        )
        _, state = step(grads, state, params=params)
        for k in grads:
            torch.testing.assert_close(state.m_fast[k], (1 - b1) * grads[k])
            torch.testing.assert_close(state.m_slow[k], (1 - b3) * grads[k])
            torch.testing.assert_close(state.nu[k], (1 - b2) * grads[k] * grads[k])

    def test_alpha_zero_reduces_to_adam(self, params, grads):
        """With α=0 and matched β's, AdEMAMix's update equals AdamW's."""
        step_ame, s_ame = ademamix(
            params,
            lr=1e-3,
            betas=(0.9, 0.999, 0.9999),
            alpha=0.0,
            weight_decay=0.01,
        )
        step_adam, s_adam = adamw(
            params,
            lr=1e-3,
            betas=(0.9, 0.999),
            weight_decay=0.01,
        )
        torch.manual_seed(7)
        for _ in range(5):
            g = {k: torch.randn_like(v) for k, v in params.items()}
            u_ame, s_ame = step_ame(g, s_ame, params=params)
            u_adam, s_adam = step_adam(g, s_adam, params=params)
            for k in params:
                torch.testing.assert_close(u_ame[k], u_adam[k])


class TestBCMode:
    def test_phi_advances_under_noisy_metadata(self, params, grads):
        b2 = 0.999
        sigma = 0.4
        step, state = ademamix(
            params, lr=1e-3, betas=(0.9, b2, 0.9999), noise_bias_correction=True
        )
        expected_phi = 0.0
        for _ in range(8):
            _, state = step(
                noised(grads, max_norm=1.0, noise_stddev=sigma),
                state,
                params=params,
            )
            expected_phi = b2 * expected_phi + (1 - b2) * (sigma**2)
        phi = state.phi
        assert isinstance(phi, dict)
        assert all(v == pytest.approx(expected_phi) for v in phi.values())

    def test_bc_flag_disables_noisy_metadata_correction(self, params, grads):
        step, state = ademamix(params, lr=1e-3, noise_bias_correction=False)
        _, state = step(
            noised(grads, max_norm=1.0, noise_stddev=0.4),
            state,
            params=params,
        )
        assert state.phi == 0.0


class TestSecondMomentMode:
    @pytest.fixture
    def sq_grads(self, grads):
        return {k: v.pow(2) + 0.05 for k, v in grads.items()}

    def test_consumes_external_g_squared(self, params, grads, sq_grads):
        b2 = 0.999
        step, state = ademamix(params, lr=1e-3, betas=(0.9, b2, 0.9999))
        output = SecondMomentNoiseOutput(
            noised(grads, max_norm=1.0, noise_stddev=0.1),
            noised(sq_grads, max_norm=1.0, noise_stddev=0.1),
        )
        _, state = step(output, state, params=params)
        for k in params:
            torch.testing.assert_close(state.nu[k], (1 - b2) * sq_grads[k])

    def test_negative_squared_stream_bounded(self, params, grads):
        sq = {k: -torch.ones_like(v) for k, v in grads.items()}
        step, state = ademamix(params, lr=1e-3)
        output = SecondMomentNoiseOutput(
            noised(grads, max_norm=1.0, noise_stddev=0.1),
            noised(sq, max_norm=1.0, noise_stddev=0.1),
        )
        updates, _ = step(output, state, params=params)
        for k in params:
            assert torch.isfinite(updates[k]).all()
            assert updates[k].abs().max() < 10.0

    def test_explicit_second_moment_kwarg_rejected(self, params, grads):
        step, state = ademamix(params, lr=1e-3)
        with pytest.raises(TypeError, match="noisy_squared_grads"):
            step(
                grads,
                state,
                params=params,
                noisy_squared_grads=grads,
            )


class TestValidation:
    def test_three_betas_required(self, params):
        with pytest.raises(ValueError, match="invalid AdEMAMix|betas"):
            ademamix(params, betas=(0.9, 0.999))

    def test_negative_alpha_raises(self, params):
        with pytest.raises(ValueError, match="invalid AdEMAMix|alpha"):
            ademamix(params, alpha=-1.0)
