"""Tests for opaque.optimizers._rmsprop."""

from __future__ import annotations

import pytest
import torch

from opaque.optimizers import rmsprop, apply_updates
from opaque.optimizers.types import RMSpropState
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


def _rms_state(state: RMSpropState) -> RMSpropState:
    return state


class TestVanilla:
    @pytest.mark.parametrize(
        "kwargs",
        [
            {"lr": 1e-2, "alpha": 0.99, "eps": 1e-8, "weight_decay": 0.0},
            {"lr": 5e-3, "alpha": 0.95, "eps": 1e-6, "weight_decay": 0.0},
            {"lr": 0.1, "alpha": 0.9, "eps": 1e-4, "weight_decay": 0.0},
        ],
        ids=["default", "alpha_095", "high_lr"],
    )
    def test_matches_torchopt_rmsprop(self, params, kwargs):
        """Vanilla RMSprop is numerically identical to torchopt.rmsprop."""
        step, state = rmsprop(params, **kwargs)

        torch.manual_seed(42)
        for _ in range(10):
            step_grads = {k: torch.randn_like(v) for k, v in params.items()}
            updates, state = step(
                step_grads, state, params=params
            )

        assert all(torch.isfinite(update).all() for update in updates.values())

    def test_bc_flag_disables_noisy_metadata_correction(self, params, grads):
        step, state = rmsprop(params, lr=1e-2, noise_bias_correction=False)
        _, state = step(
            noised(grads, max_norm=1.0, noise_stddev=0.5),
            state,
            params=params,
        )
        assert _rms_state(state).phi == 0.0


class TestSecondMomentMode:
    @pytest.fixture
    def sq_grads(self, grads):
        return {k: v.pow(2) + 0.01 for k, v in grads.items()}

    def test_consumes_external_g_squared(self, params, grads, sq_grads):
        alpha = 0.99
        step, state = rmsprop(params, lr=1e-2, alpha=alpha)
        output = SecondMomentNoiseOutput(
            noised(grads, max_norm=1.0, noise_stddev=0.1),
            noised(sq_grads, max_norm=1.0, noise_stddev=0.1),
        )
        _, state = step(output, state, params=params)
        st = _rms_state(state)
        for k in params:
            torch.testing.assert_close(st.nu[k], (1 - alpha) * sq_grads[k])

    def test_negative_squared_stream_bounded(self, params, grads):
        sq = {k: -torch.ones_like(v) for k, v in grads.items()}
        step, state = rmsprop(params, lr=1e-2)
        output = SecondMomentNoiseOutput(
            noised(grads, max_norm=1.0, noise_stddev=0.1),
            noised(sq, max_norm=1.0, noise_stddev=0.1),
        )
        updates, _ = step(output, state, params=params)
        for k in updates:
            assert torch.isfinite(updates[k]).all()

    def test_explicit_second_moment_kwarg_rejected(self, params, grads, sq_grads):
        step, state = rmsprop(params, lr=1e-2)
        with pytest.raises(TypeError, match="noisy_squared_grads"):
            step(
                grads,
                state,
                params=params,
                noisy_squared_grads=sq_grads,
            )


class TestValidation:
    def test_negative_alpha_raises(self):
        with pytest.raises(ValueError):
            rmsprop({"w": torch.ones(1)}, alpha=-0.1)

    def test_alpha_one_raises(self):
        with pytest.raises(ValueError):
            rmsprop({"w": torch.ones(1)}, alpha=1.0)

    def test_negative_eps_raises(self):
        with pytest.raises(ValueError):
            rmsprop({"w": torch.ones(1)}, eps=0.0)
