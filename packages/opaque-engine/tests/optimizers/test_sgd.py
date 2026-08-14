"""Tests for opaque.api.engine.optimizers._sgd.

Mirrors the optimizer-behavior coverage from opaque-optimizers/tests/test_sgd.py
against the Torch provider.
"""

from __future__ import annotations

import pytest
import torch

from opaque.api.engine.optimizers import sgd
from opaque.api.engine.optimizers.types import SGDState
from opaque.serialization import from_state_dict, state_dict
from opaque.types import (
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


class TestSGD:
    def test_raw_pytree_matches_torchopt(self, params, grads):
        step, state = sgd(params, lr=1e-2, momentum=0.9, weight_decay=0.01)
        ref = torchopt.sgd(lr=1e-2, momentum=0.9, weight_decay=0.01)
        ref_state = ref.init(params)

        updates, state = step(
            {k: v.clone() for k, v in grads.items()}, state, params=params
        )
        ref_updates, ref_state = ref.update(
            {k: v.clone() for k, v in grads.items()}, ref_state, params=params
        )

        for name in updates:
            torch.testing.assert_close(updates[name], ref_updates[name])

    def test_noisy_pytree_unwraps_without_warning(self, params, grads):
        step, state = sgd(params, lr=1e-2)
        updates, _ = step(
            noised(grads, max_norm=1.0, noise_stddev=0.25),
            state,
            params=params,
        )
        for name in updates:
            assert updates[name].shape == params[name].shape

    def test_clipped_updates_are_rejected(self, params, grads):
        step, state = sgd(params, lr=1e-2)
        with pytest.raises(
            TypeError, match="have not passed through a noise mechanism"
        ):
            step(clipped(grads, max_norm=1.0), state, params=params)

    def test_explicit_metadata_kwargs_are_rejected(self, params, grads):
        step, state = sgd(params, lr=1e-2)
        with pytest.raises(TypeError, match="noise_stddev"):
            step(grads, state, params=params, noise_stddev=0.5)
        with pytest.raises(TypeError, match="noisy_squared_grads"):
            step(grads, state, params=params, noisy_squared_grads={})

    def test_second_moment_output_uses_first_stream_silently(self, params, grads):
        step, state = sgd(params, lr=1e-2)
        sq = {name: value.square() for name, value in grads.items()}
        output = SecondMomentNoiseOutput(
            noised(grads, max_norm=1.0, noise_stddev=0.1),
            noised(sq, max_norm=1.0, noise_stddev=0.1),
        )
        updates, _ = step(output, state, params=params)
        for name in updates:
            assert updates[name].shape == params[name].shape

    @pytest.mark.parametrize(
        ("momentum", "dampening", "nesterov", "maximize", "weight_decay"),
        [
            (0.0, 0.0, False, False, 0.0),
            (0.9, 0.0, False, False, 0.01),
            (0.9, 0.0, True, False, 0.0),
            (0.9, 0.0, False, True, 0.01),
            (0.9, 0.1, False, False, 0.0),
        ],
    )
    def test_variants_match_torchopt(
        self, params, grads, momentum, dampening, nesterov, maximize, weight_decay
    ):
        kwargs = {
            "lr": 1e-2,
            "momentum": momentum,
            "dampening": dampening,
            "weight_decay": weight_decay,
            "nesterov": nesterov,
            "maximize": maximize,
        }
        step, state = sgd(params, **kwargs)
        ref = torchopt.sgd(**kwargs)
        ref_state = ref.init(params)

        g_new = {k: v.clone() for k, v in grads.items()}
        g_ref = {k: v.clone() for k, v in grads.items()}
        updates, state = step(g_new, state, params=params)
        ref_updates, ref_state = ref.update(g_ref, ref_state, params=params)

        for name in updates:
            torch.testing.assert_close(updates[name], ref_updates[name])

    def test_multi_step_momentum_buffer_persists(self, params, grads):
        step, state = sgd(params, lr=1e-2, momentum=0.9)
        ref = torchopt.sgd(lr=1e-2, momentum=0.9)
        ref_state = ref.init(params)

        for _ in range(5):
            g_new = {k: v.clone() for k, v in grads.items()}
            g_ref = {k: v.clone() for k, v in grads.items()}
            updates, state = step(g_new, state, params=params)
            ref_updates, ref_state = ref.update(g_ref, ref_state, params=params)
            for name in updates:
                torch.testing.assert_close(updates[name], ref_updates[name])


class TestSchedule:
    def test_callable_schedule_receives_zero_indexed_steps(self, params, grads):
        calls = []

        def schedule(step):
            calls.append(step)
            return 1e-2 / (step + 1)

        step, state = sgd(params, lr=schedule, momentum=0.0, weight_decay=0.0)
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

        step, state = sgd(params, lr=schedule, momentum=0.0, weight_decay=0.1)
        for i in range(3):
            updates, state = step(grads, state, params=params)
            expected = -schedule(i) * 0.1 * params["w"]
            torch.testing.assert_close(updates["w"], expected)


class TestSerialization:
    def test_sgd_state_round_trips(self, params, grads):
        step, state = sgd(params, lr=1e-2, momentum=0.9)
        _, state = step({k: v.clone() for k, v in grads.items()}, state, params=params)
        sd = state_dict(state)
        restored = from_state_dict(state, sd)
        assert isinstance(restored, SGDState)
        assert restored.step == state.step
        for k in params:
            torch.testing.assert_close(restored.momentum[k], state.momentum[k])
