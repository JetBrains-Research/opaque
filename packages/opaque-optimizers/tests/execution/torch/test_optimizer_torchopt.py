"""TorchOpt parity checks for non-Adam optimizer families."""

from __future__ import annotations

import pytest
import torch

from opaque.optimizers import adagrad, apply_updates, rmsprop, sgd

torchopt = pytest.importorskip("torchopt")


@pytest.fixture
def params():
    torch.manual_seed(0)
    return {"weight": torch.randn(4, 3), "bias": torch.randn(3)}


@pytest.fixture
def grads(params):
    torch.manual_seed(1)
    return {k: torch.randn_like(v) for k, v in params.items()}


class TestSGDTorchOpt:
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


class TestAdagradTorchOpt:
    @pytest.mark.parametrize(
        "kwargs",
        [
            {"lr": 1e-2, "eps": 1e-10, "initial_accumulator_value": 0.0},
            {"lr": 5e-3, "eps": 1e-8, "initial_accumulator_value": 0.1},
            {"lr": 0.1, "eps": 1e-6, "initial_accumulator_value": 1.0},
        ],
        ids=["default", "warm_accumulator", "large_accumulator"],
    )
    def test_matches_torchopt_adagrad(self, params, kwargs):
        params_opaque = {k: v.clone() for k, v in params.items()}
        params_ref = {k: v.clone() for k, v in params.items()}
        step, state = adagrad(params_opaque, **kwargs)
        opt_ref = torchopt.adagrad(**kwargs)
        state_ref = opt_ref.init(params_ref)

        torch.manual_seed(42)
        for _ in range(10):
            step_grads = {k: torch.randn_like(v) for k, v in params.items()}
            updates, state = step(step_grads, state, params=params_opaque)
            updates_ref, state_ref = opt_ref.update(
                step_grads, state_ref, params=params_ref
            )
            for name in params:
                torch.testing.assert_close(updates[name], updates_ref[name])
            params_opaque = apply_updates(params_opaque, updates)
            params_ref = torchopt.apply_updates(params_ref, updates_ref, inplace=False)
            for name in params:
                torch.testing.assert_close(params_opaque[name], params_ref[name])


class TestRMSpropTorchOpt:
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
        step, state = rmsprop(params, **kwargs)
        opt_ref = torchopt.rmsprop(**kwargs)
        state_ref = opt_ref.init(params)

        torch.manual_seed(42)
        for _ in range(10):
            step_grads = {k: torch.randn_like(v) for k, v in params.items()}
            updates, state = step(step_grads, state, params=params)
            updates_ref, state_ref = opt_ref.update(
                step_grads, state_ref, params=params
            )
            for name in params:
                torch.testing.assert_close(updates[name], updates_ref[name])
