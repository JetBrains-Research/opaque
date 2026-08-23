"""Native MLX DP-SGD integration coverage."""

from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn
import numpy as np
import pytest

from opaque import ops
from opaque.dpsgd.clipping import (
    adaptive_clipped_grad,
    auto_clipped_grad,
    clipped_grad,
    per_group,
)
from opaque.dpsgd.noise import gaussian_noise
from opaque.dpsgd.sampling import PoissonSampler
from opaque.execution import compile as opaque_compile
from opaque.mlx.functional import make_functional
from opaque.optimizers import adamw, apply_updates, sgd
from opaque.precision import loss_scaler
from opaque.random import key
from opaque.scheduling import linear_schedule
from opaque.serialization import from_state_dict, state_dict
from opaque.types import clipped


def _batch() -> tuple[mx.array, mx.array]:
    return (
        mx.array([[1.0, -1.0], [2.0, 1.0], [-1.0, 2.0], [0.5, -0.5]]),
        mx.array([[1.0], [0.0], [1.0], [-1.0]]),
    )


def _loss(functional_module):
    def loss(params, features, targets):
        return mx.mean(mx.square(functional_module(params, features) - targets))

    return loss


def _assert_tree_close(left, right, *, rtol: float = 1e-5) -> None:
    for name in left:
        np.testing.assert_allclose(
            ops.to_host(left[name]), ops.to_host(right[name]), rtol=rtol, atol=1e-6
        )


@pytest.mark.parametrize("mode", ["fixed", "auto", "adaptive"])
def test_native_mlx_module_runs_each_clipping_mode(mode: str) -> None:
    module = nn.Linear(2, 1)
    module_parameters = {
        name: ops.to_host(value) for name, value in module.parameters().items()
    }
    functional_module, params = make_functional(module)
    features, targets = _batch()
    loss = _loss(functional_module)

    if mode == "fixed":
        grad_fn, clip_state = clipped_grad(
            loss,
            clipping_norm=1.0,
            normalize_by=4.0,
            batch_argnums=(1, 2),
        )
    elif mode == "auto":
        grad_fn, clip_state = auto_clipped_grad(
            loss,
            R=1.0,
            normalize_by=4.0,
            batch_argnums=(1, 2),
        )
    else:
        grad_fn, clip_state = adaptive_clipped_grad(
            loss,
            initial_clipping_norm=1.0,
            normalize_by=4.0,
            batch_argnums=(1, 2),
            key=key(7),
        )

    clipped_grads, clip_state = grad_fn(params, features, targets, state=clip_state)
    assert clipped_grads.max_norm > 0.0
    assert clipped_grads.pytree["weight"].shape == params["weight"].shape

    noise_fn, noise_state = gaussian_noise(noise_multiplier=0.0, key=key(11))
    noised_grads, noise_state = noise_fn(clipped_grads, noise_state)
    optimizer_step, optimizer_state = adamw(params, lr=0.1)
    updates, optimizer_state = optimizer_step(
        noised_grads, optimizer_state, params=params
    )
    updated_params = apply_updates(params, updates)
    mx.eval(updated_params)

    assert optimizer_state.step == 1
    assert noise_state._step_counter == 1
    assert not np.allclose(
        ops.to_host(updated_params["weight"]), ops.to_host(params["weight"])
    )
    for name, initial_value in module_parameters.items():
        np.testing.assert_array_equal(
            ops.to_host(module.parameters()[name]), initial_value
        )


def test_compiled_mlx_clipping_closure_matches_eager() -> None:
    module = nn.Linear(2, 1)
    functional_module, params = make_functional(module)
    features, targets = _batch()
    grad_fn, clip_state = clipped_grad(
        _loss(functional_module),
        clipping_norm=1.0,
        normalize_by=4.0,
        batch_argnums=(1, 2),
    )

    eager, _ = grad_fn(params, features, targets, state=clip_state)
    compiled = opaque_compile(
        lambda explicit_params, batch_features, batch_targets: (
            grad_fn(explicit_params, batch_features, batch_targets, state=clip_state)[
                0
            ].pytree
        )
    )
    compiled_grads = compiled(params, features, targets)
    mx.eval(compiled_grads)

    _assert_tree_close(compiled_grads, eager.pytree)


def test_mlx_noise_modes_and_resume_use_native_pytrees() -> None:
    gradients = {
        "weight": mx.zeros((128,), dtype=mx.float32),
        "bias": mx.zeros((1,), dtype=mx.float32),
    }
    grouped_norm = per_group(gradients, weight=1.0, bias=0.5)
    grouped = clipped(gradients, max_norm=grouped_norm)

    noise_fn, initial_state = gaussian_noise(noise_multiplier=1.0, key=key(17))
    first, saved_state = noise_fn(grouped, initial_state)
    assert set(first.noise_stddev.values) == {"weight", "bias"}
    assert all(stddev > 0.0 for stddev in first.noise_stddev.values.values())
    assert first.noise_stddev.values["weight"] != first.noise_stddev.values["bias"]
    assert first.pytree["weight"].dtype == mx.float32
    assert np.any(ops.to_host(first.pytree["weight"]) != 0.0)

    continued, _ = noise_fn(grouped, saved_state)
    _, restore_template = gaussian_noise(noise_multiplier=1.0, key=key(17))
    restored_state = from_state_dict(restore_template, state_dict(saved_state))
    restored, _ = noise_fn(grouped, restored_state)
    _assert_tree_close(continued.pytree, restored.pytree, rtol=0.0)

    bounded_fn, bounded_state = gaussian_noise(
        noise_multiplier=1.0, bound=0.25, key=key(19)
    )
    bounded, _ = bounded_fn(clipped(gradients, max_norm=1.0), bounded_state)
    assert np.all(np.abs(ops.to_host(bounded.pytree["weight"])) <= 0.250001)


def test_mlx_loss_scaling_schedule_and_optimizer_state_restore() -> None:
    module = nn.Linear(2, 1)
    functional_module, params = make_functional(module)
    features, targets = _batch()
    unscaled_loss = _loss(functional_module)
    scaler, scaler_state = loss_scaler(init_scale=32.0, growth_interval=1)

    def scaled_loss(explicit_params, batch_features, batch_targets):
        return scaler.scale_loss(
            unscaled_loss(explicit_params, batch_features, batch_targets), scaler_state
        )

    unscaled_grad_fn, unscaled_state = clipped_grad(
        unscaled_loss,
        clipping_norm=1.0,
        normalize_by=4.0,
        batch_argnums=(1, 2),
    )
    scaled_grad_fn, scaled_state = clipped_grad(
        scaled_loss,
        clipping_norm=1.0,
        normalize_by=4.0,
        batch_argnums=(1, 2),
        pre_clipping_transform=lambda grads: scaler.unscale_grads(grads, scaler_state),
    )
    unscaled, _ = unscaled_grad_fn(params, features, targets, state=unscaled_state)
    scaled, _ = scaled_grad_fn(params, features, targets, state=scaled_state)
    _assert_tree_close(unscaled.pytree, scaled.pytree)

    noise_fn, noise_state = gaussian_noise(noise_multiplier=0.0, key=key(23))
    noised_grads, _ = noise_fn(unscaled, noise_state)
    schedule = linear_schedule(0.2, 0.0, transition_steps=2)
    optimizer_step, optimizer_state = adamw(params, lr=schedule)
    updates, optimizer_state = optimizer_step(
        noised_grads, optimizer_state, params=params
    )
    updated_params = apply_updates(params, updates)

    restored_step, restored_template = adamw(params, lr=schedule)
    restored_state = from_state_dict(restored_template, state_dict(optimizer_state))
    continued_updates, _ = optimizer_step(
        noised_grads, optimizer_state, params=updated_params
    )
    restored_updates, _ = restored_step(
        noised_grads, restored_state, params=updated_params
    )
    _assert_tree_close(continued_updates, restored_updates, rtol=0.0)

    scheduled_step, scheduled_state = sgd(params, lr=schedule)
    first_updates, scheduled_state = scheduled_step(
        noised_grads, scheduled_state, params=params
    )
    second_updates, _ = scheduled_step(noised_grads, scheduled_state, params=params)
    _assert_tree_close(
        {"weight": ops.multiply(first_updates["weight"], 0.5)},
        {"weight": second_updates["weight"]},
    )


def test_mlx_adaptive_dpsgd_checkpoint_replays_next_step() -> None:
    module = nn.Linear(2, 1)
    functional_module, params = make_functional(module)
    features, targets = _batch()
    grad_fn, clip_state = adaptive_clipped_grad(
        _loss(functional_module),
        initial_clipping_norm=1.0,
        normalize_by=4.0,
        batch_argnums=(1, 2),
        key=key(31),
    )
    noise_fn, noise_state = gaussian_noise(noise_multiplier=0.25, key=key(37))
    optimizer_step, optimizer_state = adamw(params, lr=0.1)

    def step(
        current_params, current_clip_state, current_noise_state, current_opt_state
    ):
        clipped_grads, current_clip_state = grad_fn(
            current_params, features, targets, state=current_clip_state
        )
        noised_grads, current_noise_state = noise_fn(clipped_grads, current_noise_state)
        updates, current_opt_state = optimizer_step(
            noised_grads, current_opt_state, params=current_params
        )
        return (
            apply_updates(current_params, updates),
            current_clip_state,
            current_noise_state,
            current_opt_state,
        )

    first = step(params, clip_state, noise_state, optimizer_state)
    checkpoint = state_dict(
        {
            "params": first[0],
            "clip_state": first[1],
            "noise_state": first[2],
            "optimizer_state": first[3],
        }
    )
    uninterrupted = step(*first)
    restored = from_state_dict(
        {
            "params": first[0],
            "clip_state": first[1],
            "noise_state": first[2],
            "optimizer_state": first[3],
        },
        checkpoint,
    )
    resumed = step(
        restored["params"],
        restored["clip_state"],
        restored["noise_state"],
        restored["optimizer_state"],
    )

    _assert_tree_close(uninterrupted[0], resumed[0], rtol=0.0)
    assert uninterrupted[1] == resumed[1]
    assert uninterrupted[2] == resumed[2]
    assert uninterrupted[3].step == resumed[3].step


def test_poisson_sampler_is_deterministic_and_truncates_batches() -> None:
    sampler = PoissonSampler(
        range(20), sample_rate=0.75, n_steps=4, truncated_batch_size=3, key=key(29)
    )
    batches = list(sampler)

    assert len(batches) == 4
    assert all(len(batch) <= 3 for batch in batches)
    assert batches == list(
        PoissonSampler(
            range(20),
            sample_rate=0.75,
            n_steps=4,
            truncated_batch_size=3,
            key=key(29),
        )
    )
