"""Portable DP-SGD numerical and state coverage."""

from __future__ import annotations

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
from opaque.random import key
from opaque.serialization import from_state_dict, state_dict
from opaque.types import clipped


class _PortableRuntime:
    def __init__(self, backend_case) -> None:
        self._backend_case = backend_case

    @property
    def float32(self):
        return self._backend_case.dtype("float32")

    def array(self, value, *, dtype=None):
        return self._backend_case.array(value, dtype=dtype)

    def zeros(self, shape, *, dtype=None):
        return self.array(np.zeros(shape), dtype=dtype)

    def mean(self, value):
        return value.mean()

    def square(self, value):
        return value * value

    def eval(self, *_values) -> None:
        return None


mx: _PortableRuntime


@pytest.fixture
def portable_backend(backend_case):
    global mx
    mx = _PortableRuntime(backend_case)
    return backend_case


def _batch() -> tuple[mx.array, mx.array]:
    return (
        mx.array([[1.0, -1.0], [2.0, 1.0], [-1.0, 2.0], [0.5, -0.5]]),
        mx.array([[1.0], [0.0], [1.0], [-1.0]]),
    )


def _parameters() -> dict[str, mx.array]:
    return {
        "weight": mx.array([[0.25], [-0.5]], dtype=mx.float32),
        "bias": mx.array([0.1], dtype=mx.float32),
    }


def _loss(params, features, targets):
    return mx.mean(mx.square(features @ params["weight"] + params["bias"] - targets))


def _assert_tree_close(left, right, *, rtol: float = 1e-5) -> None:
    for name in left:
        np.testing.assert_allclose(
            ops.to_host(left[name]), ops.to_host(right[name]), rtol=rtol, atol=1e-6
        )


@pytest.mark.parametrize("mode", ["fixed", "auto", "adaptive"])
def test_each_clipping_mode_runs_on_portable_arrays(
    mode: str, portable_backend
) -> None:
    params = _parameters()
    features, targets = _batch()

    if mode == "fixed":
        grad_fn, clip_state = clipped_grad(
            _loss,
            clipping_norm=1.0,
            normalize_by=4.0,
            batch_argnums=(1, 2),
        )
    elif mode == "auto":
        grad_fn, clip_state = auto_clipped_grad(
            _loss,
            R=1.0,
            normalize_by=4.0,
            batch_argnums=(1, 2),
        )
    else:
        grad_fn, clip_state = adaptive_clipped_grad(
            _loss,
            initial_clipping_norm=1.0,
            normalize_by=4.0,
            batch_argnums=(1, 2),
            key=key(7),
        )

    clipped_grads, clip_state = grad_fn(params, features, targets, state=clip_state)
    assert clipped_grads.max_norm > 0.0
    assert clipped_grads.pytree["weight"].shape == params["weight"].shape

    noise_fn, noise_state = gaussian_noise(noise_multiplier=0.25, key=key(11))
    noised_grads, noise_state = noise_fn(clipped_grads, noise_state)

    assert noise_state._step_counter == 1
    assert not np.allclose(
        ops.to_host(noised_grads.pytree["weight"]),
        ops.to_host(clipped_grads.pytree["weight"]),
    )


def test_compiled_clipping_closure_matches_eager(portable_backend) -> None:
    params = _parameters()
    features, targets = _batch()
    grad_fn, clip_state = clipped_grad(
        _loss,
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


def test_noise_modes_and_resume_use_portable_pytrees(portable_backend) -> None:
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


def test_adaptive_dpsgd_checkpoint_replays_next_step(portable_backend) -> None:
    params = _parameters()
    features, targets = _batch()
    grad_fn, clip_state = adaptive_clipped_grad(
        _loss,
        initial_clipping_norm=1.0,
        normalize_by=4.0,
        batch_argnums=(1, 2),
        key=key(31),
    )
    noise_fn, noise_state = gaussian_noise(noise_multiplier=0.25, key=key(37))

    def step(current_clip_state, current_noise_state):
        clipped_grads, current_clip_state = grad_fn(
            params, features, targets, state=current_clip_state
        )
        noised_grads, current_noise_state = noise_fn(clipped_grads, current_noise_state)
        return (
            noised_grads.pytree,
            current_clip_state,
            current_noise_state,
        )

    first = step(clip_state, noise_state)
    checkpoint = state_dict(
        {
            "clip_state": first[1],
            "noise_state": first[2],
        }
    )
    uninterrupted = step(first[1], first[2])
    restored = from_state_dict(
        {
            "clip_state": first[1],
            "noise_state": first[2],
        },
        checkpoint,
    )
    resumed = step(restored["clip_state"], restored["noise_state"])

    _assert_tree_close(uninterrupted[0], resumed[0], rtol=0.0)
    assert uninterrupted[1] == resumed[1]
    assert uninterrupted[2] == resumed[2]


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
