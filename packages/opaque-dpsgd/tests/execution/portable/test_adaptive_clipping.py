"""Provider-neutral adaptive-clipping and empty-batch contracts."""

from __future__ import annotations

import numpy as np
import pytest

from opaque.dpsgd.clipping import adaptive_clipped_grad, clipped_grad, per_group
from opaque.random import key
from opaque.serialization import from_state_dict, state_dict
from opaque.types import ClippedPytree, PerGroup


def _loss(params, features, targets):
    return ((features @ params - targets) ** 2).mean()


def _inputs(backend_case, size: int = 8):
    dtype = backend_case.dtype("float32")
    return (
        backend_case.array(np.linspace(-0.5, 0.5, 3), dtype=dtype),
        backend_case.array(
            np.arange(size * 3, dtype=np.float32).reshape(size, 3) / 10, dtype=dtype
        ),
        backend_case.array(np.linspace(-1.0, 1.0, size), dtype=dtype),
    )


def test_adaptive_clipping_updates_state_deterministically(backend_case) -> None:
    params, features, targets = _inputs(backend_case)
    first_fn, first_state = adaptive_clipped_grad(
        _loss,
        initial_clipping_norm=0.1,
        target_quantile=0.5,
        learning_rate=0.2,
        key=key(71),
        batch_argnums=(1, 2),
        return_stats=True,
    )
    second_fn, second_state = adaptive_clipped_grad(
        _loss,
        initial_clipping_norm=0.1,
        target_quantile=0.5,
        learning_rate=0.2,
        key=key(71),
        batch_argnums=(1, 2),
        return_stats=True,
    )
    (first, first_stats), first_state = first_fn(
        params, features, targets, state=first_state
    )
    (second, second_stats), second_state = second_fn(
        params, features, targets, state=second_state
    )

    assert isinstance(first, ClippedPytree)
    assert first.pytree.shape == params.shape
    assert first_state == second_state
    assert first_stats == second_stats
    np.testing.assert_array_equal(
        backend_case.to_host(first.pytree), backend_case.to_host(second.pytree)
    )


def test_adaptive_clipping_checkpoint_replays_next_state(backend_case) -> None:
    params, features, targets = _inputs(backend_case)
    grad_fn, state = adaptive_clipped_grad(
        _loss,
        initial_clipping_norm=1.0,
        fraction_noise_std=0.1,
        key=key(73),
        batch_argnums=(1, 2),
    )
    _, state = grad_fn(params, features, targets, state=state)
    checkpoint = state_dict(state)
    uninterrupted, uninterrupted_state = grad_fn(params, features, targets, state=state)
    _, template = adaptive_clipped_grad(
        _loss,
        initial_clipping_norm=1.0,
        fraction_noise_std=0.1,
        key=key(999),
        batch_argnums=(1, 2),
    )
    restored = from_state_dict(template, checkpoint)
    resumed, resumed_state = grad_fn(params, features, targets, state=restored)

    np.testing.assert_array_equal(
        backend_case.to_host(uninterrupted.pytree), backend_case.to_host(resumed.pytree)
    )
    assert uninterrupted_state == resumed_state


def test_adaptive_per_group_clipping_preserves_group_state(backend_case) -> None:
    dtype = backend_case.dtype("float32")
    params = {
        "left": backend_case.array([0.25, -0.5], dtype=dtype),
        "right": backend_case.array([0.1], dtype=dtype),
    }
    features = backend_case.array(
        np.arange(16, dtype=np.float32).reshape(8, 2) / 10, dtype=dtype
    )
    targets = backend_case.array(np.linspace(-1.0, 1.0, 8), dtype=dtype)

    def grouped_loss(current, x, y):
        return ((x @ current["left"] + current["right"] - y) ** 2).mean()

    maximum = per_group(params, left=1.0, right=0.5)
    grad_fn, state = adaptive_clipped_grad(
        grouped_loss,
        initial_clipping_norm=maximum,
        key=key(79),
        batch_argnums=(1, 2),
        return_aux=True,
    )
    (output, _), state = grad_fn(params, features, targets, state=state)

    assert isinstance(state._current_clipping_norm, PerGroup)
    assert isinstance(state._next_clipping_norm, PerGroup)
    assert set(output.pytree) == {"left", "right"}


@pytest.mark.parametrize("factory", [clipped_grad, adaptive_clipped_grad])
def test_clipping_empty_batch_returns_zero_gradient(backend_case, factory) -> None:
    params, _, _ = _inputs(backend_case)
    empty_features = backend_case.array(
        np.empty((0, 3), dtype=np.float32), dtype=backend_case.dtype("float32")
    )
    empty_targets = backend_case.array(
        np.empty((0,), dtype=np.float32), dtype=backend_case.dtype("float32")
    )
    kwargs = {"batch_argnums": (1, 2)}
    if factory is clipped_grad:
        kwargs["clipping_norm"] = 1.0
    else:
        kwargs.update(initial_clipping_norm=1.0, key=key(83))
    grad_fn, state = factory(_loss, **kwargs)
    output, _ = grad_fn(params, empty_features, empty_targets, state=state)

    assert isinstance(output, ClippedPytree)
    np.testing.assert_array_equal(backend_case.to_host(output.pytree), np.zeros(3))


def test_empty_batch_preserves_auxiliary_contract_and_clipping_state(
    backend_case,
) -> None:
    params, _, _ = _inputs(backend_case)
    empty_features = backend_case.array(
        np.empty((0, 3), dtype=np.float32), dtype=backend_case.dtype("float32")
    )
    empty_targets = backend_case.array(
        np.empty(0, dtype=np.float32), dtype=backend_case.dtype("float32")
    )
    grad_fn, state = clipped_grad(
        _loss,
        batch_argnums=(1, 2),
        clipping_norm=1.0,
        microbatch_size=4,
        return_aux=True,
    )
    (output, auxiliary), next_state = grad_fn(
        params, empty_features, empty_targets, state=state
    )

    assert next_state is state
    np.testing.assert_array_equal(backend_case.to_host(output.pytree), np.zeros(3))
    assert auxiliary.grad_norms.shape == (0,)
    assert auxiliary.loss_values.shape == (0,)
    assert auxiliary.clipped_grad_norms.shape == (0,)
    assert auxiliary.clipping_rate == 0.0
    assert auxiliary.batch_size == 0


def test_adaptive_empty_batch_preserves_threshold_then_adapts(backend_case) -> None:
    params, features, targets = _inputs(backend_case)
    empty_features = backend_case.array(
        np.empty((0, 3), dtype=np.float32), dtype=backend_case.dtype("float32")
    )
    empty_targets = backend_case.array(
        np.empty(0, dtype=np.float32), dtype=backend_case.dtype("float32")
    )
    grad_fn, state = adaptive_clipped_grad(
        _loss,
        initial_clipping_norm=1.0,
        key=key(101),
        batch_argnums=(1, 2),
        return_aux=True,
    )
    current = state._current_clipping_norm
    (empty_output, auxiliary), state = grad_fn(
        params, empty_features, empty_targets, state=state
    )

    np.testing.assert_array_equal(
        backend_case.to_host(empty_output.pytree), np.zeros(3)
    )
    assert auxiliary.grad_norms.shape == (0,)
    assert state._current_clipping_norm == current
    assert state._batch_size == state._num_clipped == 0.0
    assert state._step == 1
    _, state = grad_fn(params, features, targets, state=state)
    assert state._batch_size > 0


def test_quantile_noise_is_keyed_and_updates_adaptive_state(backend_case) -> None:
    params, features, targets = _inputs(backend_case)
    first_fn, first_state = adaptive_clipped_grad(
        _loss,
        initial_clipping_norm=1.0,
        fraction_noise_std=0.1,
        key=key(107),
        batch_argnums=(1, 2),
        return_aux=True,
    )
    second_fn, second_state = adaptive_clipped_grad(
        _loss,
        initial_clipping_norm=1.0,
        fraction_noise_std=0.1,
        key=key(107),
        batch_argnums=(1, 2),
        return_aux=True,
    )
    for step in range(1, 3):
        (first_output, first_auxiliary), first_state = first_fn(
            params, features, targets, state=first_state
        )
        (second_output, second_auxiliary), second_state = second_fn(
            params, features, targets, state=second_state
        )
        np.testing.assert_array_equal(
            backend_case.to_host(first_output.pytree),
            backend_case.to_host(second_output.pytree),
        )
        assert first_auxiliary.clipping_rate == second_auxiliary.clipping_rate
        assert first_state == second_state
        assert first_state._rng_key == key(107)
        assert first_state._step == step


@pytest.mark.parametrize("fraction_noise_std", [0.0, -1.0])
def test_quantile_noise_validates_fraction_and_requires_key(
    backend_case, fraction_noise_std
) -> None:
    del backend_case
    with pytest.raises(TypeError, match="key"):
        adaptive_clipped_grad(_loss, fraction_noise_std=0.1, batch_argnums=(1, 2))
    with pytest.raises(ValueError, match="fraction_noise_std must be positive"):
        adaptive_clipped_grad(
            _loss,
            fraction_noise_std=fraction_noise_std,
            key=key(109),
            batch_argnums=(1, 2),
        )


def test_empty_batch_pytree_params_return_zero_gradients_per_leaf(
    backend_case,
) -> None:
    dtype = backend_case.dtype("float32")

    def grouped_loss(current, x, y):
        return ((x @ current["w"] + current["b"] - y) ** 2).mean()

    params = {
        "w": backend_case.array([0.5, -0.2], dtype=dtype),
        "b": backend_case.array(0.1, dtype=dtype),
    }
    empty_features = backend_case.array(np.empty((0, 2), dtype=np.float32), dtype=dtype)
    empty_targets = backend_case.array(np.empty(0, dtype=np.float32), dtype=dtype)
    grad_fn, state = clipped_grad(grouped_loss, batch_argnums=(1, 2), clipping_norm=1.0)
    output, _ = grad_fn(params, empty_features, empty_targets, state=state)

    assert isinstance(output.pytree, dict)
    np.testing.assert_array_equal(backend_case.to_host(output.pytree["w"]), np.zeros(2))
    np.testing.assert_array_equal(backend_case.to_host(output.pytree["b"]), 0.0)


def test_adaptive_consecutive_empty_batches_do_not_drift_clipping_norm(
    backend_case,
) -> None:
    params, _, _ = _inputs(backend_case)
    empty_features = backend_case.array(
        np.empty((0, 3), dtype=np.float32), dtype=backend_case.dtype("float32")
    )
    empty_targets = backend_case.array(
        np.empty(0, dtype=np.float32), dtype=backend_case.dtype("float32")
    )
    grad_fn, state = adaptive_clipped_grad(
        _loss,
        initial_clipping_norm=1.0,
        key=key(111),
        batch_argnums=(1, 2),
    )
    initial_cn = state._current_clipping_norm

    for _ in range(10):
        _, state = grad_fn(params, empty_features, empty_targets, state=state)

    assert state._current_clipping_norm == initial_cn
    assert state._next_clipping_norm == initial_cn
    assert state._step == 10
