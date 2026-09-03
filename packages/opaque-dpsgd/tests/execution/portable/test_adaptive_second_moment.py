"""Provider-neutral second-moment adaptive-clipping and sync-passthrough contracts."""

from __future__ import annotations

import numpy as np
import pytest

from opaque.api.dpsgd.clipping._adaptive import (
    AdaptiveClippedGradAux,
    AdaptiveClipState,
)
from opaque.api.dpsgd.clipping._distributed import (
    sync_adaptive_clip_state,
    sync_adaptive_clipped_grad_aux,
)
from opaque.dpsgd.clipping import adaptive_clipped_grad
from opaque.dpsgd.noise import gaussian_noise
from opaque.random import key
from opaque.types import PerGroup, SecondMomentClippingOutput, SecondMomentNoiseOutput


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


def _empty_batch(backend_case):
    dtype = backend_case.dtype("float32")
    return (
        backend_case.array(np.empty((0, 3), dtype=np.float32), dtype=dtype),
        backend_case.array(np.empty(0, dtype=np.float32), dtype=dtype),
    )


def test_empty_batch_second_moment_returns_paired_zero_output(backend_case) -> None:
    params, _, _ = _inputs(backend_case)
    empty_features, empty_targets = _empty_batch(backend_case)
    grad_fn, state = adaptive_clipped_grad(
        _loss,
        initial_clipping_norm=1.0,
        key=key(0),
        batch_argnums=(1, 2),
        second_moment=True,
    )
    grads, _ = grad_fn(params, empty_features, empty_targets, state=state)

    assert isinstance(grads, SecondMomentClippingOutput)
    np.testing.assert_array_equal(backend_case.to_host(grads.grads.pytree), np.zeros(3))
    np.testing.assert_array_equal(
        backend_case.to_host(grads.squared_grads.pytree), np.zeros(3)
    )


def test_empty_batch_second_moment_max_norms_match_clipped_grad_convention(
    backend_case,
) -> None:
    """Both streams' ``max_norm`` must equal ``C/normalize_by`` for the first
    stream and ``C**2/normalize_by`` for the squared stream, exactly what
    ``clipped_grad`` attaches on non-empty batches."""
    params, _, _ = _inputs(backend_case)
    empty_features, empty_targets = _empty_batch(backend_case)
    clip_norm, normalize_by = 0.7, 4.0
    grad_fn, state = adaptive_clipped_grad(
        _loss,
        initial_clipping_norm=clip_norm,
        key=key(0),
        batch_argnums=(1, 2),
        second_moment=True,
        normalize_by=normalize_by,
    )
    grads, _ = grad_fn(params, empty_features, empty_targets, state=state)

    assert grads.grads.max_norm == pytest.approx(clip_norm / normalize_by)
    assert grads.squared_grads.max_norm == pytest.approx(
        (clip_norm * clip_norm) / normalize_by
    )


def test_empty_batch_second_moment_uses_updated_next_clipping_norm(
    backend_case,
) -> None:
    """After a normal batch updates ``_next_clipping_norm``, the following
    empty batch must report the updated threshold (and its square)."""
    params, features, targets = _inputs(backend_case)
    empty_features, empty_targets = _empty_batch(backend_case)
    grad_fn, state = adaptive_clipped_grad(
        _loss,
        initial_clipping_norm=1.0,
        key=key(0),
        batch_argnums=(1, 2),
        second_moment=True,
    )
    _, state = grad_fn(params, features, targets, state=state)
    next_cn = state._next_clipping_norm
    assert next_cn != 1.0

    grads, _ = grad_fn(params, empty_features, empty_targets, state=state)
    assert grads.grads.max_norm == pytest.approx(next_cn)
    assert grads.squared_grads.max_norm == pytest.approx(next_cn * next_cn)


def test_second_moment_output_type_is_stable_across_empty_and_normal_batches(
    backend_case,
) -> None:
    params, features, targets = _inputs(backend_case)
    empty_batch = _empty_batch(backend_case)
    grad_fn, state = adaptive_clipped_grad(
        _loss,
        initial_clipping_norm=1.0,
        key=key(0),
        batch_argnums=(1, 2),
        second_moment=True,
    )
    for batch in (empty_batch, (features, targets), empty_batch):
        grads, state = grad_fn(params, *batch, state=state)
        assert isinstance(grads, SecondMomentClippingOutput)


def test_second_moment_output_drives_paired_noise_dispatch(backend_case) -> None:
    """``gaussian_noise`` must emit ``SecondMomentNoiseOutput`` on both empty
    and non-empty steps when adaptive clipping runs in
    ``second_moment=True`` mode."""
    params, features, targets = _inputs(backend_case)
    empty_batch = _empty_batch(backend_case)
    grad_fn, state = adaptive_clipped_grad(
        _loss,
        initial_clipping_norm=1.0,
        key=key(0),
        batch_argnums=(1, 2),
        second_moment=True,
    )
    noise_fn, noise_state = gaussian_noise(noise_multiplier=1.1, key=key(99))

    for batch in (empty_batch, (features, targets), empty_batch):
        grads, state = grad_fn(params, *batch, state=state)
        noisy, noise_state = noise_fn(grads, noise_state)
        assert isinstance(noisy, SecondMomentNoiseOutput)


def test_per_group_empty_batch_second_moment_uses_per_group_max_norms(
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
    groups = {"w": "weights", "b": "biases"}
    init = PerGroup(groups, {"weights": 1.0, "biases": 0.5})

    grad_fn, state = adaptive_clipped_grad(
        grouped_loss,
        initial_clipping_norm=init,
        key=key(0),
        batch_argnums=(1, 2),
        second_moment=True,
        normalize_by=2.0,
    )
    grads, _ = grad_fn(params, empty_features, empty_targets, state=state)

    assert isinstance(grads, SecondMomentClippingOutput)
    first_mn, squared_mn = grads.grads.max_norm, grads.squared_grads.max_norm
    assert isinstance(first_mn, PerGroup)
    assert isinstance(squared_mn, PerGroup)
    assert first_mn.values["weights"] == pytest.approx(1.0 / 2.0)
    assert first_mn.values["biases"] == pytest.approx(0.5 / 2.0)
    assert squared_mn.values["weights"] == pytest.approx(1.0 * 1.0 / 2.0)
    assert squared_mn.values["biases"] == pytest.approx(0.5 * 0.5 / 2.0)


def test_second_moment_rejects_nonpositive_normalize_by_at_construction(
    backend_case,
) -> None:
    """``normalize_by <= 0`` must fail at factory time so empty-batch and
    non-empty-batch failure modes stay consistent (the empty-batch
    short-circuit uses ``normalize_by`` directly, bypassing the inner
    validation that otherwise fires through ``clipped_grad``)."""
    del backend_case
    with pytest.raises(ValueError, match="normalize_by must be > 0"):
        adaptive_clipped_grad(
            _loss,
            initial_clipping_norm=1.0,
            key=key(0),
            batch_argnums=(1, 2),
            second_moment=True,
            normalize_by=0.0,
        )
    with pytest.raises(ValueError, match="normalize_by must be > 0"):
        adaptive_clipped_grad(
            _loss,
            initial_clipping_norm=1.0,
            key=key(0),
            batch_argnums=(1, 2),
            normalize_by=-1.0,
        )


def test_empty_batch_second_moment_combines_with_return_aux(backend_case) -> None:
    params, _, _ = _inputs(backend_case)
    empty_features, empty_targets = _empty_batch(backend_case)
    grad_fn, state = adaptive_clipped_grad(
        _loss,
        initial_clipping_norm=1.0,
        key=key(0),
        batch_argnums=(1, 2),
        return_aux=True,
        second_moment=True,
    )
    (grads, aux), _ = grad_fn(params, empty_features, empty_targets, state=state)

    assert isinstance(grads, SecondMomentClippingOutput)
    assert aux.clipping_rate == 0.0
    assert aux.grad_norms.shape == (0,)


def test_sync_adaptive_clip_state_is_a_single_process_passthrough(
    backend_case,
) -> None:
    """Outside a distributed run, syncing must return the input unchanged."""
    state = AdaptiveClipState(
        _current_clipping_norm=1.5,
        _next_clipping_norm=1.5,
        _step=5,
        _rng_key=key(0),
        _fraction_noise_std=0.05,
        _learning_rate=0.2,
        _target_quantile=0.5,
        _clipping_norm_min=0.01,
        _clipping_norm_max=100.0,
        _num_clipped=0.0,
        _batch_size=0.0,
    )
    result = sync_adaptive_clip_state(state)
    assert result._current_clipping_norm == 1.5
    assert result._next_clipping_norm == 1.5


def test_sync_adaptive_clipped_grad_aux_is_a_single_process_passthrough(
    backend_case,
) -> None:
    dtype = backend_case.dtype("float32")
    empty = backend_case.array(np.empty(0, dtype=np.float32), dtype=dtype)
    aux = AdaptiveClippedGradAux(
        loss_values=empty,
        grad_norms=empty,
        clipped_grad_norms=empty,
        loss_aux=None,
        clipping_rate=0.0,
        batch_size=0,
    )
    result = sync_adaptive_clipped_grad_aux(aux)
    assert result.clipping_rate == 0.0
    assert result.grad_norms.shape == (0,)
