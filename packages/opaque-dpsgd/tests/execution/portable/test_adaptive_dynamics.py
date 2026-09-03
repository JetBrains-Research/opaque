"""Provider-neutral adaptive-clipping threshold dynamics and edge cases."""

from __future__ import annotations

import math

import numpy as np
import pytest

from opaque import ops
from opaque.api.engine.clipping import _clipped_fun as clipped_fun_module
from opaque.dpsgd.clipping import adaptive_clipped_grad
from opaque.dpsgd.clipping.types import ClippingStats
from opaque.dpsgd.noise import gaussian_noise
from opaque.random import key
from opaque.types import NoisedPytree


def _loss(params, x, y):
    return ((x @ params - y) ** 2).mean()


def _params_and_batch(backend_case, batch_size: int = 8, num_features: int = 10):
    dtype = backend_case.dtype("float32")
    rng = np.random.default_rng(0)
    params = backend_case.array(
        rng.standard_normal(num_features).astype(np.float32), dtype=dtype
    )
    batch_x = backend_case.array(
        rng.standard_normal((batch_size, num_features)).astype(np.float32), dtype=dtype
    )
    batch_y = backend_case.array(
        rng.standard_normal(batch_size).astype(np.float32), dtype=dtype
    )
    return params, batch_x, batch_y


def test_basic_workflow_reports_shape_and_initial_threshold(backend_case) -> None:
    grad_fn, clip_state = adaptive_clipped_grad(
        _loss,
        initial_clipping_norm=1.0,
        target_quantile=0.5,
        key=key(0),
        batch_argnums=(1, 2),
    )
    assert clip_state._next_clipping_norm == 1.0

    params, batch_x, batch_y = _params_and_batch(backend_case)
    grads, clip_state = grad_fn(params, batch_x, batch_y, state=clip_state)

    assert isinstance(clip_state._next_clipping_norm, float)
    assert grads.pytree.shape == params.shape


def test_stats_report_nonfinite_gradients_before_adaptive_clipping(
    backend_case,
) -> None:
    def loss_fn(params, data):
        return ops.sqrt(data - params)

    grad_fn, clip_state = adaptive_clipped_grad(
        loss_fn,
        initial_clipping_norm=1.0,
        key=key(0),
        batch_argnums=1,
        return_stats=True,
    )
    dtype = backend_case.dtype("float32")
    (grads, stats), _ = grad_fn(
        backend_case.array(0.0, dtype=dtype),
        backend_case.array([1.0, -1.0], dtype=dtype),
        state=clip_state,
    )

    assert isinstance(stats, ClippingStats)
    assert stats.all_finite is False
    assert bool(np.isfinite(backend_case.to_host(grads.pytree)).all())


def test_threshold_increases_when_too_many_clipped(backend_case) -> None:
    grad_fn, clip_state = adaptive_clipped_grad(
        _loss,
        initial_clipping_norm=0.01,  # Very low → many gradients clipped
        target_quantile=0.5,
        learning_rate=0.2,
        key=key(0),
        batch_argnums=(1, 2),
    )
    params, batch_x, batch_y = _params_and_batch(backend_case)
    initial = clip_state._next_clipping_norm
    _, clip_state = grad_fn(params, batch_x, batch_y, state=clip_state)
    assert clip_state._next_clipping_norm > initial


def test_threshold_decreases_when_too_few_clipped(backend_case) -> None:
    grad_fn, clip_state = adaptive_clipped_grad(
        _loss,
        initial_clipping_norm=100.0,  # Very high → few gradients clipped
        target_quantile=0.5,
        learning_rate=0.2,
        key=key(0),
        batch_argnums=(1, 2),
    )
    params, batch_x, batch_y = _params_and_batch(backend_case)
    initial = clip_state._next_clipping_norm
    _, clip_state = grad_fn(params, batch_x, batch_y, state=clip_state)
    assert clip_state._next_clipping_norm < initial


def test_threshold_is_clamped_to_configured_bounds(backend_case) -> None:
    clipping_norm_min, clipping_norm_max = 0.5, 2.0
    grad_fn, clip_state = adaptive_clipped_grad(
        _loss,
        initial_clipping_norm=0.1,  # Below min
        target_quantile=0.5,
        learning_rate=0.2,
        clipping_norm_min=clipping_norm_min,
        clipping_norm_max=clipping_norm_max,
        key=key(0),
        batch_argnums=(1, 2),
    )
    params, batch_x, batch_y = _params_and_batch(backend_case)

    for _ in range(20):
        _, clip_state = grad_fn(params, batch_x, batch_y, state=clip_state)

    assert clipping_norm_min <= clip_state._next_clipping_norm <= clipping_norm_max


def test_has_aux_and_return_aux_compose(backend_case) -> None:
    dtype = backend_case.dtype("float32")

    def loss_fn(params, x, y):
        pred = x @ params
        loss = ((pred - y) ** 2).mean()
        aux = {"accuracy": backend_case.array(0.95, dtype=dtype)}
        return loss, aux

    grad_fn, clip_state = adaptive_clipped_grad(
        loss_fn,
        has_aux=True,
        initial_clipping_norm=1.0,
        key=key(0),
        batch_argnums=(1, 2),
        return_aux=True,
    )
    params, batch_x, batch_y = _params_and_batch(backend_case)
    (grads, _grad_aux), clip_state = grad_fn(params, batch_x, batch_y, state=clip_state)
    assert grads.pytree.shape == params.shape


def test_target_quantile_controls_steady_state_threshold(backend_case) -> None:
    params, batch_x, batch_y = _params_and_batch(backend_case, batch_size=16)

    low_fn, low_state = adaptive_clipped_grad(
        _loss,
        initial_clipping_norm=1.0,
        target_quantile=0.1,
        key=key(0),
        batch_argnums=(1, 2),
    )
    high_fn, high_state = adaptive_clipped_grad(
        _loss,
        initial_clipping_norm=1.0,
        target_quantile=0.9,
        key=key(0),
        batch_argnums=(1, 2),
    )
    for _ in range(20):
        _, low_state = low_fn(params, batch_x, batch_y, state=low_state)
        _, high_state = high_fn(params, batch_x, batch_y, state=high_state)

    assert low_state._next_clipping_norm > high_state._next_clipping_norm


@pytest.mark.parametrize("target_quantile", [0.25, 0.75])
def test_converges_to_requested_clipped_fraction(backend_case, target_quantile) -> None:
    """The steady-state clipped fraction equals ``target_quantile``.

    The targets are asymmetric on purpose: 0.5 is a fixed point of
    ``x -> 1 - x``, so it cannot tell "fraction clipped" apart from the
    Andrew et al. "fraction left unclipped" convention.
    """
    dtype = backend_case.dtype("float32")
    # Per-example gradient of this loss is the example itself, so the
    # per-example gradient norms are exactly ``norms``.
    norms = np.arange(1, 101, dtype=np.float32) / 10.0
    batch = np.zeros((100, 4), dtype=np.float32)
    batch[:, 0] = norms
    batch_x = backend_case.array(batch, dtype=dtype)
    params = backend_case.array(np.zeros(4, dtype=np.float32), dtype=dtype)

    def loss_fn(params, x):
        return (params * x).sum()

    grad_fn, clip_state = adaptive_clipped_grad(
        loss_fn,
        initial_clipping_norm=1.0,
        target_quantile=target_quantile,
        learning_rate=0.5,
        fraction_noise_std=1e-6,
        clipping_norm_min=0.01,
        clipping_norm_max=100.0,
        key=key(0),
        batch_argnums=(1,),
    )
    for _ in range(200):
        _, clip_state = grad_fn(params, batch_x, state=clip_state)

    clipped_fraction = float((norms > clip_state._current_clipping_norm).mean())
    assert clipped_fraction == pytest.approx(target_quantile, abs=0.02)


def test_higher_learning_rate_adapts_faster(backend_case) -> None:
    params, batch_x, batch_y = _params_and_batch(backend_case)

    slow_fn, slow_state = adaptive_clipped_grad(
        _loss,
        initial_clipping_norm=0.01,
        learning_rate=0.05,
        key=key(0),
        batch_argnums=(1, 2),
    )
    fast_fn, fast_state = adaptive_clipped_grad(
        _loss,
        initial_clipping_norm=0.01,
        learning_rate=0.5,
        key=key(0),
        batch_argnums=(1, 2),
    )
    for _ in range(5):
        _, slow_state = slow_fn(params, batch_x, batch_y, state=slow_state)
        _, fast_state = fast_fn(params, batch_x, batch_y, state=fast_state)

    assert abs(fast_state._next_clipping_norm - 0.01) > abs(
        slow_state._next_clipping_norm - 0.01
    )


def test_return_aux_kwarg_is_forwarded_to_clipped_grad(backend_case) -> None:
    grad_fn, clip_state = adaptive_clipped_grad(
        _loss,
        initial_clipping_norm=1.0,
        key=key(0),
        batch_argnums=(1, 2),
        return_aux=True,
    )
    params, batch_x, batch_y = _params_and_batch(backend_case)
    (grads, grad_aux), clip_state = grad_fn(params, batch_x, batch_y, state=clip_state)

    assert grads.pytree.shape == params.shape
    assert grad_aux.loss_values is not None
    assert grad_aux.grad_norms is not None


def test_composes_with_gaussian_noise(backend_case) -> None:
    grad_fn, clip_state = adaptive_clipped_grad(
        _loss, initial_clipping_norm=1.0, key=key(0), batch_argnums=(1, 2)
    )
    params, batch_x, batch_y = _params_and_batch(backend_case)
    grads, clip_state = grad_fn(params, batch_x, batch_y, state=clip_state)

    noise_fn, noise_state = gaussian_noise(noise_multiplier=1.1, key=key(0))
    noisy_grads, noise_state = noise_fn(grads, noise_state)

    assert isinstance(noisy_grads, NoisedPytree)
    assert noisy_grads.noise_stddev == pytest.approx(1.1 * grads.max_norm)
    assert noisy_grads.pytree.shape == grads.pytree.shape
    assert not np.array_equal(
        backend_case.to_host(noisy_grads.pytree), backend_case.to_host(grads.pytree)
    )


def test_microbatching_matches_non_microbatched_gradients_and_state(
    backend_case,
) -> None:
    params, batch_x, batch_y = _params_and_batch(backend_case, batch_size=32)

    grad_fn_no_mb, state_no_mb = adaptive_clipped_grad(
        _loss,
        initial_clipping_norm=1.0,
        target_quantile=0.5,
        learning_rate=0.2,
        key=key(0),
        batch_argnums=(1, 2),
        microbatch_size=None,
    )
    grad_fn_mb, state_mb = adaptive_clipped_grad(
        _loss,
        initial_clipping_norm=1.0,
        target_quantile=0.5,
        learning_rate=0.2,
        key=key(0),
        batch_argnums=(1, 2),
        microbatch_size=8,
    )

    for _ in range(3):
        grads_no_mb, state_no_mb = grad_fn_no_mb(
            params, batch_x, batch_y, state=state_no_mb
        )
        grads_mb, state_mb = grad_fn_mb(params, batch_x, batch_y, state=state_mb)

        np.testing.assert_allclose(
            backend_case.to_host(grads_mb.pytree),
            backend_case.to_host(grads_no_mb.pytree),
            rtol=1e-5,
            atol=1e-6,
        )
        assert math.isclose(
            state_mb._next_clipping_norm, state_no_mb._next_clipping_norm, rel_tol=1e-5
        )


def test_microbatching_without_aux_does_not_materialize_per_example_aux(
    backend_case, monkeypatch
) -> None:
    """No-aux adaptive microbatching streams stats instead of concatenating aux."""
    original = clipped_fun_module._microbatch_accumulate

    def wrapped(*args, **kwargs):
        if kwargs.get("return_aux"):
            raise AssertionError("unexpected per-example aux materialization")
        return original(*args, **kwargs)

    monkeypatch.setattr(clipped_fun_module, "_microbatch_accumulate", wrapped)

    grad_fn, clip_state = adaptive_clipped_grad(
        _loss,
        initial_clipping_norm=1.0,
        target_quantile=0.5,
        learning_rate=0.2,
        key=key(0),
        batch_argnums=(1, 2),
        microbatch_size=4,
        return_aux=False,
    )
    params, batch_x, batch_y = _params_and_batch(backend_case, batch_size=16)
    grads, new_state = grad_fn(params, batch_x, batch_y, state=clip_state)

    assert grads.pytree.shape == params.shape
    assert new_state._step == 1


def test_pre_clipping_transform_makes_the_tracker_scale_invariant(
    backend_case,
) -> None:
    """``pre_clipping_transform`` runs before the grad norm that drives the
    adaptive quantile tracker. The canonical use is fp16 loss-scaling: the
    loss is multiplied by ``loss_scale``, and the transform divides the
    per-example gradient by the same factor before clipping, so the
    unscaled grad norms drive adaptation regardless of the chosen scale."""

    def make_loss(loss_scale: float):
        def loss_fn(params, x, y):
            return ((x @ params - y) ** 2).mean() * loss_scale

        return loss_fn

    params, batch_x, batch_y = _params_and_batch(backend_case)

    grad_fn, state = adaptive_clipped_grad(
        make_loss(1.0),
        initial_clipping_norm=0.5,
        target_quantile=0.5,
        learning_rate=0.2,
        key=key(0),
        batch_argnums=(1, 2),
    )
    _, state_base = grad_fn(params, batch_x, batch_y, state=state)

    scale = 128.0
    grad_fn_scaled, state_scaled = adaptive_clipped_grad(
        make_loss(scale),
        initial_clipping_norm=0.5,
        target_quantile=0.5,
        learning_rate=0.2,
        key=key(0),
        batch_argnums=(1, 2),
        pre_clipping_transform=lambda g: (
            tuple(t / scale for t in g) if isinstance(g, tuple) else g / scale
        ),
    )
    _, state_scaled_after = grad_fn_scaled(params, batch_x, batch_y, state=state_scaled)

    assert math.isclose(
        state_base._next_clipping_norm,
        state_scaled_after._next_clipping_norm,
        rel_tol=1e-5,
    )


def test_single_example_batch(backend_case) -> None:
    grad_fn, clip_state = adaptive_clipped_grad(
        _loss, initial_clipping_norm=1.0, key=key(0), batch_argnums=(1, 2)
    )
    params, batch_x, batch_y = _params_and_batch(backend_case, batch_size=1)
    grads, clip_state = grad_fn(params, batch_x, batch_y, state=clip_state)
    assert grads.pytree.shape == params.shape


def test_zero_gradients_at_a_constant_loss(backend_case) -> None:
    dtype = backend_case.dtype("float32")

    def loss_fn(params, x, y):
        del params, x, y
        return backend_case.array(0.0, dtype=dtype)

    grad_fn, clip_state = adaptive_clipped_grad(
        loss_fn, initial_clipping_norm=1.0, key=key(0), batch_argnums=(1, 2)
    )
    params, batch_x, batch_y = _params_and_batch(backend_case)
    grads, clip_state = grad_fn(params, batch_x, batch_y, state=clip_state)
    np.testing.assert_allclose(
        backend_case.to_host(grads.pytree), np.zeros(params.shape), atol=1e-6
    )


def test_batch_size_is_tracked_in_state(backend_case) -> None:
    grad_fn, clip_state = adaptive_clipped_grad(
        _loss, initial_clipping_norm=1.0, key=key(0), batch_argnums=(1, 2)
    )
    assert clip_state._batch_size == 0

    params, batch_x, batch_y = _params_and_batch(backend_case, batch_size=8)
    _, clip_state = grad_fn(params, batch_x, batch_y, state=clip_state)
    assert clip_state._batch_size == 8

    _, batch_x16, batch_y16 = _params_and_batch(backend_case, batch_size=16)
    _, clip_state = grad_fn(params, batch_x16, batch_y16, state=clip_state)
    assert clip_state._batch_size == 16


def test_normalize_by_does_not_affect_clipping_rate(backend_case) -> None:
    """``normalize_by`` is post-processing on gradients, not clipping rate."""
    grad_fn, clip_state = adaptive_clipped_grad(
        _loss,
        initial_clipping_norm=0.001,  # Very small → everything clips
        key=key(0),
        batch_argnums=(1, 2),
        fraction_noise_std=1e-10,  # Near-zero for deterministic testing
        normalize_by=20,
        return_aux=True,
    )
    params, batch_x, batch_y = _params_and_batch(backend_case, batch_size=8)
    (_, aux), clip_state = grad_fn(params, batch_x, batch_y, state=clip_state)

    assert clip_state._batch_size == 8
    # All 8 examples clipped → rate = 8/8 = 1.0 (normalize_by is irrelevant)
    assert aux.clipping_rate == pytest.approx(1.0, abs=1e-6)


def test_large_batch(backend_case) -> None:
    grad_fn, clip_state = adaptive_clipped_grad(
        _loss, initial_clipping_norm=1.0, key=key(0), batch_argnums=(1, 2)
    )
    params, batch_x, batch_y = _params_and_batch(backend_case, batch_size=1000)
    grads, clip_state = grad_fn(params, batch_x, batch_y, state=clip_state)
    assert grads.pytree.shape == params.shape
