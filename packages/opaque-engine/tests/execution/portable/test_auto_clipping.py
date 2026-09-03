"""Portable AUTO-S clipping contracts."""

from __future__ import annotations

import math

import pytest

from opaque import ops
from opaque.api.engine.clipping import auto_clipped_grad, clipped_grad
from opaque.api.engine.clipping._per_group import per_group
from opaque.api.engine.clipping._pytree import auto_scale_pytree
from opaque.api.engine.clipping.types import AutoClippedGradAux, AutoClipState
from opaque.pytree import global_norm
from opaque.types import PerGroup


def _linear_loss(params, x, y):
    # Elementwise linear model keeps the contract backend-neutral without matmul.
    return ops.mean(ops.square(ops.subtract(ops.multiply(params, x), y)))


def test_auto_scale_pytree_formula_bounds_and_nan_handling(backend_case) -> None:
    pytree = {"a": backend_case.array([3.0, 4.0])}
    scaled, aux = auto_scale_pytree(pytree, R=1.0, gamma=0.01)
    expected_scale = 1.0 / (5.0 + 0.01)
    backend_case.assert_allclose(
        scaled["a"], [3.0 * expected_scale, 4.0 * expected_scale]
    )
    assert float(backend_case.to_host(aux.norm)) == pytest.approx(5.0)
    assert aux.group_norms is None

    large = {"a": backend_case.array([300.0, 400.0])}
    scaled, _ = auto_scale_pytree(large, R=1.0, gamma=0.01)
    norm = float(backend_case.to_host(global_norm(scaled)))
    assert norm == pytest.approx(1.0, rel=1e-4)

    zeros = {"a": backend_case.array([0.0, 0.0, 0.0, 0.0])}
    scaled, aux = auto_scale_pytree(zeros, R=1.0, gamma=0.01)
    backend_case.assert_allclose(scaled["a"], [0.0, 0.0, 0.0, 0.0])
    assert float(backend_case.to_host(aux.norm)) == 0.0

    dirty = {"a": backend_case.array([float("nan"), float("inf"), 1.0])}
    scaled, _ = auto_scale_pytree(dirty, R=1.0, gamma=0.01)
    host = backend_case.to_host(scaled["a"])
    assert (host == host).all()  # no NaN
    assert abs(host).max() < float("inf")

    with pytest.raises(ValueError, match="gamma must be positive"):
        auto_scale_pytree({"a": backend_case.array([1.0])}, R=1.0, gamma=0.0)

    # Output remains bounded by R for large random-ish inputs.
    R = 0.7
    for values in (
        [100.0, -50.0, 25.0, -10.0, 5.0],
        [1.0, 2.0, 3.0],
        [1000.0, -1000.0],
    ):
        scaled, _ = auto_scale_pytree(
            {"a": backend_case.array(values), "b": backend_case.array([3.0, -4.0])},
            R=R,
            gamma=0.01,
        )
        total = float(backend_case.to_host(global_norm(scaled)))
        assert total <= R + 1e-5


def test_auto_scale_pytree_per_group_formula_and_validation(backend_case) -> None:
    pytree = {
        "attn.q": backend_case.array([3.0]),
        "attn.k": backend_case.array([4.0]),
        "mlp.w": backend_case.array([6.0]),
    }
    pg = PerGroup(
        groups={"attn.q": "attn", "attn.k": "attn", "mlp.w": "mlp"},
        values={"attn": 2.0, "mlp": 3.0},
    )
    scaled, aux = auto_scale_pytree(pytree, R=pg, gamma=0.01)
    attn_scale = 2.0 / (5.0 + 0.01)
    mlp_scale = 3.0 / (6.0 + 0.01)
    backend_case.assert_allclose(scaled["attn.q"], [3.0 * attn_scale])
    backend_case.assert_allclose(scaled["attn.k"], [4.0 * attn_scale])
    backend_case.assert_allclose(scaled["mlp.w"], [6.0 * mlp_scale])
    assert aux.group_norms is not None
    assert float(backend_case.to_host(aux.group_norms["attn"])) == pytest.approx(5.0)
    assert float(backend_case.to_host(aux.group_norms["mlp"])) == pytest.approx(6.0)

    nested = {
        "layer1": {
            "attn": backend_case.array([3.0, 4.0]),
            "mlp": backend_case.array([6.0]),
        },
        "layer2": {
            "attn": backend_case.array([0.0, 0.0]),
            "mlp": backend_case.array([8.0]),
        },
    }
    nested_pg = per_group(nested, attn=2.0, mlp=1.0)
    scaled, aux = auto_scale_pytree(nested, R=nested_pg, gamma=0.01)
    attn_scale = 2.0 / (5.0 + 0.01)
    mlp_scale = 1.0 / (10.0 + 0.01)
    backend_case.assert_allclose(
        scaled["layer1"]["attn"], [3.0 * attn_scale, 4.0 * attn_scale]
    )
    backend_case.assert_allclose(scaled["layer1"]["mlp"], [6.0 * mlp_scale])
    backend_case.assert_allclose(scaled["layer2"]["mlp"], [8.0 * mlp_scale])
    assert aux.group_norms is not None
    assert float(backend_case.to_host(aux.group_norms["attn"])) == pytest.approx(5.0)
    assert float(backend_case.to_host(aux.group_norms["mlp"])) == pytest.approx(10.0)

    large = {
        "a": backend_case.array([1000.0, -1000.0, 500.0, -250.0, 125.0]),
        "b": backend_case.array([800.0, -600.0, 400.0, -200.0, 100.0]),
    }
    bound_pg = PerGroup(
        groups={"a": "g1", "b": "g2"},
        values={"g1": 0.5, "g2": 1.5},
    )
    scaled, _ = auto_scale_pytree(large, R=bound_pg, gamma=0.01)
    total_norm = float(backend_case.to_host(global_norm(scaled)))
    assert total_norm <= math.sqrt(0.5**2 + 1.5**2) + 1e-4

    with pytest.raises(ValueError, match="must match the pytree tensor leaves"):
        auto_scale_pytree(
            {"a": backend_case.array([1.0]), "b": backend_case.array([2.0])},
            R=PerGroup(groups={"a": "g1"}, values={"g1": 1.0}),
        )


def test_auto_clipped_grad_bounds_aux_microbatch_and_validation(backend_case) -> None:
    assert AutoClipState() == AutoClipState()
    with pytest.raises(ValueError, match="positive"):
        auto_clipped_grad(_linear_loss, R=0.0)
    with pytest.raises(ValueError, match="gamma"):
        auto_clipped_grad(_linear_loss, R=1.0, gamma=0.0)

    params = backend_case.array(0.5, dtype=backend_case.dtype("float32"))
    batch_x = backend_case.array(
        [1.0, -2.0, 0.5, 3.0, -1.5, 2.5, -0.25, 4.0],
        dtype=backend_case.dtype("float32"),
    )
    batch_y = backend_case.array(
        [2.0, -1.0, 0.5, -0.5, 1.5, -1.5, 0.25, -0.25],
        dtype=backend_case.dtype("float32"),
    )

    grad_fn, state = auto_clipped_grad(
        _linear_loss, argnums=0, batch_argnums=(1, 2), R=1.0
    )
    grads, new_state = grad_fn(params, batch_x, batch_y, state=state)
    assert (
        backend_case.to_host(grads.pytree).shape == backend_case.to_host(params).shape
    )
    assert isinstance(new_state, AutoClipState)
    _, newer_state = grad_fn(params, batch_x, batch_y, state=new_state)
    assert new_state == newer_state

    grad_fn, state = auto_clipped_grad(
        _linear_loss,
        argnums=0,
        batch_argnums=(1, 2),
        R=2.5,
        normalize_by=10.0,
    )
    grads, _ = grad_fn(params, batch_x, batch_y, state=state)
    assert grads.max_norm == pytest.approx(0.25)

    R = 0.3
    grad_fn, state = auto_clipped_grad(
        _linear_loss, argnums=0, batch_argnums=(1, 2), R=R
    )
    # Inflate inputs so per-example grads are large.
    large_x = ops.multiply(batch_x, backend_case.array(100.0))
    large_y = ops.multiply(batch_y, backend_case.array(100.0))
    grads, _ = grad_fn(params, large_x, large_y, state=state)
    norm = float(backend_case.to_host(global_norm(grads.pytree)))
    assert norm <= 8 * R + 1e-4

    fn1, s1 = auto_clipped_grad(
        _linear_loss, argnums=0, batch_argnums=(1, 2), R=1.0, normalize_by=1.0
    )
    fn2, s2 = auto_clipped_grad(
        _linear_loss, argnums=0, batch_argnums=(1, 2), R=1.0, normalize_by=4.0
    )
    g1, _ = fn1(params, batch_x, batch_y, state=s1)
    g2, _ = fn2(params, batch_x, batch_y, state=s2)
    backend_case.assert_allclose(
        ops.divide(g1.pytree, backend_case.array(4.0)),
        g2.pytree,
        rtol=1e-5,
        atol=1e-5,
    )

    grad_fn, state = auto_clipped_grad(
        _linear_loss,
        argnums=0,
        batch_argnums=(1, 2),
        R=1.0,
        return_aux=True,
    )
    (_grads, aux), _ = grad_fn(params, batch_x, batch_y, state=state)
    assert isinstance(aux, AutoClippedGradAux)
    assert backend_case.to_host(aux.grad_norms).shape == (8,)
    assert backend_case.to_host(aux.clipped_grad_norms).shape == (8,)
    assert backend_case.to_host(aux.loss_values).shape == (8,)
    assert aux.batch_size == 8
    assert (backend_case.to_host(aux.clipped_grad_norms) <= 1.0 + 1e-5).all()

    fn_full, s_full = auto_clipped_grad(
        _linear_loss, argnums=0, batch_argnums=(1, 2), R=0.5
    )
    fn_mb, s_mb = auto_clipped_grad(
        _linear_loss,
        argnums=0,
        batch_argnums=(1, 2),
        R=0.5,
        microbatch_size=4,
    )
    g_full, _ = fn_full(params, batch_x, batch_y, state=s_full)
    g_mb, _ = fn_mb(params, batch_x, batch_y, state=s_mb)
    backend_case.assert_allclose(g_full.pytree, g_mb.pytree, rtol=1e-5, atol=1e-5)

    grad_fn, state = auto_clipped_grad(
        _linear_loss,
        argnums=0,
        batch_argnums=(1, 2),
        R=1.0,
        return_stats=True,
    )
    empty_x = backend_case.array([], dtype=backend_case.dtype("float32"))
    empty_y = backend_case.array([], dtype=backend_case.dtype("float32"))
    (grads, stats), _ = grad_fn(params, empty_x, empty_y, state=state)
    backend_case.assert_allclose(grads.pytree, 0.0)
    assert stats.all_finite is True

    def unstable_loss(param, data):
        return ops.sqrt(ops.subtract(data, param))

    grad_fn, state = auto_clipped_grad(
        unstable_loss,
        argnums=0,
        batch_argnums=1,
        R=1.0,
        return_stats=True,
    )
    (grads, stats), _ = grad_fn(
        backend_case.array(0.0),
        backend_case.array([1.0, -1.0]),
        state=state,
    )
    assert stats.all_finite is False
    assert ops.all(ops.isfinite(grads.pytree))

    # AUTO-S differs from fixed clipping on small-norm examples.
    params_scalar = backend_case.array(0.0, dtype=backend_case.dtype("float32"))
    samples = backend_case.array([0.1, 0.2], dtype=backend_case.dtype("float32"))

    def squared(param, sample):
        return ops.square(ops.subtract(param, sample))

    auto_fn, auto_state = auto_clipped_grad(
        squared, argnums=0, batch_argnums=1, R=1.0, gamma=0.01
    )
    fixed_fn, fixed_state = clipped_grad(
        squared, argnums=0, batch_argnums=1, clipping_norm=1.0
    )
    auto_g, _ = auto_fn(params_scalar, samples, state=auto_state)
    fixed_g, _ = fixed_fn(params_scalar, samples, state=fixed_state)
    assert not (
        backend_case.to_host(auto_g.pytree) == backend_case.to_host(fixed_g.pytree)
    ).all()


def test_auto_clipped_grad_per_group_metadata_and_scales(backend_case) -> None:
    def loss(params, data):
        return ops.mean(
            ops.square(
                ops.add(
                    ops.multiply(params["w1"], data),
                    ops.multiply(params["w2"], data),
                )
            )
        )

    params = {
        "w1": backend_case.array(3.0),
        "w2": backend_case.array(4.0),
    }
    pg = per_group(params, w1=1.0, w2=2.0)
    grad_fn, state = auto_clipped_grad(
        loss,
        argnums=0,
        batch_argnums=1,
        R=pg,
        normalize_by=5.0,
        return_aux=True,
    )
    data = backend_case.array([1.0, 2.0, 3.0, 4.0])
    (grads, aux), _ = grad_fn(params, data, state=state)
    assert isinstance(grads.max_norm, PerGroup)
    assert grads.max_norm.groups == pg.groups
    assert grads.max_norm.values == {"w1": pytest.approx(0.2), "w2": pytest.approx(0.4)}
    assert aux.grad_norms is not None


def test_auto_clipped_grad_has_aux_passthrough(backend_case) -> None:
    def loss_with_aux(param, sample):
        value = ops.square(ops.subtract(param, sample))
        return value, {"sample": sample}

    grad_fn, state = auto_clipped_grad(
        loss_with_aux,
        argnums=0,
        batch_argnums=1,
        R=1.0,
        has_aux=True,
        return_aux=True,
    )
    (grads, aux), _ = grad_fn(
        backend_case.array(0.0),
        backend_case.array([1.0, 2.0]),
        state=state,
    )
    assert backend_case.to_host(grads.pytree).shape == ()
    backend_case.assert_allclose(aux.loss_aux["sample"], [1.0, 2.0])
