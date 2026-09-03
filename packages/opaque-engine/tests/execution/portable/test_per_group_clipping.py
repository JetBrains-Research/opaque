"""Portable per-group clipping contracts."""

from __future__ import annotations

import pytest

from opaque import ops
from opaque.api.engine.clipping import clipped_grad
from opaque.api.engine.clipping._per_group import per_group
from opaque.api.engine.clipping._pytree import clip_pytree
from opaque.api.engine.clipping.types import FixedClipState
from opaque.types import PerGroup


def test_clip_pytree_per_group_independent_scales_and_validation(backend_case) -> None:
    pytree = {
        "attn.q": backend_case.array([1.0, 0.0]),
        "attn.k": backend_case.array([0.0, 1.0]),
        "mlp.w": backend_case.array([0.5, 0.0]),
    }
    pg = PerGroup(
        groups={"attn.q": "attn", "attn.k": "attn", "mlp.w": "mlp"},
        values={"attn": 10.0, "mlp": 10.0},
    )
    clipped, _ = clip_pytree(pytree, pg)
    for key in pytree:
        backend_case.assert_allclose(clipped[key], pytree[key])

    pytree = {
        "attn.q": backend_case.array([3.0]),
        "attn.k": backend_case.array([4.0]),
        "mlp.w": backend_case.array([3.0]),
    }
    pg = PerGroup(
        groups={"attn.q": "attn", "attn.k": "attn", "mlp.w": "mlp"},
        values={"attn": 1.0, "mlp": 6.0},
    )
    clipped, aux = clip_pytree(pytree, pg)
    backend_case.assert_allclose(clipped["attn.q"], [0.6])
    backend_case.assert_allclose(clipped["attn.k"], [0.8])
    backend_case.assert_allclose(clipped["mlp.w"], [3.0])
    assert float(backend_case.to_host(aux.norm)) == pytest.approx(
        (9.0 + 16.0 + 9.0) ** 0.5
    )

    simple = {
        "a": backend_case.array([3.0]),
        "b": backend_case.array([4.0]),
    }
    simple_pg = PerGroup(
        groups={"a": "g1", "b": "g2"},
        values={"g1": 10.0, "g2": 10.0},
    )
    _, simple_aux = clip_pytree(simple, simple_pg)
    assert float(backend_case.to_host(simple_aux.norm)) == pytest.approx(5.0)

    dirty = {
        "a": backend_case.array([float("nan"), 1.0]),
        "b": backend_case.array([2.0]),
    }
    pg = PerGroup(
        groups={"a": "g1", "b": "g2"},
        values={"g1": 10.0, "g2": 10.0},
    )
    clipped, _ = clip_pytree(dirty, pg)
    host = backend_case.to_host(clipped["a"])
    assert (host == host).all()

    zeros = {
        "a": backend_case.array([0.0, 0.0]),
        "b": backend_case.array([1.0]),
    }
    pg = PerGroup(
        groups={"a": "g1", "b": "g2"},
        values={"g1": 1.0, "g2": 10.0},
    )
    clipped, _ = clip_pytree(zeros, pg)
    backend_case.assert_allclose(clipped["a"], [0.0, 0.0])

    pytree = {
        "a": backend_case.array([5.0]),
        "b": backend_case.array([3.0]),
    }
    pg = PerGroup(
        groups={"a": "g1", "b": "g2"},
        values={"g1": 10.0, "g2": 10.0},
    )
    clipped, _ = clip_pytree(pytree, pg, return_zero=True)
    backend_case.assert_allclose(clipped["a"], [0.0])
    backend_case.assert_allclose(clipped["b"], [0.0])

    nested = {
        "layer1": {
            "attn": backend_case.array([3.0, 4.0]),
            "mlp": backend_case.array([30.0]),
        },
        "layer2": {
            "attn": backend_case.array([5.0, 12.0]),
            "mlp": backend_case.array([40.0]),
        },
    }
    nested_pg = per_group(nested, attn=1.0, mlp=1.0)
    clipped, aux = clip_pytree(nested, nested_pg)
    attn_scale = 1.0 / (9.0 + 16.0 + 25.0 + 144.0) ** 0.5
    mlp_scale = 1.0 / 50.0
    backend_case.assert_allclose(
        clipped["layer1"]["attn"], [3.0 * attn_scale, 4.0 * attn_scale]
    )
    backend_case.assert_allclose(
        clipped["layer2"]["attn"], [5.0 * attn_scale, 12.0 * attn_scale]
    )
    backend_case.assert_allclose(clipped["layer1"]["mlp"], [30.0 * mlp_scale])
    backend_case.assert_allclose(clipped["layer2"]["mlp"], [40.0 * mlp_scale])
    assert aux.group_norms is not None
    assert float(backend_case.to_host(aux.group_norms["attn"])) == pytest.approx(
        194.0**0.5
    )
    assert float(backend_case.to_host(aux.group_norms["mlp"])) == pytest.approx(50.0)

    with pytest.raises(ValueError, match="must match the pytree tensor leaves"):
        clip_pytree(
            {"a": backend_case.array([1.0]), "b": backend_case.array([2.0])},
            PerGroup(groups={"a": "g1"}, values={"g1": 1.0}),
        )
    with pytest.raises(ValueError, match="must match the pytree tensor leaves"):
        clip_pytree(
            {"a": backend_case.array([1.0]), "b": backend_case.array([2.0])},
            PerGroup(
                groups={"a": "g1", "b": "g2", "c": "g3"},
                values={"g1": 1.0, "g2": 1.0, "g3": 1.0},
            ),
        )


def test_clipped_grad_per_group_metadata_microbatch_and_equivalence(
    backend_case,
) -> None:
    assert FixedClipState() == FixedClipState()

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
    pg = per_group(params, w1=1.0, w2=1.0)
    _, state = clipped_grad(loss, argnums=0, batch_argnums=1, clipping_norm=pg)
    assert isinstance(state, FixedClipState)

    with pytest.raises(ValueError, match="positive"):
        clipped_grad(
            loss,
            clipping_norm=PerGroup(groups={"w1": "g1"}, values={"g1": -1.0}),
        )

    grad_fn, clip_state = clipped_grad(
        loss, argnums=0, batch_argnums=1, clipping_norm=pg
    )
    data = backend_case.array([1.0, 2.0, 3.0])
    grads, _ = grad_fn(params, data, state=clip_state)
    assert isinstance(grads.pytree, dict)
    assert set(grads.pytree) == {"w1", "w2"}

    def loss_ab(params, data):
        return ops.mean(ops.multiply(params["a"], data))

    params_ab = {
        "a": backend_case.array(1.0),
        "b": backend_case.array(1.0),
    }
    pg_ab = per_group(params_ab, a=2.0, b=4.0)
    grad_fn, clip_state = clipped_grad(
        loss_ab,
        argnums=0,
        batch_argnums=1,
        clipping_norm=pg_ab,
        normalize_by=10.0,
    )
    grads, _ = grad_fn(
        params_ab,
        backend_case.array([0.5, -0.25, 1.0, -1.0, 0.0, 2.0, -2.0, 0.75]),
        state=clip_state,
    )
    assert isinstance(grads.max_norm, PerGroup)
    assert grads.max_norm.groups == pg_ab.groups
    assert grads.max_norm.values == {"a": pytest.approx(0.2), "b": pytest.approx(0.4)}

    def loss_w(params, data):
        return ops.mean(ops.square(ops.subtract(params["w"], data)))

    params_w = {"w": backend_case.array(0.0)}
    pg_w = per_group(params_w, w=1.0)
    grad_fn, clip_state = clipped_grad(
        loss_w,
        argnums=0,
        batch_argnums=1,
        clipping_norm=pg_w,
        microbatch_size=2,
    )
    grads, _ = grad_fn(
        params_w, backend_case.array([1.0, 2.0, 3.0, 4.0]), state=clip_state
    )
    assert isinstance(grads.pytree, dict)

    grad_fn, clip_state = clipped_grad(
        loss_w,
        argnums=0,
        batch_argnums=1,
        clipping_norm=pg_w,
        return_aux=True,
    )
    (grads, aux), _ = grad_fn(
        params_w, backend_case.array([1.0, 2.0, 3.0]), state=clip_state
    )
    assert isinstance(grads.pytree, dict)
    assert aux.grad_norms is not None

    norm_val = 1.5
    grad_fn_g, cs_g = clipped_grad(
        loss_w, argnums=0, batch_argnums=1, clipping_norm=norm_val
    )
    grad_fn_pg, cs_pg = clipped_grad(
        loss_w, argnums=0, batch_argnums=1, clipping_norm=pg_w
    )
    # Rebuild single-group with matching bound.
    pg_match = per_group(params_w, w=norm_val)
    grad_fn_pg, cs_pg = clipped_grad(
        loss_w, argnums=0, batch_argnums=1, clipping_norm=pg_match
    )
    data = backend_case.array([5.0, 10.0, -3.0])
    grads_g, _ = grad_fn_g(params_w, data, state=cs_g)
    grads_pg, _ = grad_fn_pg(params_w, data, state=cs_pg)
    backend_case.assert_allclose(grads_g.pytree["w"], grads_pg.pytree["w"])

    def loss_a(params, data):
        return ops.mean(ops.multiply(params["a"], data))

    params_noise = {
        "a": backend_case.array(1.0),
        "b": backend_case.array(1.0),
    }
    pg_noise = per_group(params_noise, a=2.0, b=4.0)
    grad_fn, clip_state = clipped_grad(
        loss_a, argnums=0, batch_argnums=1, clipping_norm=pg_noise
    )
    grads, _ = grad_fn(
        params_noise, backend_case.array([1.0, 2.0, 3.0]), state=clip_state
    )
    scaled = 0.5 * grads.max_norm
    assert isinstance(scaled, PerGroup)
    assert scaled.values == {"a": pytest.approx(1.0), "b": pytest.approx(2.0)}
