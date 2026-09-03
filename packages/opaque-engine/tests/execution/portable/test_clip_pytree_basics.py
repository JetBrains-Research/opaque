"""Portable clip_pytree public behavior."""

from __future__ import annotations

import pytest

from opaque.api.engine.clipping._pytree import clip_pytree
from opaque.pytree import global_norm


def test_clip_pytree_zero_inf_threshold_nan_and_empty(backend_case) -> None:
    pytree = {"w": backend_case.array([3.0, 4.0])}
    clipped, aux = clip_pytree(pytree, clipping_norm=0.0)
    backend_case.assert_allclose(clipped["w"], [0.0, 0.0])
    assert float(backend_case.to_host(aux.norm)) == pytest.approx(5.0)

    clipped, _ = clip_pytree(pytree, clipping_norm=float("inf"))
    backend_case.assert_allclose(clipped["w"], [3.0, 4.0])

    clipped, _ = clip_pytree(pytree, clipping_norm=1.0)
    assert float(backend_case.to_host(global_norm(clipped))) == pytest.approx(1.0)

    small = {"w": backend_case.array([0.3, 0.4])}
    clipped, _ = clip_pytree(small, clipping_norm=1.0)
    backend_case.assert_allclose(clipped["w"], [0.3, 0.4])

    nested = {
        "layer1": {
            "w": backend_case.array([1.0, 2.0]),
            "b": backend_case.array([0.5]),
        },
        "layer2": {"w": backend_case.array([3.0, 4.0])},
    }
    clipped, _ = clip_pytree(nested, clipping_norm=1.0)
    assert set(clipped) == set(nested)
    assert set(clipped["layer1"]) == set(nested["layer1"])

    dirty = {"w": backend_case.array([float("nan"), float("inf"), 1.0])}
    clipped, aux = clip_pytree(dirty, clipping_norm=1.0)
    host = backend_case.to_host(clipped["w"])
    assert (host == host).all()
    assert float(backend_case.to_host(aux.norm)) == float(
        backend_case.to_host(aux.norm)
    )

    empty, aux = clip_pytree({}, clipping_norm=1.0)
    assert empty == {}
    assert float(backend_case.to_host(aux.norm)) == 0.0
