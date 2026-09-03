"""Portable dtype and accumulation contracts for clipping public APIs."""

from __future__ import annotations

import pytest

from opaque.api.engine.clipping._pytree import clip_pytree
from opaque.pytree import global_norm
from opaque.types import PerGroup


def test_global_norm_promotes_low_precision_by_default(backend_case) -> None:
    tree = {"a": backend_case.array([3.0, 4.0], dtype=backend_case.dtype("float16"))}

    norm = global_norm(tree)

    assert norm.dtype == backend_case.dtype("float32")
    backend_case.assert_allclose(norm, 5.0)


def test_global_norm_rejects_non_real_compute_dtypes(backend_case) -> None:
    tree = {"a": backend_case.array([3.0, 4.0], dtype=backend_case.dtype("float32"))}

    with pytest.raises(TypeError, match="real floating-point"):
        global_norm(tree, compute_dtype=backend_case.dtype("int32"))


def test_clip_pytree_preserves_output_dtype_and_honors_compute_dtype(
    backend_case,
) -> None:
    float16 = backend_case.dtype("float16")
    float32 = backend_case.dtype("float32")
    tree = {"a": backend_case.array([6.0, 8.0], dtype=float16)}

    clipped, default_aux = clip_pytree(tree, clipping_norm=5.0)
    _, explicit_aux = clip_pytree(tree, clipping_norm=10.0, compute_dtype=float32)

    assert clipped["a"].dtype == float16
    assert default_aux.norm.dtype == float32
    assert explicit_aux.norm.dtype == float32
    backend_case.assert_allclose(clipped["a"], [3.0, 4.0], atol=0.1)


def test_per_group_norm_promotes_across_mixed_low_precision_leaves(
    backend_case,
) -> None:
    float16 = backend_case.dtype("float16")
    float32 = backend_case.dtype("float32")
    groups = PerGroup(groups={"a": "g", "b": "g"}, values={"g": 100.0})
    tree = {
        "a": backend_case.array([3.0, 4.0], dtype=float16),
        "b": backend_case.array([12.0], dtype=float32),
    }

    _, aux = clip_pytree(tree, clipping_norm=groups)

    assert aux.group_norms is not None
    assert aux.group_norms["g"].dtype == float32
