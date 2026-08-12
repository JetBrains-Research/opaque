"""Seam tests: clip pytree/array math dispatches through ``active_backend()``.

``global_norm`` / ``clip_pytree`` / ``auto_scale_pytree`` (and their
per-group helpers) through the backend abstraction for their pytree ops and
elementwise / reduction / dtype math. A recording provider proves the clip math
is dispatched through the seam and
that the numeric results stay identical to the default (direct) path.
"""

from __future__ import annotations

import pytest
import torch

from opaque.api.engine.backend import clear_backend, use_backend
from opaque.api.engine.clipping._pytree import auto_scale_pytree, clip_pytree
from opaque.api.engine.primitive import CORE_PRIMITIVES
from opaque.pytree import global_norm
from opaque.torch import torch_backend
from opaque.types import PerGroup


class _RecordingBackend:
    """Delegating backend that records which primitives were invoked."""

    name = "recording"

    def __init__(self) -> None:
        self.calls: list[str] = []
        for primitive in CORE_PRIMITIVES:
            implementation = primitive.resolve("torch")
            operation_name = primitive.name.rsplit(".", 1)[-1]

            def wrapped(*args, _name=operation_name, _impl=implementation, **kwargs):
                self.calls.append(_name)
                return _impl(*args, **kwargs)

            primitive.register(self.name, wrapped, replace=True)


@pytest.fixture(autouse=True)
def _reset_backend():
    torch_backend()
    clear_backend()
    yield
    clear_backend()


def test_global_norm_dispatches_reduction_math_through_seam():
    recording = _RecordingBackend()
    tree = {"w": torch.tensor([3.0, 4.0])}

    with use_backend(recording):
        norm = global_norm(tree)

    assert norm.item() == pytest.approx(5.0)
    for prim in ("tree_leaves", "square", "sum", "sqrt"):
        assert prim in recording.calls


@pytest.mark.parametrize(
    "tree",
    [
        {"w": torch.tensor([3.0, 4.0], dtype=torch.float16)},
        {"w": torch.tensor([3.0 + 4.0j])},
    ],
    ids=["real_low_precision", "complex"],
)
def test_global_norm_dispatches_dtype_vocabulary_through_seam(tree):
    baseline = global_norm(tree)

    recording = _RecordingBackend()
    with use_backend(recording):
        norm = global_norm(tree)

    assert torch.equal(norm, baseline)
    assert norm.dtype is torch.float32
    assert "is_complex" in recording.calls


def test_clip_pytree_dispatches_clip_math_through_seam():
    recording = _RecordingBackend()
    tree = {"w": torch.tensor([3.0, 4.0]), "b": torch.tensor([0.0, 12.0])}

    with use_backend(recording):
        clip_pytree(tree, 1.0)

    # Sanitization + scaling clip math routed through the seam.
    for prim in (
        "tree_map",
        "nan_to_num",
        "scalar",
        "clamp",
        "minimum",
        "where",
        "isfinite",
        "astype",
    ):
        assert prim in recording.calls, prim
    # global_norm's reduction ran through the same backend.
    for prim in ("tree_leaves", "square", "sum", "sqrt"):
        assert prim in recording.calls, prim


def test_auto_scale_pytree_dispatches_clip_math_through_seam():
    recording = _RecordingBackend()
    tree = {"w": torch.tensor([3.0, 4.0]), "b": torch.tensor([0.0, 12.0])}

    with use_backend(recording):
        auto_scale_pytree(tree, R=1.0, gamma=0.01)

    for prim in (
        "tree_map",
        "nan_to_num",
        "scalar",
        "clamp",
        "where",
        "isfinite",
        "zeros_like",
        "astype",
    ):
        assert prim in recording.calls, prim


def test_clip_pytree_per_group_dispatches_through_seam():
    recording = _RecordingBackend()
    tree = {"a": torch.tensor([3.0, 4.0]), "b": torch.tensor([6.0, 8.0])}
    pg = PerGroup(groups={"a": "g1", "b": "g2"}, values={"g1": 1.0, "g2": 2.0})

    with use_backend(recording):
        clip_pytree(tree, pg)

    for prim in (
        "tree_flatten_with_paths",
        "is_floating",
        "astype",
        "square",
        "sum",
        "sqrt",
        "scalar",
        "clamp",
        "minimum",
        "where",
        "isfinite",
        "tree_unflatten",
    ):
        assert prim in recording.calls, prim


def test_clip_pytree_per_group_dispatches_low_precision_dtype_vocabulary():
    tree = {
        "a": torch.tensor([3.0, 4.0], dtype=torch.float16),
        "b": torch.tensor([6.0, 8.0], dtype=torch.bfloat16),
    }
    pg = PerGroup(groups={"a": "g1", "b": "g2"}, values={"g1": 1.0, "g2": 2.0})

    baseline, aux0 = clip_pytree(tree, pg)

    recording = _RecordingBackend()
    with use_backend(recording):
        through, aux1 = clip_pytree(tree, pg)

    assert "is_low_precision" in recording.calls
    assert torch.equal(through["a"], baseline["a"])
    assert torch.equal(through["b"], baseline["b"])
    assert aux0.group_norms is not None
    assert aux1.group_norms is not None
    for name in aux0.group_norms:
        assert aux0.group_norms[name].dtype is torch.float32
        assert torch.equal(aux1.group_norms[name], aux0.group_norms[name])


def test_auto_scale_pytree_per_group_dispatches_through_seam():
    recording = _RecordingBackend()
    tree = {"a": torch.tensor([3.0, 4.0]), "b": torch.tensor([6.0, 8.0])}
    pg = PerGroup(groups={"a": "g1", "b": "g2"}, values={"g1": 1.0, "g2": 2.0})

    with use_backend(recording):
        auto_scale_pytree(tree, pg, gamma=0.01)

    for prim in (
        "tree_flatten_with_paths",
        "astype",
        "square",
        "sum",
        "sqrt",
        "scalar",
        "clamp",
        "where",
        "isfinite",
        "tree_unflatten",
    ):
        assert prim in recording.calls, prim


def test_recording_backend_matches_default_numerics_global():
    tree = {"w": torch.tensor([3.0, 4.0]), "b": torch.tensor([0.0, 12.0])}

    baseline, aux0 = clip_pytree(tree, 5.0)

    recording = _RecordingBackend()
    with use_backend(recording):
        through, aux1 = clip_pytree(tree, 5.0)

    assert torch.equal(through["w"], baseline["w"])
    assert torch.equal(through["b"], baseline["b"])
    assert torch.equal(aux1.norm, aux0.norm)


def test_recording_backend_matches_default_numerics_per_group():
    tree = {"a": torch.tensor([3.0, 4.0]), "b": torch.tensor([6.0, 8.0])}
    pg = PerGroup(groups={"a": "g1", "b": "g2"}, values={"g1": 1.0, "g2": 2.0})

    baseline, aux0 = clip_pytree(tree, pg)

    recording = _RecordingBackend()
    with use_backend(recording):
        through, aux1 = clip_pytree(tree, pg)

    assert torch.equal(through["a"], baseline["a"])
    assert torch.equal(through["b"], baseline["b"])
    assert torch.equal(aux1.norm, aux0.norm)
    assert aux0.group_norms is not None
    assert aux1.group_norms is not None
    for name in aux0.group_norms:
        assert torch.equal(aux1.group_norms[name], aux0.group_norms[name])


def test_clip_pytree_nan_inf_sanitized_through_seam():
    tree = {
        "w": torch.tensor([float("nan"), 3.0]),
        "b": torch.tensor([float("inf"), -float("inf")]),
    }

    recording = _RecordingBackend()
    with use_backend(recording):
        clipped, _ = clip_pytree(tree, 1.0)

    flat = torch.cat([clipped["w"], clipped["b"]])
    assert torch.isfinite(flat).all()
    assert torch.linalg.vector_norm(flat).item() <= 1.0 + 1e-5
