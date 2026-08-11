"""Seam tests: DP gradient transforms dispatch through ``active_backend()``.

Step 2 routes ``clipped_grad`` / ``auto_clipped_grad`` differentiation and
vectorization through the backend abstraction.  A recording backend that wraps
:class:`TorchBackend` proves the two primitives (``value_and_grad`` + ``vmap``)
are dispatched through the seam and that the results are numerically identical
to the default (delegating) path.
"""

from __future__ import annotations

import pytest
import torch

from opaque.api.engine.backend import TorchBackend, set_backend, use_backend
from opaque.api.engine.clipping import auto_clipped_grad, clipped_grad
from opaque.types import ClippedPytree


class _RecordingBackend:
    """Delegating backend that records which primitives were invoked."""

    name = "recording"

    def __init__(self, inner: TorchBackend) -> None:
        self._inner = inner
        self.calls: list[str] = []

    def value_and_grad(self, *args, **kwargs):
        self.calls.append("value_and_grad")
        return self._inner.value_and_grad(*args, **kwargs)

    def vmap(self, *args, **kwargs):
        self.calls.append("vmap")
        return self._inner.vmap(*args, **kwargs)

    def __getattr__(self, name):
        # Everything else (pytree ops, array math, RNG) delegates to torch.
        return getattr(self._inner, name)


@pytest.fixture(autouse=True)
def _reset_backend():
    """Restore a fresh ``TorchBackend`` default after every test."""
    yield
    set_backend(TorchBackend())


def _loss(param, data):
    return 0.5 * ((data - param) ** 2).mean()


def _unwrap(value):
    return value.pytree if isinstance(value, ClippedPytree) else value


def test_clipped_grad_dispatches_value_and_grad_and_vmap():
    recording = _RecordingBackend(TorchBackend())
    param = torch.tensor(3.0)
    data = torch.tensor([0.0, 7.0, -2.0])

    with use_backend(recording):
        grad_fn, state = clipped_grad(
            _loss, argnums=0, batch_argnums=1, clipping_norm=1.0
        )
        grad, _ = grad_fn(param, data, state=state)

    assert "value_and_grad" in recording.calls
    assert "vmap" in recording.calls
    assert isinstance(_unwrap(grad), torch.Tensor)


def test_auto_clipped_grad_dispatches_value_and_grad_and_vmap():
    recording = _RecordingBackend(TorchBackend())
    param = torch.tensor(3.0)
    data = torch.tensor([0.0, 7.0, -2.0])

    with use_backend(recording):
        grad_fn, state = auto_clipped_grad(_loss, argnums=0, batch_argnums=1, R=1.0)
        grad, _ = grad_fn(param, data, state=state)

    assert "value_and_grad" in recording.calls
    assert "vmap" in recording.calls
    assert isinstance(_unwrap(grad), torch.Tensor)


def test_clipped_grad_microbatch_dispatches_vmap():
    """The microbatch-accumulation path also vectorizes through the seam."""
    recording = _RecordingBackend(TorchBackend())
    param = torch.tensor(3.0)
    data = torch.tensor([0.0, 7.0, -2.0, 4.0])

    with use_backend(recording):
        grad_fn, state = clipped_grad(
            _loss, argnums=0, batch_argnums=1, clipping_norm=1.0, microbatch_size=2
        )
        grad_fn(param, data, state=state)

    assert "value_and_grad" in recording.calls
    # microbatch_size=2 over a batch of 4 → two vmapped microbatches.
    assert recording.calls.count("vmap") >= 2


def test_recording_backend_matches_default_numerics():
    """Delegating recording backend yields identical grads to the default."""
    param = torch.tensor(3.0)
    data = torch.tensor([0.0, 7.0, -2.0])

    grad_fn, state = clipped_grad(_loss, argnums=0, batch_argnums=1, clipping_norm=1.0)
    baseline, _ = grad_fn(param, data, state=state)

    recording = _RecordingBackend(TorchBackend())
    with use_backend(recording):
        grad_fn2, state2 = clipped_grad(
            _loss, argnums=0, batch_argnums=1, clipping_norm=1.0
        )
        through_seam, _ = grad_fn2(param, data, state=state2)

    assert torch.equal(_unwrap(baseline), _unwrap(through_seam))
