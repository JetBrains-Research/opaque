"""Tests for the backend seam: default identity, swap/restore, RNG, façade.

Phase 0 adds a process-wide backend resolver with :class:`TorchBackend`
registered as the import-time default.  These tests pin the default identity,
the swap/restore semantics of ``set_backend`` / ``use_backend``, the RNG
primitive group (``generator`` / ``normal``), and an end-to-end proof that
``clipped_grad`` produces byte-for-byte identical output when driven through a
trivial alternate (recording) backend.
"""

from __future__ import annotations

import pytest
import torch

import opaque.backend as facade
from opaque.api.engine.backend import (
    Backend,
    TorchBackend,
    active_backend,
    set_backend,
    use_backend,
)
from opaque.api.engine.clipping import clipped_grad
from opaque.api.engine.primitive import CORE_PRIMITIVES
from opaque.random import generator_from_key, key
from opaque.types import ClippedPytree


@pytest.fixture(autouse=True)
def _reset_backend():
    """Restore a fresh ``TorchBackend`` default after every test."""
    yield
    set_backend(TorchBackend())


class _RecordingBackend:
    """Minimal stand-in backend used to prove the resolver actually swaps."""

    name = "recording"

    def __init__(self) -> None:
        _register_test_core(self.name)


def _register_test_core(backend_name: str) -> None:
    for primitive in CORE_PRIMITIVES:
        if not primitive.supports(backend_name):
            primitive.register(
                backend_name,
                lambda *args, _primitive=primitive, **kwargs: _primitive.resolve(
                    "torch"
                )(*args, **kwargs),
            )


def test_default_backend_is_torch():
    backend = active_backend()
    assert isinstance(backend, TorchBackend)
    assert backend.name == "torch"


def test_torch_backend_satisfies_protocol():
    assert isinstance(TorchBackend(), Backend)


def test_set_backend_persists():
    sentinel = _RecordingBackend()
    set_backend(sentinel)
    assert active_backend() is sentinel


def test_use_backend_swaps_and_restores():
    previous = active_backend()
    sentinel = _RecordingBackend()
    with use_backend(sentinel) as yielded:
        assert yielded is sentinel
        assert active_backend() is sentinel
    assert active_backend() is previous


def test_use_backend_restores_on_exception():
    previous = active_backend()
    sentinel = _RecordingBackend()

    def _raise_inside_block() -> None:
        with use_backend(sentinel):
            assert active_backend() is sentinel
            raise RuntimeError("boom")

    with pytest.raises(RuntimeError):
        _raise_inside_block()
    assert active_backend() is previous


def test_facade_reexports_shared_registry():
    # The façade must re-export the very same resolver objects, not copies.
    assert facade.active_backend is active_backend
    assert facade.set_backend is set_backend
    assert facade.use_backend is use_backend
    assert facade.Backend is Backend

    sentinel = _RecordingBackend()
    facade.set_backend(sentinel)
    assert active_backend() is sentinel


def test_facade_all_is_reexports_only():
    assert set(facade.__all__) == {
        "Backend",
        "active_backend",
        "set_backend",
        "use_backend",
    }


def test_torch_backend_primitives_smoke():
    backend = TorchBackend()

    # array math
    x = torch.tensor([3.0, 4.0])
    assert torch.equal(backend.square(x), torch.tensor([9.0, 16.0]))
    assert backend.sum(backend.square(x)) == pytest.approx(25.0)
    assert backend.sqrt(backend.sum(backend.square(x))) == pytest.approx(5.0)
    assert backend.is_array(x)
    assert backend.is_floating(x)
    assert backend.is_floating(torch.float32)

    # autodiff: d/dx (x**2 summed) = 2x
    def loss(t):
        return (t**2).sum()

    grad_fn = backend.value_and_grad(loss)
    grad, value = grad_fn(x)
    assert torch.equal(grad, 2 * x)
    assert value == pytest.approx(25.0)

    # vectorization
    doubled = backend.vmap(lambda t: t * 2)(x)
    assert torch.equal(doubled, x * 2)


def test_torch_backend_float32_is_torch_float32():
    assert TorchBackend().float32 is torch.float32


@pytest.mark.parametrize(
    ("dtype", "expected"),
    [
        (torch.float16, True),
        (torch.bfloat16, True),
        (torch.float32, False),
        (torch.float64, False),
        (torch.int64, False),
        (torch.complex64, False),
        (torch.complex128, False),
    ],
)
def test_torch_backend_is_low_precision_for_arrays_and_dtypes(dtype, expected):
    backend = TorchBackend()

    assert backend.is_low_precision(dtype) is expected
    assert backend.is_low_precision(torch.empty((), dtype=dtype)) is expected


@pytest.mark.parametrize(
    ("dtype", "expected"),
    [
        (torch.float16, False),
        (torch.bfloat16, False),
        (torch.float32, False),
        (torch.float64, False),
        (torch.int64, False),
        (torch.complex64, True),
        (torch.complex128, True),
    ],
)
def test_torch_backend_is_complex_for_arrays_and_dtypes(dtype, expected):
    backend = TorchBackend()

    assert backend.is_complex(dtype) is expected
    assert backend.is_complex(torch.empty((), dtype=dtype)) is expected


# --- rng primitive group -------------------------------------------------


def test_generator_matches_generator_from_key():
    """``generator(key)`` bridges the existing ``generator_from_key`` seed."""
    backend = TorchBackend()
    k = key(1234)

    drawn = torch.randn(4, generator=backend.generator(k))
    expected = torch.randn(4, generator=generator_from_key(k))

    assert torch.equal(drawn, expected)


def test_generator_is_deterministic_for_same_key():
    backend = TorchBackend()
    k = key(7)

    first = torch.randn(8, generator=backend.generator(k))
    second = torch.randn(8, generator=backend.generator(k))

    assert torch.equal(first, second)


def test_generator_distinct_keys_produce_distinct_draws():
    backend = TorchBackend()

    a = torch.randn(8, generator=backend.generator(key(1)))
    b = torch.randn(8, generator=backend.generator(key(2)))

    assert not torch.equal(a, b)


def test_normal_shape_dtype_and_reproducible():
    backend = TorchBackend()
    k = key(99)
    shape = (2, 3)

    first = backend.normal(shape, dtype=torch.float64, generator=backend.generator(k))
    second = backend.normal(shape, dtype=torch.float64, generator=backend.generator(k))

    assert tuple(first.shape) == shape
    assert first.dtype == torch.float64
    assert torch.equal(first, second)


def test_normal_matches_torch_randn():
    backend = TorchBackend()
    k = key(2024)

    drawn = backend.normal((5,), dtype=torch.float32, generator=backend.generator(k))
    expected = torch.randn(5, dtype=torch.float32, generator=generator_from_key(k))

    assert torch.equal(drawn, expected)


# --- end-to-end: clipped_grad through a trivial alternate backend --------


class _DelegatingBackend:
    """A trivial alternate backend: records every primitive it is asked for,
    then forwards to a wrapped :class:`TorchBackend`.

    It is behaviorally identical to the default backend, so anything driven
    through it must produce byte-for-byte identical results — while the
    recorded call list proves the compute actually routed through the seam.
    """

    name = "delegating"

    def __init__(self, inner: TorchBackend) -> None:
        self._inner = inner
        self.calls: list[str] = []
        for primitive in CORE_PRIMITIVES:
            implementation = primitive.resolve("torch")
            operation_name = primitive.name.rsplit(".", 1)[-1]

            def wrapped(*args, _name=operation_name, _impl=implementation, **kwargs):
                self.calls.append(_name)
                return _impl(*args, **kwargs)

            primitive.register(self.name, wrapped, replace=True)


def _loss(param, data):
    return 0.5 * ((data - param) ** 2).mean()


def _unwrap(value):
    return value.pytree if isinstance(value, ClippedPytree) else value


def test_clipped_grad_identical_through_alternate_backend():
    """Driving ``clipped_grad`` through a trivial delegating backend yields
    byte-for-byte identical gradients — proving the abstraction is transparent
    — while the recorded calls confirm autodiff + vectorization + clip math all
    routed through the seam."""
    param = torch.tensor(3.0)
    data = torch.tensor([0.0, 7.0, -2.0])

    # Baseline: the default torch backend.
    grad_fn, state = clipped_grad(_loss, argnums=0, batch_argnums=1, clipping_norm=1.0)
    baseline, _ = grad_fn(param, data, state=state)

    # Same computation, driven through the recording alternate backend.
    recording = _DelegatingBackend(TorchBackend())
    with use_backend(recording):
        grad_fn2, state2 = clipped_grad(
            _loss, argnums=0, batch_argnums=1, clipping_norm=1.0
        )
        through_seam, _ = grad_fn2(param, data, state=state2)

    assert torch.equal(_unwrap(baseline), _unwrap(through_seam))

    # The full seam was exercised: autodiff + vectorization + clip math.
    assert "grad_and_value" in recording.calls
    assert "vmap" in recording.calls
    for prim in ("square", "sum", "sqrt"):
        assert prim in recording.calls, prim
