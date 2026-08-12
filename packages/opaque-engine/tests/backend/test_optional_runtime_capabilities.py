"""Optional runtime APIs resolve through the active backend."""

from __future__ import annotations

from contextlib import contextmanager

import pytest
import torch

from opaque.api.engine import runtime
from opaque.api.engine.backend import Backend, use_backend
from opaque.api.engine.backend.torch._runtime import profiling_trace_scope
from opaque.api.engine.clipping._clipped_grad import clipped_grad
from opaque.api.engine.primitive import CORE_PRIMITIVES, UnsupportedPrimitiveError
from opaque.device import device_capabilities
from opaque.distributed import is_distributed, local_shard
from opaque.functional import make_functional
from opaque.profiling import get_memory_stats


class _CoreOnlyBackend:
    name = "core-only-runtime-test"

    def __init__(self) -> None:
        for primitive in CORE_PRIMITIVES:
            if not primitive.supports(self.name):
                primitive.register(
                    self.name,
                    lambda *args, _primitive=primitive, **kwargs: _primitive.resolve(
                        "torch"
                    )(*args, **kwargs),
                )


@pytest.mark.parametrize(
    ("call", "primitive_name"),
    [
        (lambda: is_distributed(), "opaque.runtime.distributed.is_initialized"),
        (lambda: local_shard([]), "opaque.runtime.distributed.dataset_subset"),
        (lambda: device_capabilities("cpu"), "opaque.runtime.device.capabilities"),
        (lambda: get_memory_stats("cpu"), "opaque.runtime.profiling.memory_stats"),
        (
            lambda: make_functional(object()),
            "opaque.runtime.functional.make_functional",
        ),
    ],
)
def test_optional_runtime_capabilities_fail_at_the_public_call_site(
    call, primitive_name: str
) -> None:
    backend = _CoreOnlyBackend()
    assert isinstance(backend, Backend)

    with use_backend(backend), pytest.raises(UnsupportedPrimitiveError) as error:
        call()

    assert error.value.primitive_name == primitive_name
    assert error.value.backend_name == backend.name


def test_trace_scope_fails_when_requested_by_unsupported_backend() -> None:
    backend = _CoreOnlyBackend()

    with use_backend(backend), pytest.raises(UnsupportedPrimitiveError) as error:
        runtime.profiling_trace_scope("opaque::clipped_grad")

    assert error.value.primitive_name == "opaque.runtime.profiling.trace_scope"
    assert error.value.backend_name == backend.name


def test_torch_trace_scope_uses_record_function(monkeypatch) -> None:
    marker = object()
    labels = []

    def record_function(label: str):
        labels.append(label)
        return marker

    monkeypatch.setattr(torch.autograd.profiler, "record_function", record_function)

    assert runtime.profiling_trace_scope.supports("torch")
    assert profiling_trace_scope("opaque::clipped_grad") is marker
    assert labels == ["opaque::clipped_grad"]


@pytest.mark.parametrize("return_aux", [False, True])
def test_clipped_grad_routes_through_trace_scope(monkeypatch, return_aux) -> None:
    events = []

    @contextmanager
    def trace_scope(label: str):
        events.append(("enter", label))
        yield
        events.append(("exit", label))

    monkeypatch.setattr(
        runtime.profiling_trace_scope,
        "resolve",
        lambda backend=None: trace_scope,
    )

    def loss_fn(param, data):
        return ((param - data) ** 2).mean()

    grad_fn, state = clipped_grad(
        loss_fn,
        clipping_norm=1.0,
        batch_argnums=1,
        return_aux=return_aux,
    )
    grad_fn(
        torch.tensor(0.0, requires_grad=True),
        torch.tensor([1.0, 2.0]),
        state=state,
    )

    assert events == [
        ("enter", "opaque::clipped_grad"),
        ("exit", "opaque::clipped_grad"),
    ]
