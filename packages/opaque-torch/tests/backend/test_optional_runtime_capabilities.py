"""Optional runtime APIs resolve through the active backend."""

from __future__ import annotations

from contextlib import contextmanager

import pytest
import torch

from opaque.api.engine import runtime
from opaque.api.engine.backend import Backend, ensure_backend, use_backend
from opaque.api.engine.clipping._clipped_grad import clipped_grad
from opaque.api.engine.primitive import CORE_PRIMITIVES, UnsupportedPrimitiveError
from opaque.api.torch.backend._runtime import profiling_trace_scope


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
        (lambda: runtime.distributed_rank(), "opaque.runtime.distributed.rank"),
        (
            lambda: runtime.distributed_all_gather_object(None),
            "opaque.runtime.distributed.all_gather_object",
        ),
        (lambda: runtime.synchronize(), "opaque.runtime.observability.synchronize"),
        (lambda: runtime.memory_stats(), "opaque.runtime.observability.memory_stats"),
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
        runtime.trace_scope("opaque::clipped_grad")

    assert error.value.primitive_name == "opaque.runtime.observability.trace_scope"
    assert error.value.backend_name == backend.name


def test_torch_supports_both_named_runtime_profiles() -> None:
    ensure_backend(torch.empty(0))

    assert runtime.supports_profile(runtime.RuntimeProfile.DISTRIBUTED)
    assert runtime.supports_profile(runtime.RuntimeProfile.OBSERVABILITY)


def test_torch_singleton_reduction_is_functional() -> None:
    value = torch.tensor([1.0, 2.0])

    reduced = runtime.distributed_all_reduce(value, runtime.ReduceOp.SUM)

    assert torch.equal(reduced, value)
    assert reduced is not value


def test_torch_trace_scope_uses_record_function(monkeypatch) -> None:
    marker = object()
    labels = []

    def record_function(label: str):
        labels.append(label)
        return marker

    monkeypatch.setattr(torch.autograd.profiler, "record_function", record_function)

    assert runtime.trace_scope.supports("torch")
    assert profiling_trace_scope("opaque::clipped_grad") is marker
    assert labels == ["opaque::clipped_grad"]


@pytest.mark.parametrize("return_aux", [False, True])
def test_clipped_grad_routes_through_trace_scope(monkeypatch, return_aux) -> None:
    events = []
    ensure_backend(torch.tensor(0.0))

    @contextmanager
    def trace_scope(label: str):
        events.append(("enter", label))
        yield
        events.append(("exit", label))

    monkeypatch.setattr(
        runtime.trace_scope,
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
