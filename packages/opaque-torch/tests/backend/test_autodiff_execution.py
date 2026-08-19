"""Execution-boundary tests for portable autodiff transforms."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pytest
import torch

from opaque.api.engine import autodiff
from opaque.api.engine.backend import active_backend, clear_backend, use_backend
from opaque.api.engine.clipping import clipped_grad
from opaque.api.engine.primitive import CORE_PRIMITIVES

if TYPE_CHECKING:
    from collections.abc import Callable


class _Backend:
    def __init__(self, name: str) -> None:
        self.name = name
        for operation in CORE_PRIMITIVES:
            if not operation.supports(name):
                operation.register(name, lambda *args, **kwargs: None)


@pytest.fixture(autouse=True)
def _unselected_backend():
    clear_backend()
    yield
    clear_backend()


@pytest.mark.parametrize(
    ("build", "primitive_name"),
    [
        (lambda fn: autodiff.grad_and_value(fn), "_grad_and_value_transform"),
        (lambda fn: autodiff.vmap(fn), "_vmap_transform"),
    ],
)
def test_transform_creation_is_neutral_and_provider_transforms_cache_per_backend(
    monkeypatch: pytest.MonkeyPatch,
    build: Callable[[Callable[..., Any]], Callable[..., Any]],
    primitive_name: str,
) -> None:
    first = _Backend("deferred-first")
    second = _Backend("deferred-second")
    factory_calls: list[str] = []

    def resolve(backend: _Backend):
        def factory(fn: Callable[..., Any], *args: Any, **kwargs: Any):
            del args, kwargs
            factory_calls.append(backend.name)
            return fn

        return factory

    transform_primitive = getattr(autodiff, primitive_name)
    monkeypatch.setattr(transform_primitive, "resolve", resolve)

    executable = build(lambda value: value + 1)
    assert active_backend() is None
    assert factory_calls == []

    with use_backend(first):
        assert executable(1) == 2
        assert executable(2) == 3
    with use_backend(second):
        assert executable(3) == 4
    with use_backend(first):
        assert executable(4) == 5

    assert factory_calls == ["deferred-first", "deferred-second"]


def test_transforms_infer_torch_when_the_executable_receives_arrays() -> None:
    def loss(value: torch.Tensor) -> torch.Tensor:
        return (value**2).sum()

    differentiated = autodiff.grad_and_value(loss)
    vectorized = autodiff.vmap(lambda value: value * 2)
    assert active_backend() is None

    grads, value = differentiated(torch.tensor([3.0, 4.0]))
    doubled = vectorized(torch.tensor([1.0, 2.0]))

    assert torch.equal(grads, torch.tensor([6.0, 8.0]))
    assert value.item() == pytest.approx(25.0)
    assert torch.equal(doubled, torch.tensor([2.0, 4.0]))
    assert active_backend().name == "torch"


def test_clipped_grad_composition_is_backend_neutral_until_execution() -> None:
    def loss(param: torch.Tensor, example: torch.Tensor) -> torch.Tensor:
        return 0.5 * (example - param) ** 2

    executable, state = clipped_grad(
        loss,
        argnums=0,
        batch_argnums=1,
        clipping_norm=1.0,
    )
    assert active_backend() is None

    result, returned_state = executable(
        torch.tensor(0.0), torch.tensor([1.0, 2.0]), state=state
    )

    assert result.pytree.item() == pytest.approx(-2.0)
    assert returned_state is state
    assert active_backend().name == "torch"
