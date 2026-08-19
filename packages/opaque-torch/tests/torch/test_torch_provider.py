"""Public contract for the Torch provider wheel."""

from __future__ import annotations

import torch

from opaque import ops
from opaque.api.engine.backend import active_backend, clear_backend, use_backend
from opaque.random import key
from opaque.serialization import from_state_dict, state_dict
from opaque.torch import torch_backend
from opaque.torch.random import generator_from_key, set_reproducible_pytorch_seed


def test_torch_array_activates_provider_and_dispatches() -> None:
    clear_backend()

    result = ops.square(torch.tensor([2.0, 3.0]))

    assert torch.equal(result, torch.tensor([4.0, 9.0]))
    assert active_backend() is not None
    assert active_backend().name == "torch"


def test_torch_backend_factory_returns_provider_identity() -> None:
    backend = torch_backend()

    assert backend.name == "torch"


def test_torch_backend_can_be_selected_explicitly() -> None:
    clear_backend()
    backend = torch_backend()

    with use_backend(backend):
        result = ops.square(torch.tensor([2.0, 3.0]))
        assert active_backend() is backend

    assert torch.equal(result, torch.tensor([4.0, 9.0]))
    assert active_backend() is None


def test_provider_registers_tensor_and_parameter_serializers() -> None:
    torch_backend()
    parameter = torch.nn.Parameter(torch.tensor([1.0, 2.0]))

    restored = from_state_dict(
        torch.nn.Parameter(torch.zeros(2)), state_dict(parameter)
    )

    assert isinstance(restored, torch.nn.Parameter)
    assert torch.equal(restored, parameter)


def test_torch_rng_helpers_are_provider_specific() -> None:
    rng_key = key(42)
    first = torch.randn(8, generator=generator_from_key(rng_key))
    second = torch.randn(8, generator=generator_from_key(rng_key))
    assert torch.equal(first, second)

    set_reproducible_pytorch_seed(rng_key)
    first = torch.randn(8)
    set_reproducible_pytorch_seed(rng_key)
    second = torch.randn(8)
    assert torch.equal(first, second)
