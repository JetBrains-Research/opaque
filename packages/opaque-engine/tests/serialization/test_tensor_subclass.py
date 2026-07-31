"""Tensor-subclass dispatch, shape validation, and inert tree specs."""

from __future__ import annotations

import optree
import pytest
import torch
import torch.nn as nn

from opaque.functional import make_functional
from opaque.serialization import from_state_dict, state_dict


def test_named_parameters_roundtrip() -> None:
    module = nn.Linear(3, 2)
    saved = state_dict(dict(module.named_parameters()))

    assert sorted(saved) == ["bias", "weight"]

    template = {k: nn.Parameter(torch.zeros_like(v)) for k, v in saved.items()}
    restored = from_state_dict(template, saved)
    for name, param in module.named_parameters():
        assert torch.equal(restored[name], param)


def test_parameter_template_stays_a_parameter() -> None:
    module = nn.Linear(3, 2)
    saved = state_dict(dict(module.named_parameters()))

    frozen = nn.Parameter(torch.zeros(2, 3), requires_grad=False)
    restored = from_state_dict({"weight": frozen}, saved)

    assert type(restored["weight"]) is nn.Parameter
    assert restored["weight"].requires_grad is False


def test_make_functional_params_roundtrip() -> None:
    """The tuple in the documented training loop is all ``nn.Parameter``."""
    module = nn.Linear(3, 2)
    _fmodel, params = make_functional(module)
    assert all(isinstance(p, nn.Parameter) for p in params)

    saved = state_dict({"params": params})
    assert sorted(saved) == ["params[0]", "params[1]"]

    template = {"params": tuple(torch.zeros_like(p) for p in params)}
    restored = from_state_dict(template, saved)
    for original, loaded in zip(params, restored["params"], strict=True):
        assert torch.equal(original, loaded)


def test_tensor_shape_mismatch_errors() -> None:
    saved = state_dict({"w": torch.ones(2, 3)})
    with pytest.raises(ValueError, match="shape"):
        from_state_dict({"w": torch.zeros(3, 2)}, saved)


def test_tree_spec_is_inert() -> None:
    spec = optree.tree_structure({"a": 1, "b": 2})
    saved = state_dict({"spec": spec, "step": 4})

    assert saved == {"step": 4}

    restored = from_state_dict({"spec": spec, "step": 0}, saved)
    assert restored["spec"] == spec
    assert restored["step"] == 4


def test_unregistered_leaf_raises() -> None:
    with pytest.raises(TypeError, match="no serializer is registered"):
        state_dict({"generator": torch.Generator()})
