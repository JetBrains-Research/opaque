"""PyTorch module adaptation for Opaque's callable APIs."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any, NamedTuple

import torch


class _MappingOutput(NamedTuple):
    output_type: type
    values: dict[str, Any]


def _adapt_functional_module(
    functional_module: Callable[..., Any], module: torch.nn.Module
) -> Callable[..., Any]:
    from opaque.functional import with_batch_dim

    def pytree_functional_module(*args: Any, **kwargs: Any) -> Any:
        output = functional_module(*args, **kwargs)
        if isinstance(output, Mapping) and type(output) is not dict:
            return _MappingOutput(type(output), dict(output))
        return output

    batch_kwargs = {
        "input_ids": 2,
        "attention_mask": 2,
        "labels": 2,
        "position_ids": 2,
        "inputs_embeds": 3,
    }
    main_input_name = getattr(module, "main_input_name", None)
    if isinstance(main_input_name, str):
        batch_kwargs.setdefault(
            main_input_name,
            {"input_features": 3, "pixel_values": 4}.get(main_input_name, 2),
        )

    batchified_module = with_batch_dim(
        pytree_functional_module,
        batch_argnums=(1,),
        batch_kwargs=batch_kwargs,
        min_ndim=2,
    )

    def adapted_module(*args: Any, **kwargs: Any) -> Any:
        output = batchified_module(*args, **kwargs)
        if isinstance(output, _MappingOutput):
            return output.output_type(**output.values)
        return output

    return adapted_module


def make_functional(
    module: torch.nn.Module,
    disable_autograd_tracking: bool = False,
    partition_trainable: bool = False,
) -> Any:
    """Return a callable that evaluates ``module`` with explicit parameters.

    With ``partition_trainable=False``, parameters are returned as an ordered
    tuple. With ``partition_trainable=True``, trainable and frozen parameters
    are returned as separate name-keyed dictionaries.
    """
    params = dict(module.named_parameters())
    if disable_autograd_tracking:
        params = {name: parameter.detach() for name, parameter in params.items()}

    if partition_trainable:
        original = dict(module.named_parameters())
        trainable = {
            name: params[name]
            for name, parameter in original.items()
            if parameter.requires_grad
        }
        frozen = {
            name: params[name]
            for name, parameter in original.items()
            if not parameter.requires_grad
        }

        def functional_module(explicit_params: Any, *args: Any, **kwargs: Any) -> Any:
            return torch.func.functional_call(module, explicit_params, args, kwargs)

        return _adapt_functional_module(functional_module, module), trainable, frozen

    names = tuple(params)
    values = tuple(params.values())

    def functional_module(explicit_params: Any, *args: Any, **kwargs: Any) -> Any:
        return torch.func.functional_call(
            module,
            dict(zip(names, explicit_params, strict=True)),
            args,
            kwargs,
        )

    return _adapt_functional_module(functional_module, module), values


__all__ = ["make_functional"]
