"""PyTorch module adaptation for Opaque's callable APIs."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any, NamedTuple

import torch


class _MappingOutput(NamedTuple):
    """Pytree-traversable stand-in for a non-dict Mapping model output.

    A named tuple so ``vmap``/``grad`` traverse the tensors inside, and
    dict-like (``items`` + ``__setitem__`` + ``__copy__``) so
    ``with_batch_dim`` squeezes the leading batch dimension through the
    same path it uses for the Mapping this replaces — that is, only when
    the wrapper actually added a batch dimension.
    """

    output_type: type
    values: dict[str, Any]

    def items(self) -> Any:
        return self.values.items()

    def __setitem__(self, key: str, value: Any) -> None:
        self.values[key] = value

    def __copy__(self) -> _MappingOutput:
        return _MappingOutput(self.output_type, dict(self.values))


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
            # Any batch-dim squeeze already happened inside
            # ``with_batch_dim`` via the dict-like protocol above, under
            # its ``conditional and unsqueezed`` contract; unwrap only.
            return output.output_type(**output.values)
        return output

    return adapted_module


def make_functional(
    module: torch.nn.Module,
    disable_autograd_tracking: bool = False,
    partition_trainable: bool = False,
    *,
    hf_batch_adaptation: bool = False,
) -> Any:
    """Return a callable that evaluates ``module`` with explicit parameters.

    With ``partition_trainable=False``, parameters are returned as an ordered
    tuple. With ``partition_trainable=True``, trainable and frozen parameters
    are returned as separate name-keyed dictionaries.

    By default the returned callable is a plain ``torch.func.functional_call``
    wrapper. Pass ``hf_batch_adaptation=True`` to additionally wrap it with
    per-example batch-dimension handling for Hugging Face-style keyword
    arguments (``input_ids``/``attention_mask``/``labels``/... plus the
    module's ``main_input_name``) and Mapping-output round-tripping — the
    shape the ``DPTrainer`` integration expects. Leave it off for modules
    whose forward happens to use those keyword names with other ranks.
    """
    params = dict(module.named_parameters())
    if disable_autograd_tracking:
        params = {name: parameter.detach() for name, parameter in params.items()}

    def _finalize(functional_module: Callable[..., Any]) -> Callable[..., Any]:
        if hf_batch_adaptation:
            return _adapt_functional_module(functional_module, module)
        return functional_module

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

        return _finalize(functional_module), trainable, frozen

    names = tuple(params)
    values = tuple(params.values())

    def functional_module(explicit_params: Any, *args: Any, **kwargs: Any) -> Any:
        return torch.func.functional_call(
            module,
            dict(zip(names, explicit_params, strict=True)),
            args,
            kwargs,
        )

    return _finalize(functional_module), values


__all__ = ["make_functional"]
