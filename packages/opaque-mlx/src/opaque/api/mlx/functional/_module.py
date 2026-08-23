"""MLX module adaptation for Opaque's explicit-parameter callables."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from threading import RLock
from typing import Any
from weakref import ReferenceType, ref

import mlx.core as mx
import mlx.nn as nn
from opaque.pytree import merge


@dataclass
class _BindingGuard:
    lock: Any = field(default_factory=RLock)
    bindings: list[Mapping[str, Any]] = field(default_factory=list)


_binding_guards: dict[int, tuple[ReferenceType[nn.Module], _BindingGuard]] = {}
_binding_guards_lock = RLock()


def _binding_guard(module: nn.Module) -> _BindingGuard:
    module_id = id(module)
    with _binding_guards_lock:
        entry = _binding_guards.get(module_id)
        if entry is not None and entry[0]() is module:
            return entry[1]

        def remove_guard(reference: ReferenceType[nn.Module]) -> None:
            with _binding_guards_lock:
                current = _binding_guards.get(module_id)
                if current is not None and current[0] is reference:
                    del _binding_guards[module_id]

        guard = _BindingGuard()
        _binding_guards[module_id] = ref(module, remove_guard), guard
    return guard


def _detach_tree(tree: Any) -> Any:
    if isinstance(tree, Mapping):
        return {name: _detach_tree(value) for name, value in tree.items()}
    if isinstance(tree, (list, tuple)):
        return type(tree)(_detach_tree(value) for value in tree)
    return mx.stop_gradient(tree) if isinstance(tree, mx.array) else tree


_MISSING = object()


def _frozen_parameters(parameters: Any, trainable: Any = _MISSING) -> Any:
    if isinstance(parameters, Mapping):
        frozen: dict[str, Any] = {}
        trainable_mapping = trainable if isinstance(trainable, Mapping) else {}
        for name, value in parameters.items():
            child = _frozen_parameters(value, trainable_mapping.get(name, _MISSING))
            if child is not _MISSING:
                frozen[name] = child
        return frozen
    if isinstance(parameters, (list, tuple)):
        frozen_list = []
        has_frozen = False
        for index, value in enumerate(parameters):
            child = _frozen_parameters(
                value,
                trainable[index]
                if isinstance(trainable, type(parameters)) and index < len(trainable)
                else _MISSING,
            )
            has_frozen |= child is not _MISSING
            frozen_list.append(None if child is _MISSING else child)
        return type(parameters)(frozen_list) if has_frozen else _MISSING
    if trainable is _MISSING:
        return parameters
    return _MISSING


def _complete_parameters(frozen: Any, explicit_params: Mapping[str, Any]) -> Any:
    return merge(frozen, explicit_params)


def make_functional(
    module: nn.Module,
    disable_autograd_tracking: bool = False,
    partition_trainable: bool = False,
) -> Any:
    """Return a callable that evaluates an MLX module with explicit parameters.

    Unless autograd tracking is disabled, the returned parameter tree directly
    references ``module``'s current arrays. Each invocation temporarily binds
    its explicit parameters to the original module and restores the preceding
    bindings in ``finally``. Calls for one module share a re-entrant lock, so
    nested calls restore their immediate parent bindings and concurrent callers
    are serialized.

    With ``partition_trainable=True``, trainable and frozen parameter trees
    reflect MLX's ``_no_grad`` state. The callable accepts either the complete
    parameter tree or a sparse trainable tree, merging the latter with the
    captured frozen tree before binding it.

    This adapter targets conventional eager, parameter-only MLX modules. It
    does not roll back Python-side state mutated by ``__call__``, provide
    parallel execution, or guarantee isolation under arbitrary tracing or
    compiled transformations.
    """
    if not isinstance(module, nn.Module):
        raise TypeError(f"Expected an mlx.nn.Module, got {type(module).__name__}")

    parameters = module.parameters()
    if disable_autograd_tracking:
        parameters = _detach_tree(parameters)

    guard = _binding_guard(module)

    def functional_module(
        complete_params: Mapping[str, Any], *args: Any, **kwargs: Any
    ) -> Any:
        with guard.lock:
            guard.bindings.append(module.parameters())
            try:
                module.update(complete_params)
                result = module(*args, **kwargs)
            finally:
                module.update(guard.bindings.pop())
        return result

    if partition_trainable:
        trainable = module.trainable_parameters()
        if disable_autograd_tracking:
            trainable = _detach_tree(trainable)
        frozen = _frozen_parameters(parameters, trainable)

        def partitioned_functional_module(
            explicit_params: Mapping[str, Any], *args: Any, **kwargs: Any
        ) -> Any:
            return functional_module(
                _complete_parameters(frozen, explicit_params), *args, **kwargs
            )

        return partitioned_functional_module, trainable, frozen
    return functional_module, parameters


__all__ = ["make_functional"]
