# Copyright (c) 2026 Opaque Authors
# SPDX-License-Identifier: Apache-2.0
"""Compatibility support for saved-tensor hooks under torch.func."""

from __future__ import annotations

import functools
import threading

import torch

_HIGHER_ORDER_MESSAGE = (
    "torch.func transforms (grad, vjp, jacrev, hessian) don't support saved "
    "tensor hooks under higher-order differentiation such as grad(grad) or "
    "hessian: the saved-tensor round-trip severs the higher-order graph and "
    "would silently produce incorrect gradients. First-order use is supported."
)
_COMPILE_MESSAGE = (
    "torch.func transforms (grad, vjp, jacrev, hessian) don't support saved "
    "tensor hooks under torch.compile, where saved-tensor hooks are managed by "
    "AOTAutograd. Use eager execution for saved-tensor hooks with torch.func."
)
_APPLY_LOCK = threading.Lock()
_APPLIED = False


def saved_tensor_hooks_guard_scoped() -> bool:
    """Return whether the running PyTorch scopes its hook guard natively."""
    try:
        from torch._functorch import vmap
    except Exception:  # pragma: no cover - private layout moved
        return False
    return hasattr(vmap, "disable_saved_tensors_hooks_for_higher_order")


def ensure_saved_tensor_hooks_guard() -> None:
    """Enable first-order saved-tensor hooks when PyTorch needs the backport."""
    global _APPLIED
    if saved_tensor_hooks_guard_scoped() or _APPLIED:
        return
    with _APPLY_LOCK:
        if _APPLIED:
            return
        apply_saved_tensor_hooks_guard()
        _APPLIED = True


def apply_saved_tensor_hooks_guard() -> None:
    """Install the upstream-compatible scoped saved-tensor-hooks guard."""
    import torch._functorch.eager_transforms as eager

    for name in ("grad_and_value_impl", "_vjp_with_argnums"):
        unguarded = getattr(getattr(eager, name, None), "__wrapped__", None)
        if unguarded is not None:
            setattr(
                eager, name, _disable_saved_tensor_hooks_for_higher_order(unguarded)
            )


def _disable_saved_tensor_hooks_for_higher_order(fn):
    @functools.wraps(fn)
    def dispatch(*args, **kwargs):
        if torch.compiler.is_compiling():
            with torch.autograd.graph.disable_saved_tensors_hooks(_COMPILE_MESSAGE):
                return fn(*args, **kwargs)
        try:
            from torch._C._functorch import TransformType, get_interpreter_stack

            stack = get_interpreter_stack()
            higher_order = bool(stack) and any(
                interpreter.key() in (TransformType.Grad, TransformType.Jvp)
                for interpreter in stack
            )
        except Exception:  # pragma: no cover - private API moved/unavailable
            higher_order = True
        if higher_order:
            with torch.autograd.graph.disable_saved_tensors_hooks(
                _HIGHER_ORDER_MESSAGE
            ):
                return fn(*args, **kwargs)
        return fn(*args, **kwargs)

    return dispatch
