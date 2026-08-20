# Copyright (c) 2025 Opaque Authors
# SPDX-License-Identifier: Apache-2.0
"""Backport PyTorch's scoped saved-tensor-hooks guard.

First-order checkpointing under ``torch.func`` relies on saved-tensor hooks.
PyTorch's internal compile wrapper cannot expose the transform stack, so retain
the established first-order behavior there while guarding known higher-order
compositions.
"""

from __future__ import annotations

import functools

from opaque.api.engine.functional._transform_stack import (
    under_differentiating_transform,
)

_HIGHER_ORDER_MESSAGE = (
    "torch.func transforms (grad, vjp, jacrev, hessian) don't support saved "
    "tensor hooks (e.g. torch.autograd.graph.save_on_cpu) under higher-order "
    "differentiation such as grad(grad) or hessian: the saved-tensor "
    "round-trip severs the higher-order graph and would silently produce "
    "incorrect gradients. First-order use is supported."
)


def apply() -> None:
    """Install the upstream-compatible scoped saved-tensor-hooks guard."""
    import torch._functorch.eager_transforms as eager

    for name in ("grad_and_value_impl", "_vjp_with_argnums"):
        unguarded = getattr(getattr(eager, name, None), "__wrapped__", None)
        if unguarded is not None:
            setattr(
                eager, name, _disable_saved_tensor_hooks_for_higher_order(unguarded)
            )


def _disable_saved_tensor_hooks_for_higher_order(fn):
    """Apply PyTorch's scoped hook guard to a pre-upstream transform."""

    @functools.wraps(fn)
    def dispatch(*args, **kwargs):
        if not under_differentiating_transform(when_compiling=False):
            return fn(*args, **kwargs)
        import torch

        with torch.autograd.graph.disable_saved_tensors_hooks(_HIGHER_ORDER_MESSAGE):
            return fn(*args, **kwargs)

    return dispatch
