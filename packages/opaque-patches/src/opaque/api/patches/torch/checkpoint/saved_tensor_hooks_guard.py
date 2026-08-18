# Copyright (c) 2025 Opaque Authors
# SPDX-License-Identifier: Apache-2.0
"""Restrict torch's saved-tensor-hooks guard to higher-order transforms.

``torch.func.{grad,vjp}`` wrap their internals with a decorator that disables
saved-tensor hooks unconditionally. Non-reentrant checkpoint (and
``save_on_cpu``) are built on those hooks, so the guard blocks them even for a
single first-order transform.

Rebind each decorated implementation to a dispatcher that keeps the guard while
another ``grad`` / ``vjp`` / ``jvp`` is already active and skips it otherwise:
first-order gets its hooks, higher-order keeps raising torch's message.

Applied only when torch has not already scoped the guard itself.
"""

from __future__ import annotations

import functools

from opaque.api.engine.functional._transform_stack import (
    under_differentiating_transform,
)


def apply() -> None:
    """Scope the transform guard around saved-tensor hooks to higher order."""
    import torch._functorch.eager_transforms as eager

    for name in ("grad_and_value_impl", "_vjp_with_argnums"):
        guarded = getattr(eager, name, None)
        unguarded = getattr(guarded, "__wrapped__", None)
        if unguarded is not None:
            setattr(eager, name, _higher_order_only(guarded, unguarded))


def _higher_order_only(guarded, unguarded):
    """Route to ``guarded`` only when an outer differentiating transform is active.

    The dispatcher runs before the transform pushes its own level, so anything
    it sees on the stack encloses it.
    """

    @functools.wraps(unguarded)
    def dispatch(*args, **kwargs):
        target = guarded if under_differentiating_transform() else unguarded
        return target(*args, **kwargs)

    return dispatch
