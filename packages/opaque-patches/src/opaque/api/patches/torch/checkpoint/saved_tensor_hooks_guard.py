# Copyright (c) 2025 Opaque Authors
# SPDX-License-Identifier: Apache-2.0
"""Lift torch's blanket saved-tensor-hooks guard inside first-order transforms.

``torch.func.{grad,vjp}`` wrap their internals with a decorator that disables
saved-tensor hooks unconditionally. Non-reentrant checkpoint (and ``save_on_cpu``)
are built on those hooks, so the guard blocks them even for a single first-order
transform. We strip the decorator by rebinding to its ``__wrapped__`` original.

Applied only when torch has not already scoped the guard to higher-order. opaque
is first-order only, so removing the guard wholesale is acceptable here; torch's
native fix keeps higher-order differentiation raising instead.
"""

from __future__ import annotations


def apply() -> None:
    import torch._functorch.eager_transforms as eager

    for name in ("grad_and_value_impl", "_vjp_with_argnums"):
        fn = getattr(eager, name, None)
        wrapped = getattr(fn, "__wrapped__", None)
        if wrapped is not None:
            setattr(eager, name, wrapped)
