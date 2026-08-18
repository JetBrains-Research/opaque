# Copyright (c) 2025 Opaque Authors
# SPDX-License-Identifier: Apache-2.0
"""Build a transform's internal backward graph only when grad mode asks for it.

``torch.func.grad`` always builds its internal backward with
``create_graph=True``. Under activation checkpointing that keeps every
recomputed activation alive in a graph nobody differentiates, cancelling the
saving that is the point of checkpointing under the transform.

Condition it the way ``vjp`` already conditions its own backward: on the grad
mode the transform was entered with. Nothing can differentiate the result of a
transform entered under ``no_grad`` -- not autograd, and not an enclosing
transform, since those enable grad -- so the graph is dead weight exactly there
and is built everywhere else.

Applied only when torch lacks the native conditioning.
"""

from __future__ import annotations

from opaque.api.patches.torch.functorch_transform import prev_grad_mode


def apply() -> None:
    """Install the grad-mode-conditioned first-order backward."""
    import torch._functorch.eager_transforms as eager

    orig = eager._autograd_grad

    # ``grad_outputs is None`` marks ``grad_and_value_impl``'s internal backward;
    # ``vjp``'s user-facing backward already takes ``create_graph`` from its caller.
    def _autograd_grad(
        outputs, inputs, grad_outputs=None, retain_graph=False, create_graph=True
    ):
        if create_graph and grad_outputs is None:
            create_graph = prev_grad_mode()
        return orig(
            outputs,
            inputs,
            grad_outputs,
            retain_graph=retain_graph,
            create_graph=create_graph,
        )

    eager._autograd_grad = _autograd_grad
