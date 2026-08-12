# Copyright (c) 2025 Opaque Authors
# SPDX-License-Identifier: Apache-2.0
"""Run a first-order transform's internal backward with ``create_graph=False``.

``torch.func.grad`` builds its internal backward with ``create_graph=True``. With
activation checkpointing that traps the recomputed activations in the (unused,
for first-order) inner graph, defeating the memory savings. opaque is first-order
only (``vmap(grad(...))``), so forcing ``create_graph=False`` is safe and is the
whole point of checkpointing under the transform.

Applied only on the patched path; torch's native fix conditions ``create_graph``
precisely (keeping it for higher-order), which opaque does not need.
"""

from __future__ import annotations


def apply() -> None:
    """Install the first-order checkpoint backward patch."""
    import torch._functorch.eager_transforms as eager

    orig = eager._autograd_grad

    # Keep the ``create_graph`` keyword name: functorch calls this with
    # ``create_graph=True``. The value is intentionally ignored — we always
    # force False for first-order opaque (see module docstring).
    def _autograd_grad(
        outputs, inputs, grad_outputs=None, retain_graph=False, create_graph=True
    ):
        return orig(
            outputs, inputs, grad_outputs, retain_graph=retain_graph, create_graph=False
        )

    eager._autograd_grad = _autograd_grad
