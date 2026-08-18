# Copyright (c) 2025 Opaque Authors
# SPDX-License-Identifier: Apache-2.0
"""Ask what ``torch.func`` transforms a call is nested inside.

Each transform pushes an interpreter level for the duration of the call, so a
callee can read the stack to see what encloses it. Clipping reads it to decide
whether its per-example gradients are values or an intermediate that an
enclosing transform still has to differentiate; the checkpoint patches read it
to scope torch's saved-tensor-hooks guard to higher-order differentiation.

Lives in the engine because both the clipping transform and the patches need
it, and ``opaque-patches`` depends on ``opaque-engine``, not the reverse.
"""

from __future__ import annotations


def under_differentiating_transform() -> bool:
    """Return whether a ``grad`` / ``vjp`` / ``jvp`` transform is active.

    ``vmap`` and ``functionalize`` levels are ignored: they neither consume nor
    produce derivatives. A caller running inside its own transform sees that
    transform; a caller running before its level is pushed does not.

    The stack is read through a private ``torch._C._functorch`` API. If that
    moves, answer ``True``: callers use this to decide whether to *skip*
    building a graph, and skipping one that is needed is the answer that
    silently changes results.
    """
    try:
        from torch._C._functorch import TransformType, get_interpreter_stack

        differentiating = (TransformType.Grad, TransformType.Jvp)
        stack = get_interpreter_stack() or ()
        return any(interpreter.key() in differentiating for interpreter in stack)
    except Exception:  # pragma: no cover - API moved/unavailable
        return True
