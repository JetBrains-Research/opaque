# Copyright (c) 2025 Opaque Authors
# SPDX-License-Identifier: Apache-2.0
"""Inspect the functorch interpreter stack (vmap/grad/vjp/...).

Shared by the patches that must behave differently inside a transform: batched
tensors have no storage and forbid ``requires_grad_``, and a transform's
internals depend on what is nested around it.

The "is a differentiating transform active" predicate lives in
:mod:`opaque.api.engine.functional._transform_stack` instead — clipping needs
the same answer, and the engine is the package both sides can import.
"""

from __future__ import annotations

import torch


def under_functorch_transform() -> bool:
    """Return whether a functorch transform is active on this thread."""
    try:
        from torch._C._functorch import peek_interpreter_stack

        return peek_interpreter_stack() is not None
    except Exception:  # pragma: no cover - API moved/unavailable
        return False


def prev_grad_mode() -> bool:
    """Return the grad mode captured when the innermost ``grad`` was entered.

    ``torch.func.{grad,vjp}`` unconditionally enable grad inside themselves and
    remember the caller's mode -- see torch's NOTE [grad and vjp interaction
    with no_grad]. Outside such a transform the ambient mode is the answer, and
    it is also the fallback if the private accessor moves: keeping the inner
    graph costs memory, dropping one that is needed costs correctness.
    """
    try:
        from torch._C._functorch import (
            CGradInterpreterPtr,
            TransformType,
            peek_interpreter_stack,
        )

        interpreter = peek_interpreter_stack()
        if interpreter is None or interpreter.key() != TransformType.Grad:
            return torch.is_grad_enabled()
        return CGradInterpreterPtr(interpreter).prevGradMode()
    except Exception:  # pragma: no cover - API moved/unavailable
        return torch.is_grad_enabled()
