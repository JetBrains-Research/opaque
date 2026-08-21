# Copyright (c) 2025 Opaque Authors
# SPDX-License-Identifier: Apache-2.0
"""Introspection of the active ``torch.func`` interpreter stack.

Patches that must behave differently inside a functional transform ask here
rather than probing ``torch._C._functorch`` themselves. The private API moves
between releases, so the probe is written once, in the wheel that owns torch.
"""

from __future__ import annotations

__all__ = ["under_functorch_transform"]


def under_functorch_transform() -> bool:
    """Return whether any functorch transform is active on this thread.

    True under ``vmap`` as well as under the differentiating transforms; use
    it for code that must avoid batched-tensor-hostile fast paths regardless of
    which transform is on the stack.
    """
    try:
        from torch._C._functorch import peek_interpreter_stack

        return peek_interpreter_stack() is not None
    except Exception:  # pragma: no cover - private API moved/unavailable
        return False
