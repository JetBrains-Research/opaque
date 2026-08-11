# Copyright (c) 2025 Opaque Authors
# SPDX-License-Identifier: Apache-2.0
"""Detect whether a functorch transform (vmap/grad/vjp/...) is currently active.

Shared by vmap-safety patches that must behave differently inside a transform
(where batched tensors have no storage and ``requires_grad_`` is forbidden).
"""

from __future__ import annotations


def under_functorch_transform() -> bool:
    """Return whether a functorch transform is active on this thread."""
    try:
        from torch._C._functorch import peek_interpreter_stack

        return peek_interpreter_stack() is not None
    except Exception:  # pragma: no cover - API moved/unavailable
        return False
