# Copyright (c) 2025 Opaque Authors
# SPDX-License-Identifier: Apache-2.0
"""Helpers for active functorch transforms."""

from __future__ import annotations


def under_functorch_transform() -> bool:
    """Return whether a functorch transform is active on this thread."""
    try:
        from torch._C._functorch import peek_interpreter_stack

        return peek_interpreter_stack() is not None
    except Exception:  # pragma: no cover - API moved/unavailable
        return False
