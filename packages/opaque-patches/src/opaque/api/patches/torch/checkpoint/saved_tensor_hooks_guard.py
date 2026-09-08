# Copyright (c) 2025 Opaque Authors
# SPDX-License-Identifier: Apache-2.0
"""Backport PyTorch's scoped saved-tensor-hooks guard."""

from __future__ import annotations

from opaque.api.engine.functional._saved_tensors import (
    _disable_saved_tensor_hooks_for_higher_order as _disable_saved_tensor_hooks_for_higher_order,
)
from opaque.api.engine.functional._saved_tensors import apply_saved_tensor_hooks_guard


def apply() -> None:
    """Install the upstream-compatible scoped saved-tensor-hooks guard."""
    apply_saved_tensor_hooks_guard()
