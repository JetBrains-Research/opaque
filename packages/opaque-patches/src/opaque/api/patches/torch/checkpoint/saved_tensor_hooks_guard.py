# Copyright (c) 2025 Opaque Authors
# SPDX-License-Identifier: Apache-2.0
"""Saved-tensor-hooks guard patch moved to ``opaque-torch``; this module delegates."""

from __future__ import annotations

from opaque.torch.checkpoint import (
    apply_saved_tensor_hooks_guard as apply,
)

__all__ = ["apply"]
