# Copyright (c) 2025 Opaque Authors
# SPDX-License-Identifier: Apache-2.0
"""Capability probes moved to ``opaque-torch``; this module re-exports them."""

from __future__ import annotations

from opaque.api.torch.backend._checkpoint_compat import (
    native_checkpoint_support,
    saved_tensor_hooks_guard_scoped,
)

__all__ = ["native_checkpoint_support", "saved_tensor_hooks_guard_scoped"]
