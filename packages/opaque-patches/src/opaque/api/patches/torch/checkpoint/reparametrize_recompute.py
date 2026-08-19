# Copyright (c) 2025 Opaque Authors
# SPDX-License-Identifier: Apache-2.0
"""Functional-parameter rebind patch moved to ``opaque-torch``; this module delegates."""

from __future__ import annotations

from opaque.api.torch.backend._checkpoint_compat import (
    apply_reparametrize_recompute as apply,
)

__all__ = ["apply"]
