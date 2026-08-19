# Copyright (c) 2025 Opaque Authors
# SPDX-License-Identifier: Apache-2.0
"""First-order backward patch moved to ``opaque-torch``; this module delegates."""

from __future__ import annotations

from opaque.torch.checkpoint import apply_create_graph as apply

__all__ = ["apply"]
