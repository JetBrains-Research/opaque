# Copyright (c) 2025 Opaque Authors
# SPDX-License-Identifier: Apache-2.0
"""vmap-safe ``save_on_cpu`` patch moved to ``opaque-torch``; this module delegates."""

from __future__ import annotations

from opaque.torch.checkpoint import apply_save_on_cpu as apply

__all__ = ["apply"]
