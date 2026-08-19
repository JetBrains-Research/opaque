# Copyright (c) 2025 Opaque Authors
# SPDX-License-Identifier: Apache-2.0
"""Legacy checkpoint vmap rule moved to ``opaque-torch``; this module delegates."""

from __future__ import annotations

from opaque.api.torch.backend._checkpoint_compat import (
    apply_noop_save_inputs as apply,
)

__all__ = ["apply"]
