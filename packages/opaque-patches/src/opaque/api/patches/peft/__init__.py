# Copyright (c) 2025 Opaque Authors
# SPDX-License-Identifier: Apache-2.0
"""PEFT integration patches for vmap-compatible LoRA training."""

from __future__ import annotations

from ._router import apply_peft_model_patches

__all__ = [
    "apply_peft_model_patches",
]
