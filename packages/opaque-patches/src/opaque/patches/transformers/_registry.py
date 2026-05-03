# Copyright (c) 2025 Opaque Authors
# SPDX-License-Identifier: Apache-2.0
"""Source of truth for Hugging Face model architecture support."""

from __future__ import annotations

import torch.nn as nn

SUPPORTED_FAMILIES = [
    "cohere", "cohere2", "exaone4", "gemma", "gemma2", "gemma3", 
    "glm4", "granite", "llama", "ministral", "mistral", 
    "olmo2", "olmo3", "phi3", "qwen2", "qwen3", "smollm3"
]

def supported_families() -> list[str]:
    """Return a list of all HuggingFace model families with patching support."""
    return list(SUPPORTED_FAMILIES)

def detect_family(model: nn.Module) -> str | None:
    """Detect the model family from the model config."""
    config = getattr(model, "config", None)
    if config:
        model_type = getattr(config, "model_type", None)
        if model_type == "gemma3_text":
            return "gemma3"
        return model_type
    return None

__all__ = [
    "SUPPORTED_FAMILIES",
    "supported_families",
    "detect_family",
]
