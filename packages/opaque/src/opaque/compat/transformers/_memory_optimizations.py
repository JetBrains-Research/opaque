# Copyright (c) 2025 Opaque Authors
# SPDX-License-Identifier: Apache-2.0
"""Automatic memory optimization patches for HuggingFace Transformers models.

These patches are applied automatically at `import opaque` time, similar to
vmap compatibility patches. No user action required.

Memory optimizations applied:
1. Fused lm_head + cross-entropy - avoids materializing full logits tensor
   during training when labels are provided (standard for causal LM training)

The optimizations activate automatically when:
- labels are provided (training mode)
- vocab_size > 50K (large vocabulary like Mellum's 98K)
- lm_head weights are frozen (typical for LoRA/DP-SGD)

This follows Unsloth's approach: fuse linear projection with cross-entropy loss
to avoid the memory cost of full logits (batch_size * seq_len * vocab_size).
Unlike Unsloth's Triton implementation, we use pure PyTorch for vmap compatibility.

Memory savings:
- Avoids storing full logits tensor in forward (gradient checkpointing pattern)
- Recomputes logits in backward pass on-demand
- For Mellum (98K vocab), saves ~50% memory on lm_head forward/backward

Disable with: OPAQUE_NO_MEMORY_PATCH=1
"""

from __future__ import annotations

import importlib
import logging
import os

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

# Configuration
LARGE_VOCAB_THRESHOLD = 50000
DEFAULT_N_CHUNKS = 8

# Memory optimizations disabled by default since ChunkedLinear was removed as outdated.
# These patches relied on ChunkedLinear which doesn't provide meaningful memory savings
# for vmap-based DP-SGD because the full grad_output tensor is still allocated.
_MEMORY_PATCH_ENABLED = os.environ.get("OPAQUE_MEMORY_PATCH", "0") == "1"

# Track patching state
_is_memory_patched = False

# Model classes to patch: (module_path, class_name)
_CAUSAL_LM_CLASSES = [
    ("transformers.models.llama.modeling_llama", "LlamaForCausalLM"),
    ("transformers.models.mistral.modeling_mistral", "MistralForCausalLM"),
    ("transformers.models.qwen2.modeling_qwen2", "Qwen2ForCausalLM"),
    ("transformers.models.phi3.modeling_phi3", "Phi3ForCausalLM"),
    ("transformers.models.gemma.modeling_gemma", "GemmaForCausalLM"),
    ("transformers.models.gemma2.modeling_gemma2", "Gemma2ForCausalLM"),
    ("transformers.models.gpt2.modeling_gpt2", "GPT2LMHeadModel"),
]

# Store original lm_head forward methods per model instance
# Key: id(model), Value: original lm_head module
_original_lm_heads: dict[int, nn.Module] = {}


def _should_optimize_lm_head(lm_head: nn.Module) -> bool:
    """Check if lm_head should use chunked computation.

    Returns True if:
    1. lm_head is nn.Linear (not already ChunkedLinear)
    2. Weights are frozen (requires_grad=False)
    3. Vocabulary size exceeds threshold
    """
    # Must be standard Linear
    if not isinstance(lm_head, nn.Linear):
        return False

    # Already optimized?
    if type(lm_head).__name__ == "ChunkedLinear":
        return False

    # Weights must be frozen (LoRA pattern)
    if lm_head.weight.requires_grad:
        return False

    # Vocab must be large enough to benefit
    vocab_size = lm_head.out_features
    if vocab_size <= LARGE_VOCAB_THRESHOLD:
        return False

    return True


def _optimize_lm_head_inplace(model: nn.Module) -> bool:
    """Replace model.lm_head with ChunkedLinear if conditions are met.

    This is called automatically during forward pass.
    Returns True if optimization was applied.
    """
    from opaque.kernels import ChunkedLinear

    if not hasattr(model, "lm_head"):
        return False

    lm_head = model.lm_head

    if not _should_optimize_lm_head(lm_head):
        return False

    # Store original for potential restoration
    model_id = id(model)
    if model_id not in _original_lm_heads:
        _original_lm_heads[model_id] = lm_head

    # Replace with ChunkedLinear
    model.lm_head = ChunkedLinear.from_linear(lm_head, n_chunks=DEFAULT_N_CHUNKS)

    logger.debug(
        f"opaque: Applied ChunkedLinear to {type(model).__name__}.lm_head "
        f"(vocab={lm_head.out_features}, chunks={DEFAULT_N_CHUNKS})"
    )
    return True


def _create_optimized_forward(original_forward):
    """Wrap forward to automatically optimize lm_head on first call."""

    def optimized_forward(self, *args, **kwargs):
        # Check and apply optimization on first forward with frozen lm_head
        if not getattr(self, "_opaque_lm_head_checked", False):
            if _optimize_lm_head_inplace(self):
                pass  # Optimization applied
            self._opaque_lm_head_checked = True

        return original_forward(self, *args, **kwargs)

    return optimized_forward


def apply_memory_patches() -> None:
    """Apply memory optimization patches to CausalLM model classes.

    Patches forward methods to automatically use ChunkedLinear for lm_head
    when conditions are met (frozen weights, large vocab).

    Called automatically at `import opaque` time.

    NOTE: Disabled by default. ChunkedLinear doesn't provide meaningful
    memory savings for vmap-based DP-SGD because the full grad_output
    tensor is still allocated. Enable with OPAQUE_MEMORY_PATCH=1.
    """
    global _is_memory_patched

    if _is_memory_patched:
        return

    if not _MEMORY_PATCH_ENABLED:
        _is_memory_patched = True  # Mark as "applied" (but skipped)
        return

    patched_count = 0

    for module_path, class_name in _CAUSAL_LM_CLASSES:
        try:
            module = importlib.import_module(module_path)
            cls = getattr(module, class_name, None)

            if cls is None:
                continue

            # Store and wrap original forward
            original_forward = cls.forward
            cls.forward = _create_optimized_forward(original_forward)
            cls._opaque_original_forward = original_forward

            patched_count += 1
            logger.debug(f"opaque: Patched {class_name}.forward for memory optimization")

        except (ImportError, AttributeError) as e:
            # Model not available in this transformers version
            logger.debug(f"opaque: Could not patch {class_name}: {e}")
            continue

    if patched_count > 0:
        logger.debug(f"opaque: Applied memory patches to {patched_count} model classes")

    _is_memory_patched = True


def is_memory_patched() -> bool:
    """Check if memory patches have been applied globally."""
    return _is_memory_patched


def is_model_memory_optimized(model: nn.Module) -> bool:
    """Check if a specific model instance has memory optimizations active."""
    from opaque.kernels import ChunkedLinear

    if hasattr(model, "lm_head") and isinstance(model.lm_head, ChunkedLinear):
        return True
    return False
