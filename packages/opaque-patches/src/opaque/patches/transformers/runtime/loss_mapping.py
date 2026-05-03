# Copyright (c) 2025 Opaque Authors
# SPDX-License-Identifier: Apache-2.0
"""Global loss mapping patch for HuggingFace Transformers."""

def apply_loss_mapping_patch(*, use_fused_loss: bool = True) -> None:
    """Patch HuggingFace LOSS_MAPPING with Opaque causal LM loss."""
    if use_fused_loss is False:
        return

    try:
        from transformers.loss.loss_utils import LOSS_MAPPING
        from opaque.patches.transformers.components.cross_entropy import _opaque_causal_lm_loss
        LOSS_MAPPING["ForCausalLM"] = _opaque_causal_lm_loss
    except ImportError:
        pass
