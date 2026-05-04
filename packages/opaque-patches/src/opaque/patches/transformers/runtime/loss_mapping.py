# Copyright (c) 2025 Opaque Authors
# SPDX-License-Identifier: Apache-2.0
"""Global loss mapping patch for HuggingFace Transformers."""


def apply_loss_mapping_patch(*, cross_entropy: bool = True) -> None:
    """Patch HuggingFace ``LOSS_MAPPING`` with Opaque causal LM loss.

    Triggered automatically by :func:`opaque.patches.apply_model_patches`
    when any model is patched with ``cross_entropy=True``.  Patching is
    idempotent — the global ``LOSS_MAPPING["ForCausalLM"]`` key is set
    to the opaque variant (no-op when already set).
    """
    if cross_entropy is False:
        return

    try:
        from transformers.loss.loss_utils import LOSS_MAPPING
        from opaque.patches.transformers.components.cross_entropy import (
            _opaque_causal_lm_loss,
        )

        LOSS_MAPPING["ForCausalLM"] = _opaque_causal_lm_loss
    except ImportError:
        pass
