# Copyright (c) 2025 Opaque Authors
# SPDX-License-Identifier: Apache-2.0
"""HF-golden checks for Exaone4 fused-CE forward wrapper semantics."""

from __future__ import annotations

import copy
import types

import torch

from opaque.performance.huggingface.kernel_patches import (
    _make_fused_ce_causal_lm_forward,
)


def _tiny_exaone4_for_causal_lm():
    from transformers.models.exaone4.configuration_exaone4 import Exaone4Config
    from transformers.models.exaone4.modeling_exaone4 import Exaone4ForCausalLM

    cfg = Exaone4Config(
        vocab_size=128,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        max_position_embeddings=128,
    )
    cfg._attn_implementation = "eager"
    return Exaone4ForCausalLM(cfg)


def _bind_fused_forward(model):
    # Use unbound function object, matching how kernel_patches patches classes.
    original = type(model).forward
    fused = _make_fused_ce_causal_lm_forward(original)
    model.forward = types.MethodType(fused, model)


def test_exaone4_fused_wrapper_matches_hf_with_labels_on_cpu():
    """T1: CPU path should be numerically identical to stock HF forward."""
    torch.manual_seed(0)
    base = _tiny_exaone4_for_causal_lm()
    fused = copy.deepcopy(base)
    _bind_fused_forward(fused)

    input_ids = torch.randint(0, base.config.vocab_size, (2, 9))
    attention_mask = torch.ones_like(input_ids)
    labels = input_ids.clone()

    with torch.no_grad():
        out_base = base(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            return_dict=True,
        )
        out_fused = fused(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            return_dict=True,
        )

    assert torch.allclose(out_base.logits, out_fused.logits, atol=1e-6, rtol=1e-5)
    assert torch.allclose(out_base.loss, out_fused.loss, atol=1e-6, rtol=1e-5)


def test_exaone4_fused_wrapper_matches_hf_logits_to_keep():
    """T1: wrapper must preserve HF logits slicing semantics."""
    torch.manual_seed(0)
    base = _tiny_exaone4_for_causal_lm()
    fused = copy.deepcopy(base)
    _bind_fused_forward(fused)

    input_ids = torch.randint(0, base.config.vocab_size, (2, 9))
    attention_mask = torch.ones_like(input_ids)

    with torch.no_grad():
        out_base = base(
            input_ids=input_ids,
            attention_mask=attention_mask,
            logits_to_keep=2,
            return_dict=True,
        )
        out_fused = fused(
            input_ids=input_ids,
            attention_mask=attention_mask,
            logits_to_keep=2,
            return_dict=True,
        )

    assert (
        out_base.logits.shape
        == out_fused.logits.shape
        == (2, 2, base.config.vocab_size)
    )
    assert torch.allclose(out_base.logits, out_fused.logits, atol=1e-6, rtol=1e-5)
