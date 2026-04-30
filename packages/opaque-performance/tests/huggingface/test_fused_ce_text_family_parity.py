# Copyright (c) 2025 Opaque Authors
# SPDX-License-Identifier: Apache-2.0
"""HF-golden parity checks for fused-CE wrapper on text families (CPU path)."""

from __future__ import annotations

import copy
import importlib
import types

import pytest
import torch

from opaque.performance.huggingface.kernel_patches import (
    _make_fused_ce_causal_lm_forward,
)

_FAMILY_SPECS = {
    "olmo2": {
        "module": "transformers.models.olmo2.modeling_olmo2",
        "config_module": "transformers.models.olmo2.configuration_olmo2",
        "config_cls": "Olmo2Config",
        "model_cls": "Olmo2ForCausalLM",
        "extra": {},
    },
    "olmo3": {
        "module": "transformers.models.olmo3.modeling_olmo3",
        "config_module": "transformers.models.olmo3.configuration_olmo3",
        "config_cls": "Olmo3Config",
        "model_cls": "Olmo3ForCausalLM",
        "extra": {},
    },
    "smollm3": {
        "module": "transformers.models.smollm3.modeling_smollm3",
        "config_module": "transformers.models.smollm3.configuration_smollm3",
        "config_cls": "SmolLM3Config",
        "model_cls": "SmolLM3ForCausalLM",
        "extra": {},
    },
    "ministral": {
        "module": "transformers.models.ministral.modeling_ministral",
        "config_module": "transformers.models.ministral.configuration_ministral",
        "config_cls": "MinistralConfig",
        "model_cls": "MinistralForCausalLM",
        "extra": {"head_dim": 16},
    },
    "glm4": {
        "module": "transformers.models.glm4.modeling_glm4",
        "config_module": "transformers.models.glm4.configuration_glm4",
        "config_cls": "Glm4Config",
        "model_cls": "Glm4ForCausalLM",
        "extra": {},
    },
}


def _tiny_model(family: str):
    spec = _FAMILY_SPECS[family]
    cfg_mod = importlib.import_module(spec["config_module"])
    model_mod = importlib.import_module(spec["module"])
    cfg_cls = getattr(cfg_mod, spec["config_cls"])
    model_cls = getattr(model_mod, spec["model_cls"])
    kwargs = {
        "vocab_size": 128,
        "hidden_size": 64,
        "intermediate_size": 128,
        "num_hidden_layers": 1,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "max_position_embeddings": 128,
        "pad_token_id": 0,
        "bos_token_id": 1,
        "eos_token_id": 2,
        "rope_theta": 10000.0,
    }
    kwargs.update(spec["extra"])
    cfg = cfg_cls(**kwargs)
    cfg._attn_implementation = "eager"
    return model_cls(cfg)


def _bind_fused_forward(model):
    original = type(model).forward
    fused = _make_fused_ce_causal_lm_forward(original)
    model.forward = types.MethodType(fused, model)


@pytest.mark.parametrize("family", sorted(_FAMILY_SPECS))
def test_fused_ce_wrapper_matches_hf_with_labels_on_cpu(family):
    """T1: wrapper should match stock HF numerics when CUDA fused path is disabled."""
    torch.manual_seed(0)
    base = _tiny_model(family)
    wrapped = copy.deepcopy(base)
    _bind_fused_forward(wrapped)

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
        out_wrapped = wrapped(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            return_dict=True,
        )

    assert torch.allclose(out_base.logits, out_wrapped.logits, atol=1e-6, rtol=1e-5)
    assert torch.allclose(out_base.loss, out_wrapped.loss, atol=1e-6, rtol=1e-5)


@pytest.mark.parametrize("family", sorted(_FAMILY_SPECS))
def test_fused_ce_wrapper_preserves_logits_to_keep(family):
    """T1: wrapper must preserve HF logits slicing semantics."""
    torch.manual_seed(0)
    base = _tiny_model(family)
    wrapped = copy.deepcopy(base)
    _bind_fused_forward(wrapped)

    input_ids = torch.randint(0, base.config.vocab_size, (2, 9))
    attention_mask = torch.ones_like(input_ids)

    with torch.no_grad():
        out_base = base(
            input_ids=input_ids,
            attention_mask=attention_mask,
            logits_to_keep=2,
            return_dict=True,
        )
        out_wrapped = wrapped(
            input_ids=input_ids,
            attention_mask=attention_mask,
            logits_to_keep=2,
            return_dict=True,
        )

    assert (
        out_base.logits.shape
        == out_wrapped.logits.shape
        == (
            2,
            2,
            base.config.vocab_size,
        )
    )
    assert torch.allclose(out_base.logits, out_wrapped.logits, atol=1e-6, rtol=1e-5)
