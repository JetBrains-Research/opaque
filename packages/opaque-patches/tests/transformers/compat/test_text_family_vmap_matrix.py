# Copyright (c) 2025 Opaque Authors
# SPDX-License-Identifier: Apache-2.0
"""Text-family vmap ladder checks (T2-T5) on tiny configs."""

from __future__ import annotations

import importlib

import pytest

pytest.importorskip("transformers")

import torch

from opaque.clipping import clipped_grad
from opaque.functional import make_functional
from opaque.patches.transformers.components.attention import vmap_eager_attention_forward

_TEXT_FAMILY_SPECS = {
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
    "olmo3": {
        "module": "transformers.models.olmo3.modeling_olmo3",
        "config_module": "transformers.models.olmo3.configuration_olmo3",
        "config_cls": "Olmo3Config",
        "model_cls": "Olmo3ForCausalLM",
        "extra": {},
    },
    "glm4": {
        "module": "transformers.models.glm4.modeling_glm4",
        "config_module": "transformers.models.glm4.configuration_glm4",
        "config_cls": "Glm4Config",
        "model_cls": "Glm4ForCausalLM",
        "extra": {},
    },
    "gemma3": {
        "module": "transformers.models.gemma3.modeling_gemma3",
        "config_module": "transformers.models.gemma3.configuration_gemma3",
        "config_cls": "Gemma3TextConfig",
        "model_cls": "Gemma3ForCausalLM",
        # Gemma3 needs an explicit head_dim and exercises both sliding +
        # global RoPE branches; pin sliding_window so the layer pattern
        # actually fires under vmap.
        "extra": {
            "head_dim": 16,
            "sliding_window": 8,
            "sliding_window_pattern": 2,
        },
    },
    "exaone4": {
        "module": "transformers.models.exaone4.modeling_exaone4",
        "config_module": "transformers.models.exaone4.configuration_exaone4",
        "config_cls": "Exaone4Config",
        "model_cls": "Exaone4ForCausalLM",
        # Exaone4 needs an explicit head_dim because its config validator
        # rejects ``hidden_size // num_attention_heads`` for non-power-of-2
        # combinations of the tiny test geometry.
        "extra": {"head_dim": 16},
    },
}


def _tiny_text_model(family: str):
    spec = _TEXT_FAMILY_SPECS[family]
    cfg_mod = importlib.import_module(spec["config_module"])
    model_mod = importlib.import_module(spec["module"])
    cfg_cls = getattr(cfg_mod, spec["config_cls"])
    model_cls = getattr(model_mod, spec["model_cls"])
    kwargs = {
        "vocab_size": 128,
        "hidden_size": 64,
        "intermediate_size": 128,
        # Gemma3 picks between sliding/global RoPE per layer pair; use 2
        # layers so both branches run. One layer is fine for everyone else.
        "num_hidden_layers": 2 if family == "gemma3" else 1,
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
    model = model_cls(cfg)
    from opaque.patches import apply_model_patches
    apply_model_patches(model, performance=False, compat=True)
    return model

@pytest.mark.parametrize("family", sorted(_TEXT_FAMILY_SPECS))
def test_text_family_module_eager_attention_is_patched(family):
    _tiny_text_model(family)
    module = importlib.import_module(_TEXT_FAMILY_SPECS[family]["module"])
    assert module.eager_attention_forward is vmap_eager_attention_forward


@pytest.mark.parametrize("family", sorted(_TEXT_FAMILY_SPECS))
def test_text_family_forward_no_grad(device, family):
    """T2: forward pass under no_grad should run on patched stack."""
    model = _tiny_text_model(family).to(device)
    model.eval()

    torch.manual_seed(0)
    input_ids = torch.randint(0, model.config.vocab_size, (2, 10), device=device)
    attention_mask = torch.ones_like(input_ids)

    with torch.no_grad():
        out = model(input_ids=input_ids, attention_mask=attention_mask)
    assert out.logits.shape == (2, 10, model.config.vocab_size)


@pytest.mark.parametrize("family", sorted(_TEXT_FAMILY_SPECS))
def test_text_family_forward_backward(device, family):
    """T3: patched model should support standard forward+backward."""
    model = _tiny_text_model(family).to(device)
    model.train()

    torch.manual_seed(0)
    input_ids = torch.randint(0, model.config.vocab_size, (2, 10), device=device)
    attention_mask = torch.ones_like(input_ids)
    labels = input_ids.clone()

    loss = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels).loss
    loss.backward()
    assert model.lm_head.weight.grad is not None


@pytest.mark.parametrize("family", sorted(_TEXT_FAMILY_SPECS))
def test_text_family_vmap_forward(device, family):
    """T4: per-example forward under vmap should be batch-shape stable."""
    model = _tiny_text_model(family).to(device)
    model.eval()

    torch.manual_seed(0)
    batch, seq = 4, 8
    input_ids = torch.randint(0, model.config.vocab_size, (batch, seq), device=device)
    attention_mask = torch.ones(batch, seq, dtype=torch.long, device=device)

    def per_example(ids, mask):
        with torch.no_grad():
            out = model(input_ids=ids, attention_mask=mask)
        return out.logits.mean()

    vmapped = torch.vmap(per_example)(input_ids, attention_mask)
    assert vmapped.shape == (batch,)


@pytest.mark.parametrize("family", sorted(_TEXT_FAMILY_SPECS))
def test_text_family_vmap_grad(device, family):
    """T5: clipped_grad per-example gradient path stays functional."""
    model = _tiny_text_model(family).to(device)
    model.train()

    torch.manual_seed(0)
    batch, seq, vocab = 4, 12, model.config.vocab_size
    input_ids = torch.randint(0, vocab, (batch, seq), device=device)
    attention_mask = torch.ones(batch, seq, dtype=torch.long, device=device)
    labels = input_ids.clone()

    fmodel, trainable, frozen = make_functional(
        model, disable_autograd_tracking=True, partition_trainable=True
    )

    def per_example_loss(trainable_params, frozen_params, ids, mask, lbls):
        all_params = {**frozen_params, **trainable_params}
        outputs = fmodel(all_params, ids, attention_mask=mask, labels=lbls)
        return outputs.loss

    grad_fn, clip_state = clipped_grad(
        per_example_loss, argnums=0, batch_argnums=(2, 3, 4), clipping_norm=1.0
    )
    grads, _state = grad_fn(
        trainable, frozen, input_ids, attention_mask, labels, state=clip_state
    )
    assert len(grads) > 0
