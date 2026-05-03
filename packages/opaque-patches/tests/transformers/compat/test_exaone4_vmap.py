# Copyright (c) 2025 Opaque Authors
# SPDX-License-Identifier: Apache-2.0
"""Exaone4 text decoder under per-example gradients (vmap + compat patches).

Mirrors ``test_olmo2_vmap.py``. Exaone4 has q_norm / k_norm RMSNorms inside
``Exaone4Attention``; rebinding ``Exaone4RMSNorm.forward`` at class level
covers them automatically through the OLMo2-style RMSNorm factory.
"""

from __future__ import annotations

import pytest

pytest.importorskip("transformers")

import torch

from opaque.clipping import clipped_grad
from opaque.functional import make_functional
from opaque.patches.transformers.components.attention import vmap_eager_attention_forward



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
    model = Exaone4ForCausalLM(cfg)
    from opaque.patches import apply_model_patches
    apply_model_patches(model, performance=False, compat=True)
    return model


def test_exaone4_module_eager_attention_is_patched():
    import transformers.models.exaone4.modeling_exaone4 as me4
    from opaque.patches import apply_model_patches
    model = _tiny_exaone4_for_causal_lm()
    apply_model_patches(model, performance=False, compat=True)

    assert me4.eager_attention_forward is vmap_eager_attention_forward


def test_exaone4_text_clipped_grad(device):
    model = _tiny_exaone4_for_causal_lm().to(device)
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


def test_exaone4_forward_no_grad(device):
    """T2: forward pass under no_grad should run on patched stack."""
    model = _tiny_exaone4_for_causal_lm().to(device)
    model.eval()

    torch.manual_seed(0)
    input_ids = torch.randint(0, model.config.vocab_size, (2, 10), device=device)
    attention_mask = torch.ones_like(input_ids)

    with torch.no_grad():
        out = model(input_ids=input_ids, attention_mask=attention_mask)
    assert out.logits.shape == (2, 10, model.config.vocab_size)


def test_exaone4_forward_backward(device):
    """T3: patched model should support standard forward+backward."""
    model = _tiny_exaone4_for_causal_lm().to(device)
    model.train()

    torch.manual_seed(0)
    input_ids = torch.randint(0, model.config.vocab_size, (2, 10), device=device)
    attention_mask = torch.ones_like(input_ids)
    labels = input_ids.clone()

    loss = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels).loss
    loss.backward()
    assert model.lm_head.weight.grad is not None


def test_exaone4_vmap_forward(device):
    """T4: per-example forward under vmap should be batch-shape stable."""
    model = _tiny_exaone4_for_causal_lm().to(device)
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
