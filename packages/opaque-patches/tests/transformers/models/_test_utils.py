# Copyright (c) 2025 Opaque Authors
# SPDX-License-Identifier: Apache-2.0
"""Shared utilities for testing vmap and gradients on models."""

import torch
from opaque.clipping import clipped_grad
from opaque.functional import make_functional

def get_tiny_config_kwargs():
    return {
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

def assert_forward_no_grad(model, device):
    model.eval()
    torch.manual_seed(0)
    input_ids = torch.randint(0, model.config.vocab_size, (2, 10), device=device)
    attention_mask = torch.ones_like(input_ids)
    with torch.no_grad():
        out = model(input_ids=input_ids, attention_mask=attention_mask)
    assert out.logits.shape == (2, 10, model.config.vocab_size)

def assert_forward_backward(model, device):
    model.train()
    torch.manual_seed(0)
    input_ids = torch.randint(0, model.config.vocab_size, (2, 10), device=device)
    attention_mask = torch.ones_like(input_ids)
    labels = input_ids.clone()
    loss = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels).loss
    loss.backward()
    # Assert some gradient exists
    assert any(p.grad is not None for p in model.parameters())

def assert_vmap_forward(model, device):
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

def assert_vmap_grad(model, device):
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
