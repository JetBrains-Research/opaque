# Copyright (c) 2025 Opaque Authors
# SPDX-License-Identifier: Apache-2.0
"""Shared utilities for testing vmap and gradients on models."""

import torch
from opaque.api.engine.clipping import clipped_grad
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


def assert_vmap_grad(model, device, dtype=None):
    """Run the real DP-SGD ``clipped_grad`` (vmap(grad)) pipeline.

    ``dtype`` casts the model first — pass ``torch.bfloat16`` on CUDA to exercise
    the Triton kernel paths (e.g. the fused MoE kernel), which only engage for
    bf16/fp16 CUDA tensors; the default (fp32) runs the pure-torch fallbacks.
    """
    if dtype is not None:
        model = model.to(dtype)
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
    assert len(grads.pytree) > 0
    assert all(torch.isfinite(g).all() for g in grads.pytree.values())


# ----------------------------------------------------------------------------
# MoE helpers — the stacked-weight ``*Experts`` families resolve their HF
# Config / ForCausalLM by ``model_type`` so per-family test files stay thin.
# ----------------------------------------------------------------------------


def build_moe_model(family, device, attn_impl="sdpa", **config_overrides):
    """Build + patch a tiny MoE model. Returns ``(model, modeling_module)``.

    Defaults to ``attn_impl="sdpa"`` — the transformers production default — so
    the suite exercises the SDPA path; pass ``attn_impl="eager"`` for the eager
    reference validation.

    The Opaque MoE patch targets the stacked-weight ``*Experts`` module
    (transformers v5+). On versions where the family still uses the old
    ``*SparseMoeBlock`` (a ModuleList of per-expert MLPs, no stacked experts),
    DP-SGD MoE isn't supported — skip rather than exercise the vmap-broken HF
    forward. Capability check (presence of a ``*Experts`` class), not a version
    number, so it tracks the API rather than the release.
    """
    import importlib

    import pytest

    from opaque.patches import apply_model_patches

    mod = importlib.import_module(f"transformers.models.{family}.modeling_{family}")
    if not any(n.endswith("Experts") for n in dir(mod)):
        import transformers

        pytest.skip(
            f"{family}: stacked *Experts module absent in transformers "
            f"{transformers.__version__} (v5+ feature)"
        )
    cfg_mod = importlib.import_module(
        f"transformers.models.{family}.configuration_{family}"
    )
    config_cls = next(
        getattr(cfg_mod, n)
        for n in dir(cfg_mod)
        if n.endswith("Config") and "PreTrained" not in n
    )
    causal_lm_cls = next(getattr(mod, n) for n in dir(mod) if n.endswith("ForCausalLM"))
    kwargs = get_tiny_config_kwargs()
    kwargs.update(config_overrides)
    config = config_cls(**kwargs)
    config._attn_implementation = attn_impl
    try:
        model = causal_lm_cls(config).to(device)
    except ValueError as e:
        # Some architectures (gpt_oss, deepseek_v4) are eager-only — HF rejects
        # sdpa at init. Fall back to eager rather than fail the family.
        if attn_impl == "eager" or "scaled_dot_product" not in str(e):
            raise
        config._attn_implementation = "eager"
        model = causal_lm_cls(config).to(device)
    apply_model_patches(model, eager_attention=True)
    return model, mod


def experts_forward_patched(modeling_module):
    """True if the family's stacked ``*Experts`` forward is on the Opaque kernel."""
    cls = next(
        (
            getattr(modeling_module, n)
            for n in dir(modeling_module)
            if n.endswith("Experts")
        ),
        None,
    )
    return cls is not None and hasattr(cls.forward, "__opaque_patched__")
