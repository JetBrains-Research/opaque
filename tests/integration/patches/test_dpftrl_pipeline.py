"""DP-FTRL ↔ patches end-to-end integration smoke tests.

Mirror of ``test_dpsgd_pipeline.py`` but with the DP-FTRL noise path
(``opaque.dpftrl.noise.mf_noise`` with the identity strategy as the
simplest correlated-noise mechanism). Verifies that patches don't break
the full DP-FTRL pipeline.
"""

from __future__ import annotations

import pytest
import torch

pytest.importorskip("transformers")
pytest.importorskip("peft")

from peft import LoraConfig, get_peft_model  # noqa: E402
from transformers import AutoModelForCausalLM, LlamaConfig  # noqa: E402

from opaque.api.engine.clipping import clipped_grad
from opaque.dpftrl.noise import identity_strategy, mf_noise
from opaque.functional import make_functional
from opaque.patches import apply_model_patches
from opaque.random import key


def _build_tiny_lora_model():
    config = LlamaConfig(
        vocab_size=128,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=64,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
        rope_theta=10000.0,
    )
    model = AutoModelForCausalLM.from_config(config)
    lora_config = LoraConfig(
        r=4, lora_alpha=8,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.0,
    )
    model = get_peft_model(model, lora_config)
    apply_model_patches(
        model,
        performance=False,
        compat=True,
        lora=True,
        activation=False,
        rms_norm=False,
        rope=False,
        cross_entropy=False,
        eager_attention=True,
    )
    return model


def test_dpftrl_full_step_on_patched_lora_model():
    """DP-FTRL pipeline: patched LoRA model → clip → mf_noise (identity) → manual update."""
    model = _build_tiny_lora_model()
    fmodel, trainable, frozen = make_functional(
        model, disable_autograd_tracking=True, partition_trainable=True
    )

    batch_size = 4
    seq_len = 8
    input_ids = torch.randint(0, 128, (batch_size, seq_len))
    attention_mask = torch.ones_like(input_ids)
    labels = input_ids.clone()

    def per_example_loss(params, frozen_params, ids, mask, lbls):
        all_params = {**frozen_params, **params}
        outputs = fmodel(all_params, ids, attention_mask=mask, labels=lbls)
        return outputs.loss

    grad_fn, clip_state = clipped_grad(
        per_example_loss,
        argnums=0,
        batch_argnums=(2, 3, 4),
        clipping_norm=1.0,
    )
    grads, _ = grad_fn(
        trainable, frozen, input_ids, attention_mask, labels, state=clip_state
    )

    # Identity-strategy MF noise reduces to per-step Gaussian (sensitivity
    # 1) — the simplest MF case.
    strategy = identity_strategy()
    noise_fn, noise_state = mf_noise(
        grads,
        strategy,
        noise_multiplier=1.0,
        key=key(0),
    )
    noised, _ = noise_fn(grads, noise_state)

    lr = 0.01
    updated = {
        name: trainable[name] - lr * (noised.pytree[name] / batch_size)
        for name in trainable
    }
    for name, value in updated.items():
        assert torch.isfinite(value).all(), (
            f"non-finite parameter after DP-FTRL step: {name}"
        )
        assert value.shape == trainable[name].shape
