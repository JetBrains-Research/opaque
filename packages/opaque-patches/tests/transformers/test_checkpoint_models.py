# Copyright (c) 2025 Opaque Authors
# SPDX-License-Identifier: Apache-2.0
"""Gradient checkpointing on Hugging Face models under clipped_grad.

The Hugging Face half of the checkpoint concern: forcing non-reentrant
checkpointing on ``PreTrainedModel`` and keeping gradients correct through a
PEFT-wrapped model. The torch-core half is exercised in ``opaque-torch``.
"""

import pytest
import torch

from opaque.patches import apply_runtime_patches

apply_runtime_patches(vmap_checkpointing=True)


@pytest.mark.cuda
class TestCheckpointWithClippedGrad:
    """Test checkpoint with the full clipped_grad pipeline."""

    @pytest.mark.slow
    def test_use_reentrant_override(self, device):
        """Patch overrides use_reentrant=True to False."""
        transformers = pytest.importorskip("transformers")

        config = transformers.AutoConfig.from_pretrained("Qwen/Qwen2-0.5B")
        config.num_hidden_layers = 1
        model = transformers.AutoModelForCausalLM.from_config(config).to(device)

        # Explicitly pass use_reentrant=True — patch should override.
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": True}
        )

        # Check that the stored func actually uses False.
        import functools

        for module in model.modules():
            if hasattr(module, "_gradient_checkpointing_func"):
                func = module._gradient_checkpointing_func
                if isinstance(func, functools.partial):
                    assert func.keywords.get("use_reentrant") is False, (
                        "use_reentrant was not overridden to False"
                    )

    @pytest.mark.slow
    def test_clipped_grad_with_checkpoint_model(self, device):
        """clipped_grad works on a HF model with gradient_checkpointing_enable()."""
        transformers = pytest.importorskip("transformers")
        peft = pytest.importorskip("peft")

        from opaque.api.engine.clipping import clipped_grad
        from opaque.patches import apply_model_patches
        from opaque.torch.functional import make_functional

        config = transformers.AutoConfig.from_pretrained("Qwen/Qwen2-0.5B")
        config.num_hidden_layers = 2

        model = transformers.AutoModelForCausalLM.from_config(config)
        # Model-level patches (vmap-safe RoPE, masking, etc.) must be applied
        # to the base model BEFORE wrapping with PEFT.
        apply_model_patches(model)
        model = peft.get_peft_model(
            model,
            peft.LoraConfig(
                r=8,
                lora_alpha=16,
                target_modules=["q_proj", "v_proj"],
                lora_dropout=0.0,
            ),
        ).to(device)
        # No explicit use_reentrant=False needed — our patch forces it.
        model.gradient_checkpointing_enable()

        tokenizer = transformers.AutoTokenizer.from_pretrained("Qwen/Qwen2-0.5B")
        texts = ["Hello world", "Test input", "Third sample", "Fourth one"]
        inputs = tokenizer(
            texts, return_tensors="pt", padding=True, max_length=16, truncation=True
        )
        input_ids = inputs["input_ids"].to(device)
        labels = input_ids.clone()

        fmodel, trainable, frozen = make_functional(
            model, disable_autograd_tracking=True, partition_trainable=True
        )

        def loss_fn(trainable_params, frozen_params, ids, lbls):
            all_params = {**frozen_params, **trainable_params}
            return fmodel(all_params, ids, labels=lbls).loss

        grad_fn, clip_state = clipped_grad(
            loss_fn, argnums=0, batch_argnums=(2, 3), clipping_norm=1.0
        )
        grads, _ = grad_fn(trainable, frozen, input_ids, labels, state=clip_state)

        assert len(grads.pytree) > 0
        assert all(v.shape == trainable[k].shape for k, v in grads.pytree.items())
        assert any(v.abs().max() > 0 for v in grads.pytree.values()), (
            "All gradients are zero"
        )

    @pytest.mark.slow
    def test_checkpoint_gradient_correctness(self, device):
        """Checkpointed gradients match non-checkpointed for same model."""
        transformers = pytest.importorskip("transformers")
        peft = pytest.importorskip("peft")

        from opaque.api.engine.clipping import clipped_grad
        from opaque.patches import apply_model_patches
        from opaque.torch.functional import make_functional

        config = transformers.AutoConfig.from_pretrained("Qwen/Qwen2-0.5B")
        config.num_hidden_layers = 2

        tokenizer = transformers.AutoTokenizer.from_pretrained("Qwen/Qwen2-0.5B")
        texts = ["Hello world", "Test input"]
        inputs = tokenizer(
            texts, return_tensors="pt", padding=True, max_length=16, truncation=True
        )
        input_ids = inputs["input_ids"].to(device)
        labels = input_ids.clone()

        lora_config = peft.LoraConfig(
            r=8,
            lora_alpha=16,
            target_modules=["q_proj", "v_proj"],
            lora_dropout=0.0,
        )

        # Without checkpoint
        torch.manual_seed(42)
        model_a = transformers.AutoModelForCausalLM.from_config(config)
        apply_model_patches(model_a)
        model_a = peft.get_peft_model(model_a, lora_config).to(device)
        fmodel_a, train_a, frozen_a = make_functional(
            model_a, disable_autograd_tracking=True, partition_trainable=True
        )

        def loss_a(tp, fp, ids, lbls):
            return fmodel_a({**fp, **tp}, ids, labels=lbls).loss

        grad_fn_a, state_a = clipped_grad(
            loss_a, argnums=0, batch_argnums=(2, 3), clipping_norm=float("inf")
        )
        grads_a, _ = grad_fn_a(train_a, frozen_a, input_ids, labels, state=state_a)

        # With checkpoint (same weights)
        torch.manual_seed(42)
        model_b = transformers.AutoModelForCausalLM.from_config(config)
        apply_model_patches(model_b)
        model_b = peft.get_peft_model(model_b, lora_config).to(device)
        # Bare call — patch forces use_reentrant=False automatically.
        model_b.gradient_checkpointing_enable()
        fmodel_b, train_b, frozen_b = make_functional(
            model_b, disable_autograd_tracking=True, partition_trainable=True
        )

        def loss_b(tp, fp, ids, lbls):
            return fmodel_b({**fp, **tp}, ids, labels=lbls).loss

        grad_fn_b, state_b = clipped_grad(
            loss_b, argnums=0, batch_argnums=(2, 3), clipping_norm=float("inf")
        )
        grads_b, _ = grad_fn_b(train_b, frozen_b, input_ids, labels, state=state_b)

        for k in grads_a.pytree:
            # Checkpoint recomputation under vmap can cause small numerical
            # differences (~0.5%) due to different execution contexts.
            torch.testing.assert_close(
                grads_a.pytree[k],
                grads_b.pytree[k],
                rtol=1e-2,
                atol=5e-3,
                msg=f"Gradient mismatch for {k}",
            )
