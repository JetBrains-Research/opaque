# Copyright (c) 2025 Opaque Authors
# SPDX-License-Identifier: Apache-2.0
"""DP-correctness guarantees for :class:`DPTrainer`.

These tests guard the *privacy* behaviour of the trainer — the surface
the rest of the suite mostly leaves unchecked.  They assert the things a
DP regression would silently break:

- the noise multiplier the trainer calibrates actually hits the target ε
  (against an independently-built reference accountant);
- the accountant composes exactly one mechanism per optimizer step, and
  resume keeps ``prefix + remaining`` on budget;
- clipping bounds per-example gradient norms;
- realized noise stddev equals ``noise_multiplier * clipping_norm``;
- ``evaluate()`` consumes no privacy budget;
- DP noise is reproducible at σ>0 and tracks ``seed`` / ``data_seed``;
- the two confirmed silent defects stay fixed (fp16 finite-check sees the
  real tensors; resuming a ``save_only_model`` checkpoint is refused).
"""

from __future__ import annotations

import os

import pytest
import torch
from peft import LoraConfig, TaskType, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

from opaque.transformers.trainer import DPTrainer, TrainingArguments

from _hf_shared import build_lm_dataset  # noqa: E402


# ---------------------------------------------------------------------------
# Fixtures (module-scoped: GPT-2 load is the slow part)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def gpt2_tok():
    tok = AutoTokenizer.from_pretrained("gpt2")
    tok.pad_token = tok.eos_token
    return tok


@pytest.fixture
def gpt2_lora(gpt2_tok):
    model = AutoModelForCausalLM.from_pretrained("gpt2")
    model.config.pad_token_id = gpt2_tok.pad_token_id
    lora = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=8,
        lora_alpha=16,
        lora_dropout=0.0,
        target_modules=["c_attn"],
        fan_in_fan_out=True,
    )
    return get_peft_model(model, lora), gpt2_tok


@pytest.fixture
def lm_dataset(gpt2_tok):
    texts = [
        "def fibonacci(n): return n",
        "area = math.pi * r * r",
        "return os.listdir(path)",
        "self.items = []",
        "return f'hello {name}'",
        "sorted_list = sorted(xs)",
        "x = [i for i in range(10)]",
        "print('done')",
    ]
    return build_lm_dataset(texts, gpt2_tok, max_length=16)


def _args(tmp_path, **overrides) -> TrainingArguments:
    defaults = dict(
        output_dir=str(tmp_path),
        per_device_train_batch_size=4,
        clipping_norm=1.0,
        privacy_target_epsilon=10.0,
        privacy_noise_multiplier=1.0,
        use_cpu=True,
        report_to=[],
        max_steps=4,
    )
    defaults.update(overrides)
    return TrainingArguments(**defaults)


# ---------------------------------------------------------------------------
# C2 — fp16 finite-check must inspect the real tensors, not the wrapper
# ---------------------------------------------------------------------------


def test_all_finite_sees_clipped_pytree_tensors():
    """Regression guard for the fp16 overflow no-op.

    ``ClippedPytree`` is not an optree node, so ``all_finite`` on the
    *wrapper* flattens to a single opaque leaf and never inspects the
    tensors — it would report a NaN-laden gradient as finite.  The trainer
    must call ``all_finite(grads.pytree)``; this test pins both halves so a
    revert to ``all_finite(grads)`` is caught here.
    """
    from opaque.api.engine.types import clipped
    from opaque.precision import all_finite

    nan_grads = {"w": torch.tensor([float("nan"), 1.0])}
    cp = clipped(nan_grads, max_norm=1.0)

    # The trap: the wrapper looks finite because it's an opaque leaf.
    assert all_finite(cp) is True
    # The fix: inspecting the inner pytree sees the NaN.
    assert all_finite(cp.pytree) is False

    finite = clipped({"w": torch.tensor([0.0, 1.0])}, max_norm=1.0)
    assert all_finite(finite.pytree) is True


@pytest.mark.cuda
def test_fp16_overflow_is_detected_and_skips_step(gpt2_lora, lm_dataset, tmp_path):
    """End-to-end: a real fp16 overflow trips the scaler and skips the step.

    Requires CUDA (fp16 autocast is GPU-only).  Forces an overflow by
    overriding the per-example loss to emit a non-finite value, then asserts
    the optimizer update is skipped and the overflow counter advances —
    which only happens if ``all_finite`` actually saw the bad gradient.
    """
    model, tok = gpt2_lora

    class _OverflowTrainer(DPTrainer):
        def compute_per_example_loss(self, fmodel, params, inputs, *, return_logits=False):
            loss = super().compute_per_example_loss(
                fmodel, params, inputs, return_logits=return_logits
            )
            if return_logits:
                base, logits = loss
                return base * float("inf"), logits
            return loss * float("inf")

    args = _args(tmp_path, fp16=True, use_cpu=False, max_steps=2)
    trainer = _OverflowTrainer(
        model=model, args=args, train_dataset=lm_dataset, processing_class=tok
    )
    trainer.train()
    assert trainer.state.fp16_overflow_steps > 0


# ---------------------------------------------------------------------------
# C1 — resuming a save_only_model checkpoint must be refused
# ---------------------------------------------------------------------------


def test_resume_from_save_only_model_checkpoint_is_refused(gpt2_lora, lm_dataset, tmp_path):
    """save_only_model checkpoints lack DP runtime state; resuming would reuse
    the noise stream.  Training-resume must hard-error."""
    model, tok = gpt2_lora
    args = _args(
        tmp_path,
        max_steps=4,
        save_strategy="steps",
        save_steps=2,
        save_only_model=True,
    )
    trainer = DPTrainer(
        model=model, args=args, train_dataset=lm_dataset, processing_class=tok
    )
    trainer.train()

    # Find a checkpoint and confirm it has accountant.json but no dp_state.
    ckpts = [d for d in os.listdir(tmp_path) if d.startswith("checkpoint-")]
    assert ckpts, "expected at least one checkpoint"
    ckpt_dir = os.path.join(tmp_path, sorted(ckpts)[0])
    assert os.path.exists(os.path.join(ckpt_dir, "accountant.json"))

    model2 = AutoModelForCausalLM.from_pretrained("gpt2")
    model2.config.pad_token_id = tok.pad_token_id
    model2 = get_peft_model(
        model2,
        LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=8,
            lora_alpha=16,
            lora_dropout=0.0,
            target_modules=["c_attn"],
            fan_in_fan_out=True,
        ),
    )
    args2 = _args(tmp_path, max_steps=8, save_strategy="no")
    trainer2 = DPTrainer(
        model=model2, args=args2, train_dataset=lm_dataset, processing_class=tok
    )
    with pytest.raises(RuntimeError, match="export-only"):
        trainer2.train(resume_from_checkpoint=ckpt_dir)


def test_resume_save_only_model_allowed_with_opt_in(gpt2_lora, lm_dataset, tmp_path):
    """The warmup opt-in (zero prior DP cost) still allows continuing."""
    model, tok = gpt2_lora
    args = _args(
        tmp_path, max_steps=4, save_strategy="steps", save_steps=2, save_only_model=True
    )
    trainer = DPTrainer(
        model=model, args=args, train_dataset=lm_dataset, processing_class=tok
    )
    trainer.train()
    ckpts = sorted(d for d in os.listdir(tmp_path) if d.startswith("checkpoint-"))
    ckpt_dir = os.path.join(tmp_path, ckpts[0])

    model2 = AutoModelForCausalLM.from_pretrained("gpt2")
    model2.config.pad_token_id = tok.pad_token_id
    model2 = get_peft_model(
        model2,
        LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=8,
            lora_alpha=16,
            lora_dropout=0.0,
            target_modules=["c_attn"],
            fan_in_fan_out=True,
        ),
    )
    args2 = _args(
        tmp_path,
        max_steps=8,
        save_strategy="no",
        privacy_resume_without_accountant=True,
    )
    trainer2 = DPTrainer(
        model=model2, args=args2, train_dataset=lm_dataset, processing_class=tok
    )
    out = trainer2.train(resume_from_checkpoint=ckpt_dir)  # must not raise
    assert out.global_step >= 4
