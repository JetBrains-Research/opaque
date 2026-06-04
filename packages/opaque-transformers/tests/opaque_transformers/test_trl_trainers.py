# Copyright (c) 2025 Opaque Authors
# SPDX-License-Identifier: Apache-2.0
"""Smoke + contract tests for the TRL-style DP trainers (SFT / DPO).

Hermetic: builds a tiny Llama from an in-code config (no network) and runs a
couple of DP-SGD steps through ``SFTTrainer`` / ``DPOTrainer``. The autouse
patch fixture in ``conftest.py`` installs the vmap-safety runtime patches.
"""

from __future__ import annotations

import types

import pytest

pytest.importorskip("transformers")
pytest.importorskip("datasets")

import torch  # noqa: E402
from datasets import Dataset  # noqa: E402
from transformers import LlamaConfig, LlamaForCausalLM  # noqa: E402

from opaque.transformers.trl import (  # noqa: E402
    DPOConfig,
    DPOTrainer,
    SFTConfig,
    SFTTrainer,
)


def _tiny_model() -> LlamaForCausalLM:
    cfg = LlamaConfig(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=4,
        max_position_embeddings=128,
    )
    return LlamaForCausalLM(cfg)


def _stub_tokenizer() -> types.SimpleNamespace:
    """Minimal processing_class for pre-tokenized data (no network)."""
    return types.SimpleNamespace(
        pad_token_id=0,
        pad_token="<pad>",
        eos_token="</s>",
        save_pretrained=lambda *a, **k: None,
    )


def _args(cls, tmp_path, **kw):
    return cls(
        output_dir=str(tmp_path),
        privacy_noise_multiplier=0.0,
        clipping_norm=1e9,
        per_device_train_batch_size=2,
        max_steps=2,
        logging_steps=1,
        save_strategy="no",
        report_to=[],
        seed=0,
        **kw,
    )


def _sft_dataset() -> Dataset:
    return Dataset.from_list([{"input_ids": [1, 2, 3, 4, 5, 6]} for _ in range(8)])


def _pref_dataset() -> Dataset:
    return Dataset.from_list(
        [
            {
                "chosen_input_ids": [1, 2, 3, 7, 8],
                "rejected_input_ids": [1, 2, 3, 9, 10],
                "chosen_completion_mask": [0, 0, 0, 1, 1],
                "rejected_completion_mask": [0, 0, 0, 1, 1],
            }
            for _ in range(8)
        ]
    )


# ----------------------------------------------------------------------
# Config behavior
# ----------------------------------------------------------------------
def test_sft_config_forces_remove_unused_columns_and_lr():
    cfg = SFTConfig(output_dir="x", privacy_noise_multiplier=0.0)
    assert cfg.remove_unused_columns is False
    assert cfg.learning_rate == 2e-5


def test_dpo_config_coerces_loss_type_and_defaults_weights():
    cfg = DPOConfig(output_dir="x", loss_type="sigmoid", privacy_noise_multiplier=0.0)
    assert cfg.loss_type == ["sigmoid"]
    assert cfg.loss_weights == [1.0]
    assert cfg.remove_unused_columns is False
    assert cfg.learning_rate == 1e-6

    mpo = DPOConfig(
        output_dir="x",
        loss_type=["sigmoid", "hinge"],
        privacy_noise_multiplier=0.0,
    )
    assert mpo.loss_weights == [1.0, 1.0]


def test_unsupported_param_is_absent_not_rejected():
    # No bespoke rejection: an unsupported field is simply not on the surface,
    # so passing it raises the standard dataclass TypeError. ``padding_free`` is
    # a TRL field with no DP meaning and is intentionally absent.
    with pytest.raises(TypeError):
        DPOConfig(output_dir="x", padding_free=True, privacy_noise_multiplier=0.0)


def test_duplicate_loss_type_fails_fast():
    with pytest.raises(ValueError):
        DPOConfig(
            output_dir="x",
            loss_type=["sigmoid", "sigmoid"],
            privacy_noise_multiplier=0.0,
        )


def test_sync_ref_model_incompatible_with_reference_free():
    with pytest.raises(ValueError):
        DPOConfig(
            output_dir="x",
            sync_ref_model=True,
            reference_free=True,
            privacy_noise_multiplier=0.0,
        )


def test_robust_label_smoothing_validation():
    with pytest.raises(ValueError):
        DPOConfig(
            output_dir="x",
            loss_type=["robust"],
            label_smoothing=0.7,
            privacy_noise_multiplier=0.0,
        )


def test_unknown_loss_type_fails_at_dispatch(tmp_path):
    # ``aot`` has no per-example DP meaning and no head — a standard KeyError at
    # the dispatch table, not a curated rejection.
    with pytest.raises(KeyError):
        DPOTrainer(
            model=_tiny_model(),
            args=_args(DPOConfig, tmp_path, loss_type="aot", max_length=8),
            train_dataset=_pref_dataset(),
            processing_class=_stub_tokenizer(),
        )


# ----------------------------------------------------------------------
# SFT training
# ----------------------------------------------------------------------
def test_sft_chunked_nll_trains(tmp_path):
    # chunked_nll lets the model compute its own (fused, logits-free) loss.
    torch.manual_seed(0)
    trainer = SFTTrainer(
        model=_tiny_model(),
        args=_args(SFTConfig, tmp_path, max_length=8, loss_type="chunked_nll"),
        train_dataset=_sft_dataset(),
        processing_class=_stub_tokenizer(),
    )
    out = trainer.train()
    assert out.global_step == 2
    assert torch.isfinite(torch.tensor(out.training_loss))


@pytest.mark.parametrize("loss_type", ["nll", "dft"])
def test_sft_trains_a_couple_steps(tmp_path, loss_type):
    torch.manual_seed(0)
    trainer = SFTTrainer(
        model=_tiny_model(),
        args=_args(SFTConfig, tmp_path, max_length=8, loss_type=loss_type),
        train_dataset=_sft_dataset(),
        processing_class=_stub_tokenizer(),
    )
    out = trainer.train()
    assert out.global_step == 2
    assert torch.isfinite(torch.tensor(out.training_loss))


# ----------------------------------------------------------------------
# DPO training
# ----------------------------------------------------------------------
@pytest.mark.parametrize("loss_type", ["sigmoid", "ipo", ["sigmoid", "hinge"]])
def test_dpo_trains_with_explicit_reference(tmp_path, loss_type):
    torch.manual_seed(0)
    trainer = DPOTrainer(
        model=_tiny_model(),
        ref_model=_tiny_model(),
        args=_args(DPOConfig, tmp_path, max_length=8, loss_type=loss_type),
        train_dataset=_pref_dataset(),
        processing_class=_stub_tokenizer(),
    )
    out = trainer.train()
    assert out.global_step == 2
    assert torch.isfinite(torch.tensor(out.training_loss))


def test_dpo_mixed_normalization_mpo_trains(tmp_path):
    # An MPO list mixing a summed (sigmoid) and a length-normalized (ipo) head
    # must work — normalization is per-head, not a single run-wide flag.
    torch.manual_seed(0)
    trainer = DPOTrainer(
        model=_tiny_model(),
        ref_model=_tiny_model(),
        args=_args(DPOConfig, tmp_path, max_length=8, loss_type=["sigmoid", "ipo"]),
        train_dataset=_pref_dataset(),
        processing_class=_stub_tokenizer(),
    )
    out = trainer.train()
    assert out.global_step == 2
    assert torch.isfinite(torch.tensor(out.training_loss))


def test_dpo_tr_dpo_syncs_reference(tmp_path):
    # TR-DPO: full FT, reference recomputed per step from an EMA reference that
    # tracks the policy. With sync_steps=1 the reference must move during training.
    torch.manual_seed(0)
    trainer = DPOTrainer(
        model=_tiny_model(),
        ref_model=_tiny_model(),
        args=_args(
            DPOConfig,
            tmp_path,
            max_length=8,
            loss_type="sigmoid",
            sync_ref_model=True,
            ref_model_sync_steps=1,
            ref_model_mixup_alpha=0.5,
        ),
        train_dataset=_pref_dataset(),
        eval_dataset=_pref_dataset(),
        processing_class=_stub_tokenizer(),
    )
    name, param = next(iter(trainer._tr_ref.named_parameters()))
    before = param.detach().clone()
    out = trainer.train()
    after = dict(trainer._tr_ref.named_parameters())[name].detach()
    assert out.global_step == 2
    assert not torch.allclose(before, after)  # EMA moved the reference
    # Eval scores against the current EMA reference (exercises _inject_tr_ref_logps
    # in prediction_step), not the stale seed columns.
    metrics = trainer.evaluate()
    assert "eval_loss" in metrics
    assert "eval_rewards/accuracies" in metrics


def test_dpo_logs_train_reward_metrics(tmp_path):
    # Reward telemetry rides the clipped-grad aux channel and is logged each step.
    torch.manual_seed(0)
    trainer = DPOTrainer(
        model=_tiny_model(),
        ref_model=_tiny_model(),
        args=_args(DPOConfig, tmp_path, max_length=8, loss_type="sigmoid"),
        train_dataset=_pref_dataset(),
        processing_class=_stub_tokenizer(),
    )
    trainer.train()
    logged = set().union(*(row.keys() for row in trainer.state.log_history))
    assert "rewards/chosen" in logged
    assert "rewards/rejected" in logged
    assert "rewards/accuracies" in logged
    assert "rewards/margins" in logged


def test_dpo_evaluate_logs_reward_metrics(tmp_path):
    torch.manual_seed(0)
    trainer = DPOTrainer(
        model=_tiny_model(),
        ref_model=_tiny_model(),
        args=_args(DPOConfig, tmp_path, max_length=8, loss_type="sigmoid"),
        train_dataset=_pref_dataset(),
        eval_dataset=_pref_dataset(),
        processing_class=_stub_tokenizer(),
    )
    metrics = trainer.evaluate()
    assert "eval_loss" in metrics
    assert "eval_rewards/accuracies" in metrics
    assert "eval_rewards/chosen" in metrics


def test_dpo_reference_free_trains_without_precompute(tmp_path):
    torch.manual_seed(0)
    trainer = DPOTrainer(
        model=_tiny_model(),
        args=_args(
            DPOConfig,
            tmp_path,
            max_length=8,
            loss_type="sigmoid",
            reference_free=True,
        ),
        train_dataset=_pref_dataset(),
        processing_class=_stub_tokenizer(),
    )
    out = trainer.train()
    assert out.global_step == 2


def test_dpo_wpo_weighting_trains(tmp_path):
    torch.manual_seed(0)
    trainer = DPOTrainer(
        model=_tiny_model(),
        ref_model=_tiny_model(),
        args=_args(
            DPOConfig,
            tmp_path,
            max_length=8,
            loss_type="sigmoid",
            use_weighting=True,
        ),
        train_dataset=_pref_dataset(),
        processing_class=_stub_tokenizer(),
    )
    out = trainer.train()
    assert out.global_step == 2


# ----------------------------------------------------------------------
# DP-purity: the per-example loss for example i depends only on example i.
# ----------------------------------------------------------------------
def _to_device(trainer, batch):
    device = next(trainer.model.parameters()).device
    return {
        k: (v.to(device) if isinstance(v, torch.Tensor) else v)
        for k, v in batch.items()
    }


def _per_example_losses(trainer, batch):
    """vmap ``compute_per_example_loss`` over a collated batch (the DP path)."""
    from opaque.functional import make_functional

    # The collator emits CPU tensors; move to the model device (the trainer's
    # _prepare_input does this in the real path — bypassed here).
    batch = _to_device(trainer, batch)
    fmodel, trainable, frozen = make_functional(trainer.model, partition_trainable=True)
    keys = [k for k, v in batch.items() if isinstance(v, torch.Tensor)]

    def fn(tp, *batch_args):
        merged = {**frozen, **tp}
        return trainer.compute_per_example_loss(
            fmodel, merged, dict(zip(keys, batch_args))
        )

    vmapped = torch.vmap(fn, in_dims=(None,) + (0,) * len(keys))
    return vmapped(trainable, *[batch[k] for k in keys]), keys


def test_sft_dp_purity_per_example_independence(tmp_path):
    torch.manual_seed(0)
    trainer = SFTTrainer(
        model=_tiny_model(),
        args=_args(SFTConfig, tmp_path, max_length=8, loss_type="nll"),
        train_dataset=_sft_dataset(),
        processing_class=_stub_tokenizer(),
    )
    trainer.model.eval()
    rows = [trainer.train_dataset[i] for i in range(4)]
    batch = trainer.data_collator(rows)
    losses0, _ = _per_example_losses(trainer, batch)

    # Perturb only example 0's tokens; examples 1..3 must be untouched.
    batch2 = {k: v.clone() for k, v in batch.items()}
    batch2["input_ids"][0, 1] = (batch2["input_ids"][0, 1] + 1) % 64
    losses1, _ = _per_example_losses(trainer, batch2)

    assert not torch.allclose(losses0[0], losses1[0])
    assert torch.allclose(losses0[1:], losses1[1:])


def test_dpo_dp_purity_per_example_independence(tmp_path):
    torch.manual_seed(0)
    trainer = DPOTrainer(
        model=_tiny_model(),
        ref_model=_tiny_model(),
        args=_args(DPOConfig, tmp_path, max_length=8, loss_type="sigmoid"),
        train_dataset=_pref_dataset(),
        processing_class=_stub_tokenizer(),
    )
    trainer.model.eval()
    rows = [trainer.train_dataset[i] for i in range(4)]
    batch = trainer.data_collator(rows)
    losses0, _ = _per_example_losses(trainer, batch)

    batch2 = {k: v.clone() for k, v in batch.items()}
    batch2["chosen_input_ids"][0, 3] = (batch2["chosen_input_ids"][0, 3] + 1) % 64
    losses1, _ = _per_example_losses(trainer, batch2)

    assert not torch.allclose(losses0[0], losses1[0])
    assert torch.allclose(losses0[1:], losses1[1:])


# ----------------------------------------------------------------------
# Numeric parity: the trainer wires the primitives correctly.
# ----------------------------------------------------------------------
def test_sft_loss_matches_direct_nll(tmp_path):
    from opaque.alignment.sft.loss import nll_loss

    torch.manual_seed(0)
    trainer = SFTTrainer(
        model=_tiny_model(),
        args=_args(SFTConfig, tmp_path, max_length=8, loss_type="nll"),
        train_dataset=_sft_dataset(),
        processing_class=_stub_tokenizer(),
    )
    trainer.model.eval()
    rows = [trainer.train_dataset[i] for i in range(4)]
    batch = _to_device(trainer, trainer.data_collator(rows))
    losses, _ = _per_example_losses(trainer, batch)

    with torch.no_grad():
        out = trainer.model(
            input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]
        )
        expected = nll_loss(out.logits, batch["labels"])
    assert torch.allclose(losses, expected, atol=1e-4)


def test_dpo_loss_matches_direct_sigmoid(tmp_path):
    from opaque.alignment.dpo.loss import sequence_logp, sigmoid_loss

    torch.manual_seed(0)
    beta = 0.1
    trainer = DPOTrainer(
        model=_tiny_model(),
        ref_model=_tiny_model(),
        args=_args(DPOConfig, tmp_path, max_length=8, loss_type="sigmoid", beta=beta),
        train_dataset=_pref_dataset(),
        processing_class=_stub_tokenizer(),
    )
    trainer.model.eval()
    rows = [trainer.train_dataset[i] for i in range(4)]
    batch = _to_device(trainer, trainer.data_collator(rows))
    losses, _ = _per_example_losses(trainer, batch)

    with torch.no_grad():
        c = trainer.model(
            input_ids=batch["chosen_input_ids"],
            attention_mask=batch["chosen_attention_mask"],
        )
        r = trainer.model(
            input_ids=batch["rejected_input_ids"],
            attention_mask=batch["rejected_attention_mask"],
        )
        c_lp = sequence_logp(
            c.logits, batch["chosen_input_ids"], batch["chosen_completion_mask"]
        )
        r_lp = sequence_logp(
            r.logits, batch["rejected_input_ids"], batch["rejected_completion_mask"]
        )
        expected = sigmoid_loss(
            c_lp - batch["ref_chosen_logps"],
            r_lp - batch["ref_rejected_logps"],
            beta=beta,
        )
    assert torch.allclose(losses, expected, atol=1e-4)


def test_dpo_precompute_attaches_reference_columns(tmp_path):
    # The reference precompute should add the constant ref logp columns the
    # collator emits; verified indirectly by a successful non-reference-free run
    # and directly here on the prepared dataset.
    torch.manual_seed(0)
    trainer = DPOTrainer(
        model=_tiny_model(),
        ref_model=_tiny_model(),
        args=_args(DPOConfig, tmp_path, max_length=8, loss_type="sigmoid"),
        train_dataset=_pref_dataset(),
        processing_class=_stub_tokenizer(),
    )
    cols = trainer.train_dataset.column_names
    assert "ref_chosen_logps" in cols
    assert "ref_rejected_logps" in cols
