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
    cfg = DPOConfig(
        output_dir="x", loss_type="sigmoid", privacy_noise_multiplier=0.0
    )
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
    # so passing it raises the standard dataclass TypeError.
    with pytest.raises(TypeError):
        DPOConfig(
            output_dir="x", sync_ref_model=True, privacy_noise_multiplier=0.0
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
