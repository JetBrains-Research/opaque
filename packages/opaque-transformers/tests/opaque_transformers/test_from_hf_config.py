"""Tests for ``opaque.transformers.from_hf_config`` — the single dispatch."""

from __future__ import annotations

import pytest

from opaque.transformers import from_hf_config
from opaque.transformers.trainer import TrainingArguments

hf = pytest.importorskip("transformers")
trl = pytest.importorskip("trl")

from opaque.transformers.trl import DPOConfig, SFTConfig  # noqa: E402


def test_dispatch_hf_training_arguments(tmp_path):
    """An HF ``TrainingArguments`` dispatches to ``TrainingArguments.from_hf``."""
    opaque = from_hf_config(
        hf.TrainingArguments(output_dir=str(tmp_path), report_to=[]),
        privacy_noise_multiplier=0.8,
    )
    assert isinstance(opaque, TrainingArguments)
    assert not isinstance(opaque, (SFTConfig, DPOConfig))
    assert opaque.privacy_noise_multiplier == 0.8


def test_dispatch_trl_sft_config(tmp_path):
    """A ``trl.SFTConfig`` dispatches to ``SFTConfig.from_trl`` (checked before HF)."""
    opaque = from_hf_config(
        trl.SFTConfig(output_dir=str(tmp_path), report_to=[]),
        privacy_noise_multiplier=0.8,
    )
    assert isinstance(opaque, SFTConfig)


def test_dispatch_trl_dpo_config(tmp_path):
    """A ``trl.DPOConfig`` dispatches to ``DPOConfig.from_trl`` (checked before HF)."""
    opaque = from_hf_config(
        trl.DPOConfig(output_dir=str(tmp_path), report_to=[]),
        privacy_noise_multiplier=0.8,
    )
    assert isinstance(opaque, DPOConfig)


def test_name_override_forwarded(tmp_path):
    """Arbitrary opaque overrides reach the underlying converter by name."""
    opaque = from_hf_config(
        hf.TrainingArguments(output_dir=str(tmp_path), report_to=[]),
        privacy_noise_multiplier=0.8,
        use_performance_kernels=True,
    )
    assert opaque.use_performance_kernels is True


def test_missing_dp_knob_raises(tmp_path):
    with pytest.raises(ValueError, match="privacy_noise_multiplier"):
        from_hf_config(hf.TrainingArguments(output_dir=str(tmp_path), report_to=[]))


def test_typeerror_on_unrecognized_input():
    with pytest.raises(TypeError, match="TrainingArguments"):
        from_hf_config({"learning_rate": 1e-4}, privacy_noise_multiplier=0.8)
