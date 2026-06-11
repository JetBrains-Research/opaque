"""Tests for ``opaque.transformers.trl.SFTConfig.from_trl`` (Piece 2)."""

from __future__ import annotations


import pytest

# TRL is the optional ``opaque[trl]`` extra. Skip the entire module when
# unavailable so test runs without ``trl`` installed don't error out.
trl = pytest.importorskip("trl")

from opaque.transformers.trl import SFTConfig  # noqa: E402


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _trl_args(tmp_path, **overrides):
    """Construct a TRL SFTConfig with the smallest set of explicit args."""
    return trl.SFTConfig(
        output_dir=str(tmp_path),
        per_device_train_batch_size=overrides.pop("per_device_train_batch_size", 8),
        learning_rate=overrides.pop("learning_rate", 1e-4),
        max_steps=overrides.pop("max_steps", 10),
        seed=overrides.pop("seed", 42),
        save_strategy="no",
        report_to=[],
        **overrides,
    )


# ---------------------------------------------------------------------------
# Required DP knobs
# ---------------------------------------------------------------------------


def test_missing_dp_knob_raises(tmp_path):
    with pytest.raises(ValueError, match="privacy_noise_multiplier"):
        SFTConfig.from_trl(_trl_args(tmp_path))


def test_typeerror_on_wrong_input_type(tmp_path):
    with pytest.raises(TypeError, match="SFTConfig"):
        SFTConfig.from_trl(
            {"learning_rate": 1e-4},
            privacy_noise_multiplier=0.8,
            clipping_norm=1.0,
        )


# ---------------------------------------------------------------------------
# DIRECT — TRL SFT-specific fields carry through
# ---------------------------------------------------------------------------


def test_dataset_text_field_carries_through(tmp_path):
    cfg = SFTConfig.from_trl(
        _trl_args(tmp_path, dataset_text_field="story"),
        privacy_noise_multiplier=0.8,
        clipping_norm=1.0,
    )
    assert cfg.dataset_text_field == "story"


def test_loss_type_nll_carries_through(tmp_path):
    cfg = SFTConfig.from_trl(
        _trl_args(tmp_path, loss_type="nll"),
        privacy_noise_multiplier=0.8,
        clipping_norm=1.0,
    )
    assert cfg.loss_type == "nll"


def test_loss_type_chunked_nll_carries_through(tmp_path):
    cfg = SFTConfig.from_trl(
        _trl_args(tmp_path, loss_type="chunked_nll"),
        privacy_noise_multiplier=0.8,
        clipping_norm=1.0,
    )
    assert cfg.loss_type == "chunked_nll"


def test_completion_only_loss_carries_through(tmp_path):
    cfg = SFTConfig.from_trl(
        _trl_args(tmp_path, completion_only_loss=True),
        privacy_noise_multiplier=0.8,
        clipping_norm=1.0,
    )
    assert cfg.completion_only_loss is True


def test_chat_template_path_carries_through(tmp_path):
    cfg = SFTConfig.from_trl(
        _trl_args(tmp_path, chat_template_path="/some/path"),
        privacy_noise_multiplier=0.8,
        clipping_norm=1.0,
    )
    assert cfg.chat_template_path == "/some/path"


# ---------------------------------------------------------------------------
# REJECT_IF_SET — TRL fields opaque doesn't implement
# ---------------------------------------------------------------------------


def test_reject_packing(tmp_path):
    with pytest.raises(ValueError, match="packing"):
        SFTConfig.from_trl(
            _trl_args(tmp_path, packing=True),
            privacy_noise_multiplier=0.8,
            clipping_norm=1.0,
        )


def test_reject_padding_free(tmp_path):
    with pytest.raises(ValueError, match="padding_free"):
        SFTConfig.from_trl(
            _trl_args(tmp_path, padding_free=True),
            privacy_noise_multiplier=0.8,
            clipping_norm=1.0,
        )


def test_reject_truncation_mode_keep_end(tmp_path):
    with pytest.raises(ValueError, match="keep_start"):
        SFTConfig.from_trl(
            _trl_args(tmp_path, truncation_mode="keep_end"),
            privacy_noise_multiplier=0.8,
            clipping_norm=1.0,
        )


def test_truncation_mode_keep_start_silently_accepted(tmp_path):
    """``keep_start`` is the only opaque-supported mode; should be silent."""
    cfg = SFTConfig.from_trl(
        _trl_args(tmp_path, truncation_mode="keep_start"),
        privacy_noise_multiplier=0.8,
        clipping_norm=1.0,
    )
    # keep_start is opaque's only supported value — silently accepted.
    assert cfg is not None


# ---------------------------------------------------------------------------
# HF base translation inherited
# ---------------------------------------------------------------------------


def test_inherits_hf_base_batch_collapse(tmp_path):
    cfg = SFTConfig.from_trl(
        _trl_args(
            tmp_path,
            per_device_train_batch_size=2,
            gradient_accumulation_steps=4,
        ),
        privacy_noise_multiplier=0.8,
        clipping_norm=1.0,
    )
    assert cfg.per_device_train_batch_size == 8  # 2 × 4
    assert cfg.microbatch_size == 2


def test_inherits_hf_base_rejection_of_fp16(tmp_path):
    with pytest.raises(ValueError, match="bf16"):
        SFTConfig.from_trl(
            _trl_args(tmp_path, fp16=True),
            privacy_noise_multiplier=0.8,
            clipping_norm=1.0,
        )


def test_inherits_max_grad_norm_to_clipping_norm(tmp_path):
    """HF ``max_grad_norm`` flows through the base manifest to opaque
    ``clipping_norm`` when no explicit ``clipping_norm`` override is given."""
    cfg = SFTConfig.from_trl(
        _trl_args(tmp_path, max_grad_norm=0.5),
        privacy_noise_multiplier=0.8,
    )
    assert cfg.clipping_norm == 0.5


def test_inherits_perf_kernels_off_by_default(tmp_path):
    """Converting a TRL config leaves perf-kernels OFF to match upstream,
    even though opaque's own SFT default is True. A name override wins."""
    cfg = SFTConfig.from_trl(
        _trl_args(tmp_path),
        privacy_noise_multiplier=0.8,
        clipping_norm=1.0,
    )
    assert cfg.use_performance_kernels is False
    cfg2 = SFTConfig.from_trl(
        _trl_args(tmp_path),
        privacy_noise_multiplier=0.8,
        clipping_norm=1.0,
        use_performance_kernels=True,
    )
    assert cfg2.use_performance_kernels is True


def test_inherits_liger_to_perf_kernels(tmp_path):
    """HF ``use_liger_kernel`` flows through to opaque
    ``use_performance_kernels=True``."""
    cfg = SFTConfig.from_trl(
        _trl_args(tmp_path, use_liger_kernel=True),
        privacy_noise_multiplier=0.8,
        clipping_norm=1.0,
    )
    assert cfg.use_performance_kernels is True


# ---------------------------------------------------------------------------
# Optional dependency gate
# ---------------------------------------------------------------------------


def test_import_error_when_trl_missing(tmp_path, monkeypatch):
    """Simulate missing ``trl`` via ``sys.modules`` patching."""
    # Patch ``trl`` out of sys.modules and force the import to fail.
    import builtins

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "trl":
            raise ImportError("simulated missing trl")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    # The classmethod imports ``trl`` inside the converter; the
    # ImportError should bubble out with a clear install hint.
    with pytest.raises(ImportError, match="opaque\\[trl\\]"):
        SFTConfig.from_trl(
            _trl_args(tmp_path),
            privacy_noise_multiplier=0.8,
            clipping_norm=1.0,
        )
