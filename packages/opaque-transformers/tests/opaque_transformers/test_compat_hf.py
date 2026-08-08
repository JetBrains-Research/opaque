"""Tests for ``TrainingArguments.from_hf`` — HF → opaque conversion."""

from __future__ import annotations

import warnings

import pytest

from opaque.api.transformers.trainer import TrainingArguments
from opaque.api.transformers.trainer._convert import _normalize_dp_overrides

# ``transformers`` is a required dep of opaque-transformers.
hf = pytest.importorskip("transformers")


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _hf_args(tmp_path, **overrides):
    """Construct an HF TrainingArguments with sensible defaults for tests."""
    return hf.TrainingArguments(
        output_dir=str(tmp_path),
        per_device_train_batch_size=overrides.pop("per_device_train_batch_size", 8),
        learning_rate=overrides.pop("learning_rate", 1e-4),
        max_steps=overrides.pop("max_steps", 10),
        seed=overrides.pop("seed", 42),
        save_strategy="no",
        report_to=[],
        **overrides,
    )


def _convert(tmp_path, **dp_overrides):
    """Helper: minimal default ``from_hf`` invocation."""
    return TrainingArguments.from_hf(
        _hf_args(tmp_path),
        privacy_noise_multiplier=0.8,
        clipping_norm=1.0,
        **dp_overrides,
    )


# ---------------------------------------------------------------------------
# Required DP knobs
# ---------------------------------------------------------------------------


def test_missing_dp_knob_raises(tmp_path):
    with pytest.raises(ValueError, match="privacy_noise_multiplier"):
        TrainingArguments.from_hf(_hf_args(tmp_path))


def test_either_noise_multiplier_or_target_epsilon_accepted(tmp_path):
    # noise_multiplier path
    opaque = TrainingArguments.from_hf(
        _hf_args(tmp_path),
        privacy_noise_multiplier=0.8,
        clipping_norm=1.0,
    )
    assert opaque.privacy_noise_multiplier == 0.8

    # target_epsilon path
    opaque = TrainingArguments.from_hf(
        _hf_args(tmp_path),
        privacy_target_epsilon=8.0,
        clipping_norm=1.0,
    )
    assert opaque.privacy_target_epsilon == 8.0


# ---------------------------------------------------------------------------
# DIRECT — copy as-is
# ---------------------------------------------------------------------------


def test_direct_field_learning_rate_carries_through(tmp_path):
    opaque = TrainingArguments.from_hf(
        _hf_args(tmp_path, learning_rate=3e-4),
        privacy_noise_multiplier=0.8,
        clipping_norm=1.0,
    )
    assert opaque.learning_rate == 3e-4


def test_direct_field_max_steps_carries_through(tmp_path):
    opaque = TrainingArguments.from_hf(
        _hf_args(tmp_path, max_steps=123),
        privacy_noise_multiplier=0.8,
        clipping_norm=1.0,
    )
    assert opaque.max_steps == 123


# bf16 needs a bf16-capable GPU: HF's TrainingArguments rejects ``bf16=True`` at
# construction on CPU/MPS runners, so this carry-through check can only exercise a
# real bf16 input on the CUDA lane.
@pytest.mark.cuda
def test_direct_field_bf16_carries_through(tmp_path):
    opaque = TrainingArguments.from_hf(
        _hf_args(tmp_path, bf16=True),
        privacy_noise_multiplier=0.8,
        clipping_norm=1.0,
    )
    assert opaque.bf16 is True


# ---------------------------------------------------------------------------
# RENAME
# ---------------------------------------------------------------------------


def test_evaluation_strategy_renamed_to_eval_strategy(tmp_path):
    # HF 4.41+ uses ``eval_strategy``; the rename is the inverse for old
    # configs. HF's TrainingArguments accepts both currently.
    opaque = TrainingArguments.from_hf(
        _hf_args(tmp_path, eval_strategy="steps", eval_steps=5),
        privacy_noise_multiplier=0.8,
        clipping_norm=1.0,
    )
    assert opaque.eval_strategy == "steps"
    assert opaque.eval_steps == 5


# ---------------------------------------------------------------------------
# TRANSFORM — batch collapse
# ---------------------------------------------------------------------------


def test_batch_collapse_grad_accum_multiplies_into_logical_batch(tmp_path):
    """``(per_device=2, grad_accum=4)`` → ``(opaque.per_device=8, microbatch=2)``."""
    opaque = TrainingArguments.from_hf(
        _hf_args(
            tmp_path,
            per_device_train_batch_size=2,
            gradient_accumulation_steps=4,
        ),
        privacy_noise_multiplier=0.8,
        clipping_norm=1.0,
    )
    assert opaque.per_device_train_batch_size == 8  # 2 × 4
    assert opaque.microbatch_size == 2


def test_batch_no_collapse_when_grad_accum_is_one(tmp_path):
    """At grad_accum=1, logical batch == per_device and microbatch_size
    stays at its default (None → vmap over the full batch)."""
    opaque = TrainingArguments.from_hf(
        _hf_args(
            tmp_path,
            per_device_train_batch_size=8,
            gradient_accumulation_steps=1,
        ),
        privacy_noise_multiplier=0.8,
        clipping_norm=1.0,
    )
    assert opaque.per_device_train_batch_size == 8
    assert opaque.microbatch_size is None


def test_optim_adamw_torch_collapses_to_adamw(tmp_path):
    opaque = TrainingArguments.from_hf(
        _hf_args(tmp_path, optim="adamw_torch"),
        privacy_noise_multiplier=0.8,
        clipping_norm=1.0,
    )
    assert opaque.optim == "adamw"


def test_optim_adamw_torch_fused_collapses_to_adamw_with_warning(tmp_path):
    # opaque's functional AdamW has no ``fused`` kernel arg, so the fused HF
    # optimizer must translate to plain adamw without forwarding fused — and
    # surface the dropped kernel request rather than silently rewrite (#389).
    with pytest.warns(RuntimeWarning, match="fused AdamW kernel"):
        opaque = TrainingArguments.from_hf(
            _hf_args(tmp_path, optim="adamw_torch_fused"),
            privacy_noise_multiplier=0.8,
            clipping_norm=1.0,
        )
    assert opaque.optim == "adamw"
    assert not opaque.optim_args  # nothing forwarded (no fused flag)


# ---------------------------------------------------------------------------
# REJECT_IF_SET
# ---------------------------------------------------------------------------


def test_reject_fp16(tmp_path):
    with pytest.raises(ValueError, match="bf16"):
        TrainingArguments.from_hf(
            _hf_args(tmp_path, fp16=True),
            privacy_noise_multiplier=0.8,
            clipping_norm=1.0,
        )


def test_reject_neftune_noise_alpha(tmp_path):
    with pytest.raises(ValueError, match="NEFTune"):
        TrainingArguments.from_hf(
            _hf_args(tmp_path, neftune_noise_alpha=0.5),
            privacy_noise_multiplier=0.8,
            clipping_norm=1.0,
        )


def test_auto_find_batch_size_maps_to_microbatch(tmp_path):
    """HF ``auto_find_batch_size`` → opaque ``auto_find_microbatch_size``."""
    opaque = TrainingArguments.from_hf(
        _hf_args(tmp_path, auto_find_batch_size=True),
        privacy_noise_multiplier=0.8,
        clipping_norm=1.0,
    )
    assert opaque.auto_find_microbatch_size is True


def test_max_grad_norm_maps_to_clipping_norm(tmp_path):
    """HF ``max_grad_norm`` loosely maps to opaque ``clipping_norm`` (no DP
    override given, so it isn't overwritten)."""
    opaque = TrainingArguments.from_hf(
        _hf_args(tmp_path, max_grad_norm=0.5),
        privacy_noise_multiplier=0.8,
    )
    assert opaque.clipping_norm == 0.5


def test_liger_maps_to_performance_kernels(tmp_path):
    """HF ``use_liger_kernel`` → opaque ``use_performance_kernels=True``."""
    opaque = TrainingArguments.from_hf(
        _hf_args(tmp_path, use_liger_kernel=True),
        privacy_noise_multiplier=0.8,
        clipping_norm=1.0,
    )
    assert opaque.use_performance_kernels is True


def test_performance_kernels_off_by_default_on_conversion(tmp_path):
    """Converting an HF config (no Liger) leaves perf-kernels OFF to match
    upstream, even though opaque's own default is True."""
    opaque = TrainingArguments.from_hf(
        _hf_args(tmp_path),
        privacy_noise_multiplier=0.8,
        clipping_norm=1.0,
    )
    assert opaque.use_performance_kernels is False
    # ...but a name override wins.
    opaque2 = TrainingArguments.from_hf(
        _hf_args(tmp_path),
        privacy_noise_multiplier=0.8,
        clipping_norm=1.0,
        use_performance_kernels=True,
    )
    assert opaque2.use_performance_kernels is True


def test_reject_paged_optim(tmp_path):
    with pytest.raises(ValueError, match="paged"):
        TrainingArguments.from_hf(
            _hf_args(tmp_path, optim="paged_adamw_8bit"),
            privacy_noise_multiplier=0.8,
            clipping_norm=1.0,
        )


# ---------------------------------------------------------------------------
# DROP_WITH_WARN
# ---------------------------------------------------------------------------


def test_drop_do_train_warns_when_non_default(tmp_path):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        # HF defaults do_train=False; explicitly setting it to True triggers
        # the drop warning.
        TrainingArguments.from_hf(
            _hf_args(tmp_path, do_train=True),
            privacy_noise_multiplier=0.8,
            clipping_norm=1.0,
        )
    assert any(
        "do_train" in str(w.message) and issubclass(w.category, RuntimeWarning)
        for w in caught
    )


def test_drop_silent_in_lenient_mode(tmp_path):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        TrainingArguments.from_hf(
            _hf_args(tmp_path, do_train=True),
            privacy_noise_multiplier=0.8,
            clipping_norm=1.0,
            strict=False,
        )
    # ``strict=False`` should suppress the drop warnings entirely.
    assert not any("do_train" in str(w.message) for w in caught)


# ---------------------------------------------------------------------------
# DP overrides
# ---------------------------------------------------------------------------


def test_dp_overrides_layered_on_top(tmp_path):
    opaque = TrainingArguments.from_hf(
        _hf_args(tmp_path),
        privacy_noise_multiplier=0.8,
        clipping_norm=1.0,
        privacy_noise_mechanism="gaussian",
        privacy_noise_radius=3.0,
        clipping_mode="fixed",
    )
    assert opaque.privacy_noise_multiplier == 0.8
    assert opaque.clipping_norm == 1.0
    assert opaque.privacy_noise_mechanism == "gaussian"


def test_opaque_overrides_win_over_hf_derived(tmp_path):
    # HF sets per_device=2, grad_accum=4 → opaque.microbatch_size=2 via
    # transform. Then user passes microbatch_size=1 as an opaque override,
    # which should win.
    opaque = TrainingArguments.from_hf(
        _hf_args(
            tmp_path,
            per_device_train_batch_size=2,
            gradient_accumulation_steps=4,
        ),
        privacy_noise_multiplier=0.8,
        clipping_norm=1.0,
        microbatch_size=1,
    )
    assert opaque.per_device_train_batch_size == 8  # still HF-derived
    assert opaque.microbatch_size == 1  # opaque override wins


# ---------------------------------------------------------------------------
# Type checks
# ---------------------------------------------------------------------------


def test_typeerror_on_non_hf_input(tmp_path):
    with pytest.raises(TypeError, match="TrainingArguments"):
        TrainingArguments.from_hf(
            {"learning_rate": 1e-4},
            privacy_noise_multiplier=0.8,
            clipping_norm=1.0,
        )


def test_normalize_dp_overrides_rejects_empty():
    with pytest.raises(ValueError, match="privacy_noise_multiplier"):
        _normalize_dp_overrides({})


def test_normalize_dp_overrides_accepts_noise_only():
    result = _normalize_dp_overrides({"privacy_noise_multiplier": 0.5})
    assert result["privacy_noise_multiplier"] == 0.5


def test_normalize_dp_overrides_accepts_epsilon_only():
    result = _normalize_dp_overrides({"privacy_target_epsilon": 8.0})
    assert result["privacy_target_epsilon"] == 8.0
