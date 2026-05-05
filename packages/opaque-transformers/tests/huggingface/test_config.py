"""Tests for ``DPTrainingArguments.__post_init__`` HF-parity contract.

Covers:

- Legacy alias normalisation (``no_cuda`` → ``use_cpu``) emits
  ``FutureWarning``.
- Auto-default of ``metric_for_best_model="loss"`` under
  ``load_best_model_at_end`` / ``reduce_lr_on_plateau``.
- Cadence-alignment validation for ``load_best_model_at_end``
  (``save_strategy`` must equal ``eval_strategy``;
  ``save_steps % eval_steps == 0`` when both are step-based).
- Strategy-string coercion across the four strategy fields
  (``eval_strategy`` / ``logging_strategy`` / ``save_strategy`` /
  ``lr_scheduler_type``).
- ``do_eval`` auto-flip when an eval strategy is configured.
"""

from __future__ import annotations

import warnings

import pytest
from transformers.debug_utils import DebugOption
from transformers.trainer_utils import HubStrategy

from opaque.transformers.trainer import DPTrainingArguments
from opaque.transformers.trainer._config import (
    DP_INCOMPATIBLE_PARAMETERS,
    _DP_OPTIMIZERS,
    _DP_OPTIMIZER_UNSUPPORTED,
)


class TestLegacyAliases:
    """Deprecated HF kwargs are accepted with a ``FutureWarning``."""

    def test_no_cuda_aliases_use_cpu(self):
        import dataclasses

        from transformers.training_args import TrainingArguments

        if "no_cuda" not in {f.name for f in dataclasses.fields(TrainingArguments)}:
            pytest.skip("no_cuda was removed from HF TrainingArguments in this version.")
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            args = DPTrainingArguments(no_cuda=True)
        assert args.use_cpu is True
        assert any(
            issubclass(w.category, FutureWarning) and "no_cuda" in str(w.message)
            for w in caught
        )


class TestMetricForBestModelDefaults:
    """``metric_for_best_model`` is auto-defaulted to ``"loss"`` when needed."""

    def test_load_best_model_at_end_defaults_metric(self, tmp_path):
        args = DPTrainingArguments(
            output_dir=str(tmp_path),
            load_best_model_at_end=True,
            save_strategy="steps",
            save_steps=2,
            eval_strategy="steps",
            eval_steps=2,
        )
        assert args.metric_for_best_model == "loss"
        # Loss-suffixed metric ⇒ greater_is_better=False.
        assert args.greater_is_better is False

    def test_reduce_lr_on_plateau_defaults_metric(self):
        args = DPTrainingArguments(
            lr_scheduler_type="reduce_lr_on_plateau",
            eval_strategy="steps",
            eval_steps=1,
        )
        assert args.metric_for_best_model == "loss"


class TestCadenceAlignment:
    """``load_best_model_at_end`` requires save / eval cadence alignment."""

    def test_mismatched_strategies_raises(self, tmp_path):
        # ``save_strategy=steps`` + ``eval_strategy=epoch`` ⇒ HF parity:
        # the save and eval strategies must match.
        with pytest.raises(ValueError, match="save and eval strategy to match"):
            DPTrainingArguments(
                output_dir=str(tmp_path),
                load_best_model_at_end=True,
                save_strategy="steps",
                save_steps=2,
                eval_strategy="epoch",
            )

    def test_save_steps_not_multiple_of_eval_steps_raises(self, tmp_path):
        with pytest.raises(ValueError, match="round multiple"):
            DPTrainingArguments(
                output_dir=str(tmp_path),
                load_best_model_at_end=True,
                save_strategy="steps",
                save_steps=3,
                eval_strategy="steps",
                eval_steps=2,
            )

    def test_aligned_step_strategies_pass(self, tmp_path):
        # 6 % 2 == 0 ⇒ aligned.
        args = DPTrainingArguments(
            output_dir=str(tmp_path),
            load_best_model_at_end=True,
            save_strategy="steps",
            save_steps=6,
            eval_strategy="steps",
            eval_steps=2,
        )
        assert args.save_steps == 6
        assert args.eval_steps == 2

    def test_save_strategy_best_skips_alignment_check(self, tmp_path):
        # ``save_strategy="best"`` is the only HF strategy that pairs
        # with any eval cadence regardless of save_steps.
        args = DPTrainingArguments(
            output_dir=str(tmp_path),
            load_best_model_at_end=True,
            save_strategy="best",
            eval_strategy="steps",
            eval_steps=3,
        )
        assert args.save_strategy == "best"


class TestStrategyCoercion:
    """The four strategy enum fields are coerced to canonical strings."""

    @pytest.mark.parametrize(
        "field,value,expected",
        [
            ("eval_strategy", "steps", "steps"),
            ("eval_strategy", "epoch", "epoch"),
            ("eval_strategy", "no", "no"),
            ("save_strategy", "steps", "steps"),
            ("save_strategy", "epoch", "epoch"),
            ("save_strategy", "no", "no"),
            ("save_strategy", "best", "best"),
            ("logging_strategy", "steps", "steps"),
            ("logging_strategy", "epoch", "epoch"),
            ("lr_scheduler_type", "linear", "linear"),
            ("lr_scheduler_type", "cosine", "cosine"),
        ],
    )
    def test_canonical_string_coercion(self, field, value, expected):
        kwargs = {field: value}
        # Side-args needed for some coercions (eval_strategy="steps" needs eval_steps).
        if field == "eval_strategy" and value == "steps":
            kwargs["eval_steps"] = 1
        args = DPTrainingArguments(**kwargs)
        assert getattr(args, field) == expected

    def test_unknown_strategy_raises(self):
        with pytest.raises(ValueError):
            DPTrainingArguments(eval_strategy="completely-bogus")

    def test_unknown_lr_scheduler_type_raises(self):
        with pytest.raises(ValueError):
            DPTrainingArguments(lr_scheduler_type="not-a-real-scheduler")


class TestHFFieldNormalization:
    def test_hub_strategy_coerces_to_enum(self):
        args = DPTrainingArguments(hub_strategy="end")
        assert args.hub_strategy == HubStrategy.END

    def test_debug_string_coerces_to_debug_options(self):
        args = DPTrainingArguments(debug="underflow_overflow")
        assert args.debug == [DebugOption.UNDERFLOW_OVERFLOW]

    @pytest.mark.parametrize(
        "value,expected",
        [(True, "all"), (False, "no"), ("non_padding", "non_padding")],
    )
    def test_include_num_input_tokens_seen_normalizes(self, value, expected):
        args = DPTrainingArguments(include_num_input_tokens_seen=value)
        assert args.include_num_input_tokens_seen == expected

    def test_include_num_input_tokens_seen_invalid_raises(self):
        with pytest.raises(ValueError, match="include_num_input_tokens_seen"):
            DPTrainingArguments(include_num_input_tokens_seen="sometimes")

    @pytest.mark.parametrize(
        "value,expected",
        [
            (None, []),
            ("none", []),
            ([], []),
            (["none"], []),
            ("tensorboard", ["tensorboard"]),
            (["wandb"], ["wandb"]),
        ],
    )
    def test_report_to_normalizes(self, value, expected):
        args = DPTrainingArguments(report_to=value)
        assert args.report_to == expected

    def test_lr_scheduler_kwargs_json_object_is_parsed(self):
        args = DPTrainingArguments(
            lr_scheduler_type="cosine",
            lr_scheduler_kwargs='{"num_cycles": "2"}',
        )
        assert args.lr_scheduler_kwargs == {"num_cycles": 2}

    def test_torch_empty_cache_steps_must_be_positive_int(self):
        with pytest.raises(ValueError, match="torch_empty_cache_steps"):
            DPTrainingArguments(torch_empty_cache_steps=0)


class TestUnsupportedHFKnobs:
    # Phase 11 lifted rejections for: torch_compile, torch_compile_backend,
    # torch_compile_mode, use_liger_kernel, liger_kernel_config.
    # HF 5.x removed: tpu_num_cores, mp_parameters, ray_scope, past_index,
    # torchdynamo, jit_mode_eval — they're no longer constructable as
    # kwargs, so they're outside this table's scope.
    INCOMPATIBLE_CASES = {
        "group_by_length": True,
        "dataloader_drop_last": True,
        "fsdp": "full_shard",
        "fsdp_min_num_params": 1,
        "fsdp_config": {"activation_checkpointing": True},
        "fsdp_transformer_layer_cls_to_wrap": "GPT2Block",
        "accelerator_config": {"split_batches": True},
        "parallelism_config": {"tp_size": 2},
        "deepspeed": "ds_config.json",
        "neftune_noise_alpha": 5.0,
        "eval_use_gather_object": True,
        "average_tokens_across_devices": False,
    }

    def test_incompatible_case_table_tracks_rejection_table(self):
        # Some entries in DP_INCOMPATIBLE_PARAMETERS reference HF fields
        # removed in v5 — they remain in the table for forward-compat with
        # older transformers but can't be constructed as kwargs on v5,
        # so this test verifies the constructable subset.
        assert set(self.INCOMPATIBLE_CASES).issubset(set(DP_INCOMPATIBLE_PARAMETERS))

    @pytest.mark.parametrize(
        "field,value",
        INCOMPATIBLE_CASES.items(),
    )
    def test_non_default_incompatible_fields_raise(self, field, value):
        import dataclasses

        from transformers.training_args import TrainingArguments

        if field not in {f.name for f in dataclasses.fields(TrainingArguments)}:
            pytest.skip(
                f"{field!r} was removed from HF TrainingArguments in this "
                "version; not constructable as a kwarg."
            )
        with pytest.raises(ValueError, match=field):
            DPTrainingArguments(**{field: value})


class TestOptimizerSupportSurface:
    @pytest.mark.parametrize("optim", _DP_OPTIMIZERS)
    def test_supported_dp_optimizers_construct(self, optim):
        args = DPTrainingArguments(optim=optim)
        assert args.optim == optim

    @pytest.mark.parametrize("optim,reason", _DP_OPTIMIZER_UNSUPPORTED.items())
    def test_unsupported_hf_optimizers_raise_with_redirect(self, optim, reason):
        with pytest.raises(ValueError) as exc_info:
            DPTrainingArguments(optim=optim)
        message = str(exc_info.value)
        assert optim in message
        assert "Supported optimizers" in message
        assert reason.split(".", maxsplit=1)[0] in message


class TestDoEvalAutoFlip:
    """``do_eval`` auto-flips when an eval strategy is configured."""

    def test_eval_strategy_steps_flips_do_eval(self):
        args = DPTrainingArguments(eval_strategy="steps", eval_steps=5)
        assert args.do_eval is True

    def test_eval_strategy_no_keeps_do_eval_false(self):
        args = DPTrainingArguments(eval_strategy="no")
        assert args.do_eval is False

    def test_explicit_do_eval_preserved(self):
        args = DPTrainingArguments(do_eval=True, eval_strategy="no")
        assert args.do_eval is True


class TestEvalStepsFallback:
    """``eval_strategy="steps"`` without ``eval_steps`` falls back to ``logging_steps``."""

    def test_eval_steps_falls_back_to_logging_steps(self):
        args = DPTrainingArguments(eval_strategy="steps", logging_steps=10)
        assert args.eval_steps == 10

    def test_eval_strategy_steps_zero_logging_raises(self):
        with pytest.raises(ValueError, match="non-zero"):
            DPTrainingArguments(eval_strategy="steps", logging_steps=0)


class TestMaxGradNorm:
    """DP-specific rejection of non-default ``max_grad_norm``."""

    def test_default_value_silently_accepted(self):
        args = DPTrainingArguments(max_grad_norm=1.0)
        assert args.max_grad_norm == 1.0

    def test_non_default_value_raises(self):
        with pytest.raises(TypeError, match="max_grad_norm"):
            DPTrainingArguments(max_grad_norm=0.5)
