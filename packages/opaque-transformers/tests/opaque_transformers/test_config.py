"""Tests for ``TrainingArguments.__post_init__`` HF-parity contract.

Covers:

- ``no_cuda`` is no longer accepted on the standalone dataclass.
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

import pytest
from transformers.debug_utils import DebugOption
from transformers.trainer_utils import HubStrategy

from opaque.transformers.trainer import TrainingArguments
from opaque.api.transformers.trainer._config import _DP_OPTIMIZERS
from opaque.api.transformers.trainer._optim import _DP_OPTIMIZER_UNSUPPORTED


class TestLegacyAliases:
    """Deprecated HF kwargs are not accepted on the standalone dataclass."""

    def test_no_cuda_no_longer_accepted(self):
        # Phase 10 follow-up: ``TrainingArguments`` is now standalone (no
        # HF inheritance).  ``no_cuda`` was a deprecated HF alias for
        # ``use_cpu``; passing it now raises ``TypeError`` (unexpected kwarg)
        # rather than emitting a ``FutureWarning``.  Users should use
        # ``use_cpu`` directly.
        with pytest.raises(TypeError, match="no_cuda"):
            TrainingArguments(no_cuda=True)


class TestMetricForBestModelDefaults:
    """``metric_for_best_model`` is auto-defaulted to ``"loss"`` when needed."""

    def test_load_best_model_at_end_defaults_metric(self, tmp_path):
        args = TrainingArguments(
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
        args = TrainingArguments(
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
            TrainingArguments(
                output_dir=str(tmp_path),
                load_best_model_at_end=True,
                save_strategy="steps",
                save_steps=2,
                eval_strategy="epoch",
            )

    def test_save_steps_not_multiple_of_eval_steps_raises(self, tmp_path):
        with pytest.raises(ValueError, match="round multiple"):
            TrainingArguments(
                output_dir=str(tmp_path),
                load_best_model_at_end=True,
                save_strategy="steps",
                save_steps=3,
                eval_strategy="steps",
                eval_steps=2,
            )

    def test_aligned_step_strategies_pass(self, tmp_path):
        # 6 % 2 == 0 ⇒ aligned.
        args = TrainingArguments(
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
        args = TrainingArguments(
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
        args = TrainingArguments(**kwargs)
        assert getattr(args, field) == expected

    def test_unknown_strategy_raises(self):
        with pytest.raises(ValueError):
            TrainingArguments(eval_strategy="completely-bogus")

    def test_unknown_lr_scheduler_type_raises(self):
        with pytest.raises(ValueError):
            TrainingArguments(lr_scheduler_type="not-a-real-scheduler")


class TestHFFieldNormalization:
    def test_hub_strategy_coerces_to_enum(self):
        args = TrainingArguments(hub_strategy="end")
        assert args.hub_strategy == HubStrategy.END

    def test_debug_string_coerces_to_debug_options(self):
        args = TrainingArguments(debug="underflow_overflow")
        assert args.debug == [DebugOption.UNDERFLOW_OVERFLOW]

    @pytest.mark.parametrize(
        "value,expected",
        [(True, "all"), (False, "no"), ("non_padding", "non_padding")],
    )
    def test_include_num_input_tokens_seen_normalizes(self, value, expected):
        args = TrainingArguments(include_num_input_tokens_seen=value)
        assert args.include_num_input_tokens_seen == expected

    def test_include_num_input_tokens_seen_invalid_raises(self):
        with pytest.raises(ValueError, match="include_num_input_tokens_seen"):
            TrainingArguments(include_num_input_tokens_seen="sometimes")

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
        args = TrainingArguments(report_to=value)
        assert args.report_to == expected

    def test_lr_scheduler_kwargs_json_object_is_parsed(self):
        args = TrainingArguments(
            lr_scheduler_type="cosine",
            lr_scheduler_kwargs='{"num_cycles": "2"}',
        )
        assert args.lr_scheduler_kwargs == {"num_cycles": 2}

    def test_torch_empty_cache_steps_must_be_positive_int(self):
        with pytest.raises(ValueError, match="torch_empty_cache_steps"):
            TrainingArguments(torch_empty_cache_steps=0)


class TestUnknownKwargs:
    @pytest.mark.parametrize(
        "payload",
        [
            {"group_by_length": True},
            {"deepspeed": "ds_config.json"},
            {"optim_target_modules": ["q_proj"]},
            {"ddp_find_unused_parameters": True},
            {"ddp_timeout": 3600},
            {"unknown_knob_for_test_only": 1},
        ],
    )
    def test_unknown_kwargs_raise_type_error(self, payload):
        with pytest.raises(TypeError):
            TrainingArguments(**payload)

    def test_dpargs_does_not_inherit_from_hf_training_arguments(self):
        from transformers.training_args import TrainingArguments as HFTrainingArguments

        assert HFTrainingArguments not in TrainingArguments.__mro__

    @pytest.mark.parametrize(
        "backend", ["nccl", "gloo", "mpi", "xccl", "hccl", "cncl", "mccl"]
    )
    def test_ddp_backend_surface_accepts_hf_backend_values(self, backend):
        args = TrainingArguments(ddp_backend=backend)
        assert args.ddp_backend == backend

    @pytest.mark.parametrize("backend", ["invalid-backend"])
    def test_ddp_backend_unsupported_value_raises(self, backend):
        with pytest.raises(ValueError, match="ddp_backend"):
            TrainingArguments(ddp_backend=backend)


class TestOptimizerSupportSurface:
    @pytest.mark.parametrize("optim", _DP_OPTIMIZERS)
    def test_supported_dp_optimizers_construct(self, optim):
        args = TrainingArguments(optim=optim)
        assert args.optim == optim

    @pytest.mark.parametrize("optim,reason", _DP_OPTIMIZER_UNSUPPORTED.items())
    def test_unsupported_hf_optimizers_raise_with_redirect(self, optim, reason):
        with pytest.raises(ValueError) as exc_info:
            TrainingArguments(optim=optim)
        message = str(exc_info.value)
        assert optim in message
        assert "Supported optimizers" in message
        assert reason.split(".", maxsplit=1)[0] in message


class TestDoEvalAutoFlip:
    """``do_eval`` auto-flips when an eval strategy is configured."""

    def test_eval_strategy_steps_flips_do_eval(self):
        args = TrainingArguments(eval_strategy="steps", eval_steps=5)
        assert args.do_eval is True

    def test_eval_strategy_no_keeps_do_eval_false(self):
        args = TrainingArguments(eval_strategy="no")
        assert args.do_eval is False

    def test_explicit_do_eval_preserved(self):
        args = TrainingArguments(do_eval=True, eval_strategy="no")
        assert args.do_eval is True


class TestEvalStepsFallback:
    """``eval_strategy="steps"`` without ``eval_steps`` falls back to ``logging_steps``."""

    def test_eval_steps_falls_back_to_logging_steps(self):
        args = TrainingArguments(eval_strategy="steps", logging_steps=10)
        assert args.eval_steps == 10

    def test_eval_strategy_steps_zero_logging_raises(self):
        with pytest.raises(ValueError, match="non-zero"):
            TrainingArguments(eval_strategy="steps", logging_steps=0)


class TestMaxGradNorm:
    """``clipping_norm`` is the DP clip bound: scalar or per-group dict."""

    def test_positive_scalar_accepted(self):
        args = TrainingArguments(clipping_norm=2.5)
        assert args.clipping_norm == 2.5

    def test_non_positive_scalar_raises(self):
        with pytest.raises(ValueError, match="clipping_norm"):
            TrainingArguments(clipping_norm=0.0)

    def test_dict_fallback_only_normalizes_to_float(self):
        args = TrainingArguments(clipping_norm={"fallback": 2.0})
        assert args.clipping_norm == 2.0

    def test_dict_per_group_accepted(self):
        args = TrainingArguments(
            clipping_norm={"fallback": 1.0, "linear": 2.0, "bias": 0.5},
        )
        assert args.clipping_norm == {
            "fallback": 1.0,
            "linear": 2.0,
            "bias": 0.5,
        }

    def test_json_string_object_parsed(self):
        args = TrainingArguments(
            clipping_norm='{"fallback": 1.0, "x": 2.0}',
        )
        assert args.clipping_norm == {"fallback": 1.0, "x": 2.0}

    def test_dict_missing_fallback_raises(self):
        with pytest.raises(ValueError, match="fallback"):
            TrainingArguments(clipping_norm={"linear": 1.0})


class TestClippingAndSamplingSurfaces:
    """``clipping_mode`` / ``sampling_mode`` and JSON-style arg blobs."""

    def test_invalid_clipping_mode_raises(self):
        with pytest.raises(ValueError, match="clipping_mode"):
            TrainingArguments(clipping_mode="unknown")

    def test_invalid_sampling_mode_raises(self):
        with pytest.raises(ValueError, match="sampling_mode"):
            TrainingArguments(sampling_mode="sequential")

    def test_clipping_kwargs_json_string_parsed(self):
        args = TrainingArguments(clipping_kwargs='{"norm_max": 9.0}')
        assert args.clipping_kwargs == {"norm_max": 9.0}

    def test_sampling_kwargs_json_string_parsed(self):
        args = TrainingArguments(sampling_kwargs='{"max_batch_size": 8}')
        assert args.sampling_kwargs == {"max_batch_size": 8}


class TestNoiseCalibrationKwargs:
    """``noise_calibration_kwargs`` defaults and overrides."""

    def test_defaults_and_override(self):
        base = TrainingArguments()
        assert base.noise_calibration_kwargs["min"] == 0.01
        assert base.noise_calibration_kwargs["max"] == 10.0
        assert base.noise_calibration_kwargs["tolerance"] == 1e-3
        tuned = TrainingArguments(
            noise_calibration_kwargs={"min": 0.05, "tolerance": 1e-2},
        )
        assert tuned.noise_calibration_kwargs["min"] == 0.05
        assert tuned.noise_calibration_kwargs["max"] == 10.0
        assert tuned.noise_calibration_kwargs["tolerance"] == 1e-2
