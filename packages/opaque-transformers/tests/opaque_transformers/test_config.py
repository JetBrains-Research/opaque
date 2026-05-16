"""Tests for ``TrainingArguments.__post_init__`` HF-parity contract.

Covers:

- ``no_cuda`` is no longer accepted on the standalone dataclass.
- Auto-default of ``metric_for_best_model="loss"`` under
  ``load_best_model_at_end`` / ``reduce_lr_on_plateau``.
- Cadence-alignment validation for ``load_best_model_at_end``
  (``save_strategy`` must equal ``eval_strategy``;
  ``save_steps % eval_steps == 0`` when both are step-based).
- Strategy-string validation across the four strategy fields
  (``eval_strategy`` / ``logging_strategy`` / ``save_strategy`` /
  ``lr_scheduler_type``).
"""

from __future__ import annotations

import pytest
from transformers.debug_utils import DebugOption

from opaque.transformers.trainer import TrainingArguments
from opaque.api.transformers.trainer._config import _DP_OPTIMIZERS


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
        # ``save_strategy='best'`` is a cross-field invariant: requires
        # ``eval_strategy != 'no'`` so the trainer can pick a best
        # checkpoint.  Pair with ``eval_strategy='steps'`` so the
        # coercion path exits cleanly.
        if field == "save_strategy" and value == "best":
            kwargs.update(eval_strategy="steps", eval_steps=1, save_steps=1)
        args = TrainingArguments(**kwargs)
        assert getattr(args, field) == expected

    def test_unknown_strategy_raises(self):
        with pytest.raises(ValueError):
            TrainingArguments(eval_strategy="completely-bogus")

    def test_unknown_lr_scheduler_type_raises(self):
        with pytest.raises(ValueError):
            TrainingArguments(lr_scheduler_type="not-a-real-scheduler")


class TestHFFieldNormalization:
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

    @pytest.mark.parametrize(
        "optim",
        [
            "adamw_torch_xla",
            "adamw_apex_fused",
            "adamw_bnb_8bit",
            "paged_adamw_32bit",
            "galore_adamw",
            "lomo",
            "grokadamw",
            "stable_adamw",
        ],
    )
    def test_unsupported_optimizers_raise_with_supported_list(self, optim):
        with pytest.raises(ValueError) as exc_info:
            TrainingArguments(optim=optim)
        message = str(exc_info.value)
        assert optim in message
        # Whitelist error lists the supported set.
        assert "expected one of" in message


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


# ---------------------------------------------------------------------------
# Unified dict-field input contract (Mapping / JSON / HF comma form)
# ---------------------------------------------------------------------------


from collections.abc import MutableMapping, MutableSequence  # noqa: E402


class _FakeDictConfig(MutableMapping):
    """Minimal ``Mapping`` stand-in for OmegaConf's ``DictConfig`` in tests."""

    def __init__(self, data):
        self._data = dict(data)

    def __getitem__(self, k):
        return self._data[k]

    def __setitem__(self, k, v):
        self._data[k] = v

    def __delitem__(self, k):
        del self._data[k]

    def __iter__(self):
        return iter(self._data)

    def __len__(self):
        return len(self._data)


class _FakeListConfig(MutableSequence):
    """Minimal ``Sequence`` stand-in for OmegaConf's ``ListConfig``."""

    def __init__(self, data):
        self._data = list(data)

    def __getitem__(self, i):
        return self._data[i]

    def __setitem__(self, i, v):
        self._data[i] = v

    def __delitem__(self, i):
        del self._data[i]

    def __len__(self):
        return len(self._data)

    def insert(self, i, v):
        self._data.insert(i, v)


class TestDictFieldInputContract:
    """All dict-shaped fields accept Mapping / JSON / HF comma form / None."""

    def test_mapping_input_materialises_to_dict(self):
        args = TrainingArguments(
            clipping_kwargs=_FakeDictConfig({"target_clipping_rate": 0.5}),
        )
        assert isinstance(args.clipping_kwargs, dict)
        assert args.clipping_kwargs == {"target_clipping_rate": 0.5}

    def test_nested_listconfig_materialises_to_list(self):
        # A Mapping containing a non-list Sequence (e.g. OmegaConf
        # ListConfig).  The nested container must come back as a plain
        # list — silently fixes the OmegaConf nested-ListConfig leak.
        nested = _FakeDictConfig({"items": _FakeListConfig([1, 2, 3])})
        args = TrainingArguments(clipping_kwargs=nested)
        assert args.clipping_kwargs == {"items": [1, 2, 3]}
        assert isinstance(args.clipping_kwargs["items"], list)

    def test_json_string_input_parses(self):
        args = TrainingArguments(clipping_kwargs='{"target_clipping_rate": 0.5}')
        assert args.clipping_kwargs == {"target_clipping_rate": 0.5}

    def test_hf_comma_string_input_parses(self):
        args = TrainingArguments(
            clipping_kwargs="target_clipping_rate=0.5,norm_max=10.0",
        )
        assert args.clipping_kwargs == {
            "target_clipping_rate": 0.5,
            "norm_max": 10.0,
        }

    def test_optim_args_accepts_mapping(self):
        args = TrainingArguments(optim_args={"weight_decay": 0.01})
        assert args.optim_args == {"weight_decay": 0.01}

    def test_optim_args_accepts_json_string(self):
        args = TrainingArguments(optim_args='{"weight_decay": 0.01}')
        assert args.optim_args == {"weight_decay": 0.01}

    def test_optim_args_accepts_hf_comma_string(self):
        args = TrainingArguments(optim_args="weight_decay=0.01,nesterov=True")
        assert args.optim_args == {"weight_decay": 0.01, "nesterov": True}

    def test_optim_args_none_stays_none(self):
        args = TrainingArguments(optim_args=None)
        assert args.optim_args is None

    def test_clipping_norm_accepts_dictconfig(self):
        args = TrainingArguments(
            clipping_norm=_FakeDictConfig({"fallback": 1.0, "attn": 0.5}),
        )
        assert args.clipping_norm == {"fallback": 1.0, "attn": 0.5}
