"""DP training arguments — subclass of HF ``TrainingArguments``.

``DPTrainingArguments`` inherits HF's full field set so HF tools that
introspect a ``TrainingArguments`` (callbacks, integrations, ``isinstance``
checks) keep working.  We do *not* call ``super().__post_init__()``: HF's
post-init triggers Accelerate's ``PartialState``/``AcceleratorState`` init
via ``self.device`` and bakes in env vars we don't want.  Instead we
replicate the safe HF logic ourselves and add DP-specific validation.

DP divergences from HF, all flagged at ``__post_init__`` time:

- ``max_grad_norm`` must remain at its HF default (1.0).  DP-SGD performs
  per-example gradient clipping; use ``dp_clipping_norm`` /
  ``dp_clipping_mode`` instead.
- ``gradient_accumulation_steps`` is reinterpreted as a Poisson-rate
  scaler — the expected logical batch is
  ``per_device_train_batch_size * gradient_accumulation_steps``.  One
  Poisson round = one DP-SGD step.  Warning emitted when GA != 1.
- A handful of HF parameters whose semantics break DP (``group_by_length``,
  ``dataloader_drop_last``, ``fsdp*``, ``deepspeed``, ``tpu_*``,
  ``accelerator_config``, ``parallelism_config``) raise on construction
  when set to non-default values.
- ``optim`` accepts the canonical opaque optimizer names
  (``adam``, ``adamw``, ``sgd``, ``rmsprop``, ``adagrad``, ``adafactor``,
  ``ademamix``, ``lion``, ``radam``, ``adadelta``, ``schedule_free``) and a
  curated set of HF
  aliases (``adamw_torch``, ``adamw_torch_fused``, ``adamw_hf``,
  ``adafactor``, ``lion_32bit``) that route to the same opaque
  factories.  Quantized / paged / GaLore / fused-CUDA / XLA / NPU
  variants are rejected with per-name redirection messages — see
  :mod:`opaque.transformers.trainer._optim`.
- ``metric_for_best_model`` must resolve to an eval-side metric (raise on
  bare ``"loss"`` shape that would map to *training* loss) when
  ``load_best_model_at_end`` is on.
"""

from __future__ import annotations

import dataclasses
import json
import logging
import math
import os
import warnings
from functools import cached_property
from typing import Any

import torch
from transformers.debug_utils import DebugOption
from transformers.trainer_utils import (
    HubStrategy,
    IntervalStrategy,
    SaveStrategy,
    SchedulerType,
)
from transformers.training_args import TrainingArguments
from transformers.utils import is_torch_bf16_gpu_available, is_torch_xla_available


log = logging.getLogger(__name__)


# Parameters whose HF semantics conflict with DP-SGD.  Each is rejected
# at ``__post_init__`` when set to a non-default value.  The error
# message must explain *why* it conflicts and *what* the user should do
# instead — DP correctness depends on these invariants.
DP_INCOMPATIBLE_PARAMETERS: dict[str, str] = {
    "group_by_length": (
        "length-bucketed batching violates the DP amplification-by-subsampling "
        "invariant: every training example must be included in each batch "
        "independently with equal probability p = batch_size/N.  Disable "
        "group_by_length and accept variable-size Poisson batches."
    ),
    "dataloader_drop_last": (
        "Poisson sampling produces variable-size batches by construction; "
        "drop_last has no meaning when every batch is independently sampled.  "
        "Leave at False."
    ),
    "fsdp": (
        "FSDP shards parameters across ranks, but DPTrainer computes "
        "per-example gradients via vmap over the *full* parameter tree.  "
        "Sharded parameters can't be vmapped without all-gather on every "
        "step.  Use Opaque's own DDP primitives (Phase 9) instead."
    ),
    "fsdp_min_num_params": "FSDP options have no effect — fsdp itself is not supported.",
    "fsdp_config": "FSDP options have no effect — fsdp itself is not supported.",
    "fsdp_transformer_layer_cls_to_wrap": (
        "FSDP options have no effect — fsdp itself is not supported."
    ),
    "accelerator_config": (
        "DPTrainer replaces Accelerate's backward+optimizer step with the "
        "functional DP-SGD path (vmap → clip → noise → torchopt update).  "
        "Accelerate's gradient-accumulation, mixed-precision, and "
        "distributed abstractions conflict with per-example gradient "
        "mechanics; Accelerate is not used."
    ),
    "parallelism_config": (
        "Accelerate is not used by DPTrainer (see accelerator_config); "
        "parallelism_config has no effect."
    ),
    "deepspeed": (
        "DeepSpeed ZeRO shards parameters and gradients across ranks, "
        "incompatible with vmap-based per-example gradient computation "
        "(every rank needs the full parameter tree to vmap).  Not supported."
    ),
    "tpu_num_cores": (
        "TPU/XLA execution is not supported — Opaque's vmap backend "
        "targets CUDA and CPU only."
    ),
    "mp_parameters": ("SageMaker model-parallel is not supported."),
    "ray_scope": ("Ray Tune hyperparameter search is not wired into DPTrainer."),
    "past_index": (
        "Transformer-XL style cache reuse via past_index is not supported in DPTrainer: "
        "per-step cache plumbing conflicts with vmapped per-example gradient computation."
    ),
    "torchdynamo": (
        "torchdynamo is deprecated in HF; use torch_compile / "
        "torch_compile_backend instead."
    ),
    "neftune_noise_alpha": (
        "NEFTune mutates embedding forwards with additional noise; this has not been "
        "audited against DPTrainer's functional per-example-gradient path."
    ),
    "eval_use_gather_object": (
        "eval_use_gather_object is a distributed Accelerate gather option; DPTrainer's "
        "current evaluation path is single-process."
    ),
    "average_tokens_across_devices": (
        "average_tokens_across_devices is an all-reduce option from HF Trainer; "
        "DPTrainer currently runs a single-process token counter."
    ),
}

_JSON_DICT_FIELDS: tuple[str, ...] = (
    "accelerator_config",
    "fsdp_config",
    "liger_kernel_config",
    "lr_scheduler_kwargs",
)

_INCLUDE_NUM_INPUT_TOKENS_SEEN_VALUES = frozenset({"no", "all", "non_padding"})

# HF default for ``max_grad_norm``.  Silently accepted; any other value
# raises ``TypeError`` redirecting to ``dp_clipping_norm``.
_MAX_GRAD_NORM_HF_DEFAULT: float = 1.0

# Optimizer surface — the canonical opaque names + HF compat aliases —
# is owned by ``_optim``.  ``_DP_OPTIMIZERS`` is the deduplicated list
# of names that ``args.optim`` may take; ``resolve_optimizer_name``
# is the only validator.
from opaque.transformers.trainer._optim import (  # noqa: E402
    resolve_optimizer_name as _resolve_optimizer_name,
    supported_names as _supported_optimizer_names,
)

_DP_OPTIMIZERS: tuple[str, ...] = _supported_optimizer_names()


@dataclasses.dataclass
class DPTrainingArguments(TrainingArguments):
    """Training arguments for DP-SGD training.

    Subclass of :class:`transformers.TrainingArguments`.  Inherits the
    full HF field surface; adds ``dp_*`` fields, redefines a few HF
    fields whose defaults/types we override, and validates DP-incompatible
    combinations in :meth:`__post_init__`.

    Batch-size contract (HF parity, DP-correct interpretation):

    - ``per_device_train_batch_size`` is the **physical** batch — the
      microbatch that vmap consumes in one chunk.
    - ``per_device_train_batch_size * gradient_accumulation_steps`` is the
      **logical** batch — the expected size of one Poisson-sampled round
      that defines a single DP-SGD step.

    Unlike HF's ``Trainer`` (where GA = K serial backward passes per
    optimizer step), GA here scales the Poisson sample rate so the round
    is atomic from the privacy accountant's view: one Poisson sample,
    one clip, one noise injection, one optimizer update.
    """

    # === Field overrides ============================================
    # HF's ``OptimizerNames`` enum doesn't include our DP-friendly names
    # (``adam``, ``adamw-bc``) and HF's default ``adamw_torch`` isn't in
    # our optimizer factory.  Override to a permissive ``str`` field
    # with a default our factory understands.
    optim: str = "adamw"  # type: ignore[assignment]

    # === Generic memory optimization (not DP-specific) ==============
    # Wraps the forward pass in ``torch.autograd.graph.save_on_cpu`` so
    # activations are stashed on CPU and pulled back for the backward.
    # Trades device-host transfer for GPU memory; complements
    # ``gradient_checkpointing``.
    cpu_offload_activations: bool = False

    # === DP-specific fields (no HF equivalent, prefixed with dp_) ===

    # ---- Privacy budget --------------------------------------------
    # ``dp_target_epsilon`` is the target (ε, δ) leakage over the *entire*
    # training run.  When ``dp_noise_multiplier`` is None (the default), the
    # noise scale is auto-calibrated so the planned run reaches this ε.  The
    # trainer reports ε throughout training; it does not stop on budget
    # exceedance unless a user callback requests it.
    dp_target_epsilon: float = 8.0
    # ``dp_target_delta`` is the δ in the (ε, δ)-DP guarantee; defaults
    # to ``1 / (10 * len(dataset))`` resolved at training time.  Should
    # be < 1/N where N is the dataset size (tighter delta = stronger
    # guarantee but more noise).
    dp_target_delta: float | None = None

    # ---- Clipping --------------------------------------------------
    # ``dp_clipping_mode`` selects the per-example gradient clipping
    # policy: ``"fixed"`` clips at ``dp_clipping_norm``; ``"adaptive"``
    # tracks the empirical norm quantile and adjusts toward
    # ``dp_target_clipping_rate``; ``"auto"`` uses Andrew et al.'s
    # automatic clipping (rescaling rather than threshold-clipping).
    dp_clipping_mode: str = "fixed"
    # Clip threshold C.  Single global C in fixed mode; initial C in
    # adaptive mode; ignored in auto mode.  When ``dp_per_group_clipping``
    # is set, this value also serves as the fallback C for any parameter
    # not matched by the per-group dict.
    dp_clipping_norm: float = 1.0
    # Adaptive mode only: target fraction of examples whose gradient
    # norm should *exceed* C (so 0.5 means C tracks the median norm).
    # Drives the clip-threshold update each step.
    dp_target_clipping_rate: float = 0.5
    # Adaptive mode only: hard ceiling on the adaptive C; prevents
    # runaway when norms blow up.
    dp_clipping_norm_max: float = 10.0
    # Auto mode only: soft-min epsilon in the rescaling denominator
    # (``g / (||g|| + γ)``).  Smaller γ ≈ closer to true clipping;
    # larger γ ≈ smoother but biased.
    dp_auto_clipping_gamma: float = 0.01
    # Per-parameter-group clip norms (e.g., LoRA-A vs LoRA-B); maps
    # parameter-name regex → norm.  Unmatched parameters fall back to
    # ``dp_clipping_norm``.  None = single global norm.
    dp_per_group_clipping: dict[str, float] | None = None

    # ---- Noise -----------------------------------------------------
    # ``"gaussian"`` (default) or ``"truncated_gaussian"``.  Truncated
    # uses the Bounded Gaussian Mechanism (Chen & Hale 2024).
    dp_noise_mechanism: str = "gaussian"
    # σ in the noise-on-gradients-step.  ``None`` triggers binary-search
    # auto-calibration over the run length to hit ``dp_target_epsilon``
    # at the final step.  Manual values bypass calibration; the
    # accountant still tracks ε.
    dp_noise_multiplier: float | None = None
    # Truncation radius for ``dp_noise_mechanism="truncated_gaussian"``;
    # noise is clamped to ``[-radius * σ, +radius * σ]``.  Ignored under
    # the plain Gaussian mechanism.
    dp_noise_radius: float = 3.0

    # ---- Sampling --------------------------------------------------
    # ``"poisson"`` (default) or ``"truncated_poisson"`` (cap on realized
    # batch size — see ``dp_max_batch_size``).
    dp_sampler: str = "poisson"
    # Hard upper bound on the realized batch size; only used by
    # ``dp_sampler="truncated_poisson"``.  Defaults to
    # ``expected_batch_size`` (single round size cap).
    dp_max_batch_size: int | None = None

    # ---- Auto-calibration bounds & tolerance -----------------------
    # Bracket for the σ binary-search.  Widen if the target ε is
    # unreachable inside ``[min, max]``.
    dp_calibration_min: float = 0.01
    dp_calibration_max: float = 10.0
    # ε convergence tolerance for the binary search.
    dp_calibration_tolerance: float = 1e-3

    # ---- Optimizer DP knobs ---------------------------------------
    # ``dp_noise_bias_correction`` activates the φ-EMA correction in
    # DP-aware optimizers (``adamw``, ``adam``, ``rmsprop``,
    # ``adagrad``, ``ademamix``, ``adafactor``); the optimizer reads
    # the realized per-step σ off the ``NoisedPytree`` and subtracts a
    # β₂-EMA of the noise variance from the second moment (Chooi et
    # al., arXiv:2511.07843).  No effect on ``sgd`` / ``lion``
    # (no v) or under second-moment substitution.
    dp_noise_bias_correction: bool = False
    # ``dp_decoupled_weight_decay`` flips weight-decay between L2
    # regularisation (added to the gradient) and decoupled weight
    # decay (subtracted from params after the moment update).  HF's
    # ``adamw_torch`` is decoupled by default; opaque mirrors that.
    # ``None`` lets each factory pick its own default.
    dp_decoupled_weight_decay: bool | None = None
    # ``dp_update_rms_clip`` activates StableAdamW-style update
    # rescaling in the moment-scaler stage: divides the moment-scaled
    # update by ``max(1, rms / threshold)``.  ``None`` disables.
    # Applies to ``adam`` / ``adamw`` / ``rmsprop`` / ``adafactor`` /
    # ``ademamix``.
    dp_update_rms_clip: float | None = None

    # ---- Resolved privacy metadata --------------------------------
    # Populated by DPTrainer after dataset-dependent privacy setup, before
    # ``on_train_begin`` callbacks.  These are reporting/config fields; the
    # user-input ``dp_*`` fields above remain the source of intent.
    privacy_target_delta: float | None = dataclasses.field(default=None, init=False)
    privacy_noise_multiplier: float | None = dataclasses.field(default=None, init=False)
    privacy_noise_multiplier_source: str | None = dataclasses.field(
        default=None, init=False
    )
    privacy_sample_rate: float | None = dataclasses.field(default=None, init=False)
    privacy_expected_batch_size: int | None = dataclasses.field(
        default=None, init=False
    )
    privacy_total_steps: int | None = dataclasses.field(default=None, init=False)

    # =================================================================
    # Validation / coercion
    # =================================================================

    def __post_init__(self) -> None:
        """Validate / coerce arguments and reject DP-incompatible values.

        Does *not* call ``super().__post_init__()``: HF's post-init
        initialises Accelerate state we don't want.  Instead, we
        replicate the safe HF logic step by step and add DP validation.

        Idempotent: re-entry (e.g. via ``dataclasses.replace`` or a
        manual second call) short-circuits after the first successful
        invocation, so the same ``DPTrainingArguments`` instance can be
        reused — the dataclass output stays stable across constructions
        even though many fields are mutated in-place.
        """
        # Idempotency guard: ``__post_init__`` mutates fields in place
        # (e.g. enum coercion, ``logging_steps``/``eval_steps``/
        # ``save_steps`` int-coercion, ``logging_dir`` defaulting).
        # Running it twice on the same instance would (a) re-trigger
        # ``FutureWarning`` for legacy aliases that were already folded
        # in and (b) re-run the DP-incompat rejection that already
        # passed.  We snapshot a sentinel after the first pass and bail
        # on subsequent calls; sweep parity (``DPTrainer(args=args)``
        # twice) is the canonical caller.
        if getattr(self, "_dp_post_init_done", False):
            return

        # --- 1. Output directory --------------------------------------------
        # HF parity: default ``output_dir`` to ``"trainer_output"`` and
        # expand both ``output_dir`` and ``logging_dir`` (for ``~`` etc.).
        if self.output_dir is None:
            self.output_dir = "trainer_output"
            log.info("No output directory specified, defaulting to 'trainer_output'.")
        if self.output_dir is not None:
            self.output_dir = os.path.expanduser(self.output_dir)
        if self.logging_dir is None and self.output_dir is not None:
            self.logging_dir = os.path.join(self.output_dir, "runs")
        if self.logging_dir is not None:
            self.logging_dir = os.path.expanduser(self.logging_dir)

        # HF parity: CLI dict fields can arrive as JSON strings.  Only
        # parse object literals; other strings (notably config paths) are
        # left alone and then validated/rejected by the relevant DP guard.
        for field_name in _JSON_DICT_FIELDS:
            passed_value = getattr(self, field_name, None)
            if isinstance(passed_value, str) and passed_value.startswith("{"):
                setattr(
                    self,
                    field_name,
                    _convert_str_dict(json.loads(passed_value)),
                )

        # --- 2. Legacy alias normalisation ----------------------------------

        # ``evaluation_strategy`` (deprecated in HF 4.41+, removed in 5.x):
        # fold into ``eval_strategy`` and warn.
        evaluation_strategy = getattr(self, "evaluation_strategy", None)
        if evaluation_strategy is not None:
            warnings.warn(
                "Using `evaluation_strategy` is deprecated and will be "
                "removed in version 5 of HuggingFace Transformers.  Use "
                "`eval_strategy` instead.",
                FutureWarning,
                stacklevel=2,
            )
            self.eval_strategy = evaluation_strategy
            self.evaluation_strategy = None  # type: ignore[attr-defined]

        # ``no_cuda`` was deprecated and finally removed in HF 5.x.
        # Older HF versions still emit it as a field; guard with getattr
        # so we don't crash on newer transformers.
        if getattr(self, "no_cuda", False):
            warnings.warn(
                "Using `no_cuda` is deprecated and will be removed in "
                "version 5 of HuggingFace Transformers.  Use `use_cpu` "
                "instead.",
                FutureWarning,
                stacklevel=2,
            )
            self.use_cpu = True

        # --- 3. Strategy enum coercion --------------------------------------
        # Round-trip through ``.value`` so the field stays a plain string
        # downstream.  HF's ``ExplicitEnum`` instances compare equal to
        # their string value (``IntervalStrategy.STEPS == "steps"``), so
        # storing as string keeps both HF callbacks and our internal
        # string compares working.
        self.eval_strategy = IntervalStrategy(self.eval_strategy).value
        self.logging_strategy = IntervalStrategy(self.logging_strategy).value
        self.save_strategy = SaveStrategy(self.save_strategy).value
        self.hub_strategy = HubStrategy(self.hub_strategy)
        # Keep as the SchedulerType enum (not .value) so HF utilities that
        # call ``trainer.args.lr_scheduler_type.value`` work correctly.
        self.lr_scheduler_type = SchedulerType(self.lr_scheduler_type)

        if isinstance(self.debug, str):
            self.debug = [DebugOption(s) for s in self.debug.split()]
        elif self.debug is None:
            self.debug = []

        if isinstance(self.include_num_input_tokens_seen, bool):
            self.include_num_input_tokens_seen = (
                "all" if self.include_num_input_tokens_seen else "no"
            )
        if (
            self.include_num_input_tokens_seen
            not in _INCLUDE_NUM_INPUT_TOKENS_SEEN_VALUES
        ):
            raise ValueError(
                "include_num_input_tokens_seen must be one of "
                f"{sorted(_INCLUDE_NUM_INPUT_TOKENS_SEEN_VALUES)} or a boolean; "
                f"got {self.include_num_input_tokens_seen!r}."
            )

        if self.report_to == "all" or self.report_to == ["all"]:
            from transformers.integrations import get_available_reporting_integrations

            self.report_to = get_available_reporting_integrations()
        elif self.report_to in (None, "none") or self.report_to == ["none"]:
            self.report_to = []
        elif not isinstance(self.report_to, list):
            self.report_to = [self.report_to]

        # --- 4. ``disable_tqdm`` default from log level (HF parity) ---------
        if self.disable_tqdm is None:
            self.disable_tqdm = log.getEffectiveLevel() > logging.WARN

        # --- 5. Cadence-alignment validation --------------------------------

        # ``eval_steps`` fallback to ``logging_steps`` when ``eval_strategy``
        # is "steps" but ``eval_steps`` is unset (HF parity).
        if self.eval_strategy == IntervalStrategy.STEPS.value and (
            self.eval_steps is None or self.eval_steps == 0
        ):
            if self.logging_steps and self.logging_steps > 0:
                log.info(
                    "Using `logging_steps` to initialize `eval_steps` to %s",
                    self.logging_steps,
                )
                self.eval_steps = self.logging_steps
            else:
                raise ValueError(
                    f"eval_strategy {self.eval_strategy!r} requires either a "
                    "non-zero `eval_steps` or a non-zero `logging_steps`."
                )

        # ``logging_steps`` must be non-zero when logging strategy is
        # ``"steps"``.
        if (
            self.logging_strategy == IntervalStrategy.STEPS.value
            and self.logging_steps == 0
        ):
            raise ValueError(
                f"logging strategy {self.logging_strategy} requires "
                "non-zero --logging_steps"
            )

        # Coerce >1 step fields to ``int`` (HF parity).
        if (
            self.logging_strategy == IntervalStrategy.STEPS.value
            and self.logging_steps > 1
        ):
            if self.logging_steps != int(self.logging_steps):
                raise ValueError(
                    f"--logging_steps must be an integer if bigger than 1: "
                    f"{self.logging_steps}"
                )
            self.logging_steps = int(self.logging_steps)
        if (
            self.eval_strategy == IntervalStrategy.STEPS.value
            and self.eval_steps is not None
            and self.eval_steps > 1
        ):
            if self.eval_steps != int(self.eval_steps):
                raise ValueError(
                    f"--eval_steps must be an integer if bigger than 1: "
                    f"{self.eval_steps}"
                )
            self.eval_steps = int(self.eval_steps)
        if self.save_strategy == SaveStrategy.STEPS.value and self.save_steps > 1:
            if self.save_steps != int(self.save_steps):
                raise ValueError(
                    f"--save_steps must be an integer if bigger than 1: "
                    f"{self.save_steps}"
                )
            self.save_steps = int(self.save_steps)

        # ``load_best_model_at_end`` requires save / eval cadences to match
        # (mirrors HF wording so users searching docs find the same fix).
        if (
            self.load_best_model_at_end
            and self.save_strategy != SaveStrategy.BEST.value
        ):
            if self.eval_strategy != self.save_strategy:
                raise ValueError(
                    "--load_best_model_at_end requires the save and eval "
                    f"strategy to match, but found\n  Evaluation strategy: "
                    f"{self.eval_strategy}\n  Save strategy: "
                    f"{self.save_strategy}"
                )
            if (
                self.eval_strategy == IntervalStrategy.STEPS.value
                and self.eval_steps
                and self.save_steps
            ):
                if self.save_steps % self.eval_steps != 0:
                    if self.eval_steps < 1 or self.save_steps < 1:
                        if not (self.eval_steps < 1 and self.save_steps < 1):
                            raise ValueError(
                                "--load_best_model_at_end requires the saving steps to be a multiple "
                                "of the evaluation steps, which cannot be guaranteed when mixing "
                                f"ratio and absolute steps for save_steps={self.save_steps} and "
                                f"eval_steps={self.eval_steps}."
                            )
                        large_multiplier = 1_000_000
                        if (self.save_steps * large_multiplier) % (
                            self.eval_steps * large_multiplier
                        ) != 0:
                            raise ValueError(
                                "--load_best_model_at_end requires the saving steps to be a multiple "
                                f"of the evaluation steps, but found save_steps={self.save_steps}, "
                                f"which is not a multiple of eval_steps={self.eval_steps}."
                            )
                    else:
                        raise ValueError(
                            "--load_best_model_at_end requires the saving steps to "
                            "be a round multiple of the evaluation steps, but found "
                            f"save_steps={self.save_steps}, which is not a round "
                            f"multiple of eval_steps={self.eval_steps}."
                        )

        # --- 6. Default population ------------------------------------------

        # ``do_eval`` auto-flips when an eval strategy is configured.
        if not self.do_eval and self.eval_strategy != IntervalStrategy.NO.value:
            self.do_eval = True

        # ``metric_for_best_model`` defaults to ``"loss"`` when the trainer
        # needs one (``load_best_model_at_end`` or ``reduce_lr_on_plateau``).
        if (
            self.load_best_model_at_end
            or self.lr_scheduler_type == SchedulerType.REDUCE_ON_PLATEAU
        ) and self.metric_for_best_model is None:
            self.metric_for_best_model = "loss"

        # ``greater_is_better`` defaults from the metric name suffix.
        if self.greater_is_better is None and self.metric_for_best_model is not None:
            self.greater_is_better = not self.metric_for_best_model.endswith("loss")

        # --- 7. Warmup / dataloader sanity (HF parity) ----------------------

        if (
            self.lr_scheduler_type == SchedulerType.REDUCE_ON_PLATEAU
            and self.eval_strategy == IntervalStrategy.NO.value
            and not self.eval_on_start
        ):
            raise ValueError(
                "lr_scheduler_type='reduce_lr_on_plateau' requires eval_strategy != 'no'"
            )

        if self.warmup_ratio is None:
            self.warmup_ratio = 0.0
        if self.warmup_ratio < 0 or self.warmup_ratio > 1:
            raise ValueError("warmup_ratio must lie in range [0,1]")
        if not isinstance(self.warmup_steps, int) or self.warmup_steps < 0:
            raise ValueError(
                "warmup_steps must be of type int and must be 0 or a positive integer."
            )
        if self.warmup_ratio > 0 and self.warmup_steps > 0:
            log.info(
                "Both warmup_ratio and warmup_steps given, warmup_steps "
                "will override any effect of warmup_ratio during training"
            )

        if self.torch_empty_cache_steps is not None and not (
            isinstance(self.torch_empty_cache_steps, int)
            and self.torch_empty_cache_steps > 0
        ):
            raise ValueError(
                "torch_empty_cache_steps must be an integer bigger than 0, "
                f"got {self.torch_empty_cache_steps!r}."
            )

        if self.use_cpu:
            self.dataloader_pin_memory = False

        if (
            self.dataloader_num_workers == 0
            and self.dataloader_prefetch_factor is not None
        ):
            raise ValueError(
                "--dataloader_prefetch_factor can only be set when data is "
                "loaded in a different process, i.e. when "
                "--dataloader_num_workers > 1."
            )

        if self.fp16 and self.bf16:
            raise ValueError("At most one of fp16 and bf16 can be True, but not both")
        if self.fp16_full_eval and self.bf16_full_eval:
            raise ValueError(
                "At most one of fp16 and bf16 can be True for full eval, but not both"
            )
        if (self.bf16 or self.bf16_full_eval) and not self.use_cpu:
            if not is_torch_bf16_gpu_available() and not is_torch_xla_available():
                raise ValueError(
                    "Your setup doesn't support bf16/gpu. Set use_cpu=True for CPU bf16, "
                    "or use an Ampere+ CUDA GPU."
                )

        # --- 7b. half_precision_backend / fp16_opt_level ------------------
        # We support only native PyTorch autocast.  Apex (and SageMaker's
        # ``cpu_amp``) carry their own state machines that don't compose
        # with the functional DP path.  Both fields were removed in newer
        # HF versions; ``getattr`` keeps us forward-compatible.
        hpb = getattr(self, "half_precision_backend", "auto")
        if hpb not in ("auto", "native"):
            raise ValueError(
                f"half_precision_backend={hpb!r} is "
                "not supported.  DPTrainer uses native PyTorch autocast.  "
                "Set half_precision_backend='auto' (default) or 'native'."
            )
        opt_level = getattr(self, "fp16_opt_level", "O1")
        if opt_level != "O1":
            # HF default for ``fp16_opt_level`` is "O1" (Apex semantics).
            # We reject any non-default value because the level only
            # matters for Apex, which we don't use.
            raise ValueError(
                f"fp16_opt_level={opt_level!r} is an Apex-specific "
                "optimization level; Apex is not supported by DPTrainer.  "
                "Use fp16=True with the default fp16_opt_level."
            )

        # --- 7c. torch_compile_mode whitelist ----------------------------
        # ``torch.compile`` accepts mode in {"default", "reduce-overhead",
        # "max-autotune", "max-autotune-no-cudagraphs"}.  We accept the
        # same set; reject other values up front so users see a clear
        # error rather than a deep stack from inside Inductor.
        if self.torch_compile_mode is not None and self.torch_compile_mode not in (
            "default",
            "reduce-overhead",
            "max-autotune",
            "max-autotune-no-cudagraphs",
        ):
            raise ValueError(
                f"torch_compile_mode={self.torch_compile_mode!r} is not a "
                "valid torch.compile mode.  Expected one of: 'default', "
                "'reduce-overhead', 'max-autotune', "
                "'max-autotune-no-cudagraphs'."
            )

        # --- 8. ``include_inputs_for_metrics`` ----------------------------
        # Deprecated alias for ``include_for_metrics=["inputs"]``; folded
        # in with a ``FutureWarning``.
        if getattr(self, "include_inputs_for_metrics", False):
            warnings.warn(
                "Using `include_inputs_for_metrics` is deprecated and will "
                "be removed in version 5 of HuggingFace Transformers.  "
                "Please use `include_for_metrics=['inputs']` instead.",
                FutureWarning,
                stacklevel=2,
            )
            if "inputs" not in self.include_for_metrics:
                self.include_for_metrics.append("inputs")
            self.include_inputs_for_metrics = False

        # --- 9. Optimizer name validation -----------------------------------
        # ``optim`` accepts canonical opaque names (``adamw``, ``sgd``,
        # …) and a curated set of HF aliases that map cleanly onto
        # opaque factories (``adamw_torch`` → ``adamw``, ``adafactor``
        # → ``adafactor``, ``lion_32bit`` → ``lion``).  Quantized /
        # paged / fused-CUDA / GaLore variants and optimizers without a
        # functional DP mapping (e.g. ``adamax``) are rejected by the
        # resolver with redirect messages.  ``OptimizerNames`` enum
        # inputs are normalised via ``.value`` inside the resolver.
        _resolve_optimizer_name(self.optim)

        # ``optim_target_modules`` selects low-rank / layer-wise HF optimizers;
        # DPTrainer uses a single functional transform over the full trainable
        # pytree — no GaLore-style targeting.
        otm = getattr(self, "optim_target_modules", None)
        if otm is not None:
            raise TypeError(
                "optim_target_modules is not supported by DPTrainer (no "
                "per-module functional optimizer wiring).  Leave it unset "
                "(HF default None) or subclass create_optimizer."
            )

        # --- 10. DP-incompat rejection --------------------------------------
        # ``max_grad_norm``: HF clips global gradient norm; DP clips
        # per-example.  Accept the HF default (1.0) without comment so
        # shared HF training scripts work unchanged; raise on anything
        # else with a redirection to ``dp_clipping_norm``.
        if not math.isclose(
            float(self.max_grad_norm),
            _MAX_GRAD_NORM_HF_DEFAULT,
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise TypeError(
                f"max_grad_norm={self.max_grad_norm!r} is not honored under "
                f"DP-SGD: per-example gradient clipping is governed by "
                f"dp_clipping_norm (and dp_clipping_mode), not the HF "
                f"global-norm clip.  Set dp_clipping_norm and/or "
                f"dp_clipping_mode to the desired clipping policy and "
                f"leave max_grad_norm at its HF default."
            )

        # Reject DP-incompat HF params when set to non-default values.
        for name, reason in DP_INCOMPATIBLE_PARAMETERS.items():
            value = getattr(self, name, None)
            default = _hf_field_default(name)
            if not _is_default(value, default):
                raise ValueError(
                    f"DPTrainer rejects {name}={value!r} (HF default: "
                    f"{default!r}).\n  {reason}"
                )

        # --- 11. B1: ``metric_for_best_model`` must be eval-side ------------
        # Best-model decisions on training-set metrics would leak per-example
        # information through which checkpoints survive rotation.  Require
        # an ``eval_``-prefixed name (or one we'll auto-prefix) and reject
        # anything that resolves to a training-only metric.
        if self.load_best_model_at_end and self.metric_for_best_model:
            m = self.metric_for_best_model
            # HF auto-defaulted "loss" is acceptable — it resolves to
            # ``eval_loss`` once ``evaluate()`` runs.  Same for any name
            # that the eval loop will emit; we accept anything that does
            # NOT start with ``"train_"`` (the only training-side prefix
            # we recognise).
            if m.startswith("train_"):
                raise ValueError(
                    f"metric_for_best_model={m!r} resolves to a training-set "
                    f"metric.  Best-model selection on training metrics is "
                    f"a privacy leak vector under DP-SGD: which checkpoint "
                    f"survives rotation reveals which examples the model "
                    f"memorised most.  Use an eval-side metric (default: "
                    f'"loss" → "eval_loss").'
                )

        # --- 12. G2: DP semantic-divergence warnings ------------------------
        if self.gradient_accumulation_steps != 1:
            log.warning(
                "DPTrainer reinterprets gradient_accumulation_steps=%d as a "
                "Poisson sample-rate scaler (expected logical batch = "
                "per_device_train_batch_size * gradient_accumulation_steps), "
                "NOT as K serial backward passes per optimizer step.  This "
                "differs from HF Trainer.  See "
                "docs/development/dp_training_arguments_plan.md.",
                self.gradient_accumulation_steps,
            )
        # --- 13. Single-process distributed defaults ------------------------
        # HF's ``_setup_devices`` populates these via Accelerate's
        # ``PartialState``; we don't go through Accelerate, so set them
        # directly.  Multi-rank distributed support lands in Phase 9.
        if self.local_rank == -1:
            self.local_rank = int(os.environ.get("LOCAL_RANK", -1))

        # ``distributed_state`` is set by HF's ``super().__post_init__()``
        # via Accelerate.  We skip that, but several HF utilities (e.g.
        # ``TrainingSummary.from_trainer`` → ``parallel_mode``) guard on
        # ``self.distributed_state is not None``.  Setting it to ``None``
        # here satisfies those guards without pulling in Accelerate.
        if not hasattr(self, "distributed_state"):
            self.distributed_state = None

        # Idempotency sentinel: see top-of-method docstring.
        self._dp_post_init_done = True

    # =================================================================
    # Device resolution (bypasses Accelerate)
    # =================================================================

    @cached_property
    def _setup_devices(self) -> torch.device:
        """Return the torch device for this process; bypass Accelerate.

        HF's ``TrainingArguments._setup_devices`` initialises Accelerate's
        ``PartialState``/``AcceleratorState`` and reads it for device/n_gpu
        info.  DPTrainer doesn't use Accelerate, so we resolve devices
        directly from the ``no_cuda``/``use_cpu``/``use_mps_device`` flags
        and set the relevant attributes ourselves.
        """
        if self.use_cpu:
            self._n_gpu = 0
            return torch.device("cpu")
        if getattr(self, "use_mps_device", False) or (
            hasattr(torch.backends, "mps")
            and torch.backends.mps.is_available()
            and not torch.cuda.is_available()
        ):
            self._n_gpu = 0
            return torch.device("mps")
        if torch.cuda.is_available():
            self._n_gpu = 1
            return torch.device("cuda:0")
        self._n_gpu = 0
        return torch.device("cpu")

    # =================================================================
    # Computed properties
    # =================================================================

    @property
    def expected_batch_size(self) -> int:
        """Logical batch size — the expected Poisson-sampled round size.

        ``per_device_train_batch_size * gradient_accumulation_steps``.
        Drives the DP sample rate and per-step normalization.
        """
        return self.per_device_train_batch_size * self.gradient_accumulation_steps


# =====================================================================
# Helpers
# =====================================================================


def _hf_field_default(name: str) -> Any:
    """Return the HF-side default value for a field by name."""
    for f in dataclasses.fields(TrainingArguments):
        if f.name == name:
            if f.default is not dataclasses.MISSING:
                return f.default
            if f.default_factory is not dataclasses.MISSING:  # type: ignore[misc]
                return f.default_factory()  # type: ignore[misc]
            return None
    return None


def _convert_str_dict(value: Any) -> Any:
    """Recursively coerce string scalars from JSON dict arguments."""
    if isinstance(value, dict):
        return {k: _convert_str_dict(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_convert_str_dict(v) for v in value]
    if isinstance(value, str):
        lowered = value.lower()
        if lowered == "true":
            return True
        if lowered == "false":
            return False
        if lowered in {"none", "null"}:
            return None
        try:
            return int(value)
        except ValueError:
            pass
        try:
            return float(value)
        except ValueError:
            pass
    return value


def _is_default(value: Any, default: Any) -> bool:
    """Compare ``value`` against ``default`` permissively.

    Treats ``None``, falsy iterables, and missing dicts as equivalent
    forms of "left at default" for the DP-incompat rejection list.
    """
    if value == default:
        return True
    if value is None and default in (None, [], {}, ""):
        return True
    if value in ([], {}, "") and default in (None, [], {}, ""):
        return True
    return False
