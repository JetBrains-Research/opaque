"""DP training arguments — standalone dataclass for :class:`DPTrainer`.

``DPTrainingArguments`` is a plain ``@dataclass`` (no
``transformers.TrainingArguments`` inheritance). The field surface is the
intersection of "what makes sense for DP-SGD" and "what HF utilities we
use (modelcard, reporting callbacks, ``state.compute_steps``, ...) read
off the args object". Anything we previously rejected at construction
just doesn't exist — passing the field as a kwarg now raises
``TypeError: unexpected keyword argument`` (louder, harder to miss when
porting from HF Trainer scripts).

The field surface stays close to HF for things we honour (``output_dir``,
``per_device_train_batch_size``, ``logging_strategy``, ``hub_*``,
``report_to``, ...) so HF's reporting callbacks (W&B / TensorBoard /
MLflow), ``transformers.modelcard.TrainingSummary``, and
``transformers.trainer_callback.CallbackHandler`` keep working unchanged.

Distributed property contract — replaces what HF's ``TrainingArguments``
exposed via the Accelerate ``PartialState``:

- :attr:`world_size` reads ``WORLD_SIZE`` env var (1 if unset).
- :attr:`process_index` reads ``RANK`` env var (0 if unset).
- :attr:`local_process_index` reads ``LOCAL_RANK`` env var (0 if unset).
- :attr:`should_log` / :attr:`should_save` mirror HF semantics
  (``log_on_each_node`` / ``save_on_each_node``).
- :attr:`parallel_mode` returns the HF ``ParallelMode`` enum value so
  HF utilities that probe it ("are we distributed?") see the truth.
- :attr:`device` / :attr:`n_gpu` / :attr:`train_batch_size` /
  :attr:`eval_batch_size` mirror HF property names.

DP-correct invariants worth flagging:

- ``max_grad_norm`` is the **per-example DP clipping bound** (Opaque
  reuses HF's field name for parity with Trainer-shaped configs). Pass a
  positive scalar for a single global clip, or a ``dict`` / JSON object
  with a required ``"fallback"`` key (default clip) plus substring
  pattern keys for :func:`opaque.api.engine.clipping.per_group` semantics. Adaptive /
  auto clipping hyperparameters stay in ``clipping_mode`` and
  ``clipping_kwargs`` (not per-group norms).
- ``gradient_accumulation_steps`` is reinterpreted as a Poisson-rate
  scaler — the expected logical batch is
  ``per_device_train_batch_size * gradient_accumulation_steps``. One
  Poisson round = one DP-SGD step. Warning emitted when GA != 1.
- ``optim`` accepts the torchopt-backed names DPTrainer wires
  (``adam``, ``adamw``, ``adamw-bc``, ``sgd``, ``rmsprop``, ``adagrad``,
  ``adadelta``, ``adamax``, ``radam``); HF's ``OptimizerNames`` values
  (``adamw_torch``, ``adafactor``, ``lion``, …) are rejected with a
  per-name redirection.
- ``metric_for_best_model`` must resolve to an eval-side metric (raise
  on ``"train_*"`` shape) when ``load_best_model_at_end`` is on.

Unknown kwargs are intentionally not enumerated here: unsupported HF
arguments are not part of this dataclass surface and naturally raise
``TypeError: unexpected keyword argument`` at construction.
"""

from __future__ import annotations

import dataclasses
import json
import logging
import os
import warnings
from dataclasses import field
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
from transformers.training_args import ParallelMode
from transformers.utils import is_torch_bf16_gpu_available, is_torch_xla_available


log = logging.getLogger(__name__)


_INCLUDE_NUM_INPUT_TOKENS_SEEN_VALUES = frozenset({"no", "all", "non_padding"})
_DDP_BACKEND_CHOICES: tuple[str, ...] = (
    "nccl",
    "gloo",
    "mpi",
    "xccl",
    "hccl",
    "cncl",
    "mccl",
)
_DDP_BACKEND_FIRST_CLASS: tuple[str, ...] = ("nccl", "gloo", "mpi")
_DDP_BACKEND_ENV_DEPENDENT: tuple[str, ...] = ("xccl", "hccl", "cncl", "mccl")

# JSON dict fields that may arrive as JSON strings via CLI (HF parity).
_JSON_DICT_FIELDS: tuple[str, ...] = (
    "liger_kernel_config",
    "lr_scheduler_kwargs",
    "gradient_checkpointing_kwargs",
    "clipping_kwargs",
    "sampling_kwargs",
    "noise_calibration_kwargs",
    "privacy_noise_mechanism_kwargs",
)

# Optimizer surface is owned by ``_optim``; keep validation logic and
# aliases in one place so ``DPTrainingArguments`` and optimizer factory
# stay in sync.
from ._optim import (  # noqa: E402
    resolve_optimizer_name as _resolve_optimizer_name,
    supported_names as _supported_optimizer_names,
)

_DP_OPTIMIZERS: tuple[str, ...] = _supported_optimizer_names()


@dataclasses.dataclass
class DPTrainingArguments:
    """Standalone training arguments for :class:`DPTrainer`.

    Field surface mirrors HF ``TrainingArguments`` for the subset
    DPTrainer honours, plus DPTrainer-specific fields. Unsupported HF
    knobs are intentionally omitted from this class surface.

    Batch-size contract (DP-correct interpretation):

    - ``per_device_train_batch_size`` is the **physical** batch — the
      microbatch that vmap consumes in one chunk.
    - ``per_device_train_batch_size * gradient_accumulation_steps`` is
      the **logical** batch — the expected size of one Poisson-sampled
      round that defines a single DP-SGD step.
    - Under DDP, the global expected batch is unchanged; per-rank batch
      is ``expected_batch_size / world_size``. Privacy accounting drives
      off the global expected batch.
    """

    # =================================================================
    # Output / scope
    # =================================================================
    output_dir: str | None = None
    overwrite_output_dir: bool = False
    do_train: bool = False
    do_eval: bool = False
    do_predict: bool = False

    # =================================================================
    # Batch sizes
    # =================================================================
    per_device_train_batch_size: int = 8
    per_device_eval_batch_size: int = 8
    gradient_accumulation_steps: int = 1
    eval_accumulation_steps: int | None = None
    eval_delay: float = 0.0
    auto_find_batch_size: bool = False

    # =================================================================
    # Optimizer / LR schedule
    # =================================================================
    learning_rate: float = 5e-5
    weight_decay: float = 0.0
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_epsilon: float = 1e-8
    max_grad_norm: float | dict[str, Any] | str = 1.0
    optim: str = "adamw"
    optim_args: str | None = None
    lr_scheduler_type: SchedulerType | str = "linear"
    lr_scheduler_kwargs: dict[str, Any] | str | None = field(default_factory=dict)
    warmup_ratio: float = 0.0
    warmup_steps: int = 0

    # =================================================================
    # Training duration
    # =================================================================
    num_train_epochs: float = 3.0
    max_steps: int = -1

    # =================================================================
    # Logging
    # =================================================================
    log_level: str = "passive"
    log_level_replica: str = "warning"
    log_on_each_node: bool = True
    logging_dir: str | None = None
    logging_strategy: IntervalStrategy | str = "steps"
    logging_first_step: bool = False
    logging_steps: float = 500
    logging_nan_inf_filter: bool = True

    # =================================================================
    # Saving
    # =================================================================
    save_strategy: SaveStrategy | str = "steps"
    save_steps: float = 500
    save_total_limit: int | None = None
    save_safetensors: bool = True
    save_on_each_node: bool = False
    save_only_model: bool = False
    restore_callback_states_from_checkpoint: bool = False

    # =================================================================
    # Reproducibility
    # =================================================================
    seed: int = 42
    data_seed: int | None = None
    full_determinism: bool = False

    # =================================================================
    # Precision
    # =================================================================
    use_cpu: bool = False
    use_mps_device: bool = False
    bf16: bool = False
    fp16: bool = False
    fp16_full_eval: bool = False
    bf16_full_eval: bool = False
    tf32: bool | None = None

    # =================================================================
    # Distributed (DDP)
    # =================================================================
    local_rank: int = -1
    # Distributed backend policy for current Opaque support surface.
    # Keep this aligned with tested/runtime-supported backend matrix.
    ddp_backend: str | None = None

    # =================================================================
    # Evaluation
    # =================================================================
    eval_strategy: IntervalStrategy | str = "no"
    eval_steps: float | None = None
    eval_on_start: bool = False
    eval_do_concat_batches: bool = True
    batch_eval_metrics: bool = False
    prediction_loss_only: bool = False
    include_for_metrics: list[str] = field(default_factory=list)
    include_inputs_for_metrics: bool = False  # deprecated alias
    eval_use_gather_object: bool = False
    average_tokens_across_devices: bool = True
    metric_for_best_model: str | None = None
    greater_is_better: bool | None = None
    load_best_model_at_end: bool = False
    ignore_data_skip: bool = False

    # =================================================================
    # DataLoader
    # =================================================================
    dataloader_num_workers: int = 0
    dataloader_persistent_workers: bool = False
    dataloader_pin_memory: bool = True
    dataloader_prefetch_factor: int | None = None
    dataloader_drop_last: bool = False
    remove_unused_columns: bool = True
    torch_empty_cache_steps: int | None = None

    # =================================================================
    # Labels
    # =================================================================
    label_names: list[str] | None = None
    label_smoothing_factor: float = 0.0

    # =================================================================
    # Hub
    # =================================================================
    push_to_hub: bool = False
    hub_model_id: str | None = None
    hub_strategy: HubStrategy | str = "every_save"
    hub_token: str | None = None
    hub_private_repo: bool | None = None
    hub_always_push: bool = False
    hub_revision: str | None = None

    # =================================================================
    # Reporting
    # =================================================================
    report_to: str | list[str] | None = None
    disable_tqdm: bool | None = None
    run_name: str | None = None
    project: str | None = None

    # =================================================================
    # Compile / kernels (Phase 11 owns wiring; field surface stays)
    # =================================================================
    torch_compile: bool = False
    torch_compile_backend: str | None = None
    torch_compile_mode: str | None = None
    use_liger_kernel: bool = False
    liger_kernel_config: dict[str, Any] | str | None = None

    # =================================================================
    # Misc
    # =================================================================
    gradient_checkpointing: bool = False
    gradient_checkpointing_kwargs: dict[str, Any] | str | None = None
    skip_memory_metrics: bool = True
    include_tokens_per_second: bool = False
    include_num_input_tokens_seen: bool | str = False
    debug: list | str | None = ""
    resume_from_checkpoint: str | None = None
    hp_name: str | None = None  # HPO surface; backend dispatch in _hpo.py

    # =================================================================
    # Generic memory optimization (DP-shaped, not DP-specific)
    # =================================================================
    cpu_offload_activations: bool = False

    # =================================================================
    # Differential privacy (budget, mechanisms, sampling, DDP data policy)
    # =================================================================

    # ---- Privacy budget (user targets) ----------------------------------
    privacy_target_epsilon: float = 8.0
    privacy_target_delta: float | None = None

    # ---- Clipping (mode + JSON-style args, HF ``optim_args`` pattern) ---
    clipping_mode: str = "fixed"
    clipping_kwargs: dict[str, Any] | str = field(default_factory=dict)

    # ---- Noise mechanism / fixed multiplier -------------------------------
    privacy_noise_mechanism: str = "gaussian"
    privacy_noise_multiplier: float | None = None
    privacy_noise_radius: float = 3.0
    #: Extra kwargs forwarded into :func:`opaque.dpsgd.noise.gaussian_noise`
    #: (e.g. ``bound`` for the bounded Gaussian mechanism).  JSON/HF-style
    #: parity with ``sampling_kwargs`` / ``clipping_kwargs``.
    privacy_noise_mechanism_kwargs: dict[str, Any] | str = field(default_factory=dict)

    # ---- Poisson subsampling (cap via ``sampling_kwargs``) ---------------
    sampling_mode: str = "poisson"
    sampling_kwargs: dict[str, Any] | str = field(default_factory=dict)

    # ---- DDP rank-data policy (HF-style ``ddp_`` prefix) -----------------
    # Selects how the training dataset is distributed across DDP ranks.
    # ``"per_rank"`` (default): each rank operates on a contiguous, disjoint
    # shard (``opaque.distributed.local_shard``); the Poisson sampler runs
    # locally on the shard with the same epoch-folded key on every rank, so
    # the *local* sample rate equals the *global* sample rate. Privacy
    # accountant uses the regular Poisson mechanism. Centralized DP-SGD;
    # mirrors HF's ``DistributedSampler`` rank-data semantics.
    # ``"global"``: every rank sees the full dataset and runs an
    # independent Poisson draw via ``fold_in(epoch_key, rank)``; unique
    # examples may appear on multiple ranks per step. Privacy accountant
    # switches to ``acc.parallel_poisson(..., num_workers=world_size)``.
    ddp_shard: str = "per_rank"

    # ---- Noise-multiplier calibration to ε (search bounds + tolerance) ---
    noise_calibration_kwargs: dict[str, Any] | str = field(default_factory=dict)

    # =================================================================
    # Validation / coercion
    # =================================================================

    def __post_init__(self) -> None:
        """Validate / coerce arguments.

        Idempotent: re-entry (e.g. via ``dataclasses.replace`` or a
        manual second call) short-circuits after the first successful
        invocation, so the same ``DPTrainingArguments`` instance can be
        reused — the dataclass output stays stable across constructions
        even though many fields are mutated in-place.
        """
        if getattr(self, "_dp_post_init_done", False):
            return

        # --- 1. Output directory --------------------------------------------
        if self.output_dir is None:
            self.output_dir = "trainer_output"
            log.info("No output directory specified, defaulting to 'trainer_output'.")
        if self.output_dir is not None:
            self.output_dir = os.path.expanduser(self.output_dir)
        if self.logging_dir is None and self.output_dir is not None:
            self.logging_dir = os.path.join(self.output_dir, "runs")
        if self.logging_dir is not None:
            self.logging_dir = os.path.expanduser(self.logging_dir)

        # CLI dict fields can arrive as JSON strings.  Only parse object
        # literals; other strings (notably config paths) are left alone.
        for field_name in _JSON_DICT_FIELDS:
            passed_value = getattr(self, field_name, None)
            if isinstance(passed_value, str) and passed_value.startswith("{"):
                setattr(
                    self,
                    field_name,
                    _convert_str_dict(json.loads(passed_value)),
                )

        for _name in (
            "clipping_kwargs",
            "sampling_kwargs",
            "noise_calibration_kwargs",
            "privacy_noise_mechanism_kwargs",
        ):
            _v = getattr(self, _name)
            if _v is None:
                setattr(self, _name, {})
            elif not isinstance(_v, dict):
                raise TypeError(
                    f"{_name} must be a dict or a JSON object string; "
                    f"got {type(_v).__name__}."
                )
        _nc = self.noise_calibration_kwargs
        for _k, _default in (("min", 0.01), ("max", 10.0), ("tolerance", 1e-3)):
            _nc.setdefault(_k, _default)

        # --- 1c. Deprecated DP mechanism / sampling names -------------------
        if self.privacy_noise_mechanism == "truncated_gaussian":
            warnings.warn(
                "privacy_noise_mechanism='truncated_gaussian' is deprecated; use "
                "'gaussian' with privacy_noise_mechanism_kwargs={'bound': ...} "
                "(defaults bound from privacy_noise_radius when missing).",
                DeprecationWarning,
                stacklevel=2,
            )
            self.privacy_noise_mechanism = "gaussian"
            _pnkw = dict(self.privacy_noise_mechanism_kwargs)
            _pnkw.setdefault("bound", self.privacy_noise_radius)
            self.privacy_noise_mechanism_kwargs = _pnkw
        if self.sampling_mode == "truncated_poisson":
            warnings.warn(
                "sampling_mode='truncated_poisson' is deprecated; use "
                "sampling_mode='poisson' with sampling_kwargs "
                "truncated_batch_size or max_batch_size.",
                DeprecationWarning,
                stacklevel=2,
            )
            _sk = dict(self.sampling_kwargs) if isinstance(self.sampling_kwargs, dict) else {}
            if "truncated_batch_size" not in _sk and "max_batch_size" not in _sk:
                _sk["truncated_batch_size"] = self.expected_batch_size
            self.sampling_kwargs = _sk
            self.sampling_mode = "poisson"

        # --- 2. Strategy enum coercion --------------------------------------
        # Round-trip through ``.value`` so the field stays a plain string
        # downstream.  HF's ``ExplicitEnum`` instances compare equal to
        # their string value (``IntervalStrategy.STEPS == "steps"``).
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

        # --- 3. ``disable_tqdm`` default from log level (HF parity) ---------
        if self.disable_tqdm is None:
            self.disable_tqdm = log.getEffectiveLevel() > logging.WARN

        # --- 4. Cadence-alignment validation --------------------------------
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

        # --- 5. Default population ------------------------------------------
        if not self.do_eval and self.eval_strategy != IntervalStrategy.NO.value:
            self.do_eval = True

        if (
            self.load_best_model_at_end
            or self.lr_scheduler_type == SchedulerType.REDUCE_ON_PLATEAU
        ) and self.metric_for_best_model is None:
            self.metric_for_best_model = "loss"

        if self.greater_is_better is None and self.metric_for_best_model is not None:
            self.greater_is_better = not self.metric_for_best_model.endswith("loss")

        # --- 6. Warmup / dataloader sanity ----------------------------------
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

        # --- 7. Mixed precision sanity --------------------------------------
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

        # --- 8. torch_compile_mode whitelist --------------------------------
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

        # --- 9. include_inputs_for_metrics deprecated alias ----------------
        if self.include_inputs_for_metrics:
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

        # --- 10. Optimizer name validation ---------------------------------
        # Validation/alias normalization is centralized in ``_optim``.
        _resolve_optimizer_name(self.optim)

        # --- 11. max_grad_norm (DP per-example clip, optional per-group dict) ---
        self.max_grad_norm = _coerce_max_grad_norm(self.max_grad_norm)

        # --- 11b. DP mechanism / clipping / sampling surfaces ---------------
        if self.clipping_mode not in ("fixed", "adaptive", "auto"):
            raise ValueError(
                f"clipping_mode must be 'fixed', 'adaptive', or 'auto'; "
                f"got {self.clipping_mode!r}."
            )
        if self.sampling_mode != "poisson":
            raise ValueError(
                f"sampling_mode must be 'poisson'; got {self.sampling_mode!r}."
            )
        if self.privacy_noise_mechanism != "gaussian":
            raise ValueError(
                f"privacy_noise_mechanism must be 'gaussian'; "
                f"got {self.privacy_noise_mechanism!r}."
            )

        # --- 12. metric_for_best_model must be eval-side -------------------
        if self.load_best_model_at_end and self.metric_for_best_model:
            m = self.metric_for_best_model
            if m.startswith("train_"):
                raise ValueError(
                    f"metric_for_best_model={m!r} resolves to a training-set "
                    f"metric.  Best-model selection on training metrics is "
                    f"a privacy leak vector under DP-SGD: which checkpoint "
                    f"survives rotation reveals which examples the model "
                    f"memorised most.  Use an eval-side metric (default: "
                    f'"loss" → "eval_loss").'
                )

        # --- 13. DP semantic-divergence warnings ---------------------------
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

        # --- 14. Distributed (DDP) defaults & validation -------------------
        # ``LOCAL_RANK`` env-var fallback for ``local_rank`` (HF parity).
        if self.local_rank == -1:
            self.local_rank = int(os.environ.get("LOCAL_RANK", -1))

        # ``ddp_backend`` validation mirrors HF's backend surface.
        if (
            self.ddp_backend is not None
            and self.ddp_backend not in _DDP_BACKEND_CHOICES
        ):
            raise ValueError(
                f"ddp_backend={self.ddp_backend!r} is unsupported. "
                f"Expected one of {_DDP_BACKEND_CHOICES} or None."
            )
        if (
            self.ddp_backend is not None
            and self.ddp_backend in _DDP_BACKEND_ENV_DEPENDENT
        ):
            log.warning(
                "ddp_backend=%r is accepted for HF parity but requires a vendor "
                "runtime stack that is not covered by Opaque first-class CI. "
                "Launch with an initialized process group on compatible hardware.",
                self.ddp_backend,
            )

        # ``ddp_shard`` rank-data policy gate.
        if self.ddp_shard not in ("per_rank", "global"):
            raise ValueError(
                f"ddp_shard must be 'per_rank' or 'global'; got {self.ddp_shard!r}."
            )
        # ``"global"`` only meaningful for world_size > 1.
        if self.ddp_shard == "global" and self.world_size <= 1:
            raise ValueError(
                "ddp_shard='global' requires world_size > 1; on a single "
                "process it has identical data semantics to 'per_rank' but "
                "would invoke acc.parallel_poisson which over-charges ε.  "
                "Set ddp_shard='per_rank' (the default)."
            )

        # Idempotency sentinel.
        self._dp_post_init_done = True

    # =================================================================
    # Distributed property contract (replaces HF `distributed_state`-driven
    # properties; sourced directly from env vars / device pick).
    # =================================================================

    @cached_property
    def world_size(self) -> int:
        """Number of processes in the DDP world (1 if not launched under DDP)."""
        return int(os.environ.get("WORLD_SIZE", "1") or "1")

    @cached_property
    def process_index(self) -> int:
        """Global rank of this process (0 if not launched under DDP)."""
        return int(os.environ.get("RANK", "0") or "0")

    @cached_property
    def local_process_index(self) -> int:
        """Local-node rank of this process (0 if not launched under DDP)."""
        return int(os.environ.get("LOCAL_RANK", "0") or "0")

    @property
    def parallel_mode(self) -> ParallelMode:
        """`ParallelMode.DISTRIBUTED` when world_size > 1, else NOT_DISTRIBUTED."""
        if self.world_size > 1:
            return ParallelMode.DISTRIBUTED
        return ParallelMode.NOT_DISTRIBUTED

    @property
    def should_log(self) -> bool:
        """Whether this rank should emit log payloads (HF parity)."""
        if self.log_on_each_node:
            return self.local_process_index == 0
        return self.process_index == 0

    @property
    def should_save(self) -> bool:
        """Whether this rank should write checkpoint / artefact files (HF parity)."""
        if self.save_on_each_node:
            return self.local_process_index == 0
        return self.process_index == 0

    @property
    def train_batch_size(self) -> int:
        """Cluster-wide training batch size (HF parity)."""
        return self.per_device_train_batch_size * max(1, self.world_size)

    @property
    def eval_batch_size(self) -> int:
        """Cluster-wide eval batch size (HF parity)."""
        return self.per_device_eval_batch_size * max(1, self.world_size)

    @property
    def expected_batch_size(self) -> int:
        """Logical batch size — the expected Poisson-sampled round size.

        ``per_device_train_batch_size * gradient_accumulation_steps``.
        Drives the DP sample rate and per-step normalization.
        """
        return self.per_device_train_batch_size * self.gradient_accumulation_steps

    # =================================================================
    # Device resolution (bypasses Accelerate)
    # =================================================================

    @cached_property
    def _setup_devices(self) -> torch.device:
        """Return the torch device for this process; bypass Accelerate."""
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
            # DDP-aware: bind to the rank's local GPU when launched under
            # torchrun / mp.spawn.  Falls back to ``cuda:0`` for
            # single-process runs.
            local_rank = int(os.environ.get("LOCAL_RANK", -1))
            if local_rank >= 0:
                return torch.device(f"cuda:{local_rank}")
            return torch.device("cuda:0")
        self._n_gpu = 0
        return torch.device("cpu")

    @property
    def device(self) -> torch.device:
        """Resolved device for this process (HF parity)."""
        return self._setup_devices

    @property
    def n_gpu(self) -> int:
        """Number of GPUs this process uses (HF parity)."""
        # ``_setup_devices`` populates ``_n_gpu`` as a side effect.
        _ = self._setup_devices
        return getattr(self, "_n_gpu", 0)


# =====================================================================
# Helpers
# =====================================================================


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


def _coerce_max_grad_norm(value: Any) -> float | dict[str, float]:
    """Normalize ``max_grad_norm``: positive scalar or per-group dict."""
    if isinstance(value, bool):
        raise TypeError("max_grad_norm must not be a boolean")
    if isinstance(value, str):
        stripped = value.strip()
        if stripped.startswith("{"):
            loaded = json.loads(stripped)
            if not isinstance(loaded, dict):
                raise ValueError(
                    "max_grad_norm JSON must be an object mapping strings to "
                    f"numbers; got {type(loaded).__name__}"
                )
            return _coerce_max_grad_norm(_convert_str_dict(loaded))
        try:
            value = float(stripped)
        except ValueError as exc:
            raise ValueError(
                "max_grad_norm must be a positive number or a JSON object with "
                f"a 'fallback' key; got {value!r}"
            ) from exc
    if isinstance(value, (int, float)):
        out = float(value)
        if out <= 0.0:
            raise ValueError(
                "max_grad_norm must be strictly positive for DP-SGD clipping; "
                f"got {out!r}."
            )
        return out
    if isinstance(value, dict):
        coerced: dict[str, float] = {}
        for k, v in value.items():
            if not isinstance(k, str):
                raise TypeError(
                    "max_grad_norm dict keys must be str (pattern or 'fallback'); "
                    f"got {type(k).__name__}"
                )
            if isinstance(v, bool):
                raise TypeError(f"max_grad_norm[{k!r}] must be numeric, not bool")
            fv = float(v)
            if fv <= 0.0:
                raise ValueError(f"max_grad_norm[{k!r}] must be > 0; got {v!r}")
            coerced[k] = fv
        if "fallback" not in coerced:
            raise ValueError(
                "max_grad_norm dict must include a 'fallback' key with the "
                "default per-example clip bound"
            )
        if len(coerced) == 1:
            return coerced["fallback"]
        return coerced
    raise TypeError(
        "max_grad_norm must be float, int, dict[str, float], or str; "
        f"got {type(value).__name__}"
    )
