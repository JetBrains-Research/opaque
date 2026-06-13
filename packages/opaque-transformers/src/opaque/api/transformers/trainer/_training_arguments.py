"""DP training arguments — standalone dataclass for :class:`DPTrainer`.

``TrainingArguments`` is a plain ``@dataclass`` (no
``transformers.TrainingArguments`` inheritance). The field surface is the
intersection of "what makes sense for DP-SGD" and "what HF utilities we
use (modelcard, reporting callbacks, ``state.compute_steps``, ...) read
off the args object". Unsupported HF fields don't exist on the dataclass,
so passing one as a kwarg raises ``TypeError: unexpected keyword argument``.

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

- ``clipping_norm`` is the **per-example DP clipping bound**. Pass a
  positive scalar for a single global clip, or a ``dict`` / JSON object
  with a required ``"fallback"`` key (default clip) plus substring
  pattern keys for :func:`opaque.api.engine.clipping.per_group` semantics. Adaptive /
  auto clipping hyperparameters stay in ``clipping_mode`` and
  ``clipping_kwargs`` (not per-group norms). Pass ``math.inf`` to
  **disable clipping** entirely (the single canonical no-clip bound) —
  only meaningful for a non-private baseline (``privacy_noise_multiplier=
  0``); with noise it would mean unbounded sensitivity and is rejected.
- ``per_device_train_batch_size`` is the **per-rank logical Poisson batch**
  (HF parity at ``gradient_accumulation_steps=1``). One Poisson round =
  one DP-SGD step. Cluster-wide logical batch is
  ``per_device_train_batch_size * world_size`` (the ``train_batch_size``
  HF property). Internal microbatch chunking under OOM retry never
  changes the logical batch — privacy accounting is unaffected.
- ``optim`` accepts the torchopt-backed names DPTrainer wires
  (``adam``, ``adamw``, ``adamw-bc`` = DP bias-corrected AdamW, ``sgd``,
  ``lion``, ``ademamix``, ``adafactor``, ``rmsprop``, ``adagrad``,
  ``radam``, ``adadelta``, ``schedule_free``) plus HF aliases that map
  cleanly onto them (``adamw_torch`` / ``adamw_torch_fused`` / ``adamw_hf``
  → ``adamw``, ``lion_32bit`` → ``lion``, …).  Names with no functional
  equivalent (bitsandbytes 8-bit / paged, Apex-fused, XLA/NPU variants)
  are rejected with a per-name redirection.
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
import math
import os
from collections.abc import Mapping, Sequence
from dataclasses import field
from functools import cached_property
from typing import Any

import torch
from opaque.scheduling.types import Schedule
from transformers.debug_utils import DebugOption
from transformers.trainer_utils import SchedulerType
from transformers.training_args import ParallelMode
from transformers.utils import is_torch_bf16_gpu_available, is_torch_xla_available


# Plain-string strategy domains; replaces HF's ``IntervalStrategy`` and
# ``SaveStrategy`` enums (we never need the enum form, only the value).
_INTERVAL_STRATEGIES: tuple[str, ...] = ("no", "steps", "epoch")
_SAVE_STRATEGIES: tuple[str, ...] = ("no", "steps", "epoch", "best")


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

# Dict-shaped fields.  Each accepts the same input contract via
# ``_normalize_dict_field`` at ``__post_init__`` time: a ``Mapping`` (incl.
# OmegaConf ``DictConfig``), a JSON object string (``"{...}"``), an HF-style
# comma-separated ``"key=value,key=value"`` string, or ``None``.  The
# normalised result is ``dict[str, Any] | None`` for every entry.
_DICT_FIELDS: tuple[str, ...] = (
    "performance_kernels_config",
    "lr_scheduler_kwargs",
    "gradient_checkpointing_kwargs",
    "clipping_kwargs",
    "sampling_kwargs",
    "noise_calibration_kwargs",
    "privacy_noise_mechanism_kwargs",
    "optim_args",
)

# Privacy noise mechanism surface.  ``"gaussian"`` is the DP-SGD baseline;
# ``"mf_*"`` are DP-FTRL matrix-factorization mechanisms from
# :mod:`opaque.dpftrl.noise`, dispatched through ``_dpftrl.build_strategy``
# in :meth:`DPTrainer._setup_training`.
_MECHANISMS_DPFTRL: frozenset[str] = frozenset(
    {"mf_band", "mf_blt", "mf_bisr", "mf_bsr", "mf_lambda_cgd", "mf_identity"}
)
_MECHANISMS: frozenset[str] = frozenset({"gaussian", *_MECHANISMS_DPFTRL})

# Concrete sampling modes (resolved set; ``"auto"`` is the default field
# value and is replaced by one of these in ``__post_init__``).
_SAMPLING_MODES: frozenset[str] = frozenset(
    {"poisson", "b_min_sep", "balls_in_bins", "cyclic_poisson", "sequential"}
)

# Canonical sampler pairing.  Each mechanism has a single "best" sampler;
# users opting into a mechanism shouldn't have to remember to pair the
# sampler too.  ``sampling_mode="auto"`` (the default) resolves via this
# table.  Explicit ``sampling_mode`` overrides are validated against
# :data:`_ALLOWED_SAMPLERS` below.
_SAMPLER_BY_MECHANISM: dict[str, str] = {
    "gaussian": "poisson",
    "mf_identity": "poisson",
    "mf_band": "b_min_sep",
    "mf_blt": "balls_in_bins",
    "mf_bisr": "balls_in_bins",
    "mf_bsr": "balls_in_bins",
    "mf_lambda_cgd": "balls_in_bins",
}

# Per-mechanism allow-list for explicit ``sampling_mode`` overrides.
# ``mf_band`` accepts ``"poisson"`` as a looser-but-valid alternative to
# its canonical ``"b_min_sep"`` participation pattern; everything else
# pins a single sampler.
_ALLOWED_SAMPLERS: dict[str, frozenset[str]] = {
    "gaussian": frozenset({"poisson"}),
    "mf_identity": frozenset({"poisson"}),
    "mf_band": frozenset({"b_min_sep", "poisson"}),
    "mf_blt": frozenset({"balls_in_bins"}),
    "mf_bisr": frozenset({"balls_in_bins"}),
    "mf_bsr": frozenset({"balls_in_bins"}),
    "mf_lambda_cgd": frozenset({"balls_in_bins"}),
}

# Per-mechanism kwargs defaults auto-filled into
# ``privacy_noise_mechanism_kwargs`` when the user leaves them blank.
# Tuned for a Mellum/Kstack-shaped causal-LM target; not universally
# optimal, just a sensible starting point so ``privacy_noise_mechanism=
# "mf_band"`` works out of the box.  User-supplied keys win on collision.
# Keys match the strategy factory signatures in
# :mod:`opaque.dpftrl.noise` exactly so the trainer can spread the dict
# into the factory call.
_MECH_DEFAULTS: dict[str, dict[str, Any]] = {
    "mf_band": {"bands": 16},
    # BLT buffer count is a rational-approximation degree (the optimizer
    # searches up to max_buffers and stops early), NOT a band width; the
    # BLT math rejects > 15 as ill-conditioned. 10 matches the library's
    # own optimize() default.
    "mf_blt": {"max_buffers": 10},
    "mf_bisr": {"bandwidth": 4},
    "mf_bsr": {"bandwidth": 8, "alpha": 1.0, "beta": 0.9},
    "mf_lambda_cgd": {"lambda_": 0.5},
    "mf_identity": {},
}

# Optimizer surface is owned by ``_optim``; keep validation logic and
# aliases in one place so ``TrainingArguments`` and optimizer factory
# stay in sync.
from ._optim import (  # noqa: E402
    resolve_optimizer_name as _resolve_optimizer_name,
    supported_names as _supported_optimizer_names,
)

_DP_OPTIMIZERS: tuple[str, ...] = _supported_optimizer_names()


@dataclasses.dataclass
class TrainingArguments:
    """Standalone training arguments for :class:`DPTrainer`.

    Field surface mirrors HF ``TrainingArguments`` for the subset
    DPTrainer honours, plus DPTrainer-specific fields. Unsupported HF
    knobs are intentionally omitted from this class surface.

    Batch-size contract (DP-correct interpretation):

    - ``per_device_train_batch_size`` is the **per-rank logical Poisson
      batch** — the expected size of the sample drawn on each rank for
      one DP-SGD step. Matches HF semantics at ``gradient_accumulation_steps=1``.
    - Cluster-wide logical batch is ``per_device_train_batch_size *
      world_size`` (exposed as the HF property ``train_batch_size``).
      The sample rate ``q = train_batch_size / N_total`` drives privacy
      accounting.
    - Internal microbatch chunking (only activated by
      ``auto_find_microbatch_size`` on OOM retry) splits the per-rank
      logical batch into smaller vmap calls. This never changes the
      logical batch and is privacy-neutral.
    """

    # =================================================================
    # Output / scope
    # =================================================================
    output_dir: str | None = None
    overwrite_output_dir: bool = False

    # =================================================================
    # Batch sizes
    # =================================================================
    per_device_train_batch_size: int = 8
    # ``None`` defaults to ``per_device_train_batch_size`` in
    # ``__post_init__`` so callers don't OOM when they bump the train batch
    # for a big model and forget to also bump the eval batch (HF's stock
    # default of 8 is silently retained otherwise).
    per_device_eval_batch_size: int | None = None
    eval_accumulation_steps: int | None = None
    eval_delay: float = 0.0
    # Physical vmap chunk fed into the per-example clipping path.  Default
    # ``None`` chunks at ``per_device_train_batch_size``; a smaller value
    # processes the per-rank logical batch in
    # ``per_device_train_batch_size // microbatch_size`` accumulation
    # passes, reducing peak GPU memory.  Must be in
    # [1, per_device_train_batch_size].
    microbatch_size: int | None = None
    # On a CUDA-OOM raised mid-step, halve the current microbatch and
    # retry until it fits.  Starting point is ``microbatch_size`` if set,
    # else ``per_device_train_batch_size``.
    auto_find_microbatch_size: bool = False

    # =================================================================
    # Optimizer / LR schedule
    # =================================================================
    learning_rate: float = 5e-5
    weight_decay: float = 0.0
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_epsilon: float = 1e-8
    clipping_norm: float | dict[str, Any] | str = 1.0
    optim: str = "adamw"
    optim_args: dict[str, Any] | str | None = None
    # Accepts an HF-style name (string or :class:`SchedulerType` enum)
    # *or* an :data:`~opaque.scheduling.types.Schedule` recipe directly.
    # When a recipe is supplied, ``warmup_steps``/``warmup_ratio``/
    # ``lr_scheduler_kwargs`` must be unset — the recipe owns its own
    # composition (compose ``with_warmup(...)`` yourself).
    lr_scheduler: SchedulerType | str | Schedule = "linear"
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
    logging_strategy: str = "steps"
    logging_first_step: bool = False
    logging_steps: float = 500

    # =================================================================
    # Saving
    # =================================================================
    save_strategy: str = "steps"
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
    bf16_full_eval: bool = False
    tf32: bool | None = None

    # =================================================================
    # Distributed (DDP)
    # =================================================================
    local_rank: int = -1
    # Distributed backend policy for current Opaque support surface.
    # Keep this aligned with tested/runtime-supported backend matrix.
    ddp_backend: str | None = None
    # Timeout (seconds) for the auto ``init_process_group`` call made by
    # ``resolve_ddp_state`` when ``WORLD_SIZE > 1``. Matches HF default.
    ddp_timeout: int = 1800

    # =================================================================
    # Evaluation
    # =================================================================
    eval_strategy: str = "no"
    eval_steps: float | None = None
    eval_on_start: bool = False
    eval_do_concat_batches: bool = True
    prediction_loss_only: bool = False
    include_for_metrics: list[str] = field(default_factory=list)
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
    # Reporting
    # =================================================================
    report_to: str | list[str] | None = None
    disable_tqdm: bool | None = None
    run_name: str | None = None
    project: str | None = None

    # =================================================================
    # Hub publishing (orthogonal to DP — publish the finished model)
    # =================================================================
    # Minimal "publish & manage" surface: the model is uploaded once at the
    # end of training (or via an explicit ``trainer.push_to_hub()``), with a
    # model card carrying the DP ε/δ provenance.  The HF in-training auto-push
    # machinery (``hub_strategy``, per-checkpoint async uploads) is
    # intentionally not supported — it re-couples Hub to the checkpoint loop.
    push_to_hub: bool = False
    hub_model_id: str | None = None
    hub_token: str | None = None
    hub_private_repo: bool | None = None
    hub_revision: str | None = None

    # =================================================================
    # Compile / kernels
    # =================================================================
    torch_compile: bool = False
    torch_compile_backend: str | None = None
    torch_compile_mode: str | None = None
    # ``use_performance_kernels`` gates the CUDA + Triton kernel group
    # (``rope``, ``rms_norm``, ``activation``, ``cross_entropy``).  Default
    # ``False`` because the kernels need CUDA + Triton at runtime and the
    # default cluster shape isn't guaranteed to have them.  ``kv_cache``
    # is a pure-Python ``performance`` patch and stays enabled regardless;
    # disable it explicitly via ``performance_kernels_config={"kv_cache":
    # False}`` for models whose forward depends on the HF ``DynamicCache``.
    use_performance_kernels: bool = False
    # Flat ``dict[str, bool]`` forwarded as-is to
    # ``opaque.patches.apply_model_patches`` kwargs (no key translation).
    # Supported keys: ``rope``, ``rms_norm``, ``activation``,
    # ``cross_entropy``, ``fused_linear_cross_entropy``, ``kv_cache``,
    # ``eager_attention``, ``batchify``.  ``fused_linear_cross_entropy``
    # is opt-in because the fused forward returns ``logits=None``, which
    # is incompatible with ``compute_metrics`` /
    # ``preprocess_logits_for_metrics``.
    performance_kernels_config: dict[str, Any] | str | None = None
    # Whether ``opaque.patches.apply_model_patches`` should apply compat
    # patches (vmap-safety: ``eager_attention``, ``batchify``, vmap-safe
    # masking / collator / checkpoint hooks).  Default ``True``.  Set to
    # ``False`` for custom models designed to be vmap-safe without
    # opaque's patches, or for non-HF ``nn.Module`` test fixtures —
    # silences the "no detectable family" info log opaque emits when it
    # doesn't recognise the model.
    use_compat_patches: bool = True

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

    # =================================================================
    # Generic memory optimization (DP-shaped, not DP-specific)
    # =================================================================
    #: Offload saved activations to CPU during the backward pass to extend the
    #: trainable batch past the GPU activation ceiling (host RAM is left
    #: pageable — never pinned — so the OS can swap; see ``_setup_training``).
    #: Trades host-transfer bandwidth for GPU memory; off by default.
    activation_offloading: bool = False

    # =================================================================
    # Differential privacy (budget, mechanisms, sampling, DDP data policy)
    # =================================================================

    # ---- Privacy budget (user targets) ----------------------------------
    # Set ``privacy_noise_multiplier`` (use 0.0 for non-private) or
    # ``privacy_target_epsilon`` (to calibrate noise).  When both are set,
    # training halts at the first log boundary where ε ≥ target_epsilon.
    privacy_target_epsilon: float | None = None
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

    # ---- Subsampling (cap via ``sampling_kwargs``) -----------------------
    # ``"auto"`` (default) pairs the sampler with
    # :attr:`privacy_noise_mechanism` via :data:`_SAMPLER_BY_MECHANISM` in
    # ``__post_init__``; explicit overrides are validated against
    # :data:`_ALLOWED_SAMPLERS`.  Downstream code only ever sees a
    # resolved concrete mode.
    sampling_mode: str = "auto"
    sampling_kwargs: dict[str, Any] | str = field(default_factory=dict)

    # ---- Noise-multiplier calibration to ε (search bounds + tolerance) ---
    noise_calibration_kwargs: dict[str, Any] | str = field(default_factory=dict)

    # ---- Resume policy --------------------------------------------------
    # There is no "resume without DP state" opt-in.  ``resume_from_checkpoint``
    # requires a *complete* DP checkpoint (dp_state + optimizer + accountant);
    # a weights-only export is not resumable.  To start a fresh DP run from
    # arbitrary weights (public-data warmup, an HF checkpoint, a pretrained
    # model), load them at construction via ``model=...`` — the run begins
    # with a zero accountant.

    # =================================================================
    # Validation / coercion
    # =================================================================

    def __post_init__(self) -> None:
        """Validate / coerce arguments.

        Idempotent: re-entry (e.g. via ``dataclasses.replace`` or a
        manual second call) short-circuits after the first successful
        invocation, so the same ``TrainingArguments`` instance can be
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

        # Dict-shaped fields accept Mapping (incl. OmegaConf DictConfig),
        # JSON object string, HF-style "key=value,..." string, or None.
        # Normalize once here so downstream code only ever sees
        # ``dict[str, Any] | None``.
        for field_name in _DICT_FIELDS:
            setattr(
                self,
                field_name,
                _normalize_dict_field(getattr(self, field_name)),
            )

        # Privacy / clipping / sampling kwargs default to ``{}`` rather
        # than ``None`` for the consumer's convenience (avoids ``or {}``
        # at every read site).  ``optim_args`` / ``lr_scheduler_kwargs``
        # / ``performance_kernels_config`` / ``gradient_checkpointing_kwargs``
        # may legitimately be ``None`` (= unset, fall through to factory
        # defaults) and stay as-is.
        for _name in (
            "clipping_kwargs",
            "sampling_kwargs",
            "noise_calibration_kwargs",
            "privacy_noise_mechanism_kwargs",
        ):
            if getattr(self, _name) is None:
                setattr(self, _name, {})
        _nc = self.noise_calibration_kwargs
        for _k, _default in (("min", 0.01), ("max", 10.0), ("tolerance", 1e-3)):
            _nc.setdefault(_k, _default)

        # --- 2. Strategy validation -----------------------------------------
        # Plain-string strategies (no HF enum round-trip).
        if self.eval_strategy not in _INTERVAL_STRATEGIES:
            raise ValueError(
                f"eval_strategy={self.eval_strategy!r}; "
                f"expected one of {_INTERVAL_STRATEGIES}"
            )
        if self.logging_strategy not in _INTERVAL_STRATEGIES:
            raise ValueError(
                f"logging_strategy={self.logging_strategy!r}; "
                f"expected one of {_INTERVAL_STRATEGIES}"
            )
        if self.save_strategy not in _SAVE_STRATEGIES:
            raise ValueError(
                f"save_strategy={self.save_strategy!r}; "
                f"expected one of {_SAVE_STRATEGIES}"
            )
        # Normalize string / SchedulerType forms to the enum; a
        # user-supplied ``Schedule`` recipe is left as-is and consumed
        # directly by ``build_lr_schedule``.
        if isinstance(self.lr_scheduler, (str, SchedulerType)):
            self.lr_scheduler = SchedulerType(self.lr_scheduler)
        elif not callable(self.lr_scheduler):
            raise TypeError(
                f"lr_scheduler must be a str, SchedulerType, or "
                f"Schedule callable; got "
                f"{type(self.lr_scheduler).__name__}."
            )

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
        if self.eval_strategy == "steps" and (
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

        if self.logging_strategy == "steps" and self.logging_steps == 0:
            raise ValueError(
                f"logging strategy {self.logging_strategy} requires "
                "non-zero --logging_steps"
            )

        # Coerce >1 step fields to ``int`` (HF parity).
        if self.logging_strategy == "steps" and self.logging_steps > 1:
            if self.logging_steps != int(self.logging_steps):
                raise ValueError(
                    f"--logging_steps must be an integer if bigger than 1: "
                    f"{self.logging_steps}"
                )
            self.logging_steps = int(self.logging_steps)
        if (
            self.eval_strategy == "steps"
            and self.eval_steps is not None
            and self.eval_steps > 1
        ):
            if self.eval_steps != int(self.eval_steps):
                raise ValueError(
                    f"--eval_steps must be an integer if bigger than 1: "
                    f"{self.eval_steps}"
                )
            self.eval_steps = int(self.eval_steps)
        if self.save_strategy == "steps" and self.save_steps > 1:
            if self.save_steps != int(self.save_steps):
                raise ValueError(
                    f"--save_steps must be an integer if bigger than 1: "
                    f"{self.save_steps}"
                )
            self.save_steps = int(self.save_steps)

        if self.load_best_model_at_end and self.save_strategy != "best":
            if self.eval_strategy != self.save_strategy:
                raise ValueError(
                    "--load_best_model_at_end requires the save and eval "
                    f"strategy to match, but found\n  Evaluation strategy: "
                    f"{self.eval_strategy}\n  Save strategy: "
                    f"{self.save_strategy}"
                )
            if self.eval_strategy == "steps" and self.eval_steps and self.save_steps:
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
        if self.load_best_model_at_end and self.metric_for_best_model is None:
            self.metric_for_best_model = "loss"

        if self.greater_is_better is None and self.metric_for_best_model is not None:
            self.greater_is_better = not self.metric_for_best_model.endswith("loss")

        # --- 5b. Cross-field invariants -------------------------------------
        # ``save_strategy='best'`` requires eval to be configured so we
        # can actually pick a best checkpoint.
        if self.save_strategy == "best" and self.eval_strategy == "no":
            raise ValueError("save_strategy='best' requires eval_strategy != 'no'")
        # ``load_best_model_at_end`` requires both eval and save to be on.
        if self.load_best_model_at_end:
            if self.eval_strategy == "no":
                raise ValueError(
                    "load_best_model_at_end=True requires eval_strategy != 'no'"
                )
            if self.save_strategy == "no":
                raise ValueError(
                    "load_best_model_at_end=True requires save_strategy != 'no'"
                )
        # ``save_steps`` must be positive when save_strategy != 'no'.
        if (
            self.save_strategy != "no"
            and self.save_steps is not None
            and self.save_steps <= 0
        ):
            raise ValueError(f"save_steps must be > 0, got {self.save_steps}")

        # --- 6. Warmup / dataloader sanity ----------------------------------
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
        # bf16 is the only mixed-precision mode: it needs no loss scaler and
        # has the dynamic range fp16's scaler exists to fake.  fp16 training
        # (autocast + dynamic loss scaling) is intentionally not supported —
        # it adds a per-example unscale-before-clip landmine for no benefit on
        # the bf16-capable hardware this targets.
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

        # Default eval batch to the per-device train batch when caller
        # leaves it unset, so bumping the train batch for a big model
        # doesn't silently leave eval at HF's stock 8.
        if self.per_device_eval_batch_size is None:
            self.per_device_eval_batch_size = self.per_device_train_batch_size

        # microbatch_size must fit in [1, per_device_train_batch_size]
        # so the per-rank logical batch divides cleanly into
        # ``per_device_train_batch_size // microbatch_size`` vmap chunks.
        if self.microbatch_size is not None:
            if self.microbatch_size < 1:
                raise ValueError(
                    f"microbatch_size must be >= 1; got {self.microbatch_size}."
                )
            if self.microbatch_size > self.per_device_train_batch_size:
                raise ValueError(
                    f"microbatch_size ({self.microbatch_size}) cannot exceed "
                    f"per_device_train_batch_size "
                    f"({self.per_device_train_batch_size})."
                )

        # torch.compile cannot retrace the vmap+grad closure that
        # auto_find_microbatch_size rebuilds on OOM (PyTorch #128711).
        if self.torch_compile and self.auto_find_microbatch_size:
            raise ValueError(
                "torch_compile=True is incompatible with "
                "auto_find_microbatch_size=True due to torch._dynamo "
                "functorch tracing limitations (PyTorch issue #128711). "
                "Pass an explicit microbatch_size (e.g. "
                "--microbatch-size 4) and disable "
                "--auto-find-microbatch-size, or disable --torch-compile."
            )

        # --- 9. Optimizer name validation ----------------------------------
        # Validation/alias normalization is centralized in ``_optim``.
        _resolve_optimizer_name(self.optim)

        # --- 11. clipping_norm (DP per-example clip, optional per-group dict) ---
        self.clipping_norm = _coerce_clipping_norm(self.clipping_norm)

        # --- 11b. DP mechanism / clipping / sampling surfaces ---------------
        # At least one of NM / target_epsilon must be set; NM=0.0 is
        # allowed as explicit non-private.
        if (
            self.privacy_noise_multiplier is None
            and self.privacy_target_epsilon is None
        ):
            raise ValueError(
                "Set either privacy_noise_multiplier (use 0.0 for non-private "
                "training) or privacy_target_epsilon (to calibrate noise to a "
                "budget); neither was provided."
            )
        # NM=0.0 means non-private; target_epsilon is meaningless on that path.
        if (
            self.privacy_noise_multiplier is not None
            and self.privacy_noise_multiplier == 0.0
            and self.privacy_target_epsilon is not None
        ):
            raise ValueError(
                "privacy_noise_multiplier=0.0 is the non-private path; "
                "privacy_target_epsilon is meaningless there.  Drop the target "
                "or set a positive noise multiplier."
            )
        # Calibration target must be > 0.
        if (
            self.privacy_noise_multiplier is None
            and self.privacy_target_epsilon is not None
            and self.privacy_target_epsilon <= 0
        ):
            raise ValueError(
                "privacy_target_epsilon must be > 0 when calibrating noise; "
                f"got {self.privacy_target_epsilon!r}."
            )
        if (
            self.privacy_noise_multiplier is not None
            and self.privacy_noise_multiplier < 0
        ):
            raise ValueError(
                "privacy_noise_multiplier must be >= 0; got "
                f"{self.privacy_noise_multiplier!r}."
            )
        # Disabling clipping (clip bound = +inf, via clipping_norm=math.inf or
        # a per-group dict with an inf bound) is only sound for a non-private
        # run.  With noise — nm > 0, or nm calibrated (None) — infinite
        # sensitivity yields an infinite realized noise stddev (nm * inf) and
        # inf/NaN gradients, so require an explicit non-private baseline.
        _clip_disabled = (
            isinstance(self.clipping_norm, float) and math.isinf(self.clipping_norm)
        ) or (
            isinstance(self.clipping_norm, dict)
            and any(
                isinstance(v, float) and math.isinf(v)
                for v in self.clipping_norm.values()
            )
        )
        if _clip_disabled and self.privacy_noise_multiplier != 0.0:
            raise ValueError(
                "Disabling clipping (clipping_norm=math.inf) is only valid for "
                "a non-private baseline (privacy_noise_multiplier=0.0); got "
                f"privacy_noise_multiplier={self.privacy_noise_multiplier!r}. "
                "Set privacy_noise_multiplier=0.0, or pass a finite clipping_norm."
            )
        if self.privacy_target_delta is not None and not (
            0 < self.privacy_target_delta < 1
        ):
            raise ValueError(
                "privacy_target_delta must lie in (0, 1); got "
                f"{self.privacy_target_delta!r}."
            )
        if self.clipping_mode not in ("fixed", "adaptive", "auto"):
            raise ValueError(
                f"clipping_mode must be 'fixed', 'adaptive', or 'auto'; "
                f"got {self.clipping_mode!r}."
            )
        if self.privacy_noise_mechanism not in _MECHANISMS:
            raise ValueError(
                f"privacy_noise_mechanism={self.privacy_noise_mechanism!r}; "
                f"expected one of {sorted(_MECHANISMS)}."
            )
        # ``sampling_mode="auto"`` resolves to the canonical sampler for
        # the chosen mechanism; explicit values are validated against the
        # per-mechanism allow-list.  Downstream code only ever sees a
        # resolved concrete mode.
        if self.sampling_mode == "auto":
            self.sampling_mode = _SAMPLER_BY_MECHANISM[self.privacy_noise_mechanism]
        elif self.sampling_mode not in _SAMPLING_MODES:
            raise ValueError(
                f"sampling_mode={self.sampling_mode!r}; expected 'auto' or one "
                f"of {sorted(_SAMPLING_MODES)}."
            )
        elif self.sampling_mode not in _ALLOWED_SAMPLERS[self.privacy_noise_mechanism]:
            raise ValueError(
                f"sampling_mode={self.sampling_mode!r} is not valid for "
                f"privacy_noise_mechanism={self.privacy_noise_mechanism!r}; "
                f"allowed: {sorted(_ALLOWED_SAMPLERS[self.privacy_noise_mechanism])} "
                f"(omit sampling_mode or set 'auto' to pick automatically)."
            )

        if self.privacy_noise_mechanism in _MECHANISMS_DPFTRL:
            # MF privacy proofs require constant per-step record
            # sensitivity; ``adaptive`` clipping drifts the threshold
            # across steps and breaks the analysis.  ``fixed`` and
            # ``auto`` (AUTO-S smooth scaling) both keep sensitivity
            # constant by construction.  ``adaptive`` is auto-resolved
            # to ``fixed`` with a warning so a user inheriting the
            # default from a preset isn't blocked from running MF.
            if self.clipping_mode == "adaptive":
                log.warning(
                    "clipping_mode='adaptive' is incompatible with "
                    "privacy_noise_mechanism=%r (matrix-factorization "
                    "requires constant per-step sensitivity); resolving "
                    "to clipping_mode='fixed'.",
                    self.privacy_noise_mechanism,
                )
                self.clipping_mode = "fixed"
            # Auto-fill mechanism kwargs from the Mellum-shaped defaults
            # so a one-line ``privacy_noise_mechanism='mf_band'`` works
            # out of the box.  User-supplied keys win on collision.
            defaults = _MECH_DEFAULTS[self.privacy_noise_mechanism]
            for _k, _v in defaults.items():
                self.privacy_noise_mechanism_kwargs.setdefault(_k, _v)

        # Privacy-derived sampler parameters are owned by the strategy /
        # amplifier; the trainer reads them off the built recipe at
        # sampler-construction time.  Accepting them under
        # ``sampling_kwargs`` would let the runtime sampler silently
        # desync from the accountant.
        _privacy_owned = {"bands", "sampling_prob"}
        if isinstance(self.sampling_kwargs, dict):
            _bad = _privacy_owned & self.sampling_kwargs.keys()
            if _bad:
                raise ValueError(
                    f"sampling_kwargs may not carry privacy-derived keys "
                    f"{sorted(_bad)}; these are owned by "
                    f"privacy_noise_mechanism_kwargs (the strategy recipe) "
                    f"and read off the built amplifier at runtime."
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

        # --- 13. Distributed (DDP) defaults & validation -------------------
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

        # ``include_for_metrics`` opts into populating optional fields on
        # ``EvalPrediction`` — currently ``{"inputs", "loss"}``.  Fail fast on
        # unknown keys (HF raises this at runtime; raised here instead).
        _allowed_include_for_metrics = frozenset({"inputs", "loss"})
        _bad = [
            k for k in self.include_for_metrics if k not in _allowed_include_for_metrics
        ]
        if _bad:
            raise ValueError(
                f"include_for_metrics entries must be a subset of "
                f"{sorted(_allowed_include_for_metrics)}; got unknown keys: {_bad}"
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

    # --- HF-utility compatibility shims (read-only) ---------------------
    # These are *not* user knobs (no fields, so passing them to the
    # constructor still raises ``TypeError``); they exist only so HF
    # utilities that read off the args object — ``transformers.modelcard``'s
    # ``extract_hyperparameters_from_trainer``, reporting callbacks — keep
    # working without the corresponding dataclass fields.

    @property
    def gradient_accumulation_steps(self) -> int:
        """Always 1: DP-SGD does one optimizer step per Poisson round."""
        return 1

    @property
    def lr_scheduler_type(self) -> Any:
        """HF alias for ``lr_scheduler`` (the resolved ``SchedulerType``)."""
        return self.lr_scheduler

    @property
    def fp16(self) -> bool:
        """fp16 training is unsupported (bf16 only); always ``False``."""
        return False

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

    # =================================================================
    # Serialization (HF-callback parity)
    # =================================================================

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable dict of field values.

        HF reporting callbacks (W&B / TensorBoard / MLflow) call
        ``args.to_dict()`` during ``on_train_begin`` to log the run
        configuration. Provide a shallow dataclass-asdict that handles
        the non-trivial field types we declare (``Path``, ``torch.dtype``,
        enums, recipe dataclasses) by falling back to ``str()`` on
        anything ``json`` would refuse.
        """
        out: dict[str, Any] = {}
        for f in dataclasses.fields(self):
            value = getattr(self, f.name)
            try:
                json.dumps(value)
                out[f.name] = value
            except (TypeError, ValueError):
                out[f.name] = str(value)
        return out

    @classmethod
    def from_hf(
        cls,
        hf_args: Any,
        *,
        # DP knobs (one of noise_multiplier / target_epsilon required)
        privacy_noise_multiplier: float | None = None,
        privacy_target_epsilon: float | None = None,
        privacy_target_delta: float | None = None,
        clipping_norm: float | dict[str, float] | None = None,
        privacy_noise_mechanism: str = "gaussian",
        privacy_noise_radius: float = 3.0,
        clipping_mode: str = "fixed",
        clipping_kwargs: dict[str, Any] | None = None,
        sampling_mode: str = "auto",
        sampling_kwargs: dict[str, Any] | None = None,
        noise_calibration_kwargs: dict[str, Any] | None = None,
        # Behavior
        strict: bool = True,
        **opaque_overrides: Any,
    ) -> "TrainingArguments":
        """Convert an HF ``TrainingArguments`` to opaque ``TrainingArguments``.

        Required: exactly one of ``privacy_noise_multiplier=<float>``
        (fixed-noise mode) or ``privacy_target_epsilon=<float>`` (calibrated
        noise) must be set — opaque's runtime requires one to instantiate.

        Translates HF fields by bucketed manifest (see
        ``opaque.api.transformers.trainer._hf_convert``):

        - **DIRECT**: ~80 fields with identical name and semantics — copied.
        - **RENAME**: ``evaluation_strategy`` → ``eval_strategy``,
          ``per_gpu_train_batch_size`` → ``per_device_train_batch_size``,
          ``lr_scheduler_type`` → ``lr_scheduler``.
        - **TRANSFORM**: HF's ``(per_device_train_batch_size, gradient_accumulation_steps)``
          collapses into opaque's ``per_device_train_batch_size = product``
          and ``microbatch_size = per_device``. This is *required* for
          privacy-correct ε accounting — see "Batch semantics" below.
        - **REJECT_IF_SET**: raises ``ValueError`` with a per-field
          explanation if the user set ``fp16=True``, ``fsdp=...``,
          ``deepspeed=...``, ``neftune_noise_alpha=...``,
          ``optim='paged_adamw_8bit'``, etc.
        - **DROP_WITH_WARN**: silently drops HF fields that have no opaque
          equivalent (``do_train``, ``tpu_*``, ``group_by_length``, …);
          emits ``RuntimeWarning`` if non-default and ``strict=True``.

        Batch semantics
        ----------------
        DP-SGD's sample rate (and therefore ε via subsampling
        amplification) is computed from the *logical* batch — the unit
        over which one gradient + noise step occurs. In HF, that's
        ``per_device_train_batch_size × gradient_accumulation_steps``;
        in opaque, that's ``per_device_train_batch_size`` alone. The
        converter collapses HF's two-field expression into opaque's
        one-field expression with the multiplication, and sets
        ``microbatch_size = per_device`` so the per-step memory
        footprint matches HF's microbatch.

        ``opaque_overrides`` allow setting any opaque-only field
        (e.g. ``microbatch_size=4`` to override the derived value,
        ``activation_offloading=True``). These take precedence over both
        HF-derived and DP-default values.

        Raises
        ------
        ImportError
            If ``transformers`` is not installed.
        TypeError
            If ``hf_args`` is not a ``transformers.TrainingArguments`` instance.
        ValueError
            If neither ``privacy_noise_multiplier`` nor
            ``privacy_target_epsilon`` is set, or if any REJECT_IF_SET
            HF field is set to a non-default value.
        """
        # Local import keeps module load fast and avoids a hard import cycle
        # (_hf_convert imports nothing from this module at load time).
        from ._hf_convert import _convert_hf_training_arguments

        # Collect the DP-knob kwargs into one dict the converter forwards.
        dp_overrides: dict[str, Any] = {
            "privacy_noise_multiplier": privacy_noise_multiplier,
            "privacy_target_epsilon": privacy_target_epsilon,
            "privacy_noise_mechanism": privacy_noise_mechanism,
            "privacy_noise_radius": privacy_noise_radius,
            "clipping_mode": clipping_mode,
            "sampling_mode": sampling_mode,
        }
        # ``clipping_norm`` only overrides when explicitly passed; otherwise the
        # value derived from HF ``max_grad_norm`` (or opaque's own default)
        # stands.
        if clipping_norm is not None:
            dp_overrides["clipping_norm"] = clipping_norm
        if privacy_target_delta is not None:
            dp_overrides["privacy_target_delta"] = privacy_target_delta
        if clipping_kwargs is not None:
            dp_overrides["clipping_kwargs"] = clipping_kwargs
        if sampling_kwargs is not None:
            dp_overrides["sampling_kwargs"] = sampling_kwargs
        if noise_calibration_kwargs is not None:
            dp_overrides["noise_calibration_kwargs"] = noise_calibration_kwargs

        converted = _convert_hf_training_arguments(
            hf_args,
            strict=strict,
            **dp_overrides,
        )

        # ``opaque_overrides`` win over both HF-derived values and DP defaults.
        converted.update(opaque_overrides)

        return cls(**converted)


# =====================================================================
# Helpers
# =====================================================================


def _coerce_scalar(raw: Any) -> Any:
    """Best-effort literal coercion of a user-supplied option scalar.

    Recognises ``true``/``false`` / ``none``/``null`` (case-insensitive),
    integers, and floats; falls back to the original string.  Non-string
    inputs pass through unchanged.  Used to coerce CLI-style strings
    coming from HF's ``key=value,...`` shape and from JSON values that
    are themselves strings (e.g. ``{"x": "1.0"}`` → ``{"x": 1.0}``).
    """
    if not isinstance(raw, str):
        return raw
    s = raw.strip()
    lowered = s.lower()
    if lowered in ("true", "false"):
        return lowered == "true"
    if lowered in ("none", "null"):
        return None
    try:
        return int(s)
    except ValueError:
        pass
    try:
        return float(s)
    except ValueError:
        pass
    return s


def _to_native(value: Any) -> Any:
    """Recursively materialize ``Mapping`` → dict and non-str ``Sequence`` → list.

    Silently resolves OmegaConf ``DictConfig`` / ``ListConfig`` and any
    other Mapping/Sequence container into plain Python types.  String
    scalars are coerced via :func:`_coerce_scalar` (bool / null / int /
    float fallback).  ``str``/``bytes``/``bytearray`` are intentionally
    not treated as Sequences.  Idempotent on already-native structures.
    """
    if isinstance(value, Mapping):
        return {k: _to_native(v) for k, v in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_to_native(v) for v in value]
    return _coerce_scalar(value)


def _parse_dict_string(value: str) -> dict[str, Any]:
    """Parse a CLI string into a ``dict`` — JSON first, HF comma form as fallback.

    When the stripped string starts with ``{`` it is parsed as JSON and
    materialized via :func:`_to_native`.  Otherwise it is treated as
    HF's flat ``"key1=value1,key2=value2"`` shape (the format
    :class:`~transformers.HfArgumentParser` recognises for some dict
    fields and the existing ``optim_args`` parser used).  Nested keys
    are not supported in the comma form — pass JSON or a Mapping for
    nested structure.

    Raises:
        ValueError: malformed JSON, JSON whose root is not an object,
            or a comma-form entry without a key or ``=``.
    """
    stripped = value.strip()
    if stripped.startswith("{"):
        loaded = json.loads(stripped)
        if not isinstance(loaded, Mapping):
            raise ValueError(
                f"expected a JSON object (dict); got {type(loaded).__name__}"
            )
        return _to_native(loaded)
    out: dict[str, Any] = {}
    for entry in stripped.split(","):
        if not entry.strip():
            continue
        if "=" not in entry:
            raise ValueError(f"entry {entry!r} is not in 'key=value' form")
        key, _, val = entry.partition("=")
        key = key.strip()
        if not key:
            raise ValueError(f"entry {entry!r} has an empty key")
        out[key] = _coerce_scalar(val)
    return out


def _normalize_dict_field(value: Any) -> dict[str, Any] | None:
    """Normalize a dict-shaped field to ``dict[str, Any] | None``.

    Input contract (one of):
    - ``None`` — passes through.
    - ``str`` — parsed by :func:`_parse_dict_string` (JSON or HF comma form).
    - ``Mapping`` (incl. OmegaConf ``DictConfig``) — materialized to
      plain dict via :func:`_to_native`.
    """
    if value is None:
        return None
    if isinstance(value, str):
        return _parse_dict_string(value)
    if isinstance(value, Mapping):
        return _to_native(value)
    raise TypeError(
        f"dict field must be Mapping, str (JSON or 'key=value' form), or None; "
        f"got {type(value).__name__}"
    )


def _coerce_clipping_norm(value: Any) -> float | dict[str, float]:
    """Normalize ``clipping_norm``: positive scalar or per-group dict.

    To **disable** clipping, pass ``math.inf`` (the single canonical
    no-clip bound) — only meaningful with ``privacy_noise_multiplier = 0``
    (a non-private baseline).  ``None`` is rejected; there is exactly one
    way to express "no clipping".
    """
    if value is None:
        raise ValueError(
            "clipping_norm must be a positive number (use math.inf to disable "
            "clipping for a non-private run); got None."
        )
    if isinstance(value, bool):
        raise TypeError("clipping_norm must not be a boolean")
    if isinstance(value, str):
        stripped = value.strip()
        if stripped.startswith("{"):
            return _coerce_clipping_norm(_parse_dict_string(stripped))
        try:
            value = float(stripped)
        except ValueError as exc:
            raise ValueError(
                "clipping_norm must be a positive number or a JSON object with "
                f"a 'fallback' key; got {value!r}"
            ) from exc
    if isinstance(value, (int, float)):
        out = float(value)
        if out <= 0.0:
            raise ValueError(
                "clipping_norm must be strictly positive for DP-SGD clipping; "
                f"got {out!r}."
            )
        return out
    if isinstance(value, Mapping):
        coerced: dict[str, float] = {}
        for k, v in _to_native(value).items():
            if not isinstance(k, str):
                raise TypeError(
                    "clipping_norm dict keys must be str (pattern or 'fallback'); "
                    f"got {type(k).__name__}"
                )
            if isinstance(v, bool):
                raise TypeError(f"clipping_norm[{k!r}] must be numeric, not bool")
            fv = float(v)
            if fv <= 0.0:
                raise ValueError(f"clipping_norm[{k!r}] must be > 0; got {v!r}")
            coerced[k] = fv
        if "fallback" not in coerced:
            raise ValueError(
                "clipping_norm dict must include a 'fallback' key with the "
                "default per-example clip bound"
            )
        if len(coerced) == 1:
            return coerced["fallback"]
        return coerced
    raise TypeError(
        "clipping_norm must be float, int, Mapping[str, float], or str; "
        f"got {type(value).__name__}"
    )
