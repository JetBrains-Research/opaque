"""DP-SGD Trainer for HuggingFace models.

Provides :class:`DPTrainer` — a differentially private, HF-Trainer-parity
trainer built on Opaque primitives.  See the class docstring for the
public method layout.

The trainer is shape-agnostic: any HF-style ``data_collator`` whose output
the model's forward accepts will work.  Domain-specific training that
builds on this trainer (SFT / DPO / KTO) should subclass it and override
:meth:`DPTrainer.compute_per_example_loss` — the single DP-correct
extension point that both training (vmap → grad → clip → noise) and
eval (vmap when ``include_for_metrics=['loss']``) route through.
"""

from __future__ import annotations

import contextlib
import dataclasses
import functools
import inspect
import json
import logging
import math
import os
import time
from collections.abc import Mapping
from typing import Any, Callable, NamedTuple

import opaque.accounting as acc
import opaque.dpsgd.accounting as dpsgd_acc
import torch
import torchopt
from datasets import Dataset
from opaque.accounting import Accountant
from opaque.accounting import calibration as cal
from opaque.api.engine.clipping import clipped_grad
from opaque.dpsgd.clipping import adaptive_clipped_grad, auto_clipped_grad
from opaque.dpsgd.noise import gaussian_noise
from opaque.dpftrl.noise import mf_gaussian_noise
from opaque.functional import make_functional
from opaque.profiling import PerfTracker, perf_tracker
from opaque.serialization import (
    from_state_dict as opaque_from_state_dict,
    state_dict as opaque_state_dict,
)
from . import _checkpoint as ckpt
from . import _distributed
from . import _dpftrl
from . import _eval
from ._callback import (
    BestModelSaveCallback,
    build_callback_handler,
    is_metric_improved,
    resolve_eval_metric,
)
from ._config import TrainingArguments
from ._eval import EvalPrediction, EvaluationResult
from ._precision import eval_dtype
from ._scheduler import build_lr_schedule
from ._state import DPTrainerState
from opaque.random import key
from torch import Tensor
from torch.utils.data import DataLoader
from transformers import (
    DataCollatorWithPadding,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    SequenceFeatureExtractor,
    enable_full_determinism,
    set_seed,
)
from transformers.trainer_utils import RemoveColumnsCollator
from transformers.trainer_utils import (
    TrainerMemoryTracker,
    speed_metrics,
)
from transformers.data.data_collator import default_data_collator
from transformers.trainer_callback import TrainerCallback, TrainerControl
from transformers.trainer_utils import seed_worker
from transformers.utils import find_labels

__all__ = [
    "DPTrainer",
    "EvaluationResult",
    "TrainingArguments",
    "TrainOutput",
]

log = logging.getLogger(__name__)


def _callback_matches(
    candidate: type[TrainerCallback] | TrainerCallback,
    target: type[TrainerCallback] | TrainerCallback,
) -> bool:
    """Whether ``candidate`` should be removed when removing ``target``.

    Class targets match by ``isinstance`` (so passing the class removes the
    first instance of that class); instance targets match by identity.
    """
    if isinstance(target, type):
        if isinstance(candidate, type):
            return candidate is target
        return isinstance(candidate, target)
    return candidate is target


def _is_peft_model(model: Any) -> bool:
    """Whether ``model`` is a PEFT-wrapped model.

    Local reimplementation of HF's private ``transformers.trainer._is_peft_model``
    so we don't import private API. Returns ``False`` when ``peft`` is not installed.
    """
    try:
        from peft import PeftMixedModel, PeftModel
    except ImportError:
        return False
    return isinstance(model, (PeftModel, PeftMixedModel))


def _disable_tokenizers_parallelism_before_fork() -> None:
    """Avoid HuggingFace tokenizers fork warnings from reporting integrations."""
    if "TOKENIZERS_PARALLELISM" in os.environ:
        return
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    log.debug(
        "Set TOKENIZERS_PARALLELISM=false before training; set it explicitly "
        "to override this DPTrainer default."
    )


def _effective(value: Any) -> float:
    """Extract scalar from float or PerGroup for logging."""
    return value.effective if hasattr(value, "effective") else float(value)


def _compile_with_fullgraph_fallback(
    fn: Callable, *, backend: str, mode: str
) -> Callable:
    """Compile ``fn`` with ``fullgraph=True``; on first-call failure,
    log a warning and lazily recompile with ``fullgraph=False``.

    ``torch.compile`` is lazy — the compile failure (graph break under
    ``fullgraph=True``) surfaces only when the compiled function is
    actually executed.  This wrapper catches that first-execution
    exception, records the fallback, and forwards subsequent calls to
    the more permissive variant.  ``fullgraph=True`` first catches
    silent eager-fallback regressions the user explicitly opted into
    compiling against.
    """
    full = torch.compile(fn, backend=backend, mode=mode, fullgraph=True)
    fallback: Callable | None = None

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        nonlocal fallback
        if fallback is not None:
            return fallback(*args, **kwargs)
        try:
            return full(*args, **kwargs)
        except Exception as e:  # noqa: BLE001 - any compile failure → fallback
            log.warning(
                "torch.compile fullgraph=True failed (%s: %s); "
                "falling back to fullgraph=False for subsequent steps.",
                type(e).__name__,
                e,
            )
            fallback = torch.compile(fn, backend=backend, mode=mode, fullgraph=False)
            return fallback(*args, **kwargs)

    return wrapper


class TrainOutput(NamedTuple):
    """Return type of ``DPTrainer.train()``, mirroring HF's TrainOutput."""

    global_step: int
    training_loss: float
    metrics: dict[str, float]


@dataclasses.dataclass
class _TrainingContext:
    """Mutable state carried through the training loop."""

    fmodel: Callable
    trainable_params: dict[str, Tensor]
    frozen_params: dict[str, Tensor]
    grad_fn: Callable
    clip_state: Any
    noise_fn: Callable
    noise_state: Any
    noise_multiplier: float
    opt: Any
    opt_state: Any
    lr_schedule: Callable[[int], float]
    accounting: Accountant
    mechanism: Callable
    target_delta: float
    sample_rate: float
    noise_multiplier_source: str
    expected_steps_per_epoch: int
    total_steps: int
    num_epochs: int
    collate_fn: Callable
    batch_keys: tuple[str, ...] = ()
    offload_ctx: Any = dataclasses.field(default_factory=contextlib.nullcontext)
    opt_name: str = "adamw"
    current_sampler: Any = None
    save_steps_resolved: int = 0
    # Configured clip threshold (scalar or PerGroup).  Adaptive mode
    # overrides this each step via ``clip_state.clipping_norm``; fixed
    # mode reads the configured value directly because the new
    # ``FixedClipState`` is a marker without per-state fields.
    clip_norm: Any = None
    mechanism_kind: str = "gaussian"
    mf: _dpftrl.MFContext | None = None


class DPTrainer:
    """Differentially private trainer for HuggingFace models.

    Method decomposition mirrors HF Trainer:

    - ``train()`` → ``_setup_training()`` + ``_inner_training_loop()``
    - ``training_step()`` — clip → noise → optimize (fused, unlike HF)
    - ``evaluate()`` / ``predict()`` — both return :class:`EvaluationResult`;
      shared pipeline via ``_run_evaluation_loop``
    - ``compute_per_example_loss()`` — DP-correct override hook; the
      single extension point for SFT / DPO / KTO subclasses
    - ``create_optimizer()`` — functional optimizer (torchopt)
    - ``get_train_dataloader()`` — PoissonSampler
    - ``get_eval_dataloader()`` — standard DataLoader
    - ``log()`` — append to state + fire callbacks
    """

    def __init__(
        self,
        model: PreTrainedModel | None = None,
        args: TrainingArguments | None = None,
        data_collator: Callable | None = None,
        train_dataset: Dataset | None = None,
        eval_dataset: Dataset | None = None,
        processing_class: PreTrainedTokenizerBase | None = None,
        compute_loss_func: Callable | None = None,
        compute_metrics: Callable | None = None,
        callbacks: list[Any] | None = None,
        optimizers: tuple[Any | None, Any | None] = (None, None),
        optimizer_cls_and_kwargs: tuple[Any, dict[str, Any]] | None = None,
        preprocess_logits_for_metrics: Callable | None = None,
    ) -> None:
        if args is None:
            args = TrainingArguments(output_dir="tmp_trainer")
        # HF parity (``Trainer.__init__``): seed Python / NumPy / torch
        # global RNGs from ``args.seed`` so non-DP randomness (model-
        # head init when missing weights are randomised, dataset
        # shuffling without an explicit seed, user-supplied
        # ``compute_metrics`` calling ``torch.randn``, ...) is
        # reproducible run-to-run.  The DP RNG chain seeds itself from
        # ``key(args.seed)`` independently — this call is purely for
        # the non-DP surface.
        if args.full_determinism:
            enable_full_determinism(args.seed)
            log.warning(
                "full_determinism=True: enabled deterministic algorithms; this can reduce throughput."
            )
        else:
            set_seed(args.seed)
        if model is None:
            raise RuntimeError("`DPTrainer` requires a `model` argument")
        self._functional_optimizer_factory: (
            tuple[Callable[..., Any], dict[str, Any]] | None
        ) = None
        self._functional_optimizer_name: str | None = None
        if any(item is not None for item in optimizers):
            raise RuntimeError(
                "Passing `optimizers` is not supported by DPTrainer: the DP path "
                "uses a functional torchopt optimizer built after per-example "
                "gradient clipping/noising is configured."
            )
        if optimizer_cls_and_kwargs is not None:
            from ._optim import (
                validate_functional_optimizer_cls_and_kwargs,
            )

            self._functional_optimizer_factory = (
                validate_functional_optimizer_cls_and_kwargs(optimizer_cls_and_kwargs)
            )
            _fn = self._functional_optimizer_factory[0]
            self._functional_optimizer_name = getattr(
                _fn, "__name__", type(_fn).__name__
            )
        self._model = model
        # Computed once: PEFT detection appears at multiple resume /
        # restore sites and the result depends only on the model class,
        # which doesn't change after construction.
        self._is_peft: bool = _is_peft_model(model)
        self.args = args
        self._processing_class = processing_class
        self._base_callbacks: list[Any] = list(callbacks) if callbacks else []
        self.is_in_train = False
        # HF parity: when no collator is given, use
        # ``DataCollatorWithPadding`` for tokenizer / sequence-feature
        # processors and ``default_data_collator`` otherwise.  The
        # DataLoader collator stays CPU-only; tensors are moved to the
        # trainer device in ``_prepare_input`` on the main process.
        default_collator = (
            DataCollatorWithPadding(processing_class)
            if processing_class is not None
            and isinstance(
                processing_class, (PreTrainedTokenizerBase, SequenceFeatureExtractor)
            )
            else default_data_collator
        )
        self._data_collator = data_collator or default_collator
        self._compute_metrics = compute_metrics
        self._compute_loss_func = compute_loss_func
        self._preprocess_logits = preprocess_logits_for_metrics
        # Lazily-built per-example eval-loss closure (vmap'd).  Populated
        # by ``_get_eval_per_example_loss_fn`` on first use; reset to
        # ``None`` here so model rebinding can invalidate the cache.
        self._eval_per_example_loss_fn: Callable | None = None
        self._eval_per_example_loss_fn_model: Any = None

        # Default label_names so the eval loop can identify label tensors in
        # the batch dict.  HF parity (trainer.py:789-797): inspect the
        # model's forward signature for parameters whose name contains
        # "label" — for ``*ForQuestionAnswering`` models additionally pick
        # up ``start_positions`` / ``end_positions``.  Walk through the
        # PEFT wrapper to the base model so the inspected signature is the
        # one that actually consumes the labels.  Snapshot to a private
        # attribute so the user-supplied ``args`` is never mutated.
        if args.label_names is not None:
            self._label_names: list[str] = list(args.label_names)
        else:
            inspected = model
            if self._is_peft:
                if hasattr(model, "get_base_model"):
                    inspected = model.get_base_model()
                else:
                    inspected = model.base_model.model
            discovered = find_labels(inspected.__class__)
            self._label_names = list(discovered) if discovered else ["labels"]

        # HF parity: the trainer does not tokenise.  The user supplies
        # already-prepared datasets and a matching ``data_collator``.
        self._train_dataset = train_dataset
        self._eval_dataset = eval_dataset
        if args.eval_strategy != "no" and eval_dataset is None:
            raise ValueError(
                f"You have set `args.eval_strategy` to {args.eval_strategy} but "
                "didn't pass an `eval_dataset` to `DPTrainer`. Either set "
                "`eval_strategy='no'` or pass an eval_dataset."
            )

        # Resolve device via ``args.device`` (forwards to
        # :meth:`TrainingArguments._setup_devices`, which honors
        # ``use_cpu`` / ``use_mps_device`` / ``no_cuda`` and bypasses
        # Accelerate).  HF parity: the trainer's device source of truth
        # is the args, not whatever device the user happened to leave
        # the model on.  The model is moved to ``self._device`` so the
        # batches we forward (also moved to the same device by the
        # collate-fn wrapper) line up.
        self._device = self.args.device
        _SUPPORTED_DEVICE_TYPES = {"cpu", "cuda", "mps"}
        if self._device.type not in _SUPPORTED_DEVICE_TYPES:
            raise ValueError(
                f"DPTrainer only supports cpu, cuda, and mps devices; "
                f"got device={self._device!r}. "
                f"Other backends (xpu, npu, mlu, musa, hpu, ...) are not supported."
            )
        # Phase 10a: resolve rank/world topology immediately after the device
        # pick so every subsequent setup site (sampler, checkpoint, hub,
        # logging) can read the same snapshot.  Single-process is the trivial
        # case (rank=0, world=1, is_distributed=False).
        self._ddp = _distributed.resolve_ddp_state(self._device, self.args)
        _distributed.validate_ddp_backend(self.args, self._ddp)
        # HF parity (``Trainer._wrap_model``): place the model on the
        # resolved device.  ``model.to`` is a no-op when the model is
        # already there, so this is safe for callers who pre-placed.
        self._model.to(self._device)

        # Explicit patch sites (no import-time mutation of HF globals):
        # 1) global runtime compat (masking / collator / checkpoint hooks)
        # 2) ``apply_model_patches(..., compat=use_compat_patches, performance=True, kernels=use_performance_kernels)``
        from opaque.api.transformers import _runtime_bootstrap as _opaque_rt

        _opaque_rt.apply_transformers_runtime_compat_patches()
        self._apply_opaque_model_patches()

        # Compute precision (HF parity — autocast for bf16/fp16, full-cast
        # only for the *_full_eval scope).  See _setup_precision for the
        # behavior matrix.
        self._amp_dtype: torch.dtype | None = None
        self._loss_scaler = None
        self._loss_scaler_state = None
        self._setup_precision()

        # Functional state (populated by _setup_training, used by evaluate)
        self._ctx: _TrainingContext | None = None
        # Privacy accountant lives at the trainer level so ``save_model()``
        # can write ``accountant.json`` after training finishes.  The
        # ``_setup_training`` finally block copies the live accountant
        # off the per-run context into this slot; checkpoint loads
        # restore directly into it.
        self._accountant: "Accountant | None" = None
        self._train_dataloader: DataLoader | None = None
        self._eval_dataloader: DataLoader | None = None
        self._signature_columns: list[str] | None = None
        self._signature_columns_unavailable = False

        # All cross-field validation (save/eval strategy invariants,
        # load_best_model_at_end requirements, …) lives in
        # ``TrainingArguments.__post_init__``.  The trainer reads the
        # validated ``args`` directly — no defensive snapshots.

        # Callback state.  ``state.{logging,eval,save}_steps`` are
        # resolved from ``args`` via ``state.compute_steps`` so
        # fractional values (e.g. ``logging_steps=0.5``) and ``None``
        # are handled exactly as HF does — and so callbacks running at
        # ``on_init_end`` see the post-resolution cadence rather than
        # zeros from a premature ``int(...)`` truncation.
        self.state = DPTrainerState()
        self.state.max_steps = self._predict_total_steps()
        self.state.compute_steps(args)
        self._stamp_ddp_flags(self.state)
        # Smoothed-loss bookkeeping (HF parity, trainer-internal).
        # ``_tr_loss`` accumulates per-step losses on device; ``_total_loss_scalar``
        # is the running total across all logging windows; ``_globalstep_last_logged``
        # is the step at which we last drained ``_tr_loss``.  Reset to
        # ``self.state.global_step`` on resume so the first post-resume
        # log row averages over the right window.  The tensor stays on
        # device so the DDP gather of ``tr_loss`` (Phase 10c) needs no
        # extra device migration.
        self._tr_loss = torch.tensor(0.0, device=self._device)
        self._total_loss_scalar = 0.0
        self._globalstep_last_logged: int = 0
        # Token-count bookkeeping (Phase 5c).
        # Populated during the training loop when ``include_num_input_tokens_seen``
        # is set; used by ``log()`` for live tokens/sec and by the training summary.
        self._train_start_time: float | None = None
        self._control = TrainerControl()
        # Memory tracker (HF parity: ``TrainerMemoryTracker`` handles
        # ``skip_memory_metrics``, psutil availability, CPU + GPU tracking).
        self._memory_tracker = TrainerMemoryTracker(self.args.skip_memory_metrics)
        self._memory_tracker.start()
        # Opaque step-level performance tracker: per-step wall-clock,
        # throughput, peak memory, and sub-step marks for clip / noise /
        # optimizer.  Sibling to ``_memory_tracker`` (which handles HF's
        # before/after-train memory metrics); the two surfaces are
        # complementary — HF's ``speed_metrics`` for run-level wall-clock
        # stays, the tracker adds per-step + post-warmup steady-state.
        self._perf_tracker: PerfTracker = perf_tracker(self._device)
        self._callback_handler = build_callback_handler(
            args=args,
            model=self._model,
            processing_class=processing_class,
            callbacks=self._base_callbacks,
        )
        # ``DefaultFlowCallback`` doesn't recognise ``save_strategy="best"``;
        # auto-inject the matching callback so user callbacks aren't required
        # to know about this gap.  Gated on the trainer-side snapshot so the
        # demoted-to-``"no"`` case (output_dir is None) doesn't install it.
        if self.args.save_strategy == "best":
            self._callback_handler.add_callback(BestModelSaveCallback())
        if args.debug and "underflow_overflow" in str(args.debug):
            from transformers.debug_utils import DebugUnderflowOverflow

            # Keep a strong reference: the debug object owns module hooks.
            self._debug_underflow_overflow = DebugUnderflowOverflow(self._model)

        # Warn (don't raise) when output_dir is non-empty and overwrite is off
        # — user might be intending to resume.
        self._warn_if_existing_output_dir()

        # Fire HF's ``on_init_end`` once construction is complete so callbacks
        # that pre-flight invariants (e.g. ``EarlyStoppingCallback``) can run.
        self._control = self._callback_handler.on_init_end(
            args, self.state, self._control
        )
        # HF parity: stop the init-phase memory snapshot here so training
        # (which re-starts the tracker) gets a clean baseline.
        self._memory_tracker.stop_and_update_metrics()

    # ------------------------------------------------------------------
    # Public properties (TrainerProtocol)
    # ------------------------------------------------------------------

    @property
    def model(self) -> PreTrainedModel:
        return self._model

    @model.setter
    def model(self, value: PreTrainedModel) -> None:
        self._model = value
        if hasattr(self, "_callback_handler"):
            self._callback_handler.model = value

    @property
    def processing_class(self) -> PreTrainedTokenizerBase | None:
        return self._processing_class

    @property
    def data_collator(self) -> Callable:
        return self._data_collator

    @property
    def compute_metrics(self) -> Callable | None:
        return self._compute_metrics

    @property
    def preprocess_logits_for_metrics(self) -> Callable | None:
        return self._preprocess_logits

    @property
    def label_names(self) -> list[str]:
        return self._label_names

    @property
    def callback_handler(self) -> Any:
        return self._callback_handler

    @property
    def control(self) -> TrainerControl:
        return self._control

    @control.setter
    def control(self, value: TrainerControl) -> None:
        self._control = value

    @property
    def train_dataset(self) -> Any:
        """Public alias for ``_train_dataset`` (required by ``TrainingSummary.from_trainer``)."""
        return self._train_dataset

    @property
    def eval_dataset(self) -> Any:
        """Public alias for ``_eval_dataset`` (required by ``TrainingSummary.from_trainer``)."""
        return self._eval_dataset

    # ------------------------------------------------------------------
    # Public Trainer contract helpers (Phase 9)
    # ------------------------------------------------------------------

    def add_callback(self, callback: type[TrainerCallback] | TrainerCallback) -> None:
        """Add a callback to the trainer's live callback handler.

        Also appended to ``self._base_callbacks`` so subsequent state
        resets (e.g. ``_reset_state_for_batch_size_retry``) rebuild the
        handler with this callback included.
        """
        self._callback_handler.add_callback(callback)
        self._base_callbacks.append(callback)

    def remove_callback(
        self,
        callback: type[TrainerCallback] | TrainerCallback,
    ) -> None:
        """Remove the first matching callback from the active handler."""
        self._callback_handler.remove_callback(callback)
        self._base_callbacks = [
            cb for cb in self._base_callbacks if not _callback_matches(cb, callback)
        ]

    def pop_callback(
        self,
        callback: type[TrainerCallback] | TrainerCallback,
    ) -> TrainerCallback | None:
        """Pop and return the first matching callback from the active handler."""
        popped = self._callback_handler.pop_callback(callback)
        self._base_callbacks = [
            cb for cb in self._base_callbacks if not _callback_matches(cb, callback)
        ]
        return popped

    def is_local_process_zero(self) -> bool:
        """Whether this is the local-rank-0 process (HF parity)."""
        return self._ddp.is_local_zero

    def is_world_process_zero(self) -> bool:
        """Whether this is the world-rank-0 process (HF parity)."""
        return self._ddp.is_world_zero

    def _stamp_ddp_flags(self, state: DPTrainerState) -> None:
        """Stamp per-rank ``is_*_process_zero`` flags onto ``state``.

        The flags are per-rank metadata (not part of the durable
        checkpoint contract), so any newly-constructed or
        freshly-deserialized ``DPTrainerState`` needs them set.
        """
        state.is_world_process_zero = self._ddp.is_world_zero
        state.is_local_process_zero = self._ddp.is_local_zero

    def _reset_state_for_new_run(self) -> None:
        self.state = DPTrainerState()
        self.state.max_steps = self._predict_total_steps()
        self.state.compute_steps(self.args)
        self._stamp_ddp_flags(self.state)
        self._control = TrainerControl()
        self._tr_loss = torch.tensor(0.0, device=self._device)
        self._total_loss_scalar = 0.0
        self._globalstep_last_logged = 0
        self._train_start_time = None
        self._ctx = None
        self._train_dataloader = None
        self._eval_dataloader = None

    def _apply_opaque_model_patches(self) -> None:
        """``apply_model_patches(model, compat=use_compat_patches, performance=True, kernels=use_performance_kernels)``.

        ``use_compat_patches`` (default ``True``) gates vmap-safety
        patches: ``eager_attention``, ``batchify``, vmap-safe masking /
        collator / checkpoint hooks.  ``use_performance_kernels`` (default
        ``False``) gates the CUDA + Triton kernel group (``rope``,
        ``rms_norm``, ``activation``, ``cross_entropy``).  The
        ``performance`` bucket — currently ``kv_cache`` — is always
        enabled here because ``DynamicCache`` allocation leaks vmap refs
        and inflates training memory regardless of host;
        ``performance_kernels_config={"kv_cache": False}`` opts out.

        When opaque doesn't recognise the model family it logs an
        info-level message; set ``use_compat_patches=False`` to suppress
        for custom or non-HF ``nn.Module`` fixtures.
        """
        try:
            from opaque.patches import apply_model_patches
        except ImportError:
            log.debug("opaque.patches unavailable; skipping model patches.")
            return

        kwargs = self.args.performance_kernels_config or {}
        apply_model_patches(
            self._model,
            compat=bool(self.args.use_compat_patches),
            performance=True,
            kernels=bool(self.args.use_performance_kernels),
            **kwargs,
        )

    def _setup_precision(self) -> None:
        """Resolve compute precision (TF32, bf16/fp16 autocast).

        HF parity: ``bf16=True`` and ``fp16=True`` enable autocast on the
        loss closure (forward); they do NOT cast the model.  Full-cast is
        reserved for ``bf16_full_eval`` / ``fp16_full_eval`` (eval scope
        only — see :mod:`._precision`).

        Sets:
            self._device — already resolved by caller.
            self._train_dtype — dtype the model parameters are stored in.
                Stays at whatever the caller pre-placed; autocast does
                NOT change it.
            self._amp_dtype — None | torch.bfloat16 | torch.float16.
                Driven into ``torch.autocast(device_type, dtype=self._amp_dtype)``
                inside the loss closure (Step 5) when set.
            self._loss_scaler — None | LossScaler transform from
                :mod:`opaque.precision`.  Populated for fp16 only (bf16
                has wider exponent range, no scaling needed).
            self._loss_scaler_state — None | LossScalerState.  Threaded
                through training_step; replaced wholesale each step.
        """
        a = self.args
        # ``tf32`` is a single global flag flip.  HF semantics: ``None`` =
        # leave alone; explicit ``True``/``False`` flips both flags.  No
        # restore on shutdown (matches HF).
        if a.tf32 is not None and self._device.type == "cuda":
            torch.backends.cuda.matmul.allow_tf32 = bool(a.tf32)
            torch.backends.cudnn.allow_tf32 = bool(a.tf32)

        self._train_dtype = next(self._model.parameters()).dtype

        if a.bf16:
            self._amp_dtype = torch.bfloat16
            self._loss_scaler = None
            self._loss_scaler_state = None
        elif a.fp16:
            from opaque.precision import loss_scaler

            self._amp_dtype = torch.float16
            self._loss_scaler, self._loss_scaler_state = loss_scaler()
        else:
            self._amp_dtype = None
            self._loss_scaler = None
            self._loss_scaler_state = None

    def _effective_output_dir(self) -> str | None:
        return self.args.output_dir

    # ------------------------------------------------------------------
    # train() → _setup_training() → _inner_training_loop()
    # ------------------------------------------------------------------

    def train(
        self,
        resume_from_checkpoint: str | bool | None = None,
        ignore_keys_for_eval: list[str] | None = None,
    ) -> TrainOutput:
        """Run the full DP-SGD training loop.

        Args:
            resume_from_checkpoint: ``None`` falls back to
                ``args.resume_from_checkpoint``. ``True`` auto-detects the latest
                ``checkpoint-*`` under ``args.output_dir``. A string is treated
                as the concrete checkpoint directory.

        Resume semantics under DP differ from HF's batch-replay model:

        - **Sampler resume is O(1), not batch-replay.** HF's ``Trainer``
          rebuilds the dataloader and skips ``global_step`` batches one
          by one to recover the exact data order; that's incompatible
          with our Poisson sampler whose per-iteration subsample is
          derived from ``fold_in(key, iter_count)``.  Loading the
          saved ``iter_count`` jumps the sampler directly to the right
          place.  **Privacy budget is unchanged** — every iteration
          still consumes one Poisson-amplified Gaussian step, the
          accountant composes the same number of mechanisms — but the
          *concrete batches* the resumed run sees from iteration N
          onward are the same distribution, not the same byte sequence,
          as a non-resumed run that reached iteration N organically.
          This is intentional (variance reduction, no replay cost) and
          DP-valid.
        - **``ignore_data_skip=True``** disables the sampler-state
          restore on resume.  The new run starts each epoch from
          ``iter_count=0`` with a fresh subsample sequence, again
          DP-valid; useful when the dataset shape changed since
          checkpoint write.
        - **Accountant on resume** preserves heterogeneous composition:
          the saved ``Accountant`` is loaded as the *prefix* and
          calibration of the remaining steps targets the original
          ``privacy_target_epsilon`` against that prefix.  Changing
          ``privacy_noise_multiplier`` / ``privacy_target_epsilon`` between
          checkpoint and resume warns but is allowed — the accountant
          composes whatever process the user asks for, and the warning
          guards against silent drift.
        """
        if resume_from_checkpoint is False:
            resume_from_checkpoint = None
        _disable_tokenizers_parallelism_before_fork()

        self.is_in_train = True
        try:
            return self._train_dispatch(resume_from_checkpoint, ignore_keys_for_eval)
        finally:
            self.is_in_train = False

    def _train_dispatch(
        self,
        resume_from_checkpoint: str | bool | None,
        ignore_keys_for_eval: list[str] | None,
    ) -> "TrainOutput":
        """Inner dispatch."""
        if self._train_dataset is None:
            raise ValueError("DPTrainer.train() requires a train_dataset.")
        if not self.args.auto_find_microbatch_size:
            return self._train_once(
                resume_from_checkpoint=resume_from_checkpoint,
                microbatch_size_override=None,
                ignore_keys_for_eval=ignore_keys_for_eval,
            )

        initial_microbatch_size = max(1, int(self.args.per_device_train_batch_size))
        current_microbatch_size = initial_microbatch_size
        state_snapshot = DPTrainerState.from_json(self.state.to_json())
        model_snapshot = {
            k: v.detach().to("cpu").clone() for k, v in self._model.state_dict().items()
        }
        rng_snapshot = ckpt.snapshot_rng_state()

        while True:
            try:
                return self._train_once(
                    resume_from_checkpoint=resume_from_checkpoint,
                    microbatch_size_override=current_microbatch_size,
                    ignore_keys_for_eval=ignore_keys_for_eval,
                )
            except RuntimeError as err:
                if not self._is_retryable_oom(err):
                    raise
                if current_microbatch_size <= 1:
                    raise

                next_microbatch_size = max(1, current_microbatch_size // 2)
                if next_microbatch_size == current_microbatch_size:
                    raise

                log.warning(
                    "auto_find_microbatch_size: OOM at microbatch_size=%d, retrying with microbatch_size=%d",
                    current_microbatch_size,
                    next_microbatch_size,
                )
                self._model.load_state_dict(model_snapshot, strict=False)
                ckpt.restore_rng_state(rng_snapshot)
                self._reset_state_for_batch_size_retry(state_snapshot)
                self._empty_device_cache_for_retry()
                current_microbatch_size = next_microbatch_size

    def _train_once(
        self,
        *,
        resume_from_checkpoint: str | bool | None,
        microbatch_size_override: int | None,
        ignore_keys_for_eval: list[str] | None,
    ) -> TrainOutput:
        if resume_from_checkpoint is None:
            resume_from_checkpoint = self.args.resume_from_checkpoint
        resume_path = self._resolve_resume_path(resume_from_checkpoint)

        # Pre-load weights so make_functional starts from the saved values.
        prefix_accountant: Accountant | None = None
        runtime_payload: ckpt.RuntimeCheckpoint | None = None
        trainer_state_json: dict[str, Any] | None = None
        if resume_path is not None:
            self._load_model_weights(resume_path)
            runtime_payload, prefix_accountant = self._read_runtime_for_resume(
                resume_path
            )
            trainer_state_json = self._read_trainer_state(resume_path)
            if trainer_state_json is not None:
                self.state = DPTrainerState.from_json(trainer_state_json)
                self._stamp_ddp_flags(self.state)
                # Re-bind callback handler to the new state object.
                self._callback_handler.state = self.state

        ctx = self._setup_training(
            prefix_accountant=prefix_accountant,
            global_step_already_done=(
                self.state.global_step if resume_path is not None else 0
            ),
            microbatch_size_override=microbatch_size_override,
        )
        self._ctx = ctx

        if resume_path is not None:
            if runtime_payload is not None:
                self._apply_runtime_state(
                    ctx, runtime_payload, prefix_accountant, resume_path
                )
                self._warn_on_arg_drift(runtime_payload)
            elif prefix_accountant is not None:
                # No ``dp_state.pt`` in the checkpoint.  Reaching here with
                # ``privacy_resume_without_accountant=False`` means
                # ``accountant.json`` *was* present (otherwise
                # ``_read_runtime_for_resume`` would have raised) but the DP
                # runtime state was not saved — i.e. a ``save_only_model``
                # checkpoint of a real DP run.  Continuing training from it
                # would rebuild the noise state at ``_step_counter=0`` and
                # restart the Poisson cursor, **reusing the same noise draws**
                # the original run already released on (re-sampled) data.  An
                # observer of both runs could cancel the shared noise and
                # recover a noiseless data-dependent gradient — a silent
                # privacy violation while the accountant still reports a clean
                # ε.  Refuse it: ``save_only_model`` checkpoints are
                # export-only.
                if not self.args.privacy_resume_without_accountant:
                    raise RuntimeError(
                        f"Cannot resume training from {resume_path}: the "
                        "checkpoint has no DP runtime state "
                        f"({ckpt.DP_STATE_NAME} is missing, as written by "
                        "save_only_model=True), but it does carry a privacy "
                        "accountant with prior DP cost.  Continuing training "
                        "would reuse the original run's noise stream and "
                        "silently break the privacy guarantee.  save_only_model "
                        "checkpoints are export-only; resume from a full "
                        "checkpoint (save_only_model=False).  Only if the "
                        "checkpoint genuinely has zero prior DP cost (e.g. a "
                        "public-data warmup) pass "
                        "privacy_resume_without_accountant=True to opt in."
                    )
                # Warmup opt-in path: the user asserts zero prior DP cost, so
                # there is no released DP noise to collide with — installing the
                # empty prefix accountant and starting the noise stream fresh is
                # correct.
                ctx.accounting = prefix_accountant
            self._load_rng_state(resume_path)
            self._load_callback_states()

        try:
            return self._inner_training_loop(
                ctx,
                resume_path=resume_path,
                saved_sampler_state=(
                    runtime_payload.sampler_state if runtime_payload else None
                ),
                ignore_keys_for_eval=ignore_keys_for_eval,
            )
        finally:
            self._restore_params(ctx.trainable_params)
            # Promote accountant to trainer-level so save_model() can write
            # ``accountant.json`` after train() returns.
            self._accountant = ctx.accounting
            self._ctx = None
            self._train_dataloader = None
            self._eval_dataloader = None

    def _is_retryable_oom(self, err: RuntimeError) -> bool:
        """``True`` for an OOM that justifies a microbatch retry.

        ``torch.OutOfMemoryError`` (torch >= 2.4) is the typed exception
        emitted by both CUDA and MPS backends; ``isinstance`` is robust
        across torch versions and avoids matching unrelated runtime
        errors that happen to mention "out of memory" in the message.
        """
        return isinstance(err, torch.OutOfMemoryError)

    def _reset_state_for_batch_size_retry(self, snapshot: DPTrainerState) -> None:
        self.state = DPTrainerState.from_json(snapshot.to_json())
        self._stamp_ddp_flags(self.state)
        self._callback_handler.state = self.state
        self._control = TrainerControl()
        self._ctx = None
        self._train_dataloader = None
        self._eval_dataloader = None
        self._tr_loss = torch.tensor(0.0, device=self._device)
        self._total_loss_scalar = 0.0
        self._globalstep_last_logged = int(self.state.global_step)

    def _empty_device_cache_for_retry(self) -> None:
        if self._device.type == "cuda":
            torch.cuda.empty_cache()
        elif self._device.type == "mps":
            torch.mps.empty_cache()

    def _set_resolved_privacy_args(
        self,
        *,
        target_delta: float,
        noise_multiplier: float,
        noise_multiplier_source: str,
        sample_rate: float,
        expected_batch_size: int,
        total_steps: int,
    ) -> None:
        """Expose run-resolved privacy constants for reporting callbacks."""
        self.state.privacy_resolved_delta = float(target_delta)
        self.state.privacy_resolved_noise_multiplier = float(noise_multiplier)
        self.state.privacy_noise_multiplier_source = noise_multiplier_source
        self.state.privacy_sample_rate = float(sample_rate)
        self.state.privacy_expected_batch_size = int(expected_batch_size)
        self.state.privacy_total_steps = int(total_steps)

    def _setup_training(
        self,
        *,
        prefix_accountant: "Accountant | None" = None,
        global_step_already_done: int = 0,
        microbatch_size_override: int | None = None,
    ) -> _TrainingContext:
        """Functional conversion, clipping, calibration, optimizer.

        On resume (``prefix_accountant`` is not None), calibration is performed
        over the *remaining* steps with the prefix accountant's process composed
        on the left, so the final ε reaches ``privacy_target_epsilon`` once the run
        completes.
        """
        a = self.args
        # --- Gradient checkpointing ---
        if a.gradient_checkpointing:
            gc_kwargs = a.gradient_checkpointing_kwargs or {"use_reentrant": False}
            self._model.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs=gc_kwargs,
            )
            log.info("Gradient checkpointing: enabled")

        # --- CPU offload context ---
        offload_ctx: Any = contextlib.nullcontext()
        if a.cpu_offload_activations:
            offload_ctx = torch.autograd.graph.save_on_cpu(pin_memory=True)
            log.info("CPU offload: enabled")

        # --- Functional conversion ---
        log.info("Converting model to functional form...")
        fmodel, trainable_params, frozen_params = make_functional(
            self._model,
            disable_autograd_tracking=True,
            partition_trainable=True,
        )
        log.info("Trainable parameters: %d tensors", len(trainable_params))

        # --- Discover batch shape via a one-example collator dry run ---
        # The collator's tensor-valued output keys define the per-
        # example loss surface.  Non-tensor metadata (string IDs, raw
        # text) is allowed in the batch but excluded from the vmapped
        # loss closure.
        batch_keys = self._discover_batch_keys()

        # --- Build the per-example loss closure ---
        # Default behaviour: forward through ``fmodel`` and read
        # ``output["loss"]``, which transparently inherits HF's
        # ``LOSS_MAPPING`` dispatch (causal-LM gets ``ForCausalLMLoss``,
        # classification gets ``ForSequenceClassificationLoss``, …).
        # Subclasses override :meth:`compute_per_example_loss` for
        # domain-specific losses; ``_build_per_example_loss`` here just
        # wraps it with autocast / fp16 scaling / torch.compile.
        per_example_loss_fn, batch_argnums = self._build_per_example_loss(
            fmodel,
            frozen_params,
            batch_keys,
        )

        # --- Sampling & step calculations ---
        # Logical batch (Poisson round size) drives DP accounting; physical
        # batch (per-device size) is the vmap chunk fed into clipping.
        expected_batch_size = a.train_batch_size
        microbatch_size = (
            int(microbatch_size_override)
            if microbatch_size_override is not None
            else a.per_device_train_batch_size
        )

        if self._train_dataset is None:
            raise ValueError("DPTrainer.train() requires a train_dataset.")
        dataset_size = self._effective_train_dataset_size()
        if dataset_size <= 0:
            raise ValueError(
                "DPTrainer requires a non-empty train_dataset: DP-SGD needs "
                "at least one example to build the per-example loss surface "
                "and calibrate Poisson sampling."
            )
        # Rank-local sample rate.  The user's ``expected_batch_size`` is the
        # *global* (cluster-wide) expected Poisson round size.  Under DDP we
        # shard the dataset (``opaque.distributed.local_shard``) and run the
        # Poisson sampler on each shard with the same epoch-folded key, so
        # the *local* rate equals the *global* rate by construction.  The
        # accountant uses regular ``acc.poisson`` over the global rate.
        #
        # ``dataset_size`` is the *post-trim* effective size — under DDP we
        # drop ``len(train_dataset) % world_size`` tail examples so every
        # rank ends up with an identical-length shard (avoids deadlocks for
        # fixed-order FTRL samplers).  Computing ``sample_rate`` here from
        # the trimmed denominator means the accountant calibrates noise for
        # exactly the ``q`` the sampler will use — there is no "actual q
        # vs accounted q" drift.
        sample_rate = expected_batch_size / dataset_size
        if sample_rate > 1.0:
            raise ValueError(
                "DPTrainer requires expected_batch_size <= len(train_dataset) "
                "for Poisson sampling; got expected_batch_size="
                f"{expected_batch_size} and len(train_dataset)={dataset_size}."
            )
        expected_steps_per_epoch, total_steps, num_epochs = self._steps_breakdown(
            dataset_size
        )

        self.state.max_steps = total_steps
        # HF-parity bookkeeping for ``trainer_state.json``.
        self.state.train_batch_size = int(microbatch_size)
        self.state.num_train_epochs = float(num_epochs)
        # Resolve fractional ``logging_steps`` / ``eval_steps`` /
        # ``save_steps`` to absolute step counts on ``state`` so
        # ``DefaultFlowCallback`` can read them.  Already called in
        # ``__init__`` with the same ``total_steps``, but repeated here
        # so subclasses overriding ``_predict_total_steps`` to no-op
        # still get a populated cadence by ``train()`` time.
        self.state.compute_steps(a)

        # Resolve save_steps from a fraction now that total_steps is known.
        save_steps_resolved = self._resolve_save_steps_int(total_steps)
        self.state.save_steps = save_steps_resolved

        # --- Clipping norm (scalar ``clipping_norm`` or per-group dict) ---
        mgn = a.clipping_norm
        if isinstance(mgn, dict):
            from opaque.api.engine.clipping import per_group as per_group_clipper

            fb = float(mgn["fallback"])
            patterns = {k: float(v) for k, v in mgn.items() if k != "fallback"}
            if not patterns:
                clip_norm: Any = fb
            else:
                clip_norm = per_group_clipper(
                    trainable_params,
                    fallback=fb,
                    **patterns,
                )
                log.info("Per-group clipping: %d groups", len(clip_norm.values))
        else:
            clip_norm = float(mgn)

        # --- Clipping ---
        grad_fn, clip_state = self._create_grad_fn(
            per_example_loss_fn,
            batch_argnums,
            a,
            clip_norm,
            expected_batch_size,
            microbatch_size,
        )

        # --- LR schedule ---
        # Built early so MF strategies (BandMF / BLT) can consume it for
        # workload-aware Toeplitz tuning.  The schedule only depends on
        # ``total_steps``, so moving it ahead of ``_build_mechanism`` is
        # a no-op for DP-SGD.
        lr_schedule = self.create_scheduler(num_training_steps=total_steps)

        # --- MF strategy (DP-FTRL only) ---
        mechanism_kind = a.privacy_noise_mechanism
        mf: _dpftrl.MFContext | None = None
        if mechanism_kind != "gaussian":
            mf_strategy = _dpftrl.build_strategy(
                mechanism_kind,
                (
                    a.privacy_noise_mechanism_kwargs
                    if isinstance(a.privacy_noise_mechanism_kwargs, dict)
                    else None
                ),
                lr_schedule=lr_schedule,
            )
            sk = a.sampling_kwargs if isinstance(a.sampling_kwargs, dict) else {}
            tb_raw = sk.get("truncated_batch_size", sk.get("max_batch_size"))
            mf_amplifier_factory = _dpftrl.build_amplifier_factory(
                sampling_mode=a.sampling_mode,
                strategy=mf_strategy,
                sample_rate=sample_rate,
                n_steps=total_steps,
                num_bins=expected_steps_per_epoch,
                dataset_size=dataset_size,
                truncated_batch_size=int(tb_raw) if tb_raw is not None else None,
            )
            mf = _dpftrl.MFContext(
                strategy=mf_strategy, amplifier_factory=mf_amplifier_factory
            )

        # --- Privacy calibration ---
        target_delta = (
            a.privacy_target_delta
            if a.privacy_target_delta is not None
            else 1.0 / (dataset_size**1.1)
        )
        mechanism = self._build_mechanism(
            a,
            expected_batch_size,
            sample_rate,
            clip_norm,
            dataset_size,
            mf_amplifier_factory=mf.amplifier_factory if mf is not None else None,
        )
        noise_multiplier = self._calibrate_noise(
            a,
            mechanism,
            total_steps,
            target_delta,
            prefix_accountant=prefix_accountant,
            global_step_already_done=global_step_already_done,
        )
        noise_multiplier_source = (
            "fixed" if a.privacy_noise_multiplier is not None else "calibrated"
        )
        self._set_resolved_privacy_args(
            target_delta=target_delta,
            noise_multiplier=noise_multiplier,
            noise_multiplier_source=noise_multiplier_source,
            sample_rate=sample_rate,
            expected_batch_size=expected_batch_size,
            total_steps=total_steps,
        )
        log.info(
            "Resolved privacy config: delta=%.2e, noise_multiplier=%.4f (%s), "
            "sample_rate=%.6f, total_steps=%d",
            target_delta,
            noise_multiplier,
            noise_multiplier_source,
            sample_rate,
            total_steps,
        )

        # --- Optimizer ---
        opt, opt_state = self.create_optimizer(
            trainable_params,
            lr_schedule,
            clip_state,
            noise_multiplier,
        )
        accounting = Accountant()

        # --- Noise ---
        # Sensitivity flows through the ``ClippedPytree`` returned by
        # ``clipped_grad`` (its ``.max_norm`` field).  ``noise_fn`` reads
        # that wrapper at call time and emits a ``NoisedPytree`` whose
        # ``.noise_stddev`` carries the realized σ for downstream
        # consumers (e.g. opaque optimizers' DP bias correction).
        if mechanism_kind == "gaussian":
            _gn_extra: dict[str, Any] = {}
            if isinstance(a.privacy_noise_mechanism_kwargs, dict):
                for _k, _v in a.privacy_noise_mechanism_kwargs.items():
                    if _k in ("bound", "compute_dtype"):
                        _gn_extra[_k] = _v
            make_noise = (
                functools.partial(gaussian_noise, **_gn_extra)
                if _gn_extra
                else gaussian_noise
            )
            noise_fn, noise_state = make_noise(
                noise_multiplier=noise_multiplier,
                key=key(a.seed),
            )
        else:
            # DP-FTRL: pull the participation context off the raw
            # amplifier (matches the legacy script's
            # ``_amp.n_steps`` / ``min_sep`` / ``max_participations``
            # pattern) so the streaming noise matrix tracks the
            # calibrated PLD exactly.
            assert mf is not None
            _amp = mf.amplifier_factory(noise_multiplier)
            noise_fn, noise_state = mf_gaussian_noise(
                trainable_params,
                mf.strategy,
                n_steps=int(_amp.n_steps),
                min_sep=int(_amp.min_sep),
                max_participations=int(_amp.max_participations),
                noise_multiplier=noise_multiplier,
                key=key(a.seed),
            )

        # --- Collate ---
        # Same wrapper used by the eval dataloader so train and eval
        # share key validation + device move (no asymmetric crash modes).
        collate_fn = self._resolve_collate_fn()

        return _TrainingContext(
            fmodel=fmodel,
            trainable_params=trainable_params,
            frozen_params=frozen_params,
            grad_fn=grad_fn,
            clip_state=clip_state,
            noise_fn=noise_fn,
            noise_state=noise_state,
            noise_multiplier=noise_multiplier,
            opt=opt,
            opt_state=opt_state,
            lr_schedule=lr_schedule,
            accounting=accounting,
            mechanism=mechanism,
            target_delta=target_delta,
            sample_rate=sample_rate,
            noise_multiplier_source=noise_multiplier_source,
            expected_steps_per_epoch=expected_steps_per_epoch,
            total_steps=total_steps,
            num_epochs=num_epochs,
            collate_fn=collate_fn,
            batch_keys=batch_keys,
            offload_ctx=offload_ctx,
            opt_name=(
                self._functional_optimizer_name
                if self._functional_optimizer_factory is not None
                else a.optim
            ),
            save_steps_resolved=save_steps_resolved,
            clip_norm=clip_norm,
            mechanism_kind=mechanism_kind,
            mf=mf,
        )

    def _inner_training_loop(
        self,
        ctx: _TrainingContext,
        *,
        resume_path: str | None = None,
        saved_sampler_state: dict[str, Any] | None = None,
        ignore_keys_for_eval: list[str] | None = None,
    ) -> TrainOutput:
        """Epoch/step loop with Poisson sampling."""
        a = self.args

        self._control = self._callback_handler.on_train_begin(
            self.args, self.state, self._control
        )

        log.info(
            "Starting DP-SGD training: %d epochs, ~%d steps/epoch, %d total",
            ctx.num_epochs,
            ctx.expected_steps_per_epoch,
            ctx.total_steps,
        )

        # On resume, pick up from the saved global_step/epoch.
        global_step = self.state.global_step if resume_path is not None else 0
        last_loss = 0.0
        last_step_result: dict[str, Any] = {}
        # HF parity: derive ``start_epoch`` from ``global_step``, not
        # ``state.epoch``.  ``state.epoch`` reaches an integer value at
        # epoch boundaries (e.g. step N of a 2-step epoch sets
        # ``state.epoch=1.0``); rounding it down would skip the epoch
        # we *just* finished.  ``global_step // steps_per_epoch`` is
        # what HF's ``Trainer._inner_training_loop`` uses
        # (``epochs_trained``).
        start_epoch = (
            global_step // max(1, ctx.expected_steps_per_epoch)
            if resume_path is not None
            else 0
        )
        # HF parity: anchor the smoothing window at the resume / start
        # boundary so the first post-resume log row averages over the
        # post-resume window only.
        self._globalstep_last_logged = global_step
        self._tr_loss = torch.tensor(0.0, device=self._device)
        self._total_loss_scalar = 0.0
        self._train_start_time = time.time()
        self._memory_tracker.start()
        if resume_path is not None:
            log.info(
                "Resuming from %s: epoch=%d, global_step=%d",
                resume_path,
                start_epoch,
                global_step,
            )

        # ``eval_on_start``: fires once before the inner loop runs.  HF
        # parity (``trainer.py`` ``_inner_training_loop``): the flag is
        # honored regardless of resume — users opting in want a baseline
        # eval at the *start* of each ``train()`` call, not only on
        # fresh runs.  Disable ``eval_on_start`` if you don't want it
        # on resume.
        if a.eval_on_start:
            self.evaluate(ignore_keys=ignore_keys_for_eval)

        # Build the train loader ONCE.  Under the new sampler contract,
        # a single ``PoissonSampler(n_steps=total_steps)`` drives every
        # epoch; the outer loop's role is purely callback synthesis
        # (``on_epoch_begin`` / ``on_epoch_end``) and per-epoch break
        # handling.  Resume restores the sampler's ``consumed`` cursor
        # via the opaque.serialization registry; the restored sampler
        # is installed on ``ctx.current_sampler`` *before* loader
        # construction so ``DataLoader`` binds to it (the
        # ``batch_sampler`` attribute is immutable post-init).
        if (
            resume_path is not None
            and saved_sampler_state is not None
            and not a.ignore_data_skip
        ):
            from opaque.serialization import from_state_dict

            # Need a template sampler whose ``data_source`` matches the
            # saved length so ``from_state_dict`` can validate.  Build
            # one (without caching the loader yet), then replace it
            # with the restored cursor before the actual loader binds.
            if ctx.current_sampler is None:
                self._train_dataloader = None
                self.get_train_dataloader()  # populates ctx.current_sampler
                self._train_dataloader = None  # drop the cached loader
            ctx.current_sampler = from_state_dict(
                ctx.current_sampler, saved_sampler_state
            )

        train_loader = self.get_train_dataloader()
        train_loader_iter = iter(train_loader)

        for epoch in range(start_epoch, ctx.num_epochs):
            self.state.epoch = float(epoch)
            self._control = self._callback_handler.on_epoch_begin(
                self.args, self.state, self._control
            )
            if self._control.should_training_stop:
                break

            for step_idx in range(ctx.expected_steps_per_epoch):
                try:
                    batch = next(train_loader_iter)
                except StopIteration:
                    # Sampler exhausted before this epoch's quota — happens
                    # when ``total_steps`` doesn't divide evenly into the
                    # ``num_epochs × expected_steps_per_epoch`` budget, or
                    # on resume past the recorded ``total_steps``.
                    break
                batch = self._prepare_input(batch)
                batch_size = _eval.find_batch_size(batch) or 0

                self._control = self._callback_handler.on_step_begin(
                    self.args, self.state, self._control
                )

                # Privacy accounting (data-independent, before execution)
                ctx.accounting |= ctx.mechanism(ctx.noise_multiplier)

                # Training step: clip → noise → optimize.  DP-SGD has no
                # substep concept; each iteration is a full optimizer step
                # over one Poisson-sampled logical batch.  ``on_substep_end``
                # is therefore not fired.
                step_result = self.training_step(self._model, batch)

                global_step += 1
                self.state.global_step = global_step
                # HF parity: ``state.epoch`` is fractional during the inner
                # loop — ``epoch + (step + 1) / steps_per_epoch``.  This is
                # what ``ProgressCallback``, W&B / TensorBoard timeline, and
                # any callback-driven cadence (``eval_delay``) read.
                self.state.epoch = float(epoch) + (
                    (step_idx + 1) / max(1, ctx.expected_steps_per_epoch)
                )

                if batch_size == 0:
                    # Still fire on_step_end so callbacks observe a step boundary.
                    self._control = self._callback_handler.on_step_end(
                        self.args, self.state, self._control
                    )
                    if self._control.should_training_stop:
                        break
                    continue

                last_loss = step_result["loss"]
                last_step_result = step_result
                # Loss accumulator stays on device for DDP gather.  fp16
                # overflows already short-circuit upstream (the scaler
                # returns ``batch_size=0`` which hits the empty-step
                # ``continue`` above), so any NaN reaching here reflects
                # a genuine forward / loss-math divergence — propagate
                # it through the running average so the user sees the
                # honest signal instead of a smoothed-over fake curve.
                tr_loss_step = torch.tensor(float(last_loss), device=self._device)
                self._tr_loss = self._tr_loss + tr_loss_step
                # Token counting (Phase 5c).
                if a.include_num_input_tokens_seen != "no":
                    main_input_name = getattr(
                        self._model, "main_input_name", "input_ids"
                    )
                    if main_input_name in batch:
                        if a.include_num_input_tokens_seen == "non_padding":
                            if "attention_mask" in batch:
                                n_tokens = int(batch["attention_mask"].sum().item())
                            elif (
                                self._processing_class is not None
                                and hasattr(self._processing_class, "pad_token_id")
                                and self._processing_class.pad_token_id is not None
                            ):
                                n_tokens = int(
                                    (
                                        batch[main_input_name]
                                        != self._processing_class.pad_token_id
                                    )
                                    .sum()
                                    .item()
                                )
                            else:
                                log.warning(
                                    "include_num_input_tokens_seen='non_padding': "
                                    "no attention_mask and no pad_token_id on "
                                    "processing_class — falling back to all tokens."
                                )
                                n_tokens = batch[main_input_name].numel()
                        else:  # "all"
                            n_tokens = batch[main_input_name].numel()
                        # Phase 10c: ``average_tokens_across_devices=True``
                        # (HF parity) sums the per-rank token count into the
                        # cluster-wide total so ``num_input_tokens_seen`` and
                        # the live tokens/sec rate reflect the whole DDP
                        # batch.  The flag default is True in HF; we respect
                        # whatever the user set on ``args``.
                        if self._ddp.is_distributed and getattr(
                            a, "average_tokens_across_devices", True
                        ):
                            from opaque.api.engine.distributed._state import (
                                reduce_scalar,
                            )

                            n_tokens = int(
                                reduce_scalar(
                                    float(n_tokens),
                                    op="sum",
                                    device=self._device,
                                )
                            )
                        self.state.num_input_tokens_seen += n_tokens
                # ``DefaultFlowCallback.on_step_end`` populates the
                # ``should_log/save/evaluate/training_stop`` flags from
                # ``state.{logging,eval,save}_steps``; user callbacks
                # registered after it can override.
                self._control = self._callback_handler.on_step_end(
                    self.args, self.state, self._control
                )
                self._maybe_log_save_evaluate(
                    ctx,
                    step_result,
                    global_step,
                    ignore_keys_for_eval=ignore_keys_for_eval,
                )

                if self._control.should_training_stop:
                    break
                if self._control.should_epoch_stop:
                    break
                if a.max_steps > 0 and global_step >= a.max_steps:
                    break

            # Update state.epoch first so log_history rows are tagged correctly.
            self.state.epoch = float(epoch + 1)
            # ``DefaultFlowCallback.on_epoch_end`` populates
            # ``should_evaluate`` / ``should_save`` / ``should_log`` for
            # epoch-strategy cadence; ``_maybe_log_save_evaluate`` then acts
            # on the flags exactly as the step-end path does, so the
            # epoch-boundary log / eval / save sequence shares one call site.
            self._control = self._callback_handler.on_epoch_end(
                self.args, self.state, self._control
            )
            self._maybe_log_save_evaluate(
                ctx,
                last_step_result,
                global_step,
                ignore_keys_for_eval=ignore_keys_for_eval,
            )

            if self._control.should_training_stop:
                break
            if a.max_steps > 0 and global_step >= a.max_steps:
                break

        # Final save — parity with HF: when saving is enabled, the last step always
        # produces (or refreshes) a checkpoint, even if it doesn't align with save_steps.
        self._maybe_final_save(ctx, global_step)

        # Best-model rewind happens after the final save so the checkpoint dir exists.
        if a.load_best_model_at_end:
            self._load_best_model(ctx)

        # Final metrics
        final_epsilon = ctx.accounting.epsilon_at(ctx.target_delta)
        # HF parity: add any remaining tr_loss to the total before computing avg.
        # This ensures that even if logging_steps didn't align perfectly with the
        # final step, the total training loss includes all steps.
        self._total_loss_scalar += self._tr_loss.item()
        effective_global_step = max(global_step, 0.001)  # Avoid ZeroDivisionError
        train_loss = self._total_loss_scalar / effective_global_step
        train_start = self._train_start_time or time.time()
        metrics: dict[str, Any] = speed_metrics(
            "train",
            train_start,
            num_samples=len(self._train_dataset) * ctx.num_epochs,
            num_steps=global_step,
            num_tokens=(
                self.state.num_input_tokens_seen
                if a.include_tokens_per_second
                else None
            ),
        )
        metrics.update(
            {
                "train_loss": train_loss,
                "train_steps": global_step,
                "privacy_epsilon": final_epsilon,
                "privacy_delta": ctx.target_delta,
                "privacy_noise_multiplier": ctx.noise_multiplier,
            }
        )
        # Surface fp16 overflow counter for stability auditing.  Only
        # emitted on runs where the loss-scaler was actually active
        # (avoids a noisy zero on bf16 / fp32 / no-scaler runs).
        if self._loss_scaler is not None:
            metrics["train_fp16_overflow_steps"] = self.state.fp16_overflow_steps
        if a.include_num_input_tokens_seen != "no":
            metrics["num_input_tokens_seen"] = self.state.num_input_tokens_seen
        self._memory_tracker.stop_and_update_metrics(metrics)

        log.info(
            "Training complete: %d steps, final loss=%.4f, epsilon=%.3f",
            global_step,
            train_loss,
            final_epsilon,
        )
        self.log(metrics, start_time=self._train_start_time)
        self._refresh_final_checkpoint_state(global_step)

        self._control = self._callback_handler.on_train_end(
            self.args, self.state, self._control
        )

        return TrainOutput(
            global_step=global_step,
            training_loss=train_loss,
            metrics=metrics,
        )

    # ------------------------------------------------------------------
    # training_step() — single DP-SGD step
    # ------------------------------------------------------------------

    def training_step(
        self,
        model: Any,
        inputs: dict[str, Tensor],
    ) -> dict[str, Any]:
        """One DP-SGD step: clipped grad → noise → optimizer update.

        HF-shaped signature; the ``model`` argument is accepted for
        subclass-override compatibility but the actual forward goes
        through the functional fmodel held on ``self._ctx`` (the model
        is mutated in place by the trainer's training-context wiring,
        so passing it through is redundant).
        """
        ctx = self._ctx
        if ctx is None:
            raise RuntimeError(
                "training_step called outside an active training run; "
                "DPTrainer's functional context is not initialised."
            )
        inputs = self._prepare_input(inputs)
        # Positional batch tensors in the order discovered at
        # ``_setup_training`` time.  Matches the ``batch_argnums`` the
        # loss builder published, so ``vmap`` batches correctly.
        try:
            batch_args = tuple(inputs[k] for k in ctx.batch_keys)
        except KeyError as missing:
            raise TypeError(
                f"data_collator output is missing required key {missing!s}; "
                f"expected keys {list(ctx.batch_keys)!r} (discovered at "
                f"_setup_training time from a dry run on one example)."
            ) from None
        # Tracked separately for batch-size accounting; the first
        # tensor's leading dim is what HF's ``find_batch_size`` would
        # return.  ``clipped_grad`` short-circuits on empty batches
        # internally (returning zero grads + empty aux), and the DDP
        # collectives below run unchanged on a zero-grad pytree — every
        # rank issues a SUM AllReduce on identical-zero tensors, so the
        # cluster stays in lockstep even when individual ranks see empty
        # Poisson rounds.  This matches ``examples/train_causal_lm.py``'s
        # handling: privacy budget is consumed for every step regardless
        # of realized batch size (Poisson accounting is data-independent).
        leading = batch_args[0]
        step_batch_size = int(leading.shape[0])
        # Per-step perf tracker covers clip → DDP sync → noise → optimizer;
        # post-step metric bookkeeping below stays outside the scope.
        # ``sp.mark`` records the elapsed time since the previous mark.
        with self._perf_tracker.train(batch_size=step_batch_size) as sp:
            # Clipped gradients (with optional CPU offload)
            with ctx.offload_ctx:
                (grads, aux), ctx.clip_state = ctx.grad_fn(
                    ctx.trainable_params,
                    *batch_args,
                    state=ctx.clip_state,
                )

            # Phase 10c: DDP collectives between clipping and noise.
            # 1. ``sum_gradients_`` — AllReduce SUM the clipped per-example sum;
            #    after this every rank holds the cluster-wide gradient sum.
            # 2. ``sync(clip_state, aux)`` — under fixed clipping ``clip_state``
            #    sync asserts ``clipping_norm`` matches across ranks; under
            #    adaptive clipping it aggregates the clipped-count and
            #    recomputes the norm.  ``sync(aux)`` gathers the per-example
            #    tensor fields (``grad_norms`` / ``loss_values`` / …) and
            #    weights scalar fields (``clipping_rate``) across ranks so
            #    metrics reported below reflect the cluster-wide batch.
            # Noise can then be added independently on every rank (shared key
            # ⇒ identical noise) and the optimizer update is a pure function of
            # an identical input on every rank, so parameters stay in lockstep.
            if self._ddp.is_distributed:
                from opaque.distributed import sum_gradients_, sync as _opaque_sync

                sum_gradients_(grads)
                ctx.clip_state, aux = _opaque_sync(ctx.clip_state, aux)
            sp.mark("clip")

            # fp16 dynamic-loss-scale: detect overflow on the (post-AllReduce)
            # gradient.  ``pre_clipping_transform`` ran per-example before the
            # clip-norm so the unscale is done; an inf/nan here reflects a real
            # forward/backward overflow.  Mirrors ``torch.amp.GradScaler.step``:
            # skip the optimizer update and back off the scale.  Under DDP, an
            # overflow on **any** rank must trip every rank or parameter trees
            # diverge — see ``_distributed.reduce_step_finite``.
            if self._loss_scaler is not None:
                from opaque.precision import all_finite

                # ``grads`` is a ``ClippedPytree`` wrapper, which is *not* an
                # optree node — flattening it yields a single opaque leaf, so
                # ``all_finite(grads)`` would never see the tensors (and always
                # report finite).  Inspect ``grads.pytree`` (the raw tensor
                # tree) so a real fp16 overflow is actually detected.
                grads_finite = all_finite(grads.pytree)
                grads_finite = _distributed.reduce_step_finite(grads_finite, self._ddp)
                self._loss_scaler_state = self._loss_scaler.update(
                    self._loss_scaler_state, grads_finite
                )
                if not grads_finite:
                    # Overflow → optimizer update is skipped; flow through the
                    # outer-loop's empty-step gate by returning ``batch_size=0``
                    # (same sentinel an empty Poisson sample uses).  Drops the
                    # historical ``loss=NaN`` return — propagating NaN into the
                    # log path forced ``logging_nan_inf_filter`` to substitute
                    # the running average and racing with this code; now the
                    # outer loop sees a clean "no step happened" signal.  The
                    # overflow is still surfaced via ``state.fp16_overflow_steps``
                    # for end-of-train auditing.
                    self.state.fp16_overflow_steps += 1
                    return {
                        "loss": 0.0,
                        "batch_size": 0,
                        "loss_scale": self._loss_scaler_state.scale,
                        "overflow": True,
                    }

            # Noise injection — ``grads`` is a ``ClippedPytree`` whose
            # ``.max_norm`` carries the per-step sensitivity; ``noise_fn``
            # reads it directly and returns a ``NoisedPytree``.  Adaptive
            # clipping flows through unchanged because the wrapper updates
            # ``max_norm`` per call.
            noisy_grads, ctx.noise_state = ctx.noise_fn(grads, ctx.noise_state)
            sp.mark("noise")

            # HF parity: empty device cache *after* the forward/backward pass
            # (activations are freed) but *before* the optimizer update.
            # Uses global_step (pre-increment) to match HF's cadence.
            self._maybe_empty_device_cache()

            # Pre-optimizer hook fires *after* clipping+noise but *before* the
            # optimizer update.  ``grads`` exposes the clipped-and-noised
            # gradients keyed by parameter name so callbacks (e.g. NES's
            # ``OptimizationCallback``) can compute group norms without
            # touching ``param.grad`` (which doesn't exist in the functional
            # path).
            # ``call_event`` rather than the per-hook method so we can forward
            # DP-specific kwargs (``grads``, ``trainable_params``) — HF's
            # ``CallbackHandler.on_pre_optimizer_step`` has a fixed signature.
            self._control = self._callback_handler.call_event(
                "on_pre_optimizer_step",
                self.args,
                self.state,
                self._control,
                grads=noisy_grads,
                trainable_params=ctx.trainable_params,
            )

            # Optimizer step — DP-aware optimizers read σ directly off the
            # ``NoisedPytree`` when ``noise_bias_correction=True`` was set at
            # construction (via ``optim_args``); no per-step kwargs are accepted.
            updates, ctx.opt_state = ctx.opt.update(
                noisy_grads,
                ctx.opt_state,
                params=ctx.trainable_params,
            )
            ctx.trainable_params = torchopt.apply_updates(ctx.trainable_params, updates)
            sp.mark("optimizer")

            # Post-optimizer hook: surface the post-update parameters so
            # callbacks tracking weight-update norms can snapshot them.
            self._control = self._callback_handler.call_event(
                "on_optimizer_step",
                self.args,
                self.state,
                self._control,
                trainable_params=ctx.trainable_params,
            )

        # After ``sync(aux)`` in distributed mode, ``aux.batch_size`` is
        # the cluster-wide realized batch size (sum across ranks); on a
        # single process it equals the local batch.  Use it as the truth
        # for the empty-step gate so a rank with local_bs=0 still reports
        # the cluster-wide loss when other ranks contributed examples.
        batch_size = int(getattr(aux, "batch_size", 0) or 0)
        if batch_size == 0:
            return {"loss": 0.0, "batch_size": 0}

        # Noise σ travels on the ``NoisedPytree`` wrapper now;
        # ``_effective`` handles both scalar and ``PerGroup`` shapes.
        # Adaptive mode publishes the live threshold via
        # ``clip_state.clipping_norm``; fixed mode keeps the
        # configured value on the training context (the new
        # ``FixedClipState`` carries no fields).
        noise_std = noisy_grads.noise_stddev
        clipping_norm = getattr(ctx.clip_state, "clipping_norm", ctx.clip_norm)
        metrics: dict[str, Any] = {
            "loss": aux.loss_values.mean().item(),
            "batch_size": batch_size,
            "grad_norm": aux.grad_norms.mean().item(),
            "clip_rate": aux.clipping_rate,
            "clipping_norm": _effective(clipping_norm),
            "noise_std": _effective(noise_std),
        }
        if aux.clipped_grad_norms is not None and aux.clipped_grad_norms.numel() > 0:
            metrics["clipped_grad_norm"] = aux.clipped_grad_norms.mean().item()

        if aux.group_norms is not None and hasattr(clipping_norm, "values"):
            group_noise_std = noise_std if hasattr(noise_std, "values") else None
            group_metrics: dict[str, dict[str, float]] = {}
            for group_name, group_norms in aux.group_norms.items():
                if group_norms.numel() == 0:
                    continue
                group_bound = float(clipping_norm.values[group_name])
                group_values = {
                    "grad_norm": group_norms.mean().item(),
                    "clip_rate": float((group_norms > group_bound).sum().item())
                    / max(1.0, float(batch_size)),
                    "clipping_norm": group_bound,
                }
                if group_noise_std is not None:
                    group_values["noise_std"] = float(
                        group_noise_std.values[group_name]
                    )
                group_metrics[group_name] = group_values
            if group_metrics:
                metrics["group_metrics"] = group_metrics

        return metrics

    # ------------------------------------------------------------------
    # evaluate() — functional forward, no param restoration
    # ------------------------------------------------------------------

    def compute_per_example_loss(
        self,
        fmodel: Callable[..., Any],
        params: dict[str, Tensor],
        inputs: dict[str, Tensor],
        *,
        return_logits: bool = False,
    ) -> Tensor | tuple[Tensor, Any]:
        """Compute one example's loss; vmap-batched by the caller.

        This is the unified DP-correct override hook.  The trainer wraps
        it with ``vmap`` for training (then ``grad`` → clip → noise) and
        for per-example eval (when ``'loss' in include_for_metrics``).
        Subclasses (SFT, DPO, KTO, …) override this method to compute
        domain-specific per-example losses — the same override point
        covers training and eval semantics by construction.

        Default behavior: forward through ``fmodel(params, **inputs)``
        and read ``output["loss"]`` (HF's per-model ``LOSS_MAPPING``
        dispatch is inherited automatically — causal-LM, classification,
        seq2seq, etc.).  When a ``compute_loss_func`` was supplied at
        construction it is called as
        ``compute_loss_func(outputs, labels) -> scalar`` instead — a
        no-subclass escape hatch for one-off custom losses.

        ``args.label_smoothing_factor > 0`` is honored by both the
        Opaque CE kernels (which accept ``label_smoothing`` natively —
        the trainer pushes it through as a ``loss_kwarg``) and by a
        trainer-side ``cross_entropy(..., label_smoothing=...)`` rebuild
        that overrides the model's loss whenever logits are exposed.
        The rebuild covers the no-patches case (HF's native
        ``ForCausalLMLoss`` silently drops the kwarg); for the
        opt-in ``fused_linear_cross_entropy`` path (``logits=None``),
        the kernel applies smoothing inside the fused kernel.
        Subclasses overriding this method are responsible for
        smoothing semantics themselves.

        Args:
            fmodel: Functional model from
                :func:`opaque.functional.make_functional` (called as
                ``fmodel(params, **inputs)``).
            params: All model parameters merged
                (``frozen | trainable``).  Under vmap, ``trainable`` is
                replicated per example and ``frozen`` is broadcast.
            inputs: One example's input dict (under vmap; the caller
                stripped the leading batch dim before invoking this
                method).
            return_logits: When ``True``, also return the model's
                ``logits`` tensor — used by the per-example eval path
                so a single forward yields both per-example losses and
                predictions.

        Returns:
            Scalar ``loss`` (or ``(loss, logits)`` when
            ``return_logits=True``).
        """
        smoothing = float(self.args.label_smoothing_factor)
        # Push smoothing through to the loss function as a kwarg so the
        # Opaque CE kernels (both non-fused and fused-linear) apply it
        # natively.  Vmap-safe — the value is a scalar constant, not a
        # batched tensor.  HF model forwards have ``**kwargs`` that
        # propagate to ``loss_function``; HF's native CE silently drops
        # the kwarg but the trainer-side rebuild below corrects that.
        if smoothing > 0.0:
            inputs = {**inputs, "label_smoothing": smoothing}

        output = fmodel(params, **inputs)
        # Output is required to be dict-like (``ModelOutput`` /
        # ``Mapping``).  ``prediction_step`` enforces this contract at
        # the eval boundary; the training path lands here through
        # ``_build_per_example_loss`` which calls ``fmodel`` (and HF
        # families uniformly emit ``ModelOutput``).  A non-Mapping
        # surfaces as ``AttributeError`` on ``.get`` below — louder than
        # a silent fallback would be.
        output_logits = output.get("logits")

        if self._compute_loss_func is not None:
            labels = next((inputs[k] for k in self._label_names if k in inputs), None)
            loss = self._compute_loss_func(output, labels)
        else:
            loss = output.get("loss")
        if loss is None:
            raise RuntimeError(
                "DPTrainer.compute_per_example_loss: model forward returned no "
                "`loss` field.  Pass `compute_loss_func=` for a custom loss, "
                "or override `compute_per_example_loss` in a subclass."
            )

        # Trainer-side rebuild: when logits are exposed, rewrite the
        # loss with ``F.cross_entropy(..., label_smoothing=...)`` so the
        # no-patches case (HF's native CE drops the kwarg) is also
        # honored.  Idempotent w.r.t. the kernel's smoothed loss: both
        # paths converge to the same math when ``label_smoothing > 0``.
        if smoothing > 0.0 and output_logits is not None:
            label_key = next((k for k in self._label_names if k in inputs), None)
            labels_tensor = (
                inputs.get(label_key) if label_key is not None else inputs.get("labels")
            )
            if labels_tensor is not None:
                if (
                    output_logits.ndim >= 2
                    and output_logits.shape[:-1] == labels_tensor.shape
                    and output_logits.shape[-2] > 1
                ):
                    smooth_logits = output_logits[..., :-1, :].contiguous()
                    smooth_labels = labels_tensor[..., 1:].contiguous()
                else:
                    smooth_logits = output_logits
                    smooth_labels = labels_tensor
                loss = torch.nn.functional.cross_entropy(
                    smooth_logits.view(-1, smooth_logits.size(-1)),
                    smooth_labels.view(-1),
                    ignore_index=-100,
                    label_smoothing=smoothing,
                )

        if return_logits:
            return loss, output_logits
        return loss

    def prediction_step(
        self,
        model: Any,
        inputs: dict[str, Tensor],
        prediction_loss_only: bool,
        ignore_keys: list[str] | None = None,
    ) -> tuple[Tensor | None, Tensor | tuple[Tensor, ...] | None, Any | None]:
        """Run one eval batch and return ``(loss, logits, labels)``.

        HF-shaped signature; the ``model`` argument is accepted for
        subclass-override compatibility.  The actual forward goes
        through ``self._ctx.fmodel`` mid-training (functional path) or
        ``self._model`` post-training (``nn.Module`` path) — both
        produce identically-shaped tensors so the functional/``nn.Module``
        choice is downstream-transparent.

        Two prediction shapes are possible, selected by
        ``include_for_metrics``:

        - **Standard path** (default): ``logits`` is a ``tuple`` of every
          output field surviving the ``ignore_keys + ["loss"]`` filter,
          collapsed to a bare tensor when length 1 — so multi-output
          models (seq2seq / vision) expose every auxiliary tensor to
          ``compute_metrics``.
        - **Per-example-loss path** (``"loss" in include_for_metrics``):
          the vmap'd closure returns real per-example losses plus the
          model's ``logits`` tensor *only*.  Auxiliary outputs are not
          collected on this path, so ``predictions`` is logits-only even
          for multi-output models.  A one-time warning is emitted; if you
          need full multi-output predictions, drop ``"loss"`` from
          ``include_for_metrics`` and use the standard path.

        ``inputs`` is forwarded to the model via ``**inputs`` after
        popping label tensors named in ``self._label_names`` (default:
        ``["labels"]``).  Any column the model ``forward`` accepts
        (``decoder_input_ids``, ``pixel_values``, ``token_type_ids``, …)
        is forwarded unchanged.

        When ``prediction_loss_only`` is ``True``, ``logits`` and ``labels``
        are returned as ``None`` and ``self._preprocess_logits`` is not
        called — the loop never materializes prediction tensors.

        ``ignore_keys`` filters keys out of ``ModelOutput`` containers
        before logits are extracted.  Defaults to ``[]``; the kv_cache
        patch (always-on under DPTrainer) already prevents
        ``past_key_values`` from landing in outputs, so there's nothing
        to filter by default.

        ``logits`` is a ``tuple`` of every output field *not* in
        ``ignore_keys + ["loss"]``, collapsed to a bare tensor when the
        tuple has length 1.  This exposes auxiliary outputs
        (``hidden_states``, ``attentions``,
        ``encoder_last_hidden_state``, …) to ``compute_metrics`` for
        seq2seq / encoder-decoder / vision models.

        The model ``forward`` is required to return a dict-like
        ``ModelOutput`` (or any ``Mapping`` of name → tensor).  Bare
        tuples, plain dataclasses, and bare tensors are rejected with
        :class:`TypeError`; wrap your forward to return a dict if you
        need a custom output shape.
        """
        label_keys = list(self._label_names) if self._label_names else []
        has_labels = bool(label_keys) and all(
            inputs.get(k) is not None for k in label_keys
        )

        inputs = self._prepare_input(inputs)
        # Split the batch into ``label_kwargs`` and model inputs.  Labels are
        # captured before the forward because user loss functions may pop
        # or otherwise consume them.
        model_inputs: dict[str, Any] = dict(inputs)
        labels_kwargs: dict[str, Tensor] = {}
        for label_key in label_keys:
            if label_key in model_inputs:
                labels_kwargs[label_key] = model_inputs.pop(label_key)
        if has_labels:
            label_values = tuple(labels_kwargs[k] for k in label_keys)
            labs: Any | None = _eval.nested_detach(
                label_values[0] if len(label_values) == 1 else label_values
            )
        else:
            labs = None

        # ``'loss' in include_for_metrics`` opts into real per-example
        # losses via the vmap'd eval closure — one forward pass returns
        # ``(per_example_loss_1d, logits_batched)``.  Requires labels
        # (loss-without-labels falls through to the standard path) and
        # not ``prediction_loss_only`` (the vmap path produces logits
        # anyway — there's no separate loss-only fast path).
        use_per_example_loss = (
            "loss" in (self.args.include_for_metrics or [])
            and has_labels
            and not prediction_loss_only
        )

        if use_per_example_loss:
            vmapped_fn, batch_argnums, batch_keys = self._get_eval_per_example_loss_fn()
            if self._ctx is not None:
                trainable = self._ctx.trainable_params
            else:
                trainable = {
                    name: p
                    for name, p in self._model.named_parameters()
                    if p.requires_grad
                }
            batch_args = tuple(inputs.get(k) for k in batch_keys)
            with torch.no_grad():
                was_training = self._model.training
                if was_training:
                    self._model.eval()
                try:
                    per_example_loss, logits_tensor = vmapped_fn(trainable, *batch_args)
                finally:
                    if was_training:
                        self._model.train()
            loss = per_example_loss.detach()
            if prediction_loss_only:
                return loss, None, None
            # The per-example path collects only the model's ``logits``
            # tensor (not the full ``ignore_keys``-filtered output tuple the
            # standard path builds).  Surface that once so multi-output
            # users aren't silently handed logits-only predictions, and
            # honour an explicit ``logits`` entry in ``ignore_keys``.
            if not getattr(self, "_warned_per_example_logits_only", False):
                log.warning(
                    "include_for_metrics=['loss'] uses the per-example eval "
                    "path, which returns logits-only predictions (auxiliary "
                    "model outputs are not collected).  Drop 'loss' from "
                    "include_for_metrics for the full multi-output prediction "
                    "tuple."
                )
                self._warned_per_example_logits_only = True
            preds = None if ignore_keys and "logits" in ignore_keys else (
                logits_tensor.detach() if logits_tensor is not None else None
            )
            return loss, preds, labs

        # Batched forward: reduced eval path reads ``output["loss"]``
        # directly (no per-example vmap).  This is the HF-equivalent fast
        # path; users who want ``compute_loss_func`` honoured at eval set
        # ``include_for_metrics=["loss"]`` to take the per-example path
        # above.
        with torch.no_grad():
            was_training = self._model.training
            if was_training:
                self._model.eval()
            try:
                if self._ctx is not None:
                    merged = {**self._ctx.frozen_params, **self._ctx.trainable_params}
                    output = self._ctx.fmodel(
                        merged, **{**model_inputs, **labels_kwargs}
                    )
                else:
                    output = model(**{**model_inputs, **labels_kwargs})
            finally:
                if was_training:
                    self._model.train()
            if has_labels and isinstance(output, Mapping):
                loss = output.get("loss")
                if loss is not None:
                    loss = loss.detach().mean()
            else:
                loss = None
        if prediction_loss_only:
            return loss, None, None

        if not isinstance(output, Mapping):
            raise TypeError(
                "DPTrainer requires model.forward to return a dict-like "
                "ModelOutput (or Mapping). "
                f"Got {type(output).__name__}; wrap forward to return a dict."
            )

        # Collect every output field that survives the ``ignore_keys +
        # ["loss"]`` filter into a tuple, preserving model-defined order.
        # Non-tensor values are dropped (eval pipeline consumes tensor
        # predictions only).
        ignore_keys = list(ignore_keys) if ignore_keys is not None else []
        skip = set(ignore_keys)
        if has_labels:
            skip.add("loss")
        logits_tuple: tuple[Tensor, ...] = tuple(
            v for k, v in output.items() if k not in skip and isinstance(v, Tensor)
        )

        # Collapse a length-1 tuple to a bare tensor so single-output
        # models keep the simple ``compute_metrics(EvalPrediction(
        # predictions=tensor, ...))`` contract.  Length-0 → ``None``.
        if len(logits_tuple) == 0:
            logits: Tensor | tuple[Tensor, ...] | None = None
        elif len(logits_tuple) == 1:
            logits = logits_tuple[0]
        else:
            logits = logits_tuple

        return loss, logits, labs

    def evaluation_loop(
        self,
        dataloader: DataLoader,
        *,
        description: str = "Evaluation",
        prediction_loss_only: bool | None = None,
        ignore_keys: list[str] | None = None,
        metric_key_prefix: str = "eval",
    ) -> EvaluationResult:
        """Drive a full eval pass and return an :class:`EvaluationResult`.

        The loop produces ``(loss, logits, labels)`` per batch via
        :meth:`prediction_step` and feeds them to a
        :class:`~opaque.api.transformers.trainer._eval._PredictionAccumulator`;
        ``compute_metrics`` is invoked once at the end with a single
        :class:`EvalPrediction`.

        Optional ``EvalPrediction`` fields are opt-in via
        ``args.include_for_metrics`` (the HF-canonical knob):

        - ``'inputs'`` — populates ``EvalPrediction.inputs`` with the
          model's primary input column (sniffed from
          ``model.main_input_name``; ``"input_ids"`` for text models,
          ``"pixel_values"`` for vision, etc.).
        - ``'loss'`` — populates ``EvalPrediction.losses`` with **real
          per-example losses** computed via the vmap'd eval closure
          (:meth:`prediction_step` switches to a per-example forward
          when this is requested).

        ``prediction_loss_only`` (when not ``None``) overrides
        ``args.prediction_loss_only`` — used by HF when callers want a
        loss-only pass even though metrics are configured.

        ``description`` is logged at entry alongside ``Num examples`` and
        ``Batch size`` (HF parity).  Returned ``metrics`` contain only
        model/metric outputs; ``evaluate`` and ``predict`` add throughput
        and memory metrics around this raw loop, matching HF's layering.
        """
        a = self.args
        ploss_only = (
            bool(prediction_loss_only)
            if prediction_loss_only is not None
            else bool(a.prediction_loss_only)
        )

        include_for = set(a.include_for_metrics or [])
        include_inputs = "inputs" in include_for
        include_losses = "loss" in include_for

        # HF parity (trainer.py:4863-4866 → ``EvalPrediction.inputs``):
        # ``inputs`` exposed to ``compute_metrics`` carries only the
        # model's *primary* input column, not the entire batch dict.
        main_input_name = getattr(self._model, "main_input_name", "input_ids")

        # HF-parity entry log so train-time eval, final eval, and predict
        # show up distinctly in train logs.
        try:
            num_examples = len(getattr(dataloader, "dataset", []) or [])
        except TypeError:
            num_examples = 0
        per_device_eval_bs = int(a.per_device_eval_batch_size or 0)
        log.info("***** Running %s *****", description)
        if num_examples:
            log.info("  Num examples = %d", num_examples)
        if per_device_eval_bs:
            log.info("  Batch size = %d", per_device_eval_bs)

        accumulator = _eval._PredictionAccumulator(
            prediction_loss_only=ploss_only,
            eval_accumulation_steps=a.eval_accumulation_steps,
            eval_do_concat_batches=bool(a.eval_do_concat_batches),
            include_inputs=include_inputs,
            include_losses=include_losses,
        )
        if a.eval_accumulation_steps:
            log.info(
                "Eval CPU offload engaged: flushing every %d batches",
                a.eval_accumulation_steps,
            )

        total_loss = 0.0
        loss_samples = 0
        total_samples = 0

        for batch in dataloader:
            bs = _eval.find_batch_size(batch) or 0
            if bs == 0:
                continue
            with self._perf_tracker.eval(batch_size=bs):
                loss, logits, labels = self.prediction_step(
                    self._model,
                    batch,
                    prediction_loss_only=ploss_only,
                    ignore_keys=ignore_keys,
                )

            # Per-batch progress hook (HF parity); progress bars / NES
            # callbacks rely on this firing once per eval batch.
            self._control = self._callback_handler.on_prediction_step(
                self.args,
                self.state,
                self._control,
            )

            # ``loss`` is scalar (default forward) or 1-D per-example
            # (when ``'loss' in include_for_metrics`` triggers the
            # vmap'd eval closure).  Either way, sum-into-total uses the
            # same closed form: scalar contributes ``loss * bs``; 1-D
            # contributes ``loss.sum()`` which equals the same total.
            if loss is not None:
                total_loss += (
                    float(loss.sum().item())
                    if loss.ndim > 0
                    else float(loss.item()) * bs
                )
                loss_samples += bs
            total_samples += bs

            if logits is not None and self._preprocess_logits is not None:
                logits_for_hook: Tensor | tuple[Tensor, ...]
                logits_for_hook = (
                    logits[0]
                    if isinstance(logits, tuple) and len(logits) == 1
                    else logits
                )
                logits = self._preprocess_logits(logits_for_hook, labels)

            main_input = batch.get(main_input_name) if include_inputs else None
            accumulator.add(
                loss=loss,
                logits=logits,
                labels=labels,
                inputs=main_input,
                batch_size=bs,
            )

        # ----- Finalize metrics -----
        # Phase 10c: under DDP each rank evaluated a disjoint shard of the
        # eval dataset.  Reduce the per-rank loss totals to a cluster-wide
        # mean before reporting; the prediction accumulator's tensors are
        # gathered inside ``finalize()`` below.
        if self._ddp.is_distributed:
            from opaque.api.engine.distributed._state import reduce_scalar

            total_loss = reduce_scalar(float(total_loss), op="sum", device=self._device)
            loss_samples = int(
                reduce_scalar(float(loss_samples), op="sum", device=self._device)
            )
            total_samples = int(
                reduce_scalar(float(total_samples), op="sum", device=self._device)
            )
        metrics: dict[str, Any] = {}
        if loss_samples > 0:
            metrics["loss"] = total_loss / loss_samples

        # HF parity: under DDP each rank's dataloader sees a per-rank
        # shard (``local_shard``) so ``len(dataset)`` reports per-rank
        # count, not cluster-wide.  Use the AllReduce'd ``total_samples``
        # directly so truncation after gather doesn't slice the
        # cluster-wide tensor back down to one shard.
        if self._ddp.is_distributed:
            num_samples = total_samples
        else:
            num_samples = _eval.resolve_eval_num_samples(
                dataloader,
                observed=total_samples,
            )

        predictions, label_ids, inputs_arr, losses_tensor = accumulator.finalize(
            num_samples=num_samples,
            gather=self._ddp.is_distributed,
        )

        if total_samples > 0 and (
            self._compute_metrics is not None
            and not ploss_only
            and predictions is not None
        ):
            ep = EvalPrediction(
                predictions=predictions,
                label_ids=label_ids,
                inputs=inputs_arr,
                losses=losses_tensor,
            )
            user_metrics = self._compute_metrics(ep)
            if user_metrics:
                metrics.update(user_metrics)
        # Empty-dataset path: HF silently skips ``compute_metrics``;
        # we mirror that — ``metrics`` carries only ``loss`` (absent
        # when total_samples == 0).  Caller-level evaluate/predict
        # wrappers add throughput metrics.

        # Opaque per-step performance metrics for the eval pass.
        # ``last`` is the most recent batch's StepPerf (post-warmup);
        # we surface ``step_time_sec`` / ``memory_*`` as new fields.
        # ``samples_per_second`` is emitted here too but the caller's
        # later ``speed_metrics`` update overwrites it with the
        # wall-clock aggregate — HF parity wins on the colliding key,
        # everything else is added by us.
        if self._perf_tracker.eval.last is not None:
            metrics.update(self._perf_tracker.eval.last.to_dict())

        # HF parity: scalarize numpy / tensor scalars before serialization.
        metrics = _eval.denumpify_detensorize(metrics)
        prefixed = _eval.with_metric_prefix(metrics, metric_key_prefix)

        return EvaluationResult(
            predictions=predictions,
            label_ids=label_ids,
            metrics=prefixed,
            num_samples=num_samples,
        )

    def evaluate(
        self,
        eval_dataset: Dataset | None = None,
        ignore_keys: list[str] | None = None,
        metric_key_prefix: str = "eval",
    ) -> dict[str, float]:
        """Evaluate by functional forward pass.

        During training, uses the functional model with current
        ``trainable_params``. After training (no active context),
        falls back to the ``nn.Module``.

        Installs the cached-accountant barrier on the active training
        context (when present), appends the metrics to
        ``state.log_history`` via :meth:`log`, fires the ``on_evaluate``
        callback, and updates ``state.best_metric`` / ``best_global_step``.
        Direct user calls behave identically to the eval calls the
        training loop makes.

        Returns a metrics dict with ``{prefix}_loss`` and any keys
        produced by a user-supplied ``compute_metrics`` (auto-prefixed
        with ``metric_key_prefix`` where missing — HF parity).
        """
        dataset = eval_dataset if eval_dataset is not None else self._eval_dataset
        if dataset is None:
            raise ValueError("DPTrainer.evaluate() requires an eval_dataset.")

        result = self._run_evaluation_loop(
            dataset,
            prediction_loss_only=True if self._compute_metrics is None else None,
            ignore_keys=ignore_keys,
            metric_key_prefix=metric_key_prefix,
        )
        self._after_evaluate(result.metrics)
        return result.metrics

    def predict(
        self,
        test_dataset: Dataset,
        ignore_keys: list[str] | None = None,
        metric_key_prefix: str = "test",
    ) -> EvaluationResult:
        """Run prediction loop and return predictions + labels + metrics."""
        result = self._run_evaluation_loop(
            test_dataset,
            prediction_loss_only=None,
            ignore_keys=ignore_keys,
            metric_key_prefix=metric_key_prefix,
            description="Prediction",
        )
        self._control = self._callback_handler.on_predict(
            self.args,
            self.state,
            self._control,
            metrics=result.metrics,
        )
        return result

    def log_metrics(self, split: str, metrics: dict[str, float]) -> None:
        """Log metrics in HF's public helper shape."""
        if not _distributed.should_log(self.args, self._ddp):
            return
        log.info("***** %s metrics *****", split)
        for key_name in sorted(metrics):
            log.info("  %s = %s", key_name, metrics[key_name])

    def save_metrics(
        self,
        split: str,
        metrics: dict[str, float],
        combined: bool = True,
    ) -> None:
        """Save split metrics as JSON files under the effective output directory."""
        if not _distributed.should_save(self.args, self._ddp):
            return
        output_dir = self._effective_output_dir()
        if output_dir is None:
            raise ValueError("save_metrics requires args.output_dir to be set")
        os.makedirs(output_dir, exist_ok=True)
        path = os.path.join(output_dir, f"{split}_results.json")
        with open(path, "w") as f:
            json.dump(metrics, f, indent=2, sort_keys=True, default=str)
        if combined:
            all_path = os.path.join(output_dir, "all_results.json")
            if os.path.exists(all_path):
                with open(all_path) as f:
                    all_metrics = json.load(f)
            else:
                all_metrics = {}
            all_metrics.update(metrics)
            with open(all_path, "w") as f:
                json.dump(all_metrics, f, indent=2, sort_keys=True, default=str)

    def save_state(self) -> None:
        """Save ``trainer_state.json`` under the effective output directory."""
        if not _distributed.should_save(self.args, self._ddp):
            return
        output_dir = self._effective_output_dir()
        if output_dir is None:
            raise ValueError("save_state requires args.output_dir to be set")
        os.makedirs(output_dir, exist_ok=True)
        self._save_trainer_state(output_dir)

    def _run_evaluation_loop(
        self,
        dataset: Dataset,
        *,
        prediction_loss_only: bool | None,
        ignore_keys: list[str] | None,
        metric_key_prefix: str,
        description: str = "Evaluation",
    ) -> EvaluationResult:
        """Drive a single :meth:`evaluation_loop` and return its result.

        Pure forward + accumulator; no callback / log / state-mutation
        side effects.  Shared by :meth:`evaluate` and :meth:`predict`.
        """
        # HF parity: start/stop memory tracker around the eval loop so
        # ``skip_memory_metrics=False`` captures eval-phase memory usage.
        self._memory_tracker.start()
        loader = self.get_eval_dataloader(dataset)
        self._callback_handler.eval_dataloader = loader
        start_time = time.time()
        with eval_dtype(self._model, self.args, self._train_dtype):
            result = self.evaluation_loop(
                loader,
                description=description,
                prediction_loss_only=prediction_loss_only,
                ignore_keys=ignore_keys,
                metric_key_prefix=metric_key_prefix,
            )
        total_batch_size = max(1, int(self.args.per_device_eval_batch_size))
        result.metrics.update(
            speed_metrics(
                metric_key_prefix,
                start_time,
                num_samples=result.num_samples,
                num_steps=math.ceil(result.num_samples / total_batch_size),
            )
        )
        self._memory_tracker.stop_and_update_metrics(result.metrics)
        return result

    def _after_evaluate(self, metrics: dict[str, float]) -> None:
        """Apply the post-eval side effects HF parity requires.

        - Wraps the active accountant in :func:`acc.cached` so subsequent
          ε queries reuse the PLD up to this point.
        - Appends the metrics to ``state.log_history`` via :meth:`log`.
        - Fires ``on_evaluate``.
        Best-model tracking and metric-driven LR schedules are handled by
        the training loop after train-triggered evaluation, matching HF.
        """
        ctx = self._ctx
        if ctx is not None:
            ctx.accounting = acc.cached(ctx.accounting)
        self.log(dict(metrics))
        self._control = self._callback_handler.on_evaluate(
            self.args,
            self.state,
            self._control,
            metrics=metrics,
        )

    # ------------------------------------------------------------------
    # Data loading
    # ------------------------------------------------------------------

    def _pin_memory_enabled(self) -> bool:
        """Resolve effective ``pin_memory`` for DataLoaders.

        Pinned host memory is a CUDA-only optimization (it accelerates host
        → device async copies via ``cudaMemcpyAsync``); enabling it on CPU is
        a no-op and on MPS triggers a noisy PyTorch warning on every loader
        construction.  Mirror HF's behavior: only honor the user setting when
        the trainer is bound to a CUDA device.
        """
        return bool(self.args.dataloader_pin_memory) and self._device.type == "cuda"

    def _resolve_collate_fn(self, base_collator: Callable | None = None) -> Callable:
        """Resolve the collate function for both train and eval loaders.

        Wraps the HF collator with :func:`opaque.functional.empty_collate` so
        Poisson subsampling can yield an empty index list without crashing
        collators that assume ``examples[0]`` exists.

        The collator is deliberately CPU-only so DataLoader workers never touch
        CUDA/MPS; tensors move to ``self._device`` in ``_prepare_input`` on the
        main process.
        """
        from opaque.functional import empty_collate

        base = self._data_collator if base_collator is None else base_collator
        return empty_collate(base)

    def _maybe_prime_collate(self, collate_fn: Callable, dataset: Any) -> Callable:
        """Warm ``empty_collate`` with one real example so the first Poisson
        batch can be empty without needing ``examples[0]`` for a template.

        The wrapped ``empty_collate`` is stateful — calling it once with a
        real row caches the template tensor's shape so a later empty batch
        can be synthesized.  Returns the collator unmodified (priming is a
        side effect on the closure state).

        Falls back to unprimed when probing the dataset isn't possible
        (streaming iterables have no ``__len__`` / indexing; custom
        collators may reject a single-row list).  Each ``except`` is
        narrowed to the exact failure mode it's guarding against so that
        unrelated exceptions still surface.
        """
        if dataset is None:
            return collate_fn
        try:
            n = len(dataset)  # type: ignore[arg-type]
        except TypeError:
            # Streaming / iterable dataset without ``__len__``.
            return collate_fn
        if n == 0:
            return collate_fn
        try:
            row = dataset[0]  # type: ignore[index]
        except (TypeError, IndexError, KeyError):
            # Non-subscriptable dataset, or row lookup failed.
            return collate_fn
        try:
            collate_fn([row])
        except (TypeError, ValueError, KeyError):
            # Collator rejected the single-row list shape (rare —
            # default collators accept lists of any length including 1).
            return collate_fn
        return collate_fn

    def _prepare_input(self, value: Any) -> Any:
        """Recursively move tensor inputs to the trainer device.

        Mirrors HF's main-process input preparation while keeping
        DataLoader workers CPU-only.
        """
        if isinstance(value, Tensor):
            return value.to(self._device)
        if isinstance(value, Mapping):
            return {k: self._prepare_input(v) for k, v in value.items()}
        if isinstance(value, tuple):
            return tuple(self._prepare_input(v) for v in value)
        if isinstance(value, list):
            return [self._prepare_input(v) for v in value]
        return value

    def _set_signature_columns_if_needed(self) -> None:
        if self._signature_columns is not None or self._signature_columns_unavailable:
            return
        model_to_inspect = self._model
        if self._is_peft:
            if hasattr(self._model, "get_base_model"):
                model_to_inspect = self._model.get_base_model()
            else:
                model_to_inspect = self._model.base_model.model
        signature = inspect.signature(model_to_inspect.forward)
        signature_columns = list(signature.parameters.keys())
        signature_columns += list(set(["label", "label_ids"] + self._label_names))

        # Opaque's import-time model patches may wrap ``forward`` into a generic
        # ``(*args, **kwargs)`` callable, which makes HF-style column pruning unsafe.
        label_columns = {"label", "label_ids", *self._label_names}
        semantic_columns = [
            name
            for name in signature_columns
            if name not in {"args", "kwargs", "self", *label_columns}
        ]
        if len(semantic_columns) == 0:
            self._signature_columns_unavailable = True
            self._signature_columns = []
            return

        self._signature_columns = signature_columns

    def _remove_unused_columns(
        self,
        dataset: Dataset,
        description: str | None = None,
    ) -> Dataset:
        if not self.args.remove_unused_columns:
            return dataset
        self._set_signature_columns_if_needed()
        if self._signature_columns_unavailable:
            return dataset
        signature_columns = self._signature_columns or []

        ignored_columns = [
            k for k in dataset.column_names if k not in signature_columns
        ]
        if ignored_columns:
            dset_description = (
                "" if description is None else f"in the {description} set"
            )
            log.info(
                "The following columns %s don't have a corresponding argument in "
                "`%s.forward` and have been ignored: %s. If %s are not expected by "
                "`%s.forward`, you can safely ignore this message.",
                dset_description,
                self._model.__class__.__name__,
                ", ".join(ignored_columns),
                ", ".join(ignored_columns),
                self._model.__class__.__name__,
            )

        columns = [k for k in signature_columns if k in dataset.column_names]
        if not columns:
            raise ValueError(
                "No columns in the dataset match the model's forward method signature: "
                f"({', '.join(signature_columns)}). The following columns have "
                f"been ignored: [{', '.join(ignored_columns)}]. Please check the "
                "dataset and model. You may need to set `remove_unused_columns=False` "
                "in TrainingArguments."
            )
        return dataset.remove_columns(ignored_columns)

    def _get_collator_with_removed_columns(
        self,
        data_collator: Callable,
        description: str | None = None,
    ) -> Callable:
        if not self.args.remove_unused_columns:
            return data_collator
        self._set_signature_columns_if_needed()
        if self._signature_columns_unavailable:
            return data_collator
        return RemoveColumnsCollator(
            data_collator=data_collator,
            signature_columns=self._signature_columns or [],
            logger=log,
            description=description,
            model_name=self._model.__class__.__name__,
        )

    def _prepare_dataset_and_collator(
        self,
        dataset: Any,
        *,
        description: str,
        collate_fn: Callable,
    ) -> tuple[Any, Callable]:
        if isinstance(dataset, Dataset):
            return self._remove_unused_columns(
                dataset, description=description
            ), collate_fn
        return dataset, self._get_collator_with_removed_columns(
            collate_fn,
            description=description,
        )

    def _build_per_example_loss(
        self,
        fmodel: Callable[..., Any],
        frozen_params: dict[str, Tensor],
        batch_keys: tuple[str, ...],
        *,
        return_logits: bool = False,
    ) -> tuple[Callable[..., Any], tuple[int, ...]]:
        """Wrap :meth:`compute_per_example_loss` for ``vmap(grad(...))``.

        Bridges the user-facing override hook (``compute_per_example_loss``,
        kwargs-style) to ``clipped_grad``'s positional contract:
        ``(trainable_params, *batch_args) -> scalar_loss``.  The training
        loop concerns — autocast, fp16 loss scaling, ``torch.compile`` —
        wrap around the user's per-example loss math here so subclasses
        don't have to reimplement them.

        Args:
            fmodel: Functional model from
                :func:`opaque.functional.make_functional`.
            frozen_params: Non-trainable parameters merged with
                ``trainable_params`` at every forward.
            batch_keys: Ordered tuple of tensor keys the collator emits
                (discovered via :meth:`_discover_batch_keys`).
            return_logits: When ``True``, the closure returns
                ``(loss, logits)`` instead of just ``loss``.  fp16 loss
                scaling is skipped in this mode (eval doesn't compose
                with dynamic loss scaling).

        Returns:
            ``(per_example_loss_fn, batch_argnums)``.
        """
        keys = batch_keys
        amp_dtype = self._amp_dtype
        device_type = self._device.type
        # Skip loss scaling on the eval closure: fp16 dynamic scaling is
        # a training-side mechanism for gradient underflow, irrelevant
        # to eval and incompatible with the ``return_logits`` shape.
        loss_scaler = None if return_logits else self._loss_scaler
        # autocast(device_type="cpu") only supports bf16; fp16 autocast is
        # a CUDA-only path.  We let torch raise a clear error on misuse;
        # nothing extra to validate here.
        autocast_active = amp_dtype is not None

        def per_example_loss(
            trainable: dict[str, Tensor],
            *batch_args: Tensor,
        ) -> Any:
            merged = {**frozen_params, **trainable}
            inputs = dict(zip(keys, batch_args, strict=True))
            if autocast_active:
                with torch.autocast(device_type=device_type, dtype=amp_dtype):
                    result = self.compute_per_example_loss(
                        fmodel, merged, inputs, return_logits=return_logits
                    )
            else:
                result = self.compute_per_example_loss(
                    fmodel, merged, inputs, return_logits=return_logits
                )

            if return_logits:
                loss, logits = result
            else:
                loss = result

            # fp16 dynamic-loss-scale: multiply the loss by the current
            # scale before returning to vmap(grad(...)).  The matching
            # unscale runs inside `clipped_grad`'s `pre_clipping_transform`
            # — applied per-example, before the clip-norm — so the
            # accountant's sensitivity calibration sees unscaled grads.
            if loss_scaler is not None:
                loss = loss_scaler.scale_loss(loss, self._loss_scaler_state)
            if return_logits:
                return loss, logits
            return loss

        # When `args.torch_compile=True`, compile the loss closure (NOT
        # the model — opaque's functional path goes through
        # `functional_call`, which doesn't compose with model-compile).
        # Try ``fullgraph=True`` first so graph breaks surface as errors
        # at first call (otherwise torch.compile silently fragments and
        # falls back to eager, giving the user no signal); lazily
        # downgrade to ``fullgraph=False`` with a warning if the closure
        # can't be traced fully.  Backends ``aot_eager`` / ``inductor``
        # are validated upstream; users can set
        # ``torch_compile_backend`` to anything torch.compile accepts.
        compile_args = self.args
        if compile_args.torch_compile:
            backend = compile_args.torch_compile_backend or "inductor"
            mode = compile_args.torch_compile_mode or "default"
            per_example_loss = _compile_with_fullgraph_fallback(
                per_example_loss, backend=backend, mode=mode
            )

        return per_example_loss, tuple(range(1, 1 + len(keys)))

    def _get_eval_per_example_loss_fn(
        self,
    ) -> tuple[Callable[..., Any], tuple[int, ...], tuple[str, ...]]:
        """Return a vmap'd per-example eval closure (cached) plus its batch keys.

        Used by :meth:`prediction_step` when the caller has opted into
        real per-example losses via
        ``args.include_for_metrics=['loss']``.  The closure returns
        ``(per_example_loss, logits)`` for one example; ``vmap`` over
        the batch produces a 1-D loss tensor + batched logits in a
        single forward pass.

        Cached per model identity — invalidated when ``self._model``
        rebinds (the cache key is the live ``self._model`` reference).
        During an active training run the trainer already has
        ``ctx.fmodel`` / ``ctx.frozen_params`` populated; outside
        training we run ``make_functional(self._model)`` once on first
        use and reuse the result.
        """
        if (
            self._eval_per_example_loss_fn is not None
            and self._eval_per_example_loss_fn_model is self._model
        ):
            fn, batch_argnums, batch_keys = self._eval_per_example_loss_fn
            return fn, batch_argnums, batch_keys

        ctx = self._ctx
        if ctx is not None:
            fmodel = ctx.fmodel
            frozen_params = ctx.frozen_params
            batch_keys = ctx.batch_keys
        else:
            from opaque.functional import make_functional

            fmodel, _trainable, frozen_params = make_functional(
                self._model, partition_trainable=True
            )
            batch_keys = self._discover_batch_keys()
        fn, batch_argnums = self._build_per_example_loss(
            fmodel, frozen_params, batch_keys, return_logits=True
        )
        vmapped = torch.vmap(fn, in_dims=(None,) + (0,) * len(batch_argnums))
        self._eval_per_example_loss_fn = (vmapped, batch_argnums, batch_keys)
        self._eval_per_example_loss_fn_model = self._model
        return vmapped, batch_argnums, batch_keys

    def _discover_batch_keys(self) -> tuple[str, ...]:
        """Discover the ordered tuple of tensor keys the collator emits.

        Runs the resolved collator on a single example and returns the
        keys whose values are ``torch.Tensor``.  This drives both the
        ``batch_argnums`` returned by :meth:`_build_per_example_loss`
        and the positional batch-arg ordering in :meth:`training_step`.
        Non-tensor metadata is excluded — the per-example loss closure
        operates only on tensors.
        """
        if len(self._train_dataset) == 0:
            raise ValueError("Cannot discover batch keys: train_dataset is empty.")
        prepared_dataset, prepared_collator = self._prepare_dataset_and_collator(
            self._train_dataset,
            description="training",
            collate_fn=self._data_collator,
        )
        sample = prepared_collator([prepared_dataset[0]])
        if not isinstance(sample, Mapping):
            raise TypeError(
                f"data_collator must return a mapping; got {type(sample).__name__}."
            )
        keys = tuple(k for k, v in sample.items() if isinstance(v, Tensor))
        if not keys:
            raise TypeError(
                "data_collator produced no tensor outputs on a one-example dry "
                f"run; got keys={list(sample.keys())!r}.  The DP path requires "
                "at least one batched tensor."
            )
        return keys

    def get_train_dataloader(self) -> DataLoader:
        """Train DataLoader.

        During training (``self._ctx`` populated) returns a
        single-pass :class:`PoissonSampler`-backed DataLoader bounded
        by ``ctx.total_steps`` (one sampler instance for the whole
        run; the outer epoch loop just synthesises boundaries for the
        HF callback surface).  Outside training (``ctx is None``,
        inspection mode) returns a standard DataLoader.

        Override in a subclass to plug in a custom sampler — DP
        correctness depends on the sampler producing each example with
        independent probability ``ctx.sample_rate``, so any override
        must preserve that invariant.
        """
        a = self.args
        ctx = self._ctx
        worker_init = self._dataloader_worker_init_fn()
        if ctx is None:
            dataset, base_collator = self._prepare_dataset_and_collator(
                self._train_dataset,
                description="training",
                collate_fn=self._data_collator,
            )
            return DataLoader(
                dataset,
                batch_size=a.per_device_train_batch_size,
                shuffle=False,
                collate_fn=self._maybe_prime_collate(
                    self._resolve_collate_fn(base_collator), dataset
                ),
                num_workers=a.dataloader_num_workers,
                pin_memory=self._pin_memory_enabled(),
                worker_init_fn=worker_init,
            )
        if self._train_dataloader is not None:
            self._callback_handler.train_dataloader = self._train_dataloader
            return self._train_dataloader

        dataset, base_collator = self._prepare_dataset_and_collator(
            self._train_dataset,
            description="training",
            collate_fn=self._data_collator,
        )
        collate_fn = self._resolve_collate_fn(base_collator)
        collate_fn = self._maybe_prime_collate(collate_fn, dataset)

        # Under DDP each rank operates on a disjoint shard of the dataset
        # (``opaque.distributed.local_shard``); the Poisson sampler runs
        # with the same key on every rank so the union of per-rank batches
        # is a single global Poisson draw at the user's
        # ``expected_batch_size`` rate.  ``ctx.sample_rate`` was computed in
        # ``_setup_training`` from the same trimmed denominator we use here
        # (see :meth:`_effective_train_dataset_size`), so the rate the
        # sampler is configured with matches the rate the accountant
        # calibrated against — both bind to the post-trim ``q``.
        if self._ddp.world_size > 1:
            from torch.utils.data import Subset

            from opaque.distributed import local_shard

            world_size = self._ddp.world_size
            trim_to = self._effective_train_dataset_size()
            if trim_to < len(dataset):
                dataset = Subset(dataset, range(trim_to))
            dataset = local_shard(
                dataset,
                rank=self._ddp.rank,
                world_size=world_size,
            )

        # ONE sampler for the whole run.  Resume installs
        # ``ctx.current_sampler`` from a registry-deserialised snapshot
        # before calling here, so the loader picks up the right cursor;
        # otherwise build a fresh sampler bound to the resolved
        # ``sampling_mode`` (one of the five supported modes —
        # ``poisson``, ``b_min_sep``, ``balls_in_bins``,
        # ``cyclic_poisson``, ``sequential``) which iterates end-to-end
        # without per-epoch re-instantiation.  The outer epoch loop is
        # purely a synthetic boundary layer for HF callbacks, not a
        # sampling-side concept.
        if ctx.current_sampler is None:
            from opaque.random import key

            sampler_key = key(a.data_seed if a.data_seed is not None else a.seed)
            ctx.current_sampler = _dpftrl.build_sampler(
                sampling_mode=a.sampling_mode,
                dataset=dataset,
                sample_rate=ctx.sample_rate,
                n_steps=ctx.total_steps,
                key=sampler_key,
                sampling_kwargs=(
                    a.sampling_kwargs if isinstance(a.sampling_kwargs, dict) else None
                ),
                mf=ctx.mf,
                noise_multiplier=ctx.noise_multiplier,
                num_bins=ctx.expected_steps_per_epoch,
                expected_batch_size=int(a.train_batch_size),
            )
        sampler = ctx.current_sampler

        # HF parity: MPS requires fork start method when using multiple workers
        # (PyTorch's default spawn method does not work with MPS).
        should_fork = self._device.type == "mps" and a.dataloader_num_workers > 1
        kwargs: dict[str, Any] = {
            "batch_sampler": sampler,
            "collate_fn": collate_fn,
            "num_workers": a.dataloader_num_workers,
            "pin_memory": self._pin_memory_enabled(),
            "worker_init_fn": worker_init,
            "multiprocessing_context": "fork" if should_fork else None,
        }
        if a.dataloader_num_workers > 0:
            kwargs["persistent_workers"] = a.dataloader_persistent_workers
            if a.dataloader_prefetch_factor is not None:
                kwargs["prefetch_factor"] = a.dataloader_prefetch_factor

        self._train_dataloader = DataLoader(dataset, **kwargs)
        self._callback_handler.train_dataloader = self._train_dataloader
        return self._train_dataloader

    def get_eval_dataloader(self, eval_dataset: Dataset | None = None) -> DataLoader:
        """Standard DataLoader for evaluation.

        Phase 10c: under DDP, the eval dataset is sharded into a contiguous
        per-rank slice via ``opaque.distributed.local_shard`` so each rank
        evaluates a disjoint subset.  Per-batch losses / predictions are
        reduced / gathered in :meth:`evaluation_loop` before metrics are
        computed.
        """
        raw_dataset = eval_dataset if eval_dataset is not None else self._eval_dataset
        if eval_dataset is None and self._eval_dataloader is not None:
            return self._eval_dataloader

        dataset, base_collator = self._prepare_dataset_and_collator(
            raw_dataset,
            description="evaluation",
            collate_fn=self._data_collator,
        )

        if self._ddp.world_size > 1:
            from opaque.distributed import local_shard

            dataset = local_shard(
                dataset,
                rank=self._ddp.rank,
                world_size=self._ddp.world_size,
            )

        # HF parity: MPS requires fork start method when using multiple workers.
        should_fork = (
            self._device.type == "mps" and self.args.dataloader_num_workers > 1
        )
        eval_collate = self._resolve_collate_fn(base_collator)
        eval_collate = self._maybe_prime_collate(eval_collate, dataset)
        kwargs: dict[str, Any] = {
            "batch_size": self.args.per_device_eval_batch_size,
            "shuffle": False,
            "collate_fn": eval_collate,
            "num_workers": self.args.dataloader_num_workers,
            "pin_memory": self._pin_memory_enabled(),
            "drop_last": self.args.dataloader_drop_last,
            "worker_init_fn": self._dataloader_worker_init_fn(),
            "multiprocessing_context": "fork" if should_fork else None,
        }
        if self.args.dataloader_num_workers > 0:
            kwargs["persistent_workers"] = self.args.dataloader_persistent_workers
            if self.args.dataloader_prefetch_factor is not None:
                kwargs["prefetch_factor"] = self.args.dataloader_prefetch_factor

        loader = DataLoader(dataset, **kwargs)
        if eval_dataset is None and self.args.dataloader_persistent_workers:
            self._eval_dataloader = loader
        return loader

    def _dataloader_worker_init_fn(self) -> Callable[[int], None] | None:
        """Build a worker-init callable that re-seeds each worker process.

        HF parity (``Trainer._get_dataloader``): we forward each worker
        a deterministic seed derived from ``args.seed`` so that
        non-DP randomness inside a custom ``data_collator`` (or any
        torch / numpy / random call inside ``__getitem__``) is
        reproducible run-to-run.  Returns ``None`` when single-process
        loading is in use — :class:`torch.utils.data.DataLoader`
        ignores ``worker_init_fn`` in that case anyway, but spelling
        ``None`` keeps the spawn path explicit.

        DP correctness is unaffected: the per-step DP RNG chain
        (``key(args.seed)`` folded with ``state.epoch`` /
        ``iter_count``) is independent of ``torch`` / NumPy / Python's
        global generators.
        """
        a = self.args
        if a.dataloader_num_workers <= 0:
            return None
        # ``seed_worker(worker_id, num_workers, rank)`` consumes
        # ``torch.initial_seed()`` inside the worker process — which
        # itself is set by PyTorch from the base RNG seeded via
        # ``set_seed(args.seed)`` at trainer construction.  ``rank=0``
        # is correct for the current single-process path (Phase 9
        # extends this for DDP).
        return functools.partial(
            seed_worker,
            num_workers=a.dataloader_num_workers,
            rank=self._ddp.rank,
        )

    def _maybe_empty_device_cache(self) -> None:
        steps = self.args.torch_empty_cache_steps
        if steps is None or steps <= 0:
            return
        if self.state.global_step % steps != 0:
            return
        device_type = self._device.type
        if device_type == "cuda":
            torch.cuda.empty_cache()
        elif device_type == "mps":
            torch.mps.empty_cache()
        # cpu: no-op (nothing to evict)

    # ------------------------------------------------------------------
    # Optimizer
    # ------------------------------------------------------------------

    def create_optimizer(
        self,
        trainable_params: dict[str, Tensor],
        lr_schedule: Callable[[int], float],
        clip_state: Any = None,
        noise_multiplier: float = 0.0,
    ) -> tuple[Any, Any]:
        """Create functional optimizer and initial state.

        Dispatches via :func:`opaque.api.transformers.trainer._optim.build_optimizer`,
        which resolves ``args.optim`` (canonical opaque name or HF alias)
        and forwards HF-canonical fields (``learning_rate``,
        ``weight_decay``, ``adam_beta1``/``adam_beta2``, ``adam_epsilon``)
        to the underlying opaque factory.  Opaque-only knobs (for example
        ``noise_bias_correction``, ``decoupled_weight_decay``,
        ``update_rms_clip``) are supplied through ``args.optim_args``,
        which ``TrainingArguments.__post_init__`` has already normalized
        from any of {Mapping, JSON object string, HF "key=value,..."
        string, None} to ``dict[str, Any] | None``.  Opaque factories
        raise ``TypeError`` on unknown keys, so typos surface immediately.

        ``clip_state`` and ``noise_multiplier`` are accepted for
        signature stability with subclasses; the wrapper-pytree noise
        flow makes them irrelevant at construction time (sensitivity
        flows through ``ClippedPytree.max_norm`` at every step).
        """
        from ._optim import build_optimizer

        del clip_state, noise_multiplier  # see docstring
        a = self.args
        extra = dict(a.optim_args or {})
        fac = self._functional_optimizer_factory
        if fac is not None:
            factory, init_kw = fac
            merged = {**dict(init_kw), **extra}
            merged.pop("lr", None)
            opt = factory(lr=lr_schedule, **merged)
            return opt, opt.init(trainable_params)
        opt = build_optimizer(a, lr_schedule, extra_kwargs=extra)
        return opt, opt.init(trainable_params)

    def create_scheduler(self, num_training_steps: int) -> Callable[[int], float]:
        """Build the LR schedule for the run.

        Dispatches via :func:`opaque.api.transformers.trainer._scheduler.build_lr_schedule`,
        which reads ``args.lr_scheduler``, ``args.lr_scheduler_kwargs``,
        and ``args.warmup_steps`` / ``args.warmup_ratio``.  Override in a
        subclass to supply a different schedule.
        """
        return build_lr_schedule(self.args, num_training_steps)

    # ------------------------------------------------------------------
    # Logging
    # ------------------------------------------------------------------

    def log(self, logs: dict[str, Any], start_time: float | None = None) -> None:
        """Record metrics and fire ``on_log`` callback.

        HF parity:
        - ``start_time`` is the training-loop start timestamp; when
          provided and ``include_num_input_tokens_seen`` is set, the
          live tokens/sec rate is injected into the log row (mirrors
          HF 5.1's ``Trainer.log`` signature).
        - ``num_input_tokens_seen`` is appended to every log row when
          tracking is enabled — not just training-step rows.
        """
        if not _distributed.should_log(self.args, self._ddp):
            return
        if self.state.epoch is not None:
            logs["epoch"] = self.state.epoch
        if self.args.include_num_input_tokens_seen != "no":
            logs["num_input_tokens_seen"] = self.state.num_input_tokens_seen
            if start_time is not None:
                logs.update(
                    speed_metrics(
                        "train",
                        start_time,
                        num_tokens=self.state.num_input_tokens_seen,
                    )
                )
        output = {**logs, "step": self.state.global_step}
        self.state.log_history.append(output)
        self._control = self._callback_handler.on_log(
            self.args, self.state, self._control, logs=output
        )

    def _maybe_log_save_evaluate(
        self,
        ctx: _TrainingContext,
        step_result: dict[str, Any],
        global_step: int,
        *,
        ignore_keys_for_eval: list[str] | None = None,
    ) -> None:
        """Act on ``control.should_log/evaluate/save`` populated by
        :class:`DefaultFlowCallback.on_step_end` (and possibly overridden
        by user callbacks)."""
        ctrl = self._control

        if ctrl.should_log:
            epsilon = ctx.accounting.epsilon_at(ctx.target_delta)
            # HF parity: ``loss`` is the *average* per-step loss across the
            # window since the last log boundary, not the per-step
            # instantaneous value.  Smooths out per-step variance that
            # would otherwise dominate the displayed curve.
            window = max(1, global_step - self._globalstep_last_logged)
            tr_loss_scalar = self._tr_loss.item()
            smoothed_loss = tr_loss_scalar / window
            # HF parity: accumulate into _total_loss_scalar, then reset tr_loss.
            self._total_loss_scalar += tr_loss_scalar
            self._tr_loss -= self._tr_loss  # zero in place
            self._globalstep_last_logged = global_step
            # HF parity: log the LR that was just applied to the optimizer
            # update we performed for ``global_step``.  Inside torchopt the
            # schedule's step counter is incremented on every ``update`` so
            # the LR consumed by the update at iteration N was
            # ``schedule(N - 1)``; ``global_step`` has already been bumped
            # to N by the time we log here.
            #
            # ``ctrl.should_log`` is only set after at least one optimizer
            # update has fired (the only setter is
            # :class:`DefaultFlowCallback.on_step_end`, which runs *after*
            # ``global_step`` is bumped to ``>=1``).  Assert the invariant
            # here so a future caller that flips ``should_log`` from
            # ``on_train_begin`` (where ``global_step == 0``) gets a
            # loud failure instead of a silent ``schedule(-1)`` lookup.
            assert global_step >= 1, (
                f"_maybe_log_save_evaluate fired with global_step="
                f"{global_step}; expected >=1 (DefaultFlowCallback only "
                f"sets should_log post-step)."
            )
            logs = {
                "loss": smoothed_loss,
                "batch_size": step_result.get("batch_size", 0),
                "grad_norm": step_result.get("grad_norm", 0.0),
                "learning_rate": ctx.lr_schedule(global_step - 1),
                "privacy_epsilon": epsilon,
                "privacy_delta": ctx.target_delta,
                "privacy_clip_rate": step_result.get("clip_rate", 0.0),
                "privacy_clipping_norm": step_result.get("clipping_norm", 0.0),
                "privacy_noise_std": step_result.get("noise_std", 0.0),
                "privacy_noise_multiplier": ctx.noise_multiplier,
            }
            if "clipped_grad_norm" in step_result:
                logs["privacy_clipped_grad_norm"] = step_result["clipped_grad_norm"]
            for group_name, group_values in step_result.get(
                "group_metrics", {}
            ).items():
                for metric_name, value in group_values.items():
                    logs[f"privacy_group_{group_name}_{metric_name}"] = value
            # Opaque per-step performance metrics (step_time_sec,
            # samples_per_second, memory_*, clip_sec / noise_sec /
            # optimizer_sec from ``sp.mark(...)``).  Bare keys; the
            # reporting-callback rewriter wraps them under ``train/``.
            if self._perf_tracker.train.last is not None:
                logs.update(self._perf_tracker.train.last.to_dict())
            self.log(logs, start_time=self._train_start_time)
            ctrl = self._control
            ctrl.should_log = False

        if ctrl.should_evaluate:
            metrics = self.evaluate(ignore_keys=ignore_keys_for_eval)
            # ``evaluate()`` fires ``on_evaluate`` from inside
            # ``_after_evaluate``; :class:`BestModelSaveCallback` (auto-
            # injected when ``save_strategy="best"``) may have set
            # ``should_save`` there.  Refresh the local handle accordingly.
            ctrl = self._control
            self._update_best_metric(metrics, global_step)
            ctrl.should_evaluate = False

        if ctrl.should_save:
            self._save_checkpoint(ctx, global_step)
            ctrl.should_save = False

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _create_grad_fn(
        self,
        loss_fn: Callable[..., Any],
        batch_argnums: tuple[int, ...],
        a: TrainingArguments,
        clip_norm: Any,
        expected_batch_size: int,
        microbatch_size: int,
    ) -> Callable[..., Any]:
        """Create the clipped gradient function based on clipping mode.

        When fp16 autocast is active, the loss closure scales the loss by
        ``self._loss_scaler_state.scale``; the matching unscale runs as
        ``pre_clipping_transform`` *inside* vmap, *before* the clip-norm.
        This preserves the DP sensitivity invariant the accountant relies on.
        """
        if self._loss_scaler is not None:
            scaler = self._loss_scaler

            def pre_clip(g):
                return scaler.unscale_grads(g, self._loss_scaler_state)
        else:

            def pre_clip(g):
                return g

        ca = a.clipping_kwargs
        target_clip_rate = float(ca.get("target_clipping_rate", 0.5))
        clip_norm_max = float(ca.get("norm_max", 10.0))
        auto_gamma = float(ca.get("gamma", 0.01))

        if a.clipping_mode == "adaptive":
            return adaptive_clipped_grad(
                loss_fn,
                argnums=0,
                batch_argnums=batch_argnums,
                initial_clipping_norm=clip_norm,
                target_quantile=1.0 - target_clip_rate,
                clipping_norm_max=clip_norm_max,
                microbatch_size=microbatch_size,
                return_aux=True,
                key=key(a.seed),
                normalize_by=expected_batch_size,
                pre_clipping_transform=pre_clip,
            )
        elif a.clipping_mode == "auto":
            return auto_clipped_grad(
                loss_fn,
                argnums=0,
                batch_argnums=batch_argnums,
                R=clip_norm,
                gamma=auto_gamma,
                normalize_by=expected_batch_size,
                microbatch_size=microbatch_size,
                return_aux=True,
                pre_clipping_transform=pre_clip,
            )
        else:
            return clipped_grad(
                loss_fn,
                argnums=0,
                batch_argnums=batch_argnums,
                clipping_norm=clip_norm,
                normalize_by=expected_batch_size,
                microbatch_size=microbatch_size,
                return_aux=True,
                pre_clipping_transform=pre_clip,
            )

    def _build_mechanism(
        self,
        a: TrainingArguments,
        expected_batch_size: int,
        sample_rate: float,
        clip_norm: Any,
        dataset_size: int,
        *,
        mf_amplifier_factory: Callable[[float], Any] | None = None,
    ) -> Callable[..., Any]:
        """Build the privacy accounting mechanism chain.

        DP-SGD branch (``privacy_noise_mechanism == "gaussian"``): the
        Poisson amplification covers both plain Poisson sampling and the
        truncated variant; ``dpsgd_acc.poisson`` dispatches internally
        when ``truncated_batch_size`` / ``dataset_size`` are supplied.

        DP-FTRL branch (``mf_*`` mechanism): wraps the supplied raw
        amplifier factory (built in :meth:`_setup_training`) with
        :func:`opaque.dpftrl.accounting.per_step` so each call returns a
        per-step composable :class:`DpProcess` that materialises as the
        true K-step PLD of the deployed N-step mechanism under
        ``acc |= step`` accumulation.
        """
        if a.privacy_noise_mechanism != "gaussian":
            if mf_amplifier_factory is None:
                raise RuntimeError(
                    "_build_mechanism reached the DP-FTRL branch without an "
                    "amplifier factory; _setup_training should populate it."
                )
            return _dpftrl.build_step_mechanism_factory(mf_amplifier_factory)

        num_groups = len(clip_norm.values) if hasattr(clip_norm, "values") else 1

        base = dpsgd_acc.gaussian
        if a.clipping_mode == "adaptive":
            _base = base

            def base(nm, _b=_base):
                return dpsgd_acc.adaclip(
                    _b(nm),
                    expected_batch_size=expected_batch_size,
                    num_groups=num_groups,
                )

        _unamplified = base
        sk = a.sampling_kwargs if isinstance(a.sampling_kwargs, dict) else {}
        tb_raw = sk.get("truncated_batch_size", sk.get("max_batch_size"))
        tb_cap = int(tb_raw) if tb_raw is not None else None

        if tb_cap is not None:

            def mechanism(
                nm,
                _u=_unamplified,
                _cap=tb_cap,
                _n=dataset_size,
                _rate=sample_rate,
            ):
                return dpsgd_acc.poisson(
                    _u(nm),
                    _rate,
                    truncated_batch_size=_cap,
                    dataset_size=_n,
                )
        else:

            def mechanism(nm, _u=_unamplified, _rate=sample_rate):
                return dpsgd_acc.poisson(_u(nm), sample_rate=_rate)

        return mechanism

    def _calibrate_noise(
        self,
        a,
        mechanism,
        total_steps,
        target_delta,
        *,
        prefix_accountant: "Accountant | None" = None,
        global_step_already_done: int = 0,
    ):
        """Calibrate or return fixed noise multiplier.

        When ``prefix_accountant`` is given, calibrates over the *remaining*
        steps with the saved process composed on the left, so the run's final ε
        equals ``privacy_target_epsilon``.
        """
        if a.privacy_noise_multiplier is not None:
            log.info("Using fixed noise multiplier: %.4f", a.privacy_noise_multiplier)
            return a.privacy_noise_multiplier

        if prefix_accountant is None or global_step_already_done == 0:
            log.info(
                "Calibrating privacy (target eps=%.2f, delta=%.2e)...",
                a.privacy_target_epsilon,
                target_delta,
            )

            def objective(nm, _mechanism=mechanism, _steps=total_steps):
                return _mechanism(nm) * _steps
        else:
            remaining_steps = max(1, total_steps - global_step_already_done)
            log.info(
                "Recalibrating remaining %d/%d steps over saved accountant "
                "(target eps=%.2f, delta=%.2e)...",
                remaining_steps,
                total_steps,
                a.privacy_target_epsilon,
                target_delta,
            )
            prefix_process = prefix_accountant.process

            def objective(nm, _prefix=prefix_process, _rem=remaining_steps):
                return _prefix | (mechanism(nm) * _rem)

        ecal = a.noise_calibration_kwargs
        result = cal.calibrate(
            cal.epsilon_budget(a.privacy_target_epsilon, delta=target_delta),
            objective,
            param_min=float(ecal["min"]),
            param_max=float(ecal["max"]),
            tolerance=float(ecal["tolerance"]),
        )
        log.info(
            "Calibrated: noise_multiplier=%.4f, achieved eps=%.3f (converged=%s)",
            result.param,
            result.achieved,
            result.converged,
        )
        return result.param

    def _restore_params(self, trainable_params: dict[str, Tensor]) -> None:
        """Load trained parameters back into the nn.Module.

        Validates that ``trainable_params`` keys exactly match the
        model's current set of ``requires_grad=True`` parameters before
        loading; any divergence raises ``RuntimeError``.  This catches
        typo'd parameter names in subclass overrides (which
        ``strict=False`` would silently ignore, leaving the parameter
        at its initial value) and surfaces mid-run requires_grad
        churn (a callback freezing / unfreezing layers between snapshot
        and restore — the snapshot no longer matches the model).

        After validation the merged state dict (model's current state
        with trainable entries overwritten) is loaded with
        ``strict=True``; the state dict is sourced from the model
        itself so every key the model expects is already present.
        """
        expected = {
            name for name, p in self._model.named_parameters() if p.requires_grad
        }
        got = set(trainable_params)
        if got != expected:
            missing = expected - got
            unexpected = got - expected
            details = []
            if missing:
                details.append(f"missing from trainable_params: {sorted(missing)}")
            if unexpected:
                details.append(f"not in model.named_parameters(): {sorted(unexpected)}")
            raise RuntimeError(
                "DPTrainer._restore_params: trainable_params keys do not match "
                "the model's current requires_grad set. " + "; ".join(details)
            )
        state_dict = self._model.state_dict()
        for name, tensor in trainable_params.items():
            state_dict[name] = tensor.detach()
        self._model.load_state_dict(state_dict, strict=True)

    # ------------------------------------------------------------------
    # Save / checkpoint  (Phase 2a)
    # ------------------------------------------------------------------

    def _effective_train_dataset_size(self) -> int:
        """Length of ``self._train_dataset`` after the DDP equal-shard trim.

        Single source of truth for the training-time dataset size: under DDP
        the trainer drops ``len(train_dataset) % world_size`` tail examples
        before sharding so every rank ends up with an identical-length local
        shard (avoids batch-count desynchronisation under fixed-order
        samplers).  Callers that drive privacy accounting and the Poisson
        sampler must agree on which denominator they're using; routing both
        through this helper guarantees that.

        Raises:
            ValueError: If ``len(train_dataset) < world_size``, which would
                trim the whole dataset away.
        """
        if self._train_dataset is None:
            return 0
        n = len(self._train_dataset)
        world_size = self._ddp.world_size
        if world_size <= 1:
            return n
        trimmed = (n // world_size) * world_size
        if trimmed < 1:
            raise ValueError(
                f"Train dataset has {n} example(s), fewer than "
                f"world_size={world_size}; every rank requires at least one "
                "example after sharding."
            )
        return trimmed

    def _steps_breakdown(
        self,
        dataset_size: int,
    ) -> tuple[int, int, int]:
        """Compute ``(steps_per_epoch, total_steps, num_epochs)``.

        Single source of truth for the step math; called from
        ``__init__`` (to populate ``state.max_steps`` /
        ``state.{logging,eval,save}_steps`` before ``on_init_end``)
        and again from ``_setup_training`` (where ``state`` is fully
        repopulated for the actual run).

        Subclasses that resize the train dataset between construction
        and ``train()`` may override ``_predict_total_steps`` to defer
        cadence resolution.
        """
        a = self.args
        sample_rate = a.train_batch_size / max(1, dataset_size)
        steps_per_epoch = math.ceil(1.0 / sample_rate)
        if a.max_steps > 0:
            total = a.max_steps
            num_epochs = math.ceil(total / max(1, steps_per_epoch))
        else:
            num_epochs = int(a.num_train_epochs)
            total = num_epochs * steps_per_epoch
        return steps_per_epoch, total, num_epochs

    def _predict_total_steps(self) -> int:
        """Predict ``total_steps`` for ``state.compute_steps`` at init time.

        Returns the same value as the ``total_steps`` field of
        ``_steps_breakdown(self._effective_train_dataset_size())`` — the
        post-trim denominator the actual training run will see, so
        ``state.max_steps`` matches the cadence ``_setup_training`` produces.
        Override in subclasses where the dataset isn't sized at construction
        time (e.g. streaming datasets) — return ``0`` to signal "unknown" and
        ``state.compute_steps`` will leave ``state.{logging,eval,save}_steps``
        at HF defaults until ``_setup_training`` recomputes.
        """
        try:
            n = self._effective_train_dataset_size()
        except TypeError:
            return 0
        return self._steps_breakdown(n)[1]

    def _warn_if_existing_output_dir(self) -> None:
        a = self.args
        output_dir = self._effective_output_dir()
        if output_dir is None or a.overwrite_output_dir:
            return
        if not os.path.isdir(output_dir):
            return
        existing = ckpt.list_checkpoints(output_dir)
        if existing:
            log.warning(
                "Output directory %s already contains %d checkpoint(s) and "
                "overwrite_output_dir=False. Pass resume_from_checkpoint to continue, "
                "or set overwrite_output_dir=True to ignore them.",
                output_dir,
                len(existing),
            )

    def _resolve_save_steps_int(self, total_steps: int) -> int:
        """Resolve ``save_steps`` to an integer.

        ``save_steps`` is a fraction of ``total_steps`` when 0 < value < 1
        (HF parity); otherwise treated as an absolute integer step count.
        """
        v = self.args.save_steps
        if v is None:
            return 0
        if 0 < v < 1:
            return max(1, int(round(total_steps * float(v))))
        return max(1, int(v))

    def _update_best_metric(
        self,
        eval_metrics: dict[str, Any],
        global_step: int,
    ) -> None:
        """Update ``state.best_*`` if the eval metric improved.

        ``BestModelSaveCallback`` independently decides whether to set
        ``control.should_save`` for ``save_strategy='best'`` (it runs at
        ``on_evaluate``, before this method updates ``state.best_metric``);
        both use :func:`is_metric_improved` against the same operands so
        the two decisions can't drift.
        """
        a = self.args
        if a.metric_for_best_model is None:
            return
        resolved = resolve_eval_metric(eval_metrics, a.metric_for_best_model)
        if resolved is None:
            key = a.metric_for_best_model
            if not key.startswith("eval_"):
                key = f"eval_{key}"
            raise KeyError(
                f"The `metric_for_best_model` training argument is set to {key!r}, "
                "which is not found in the evaluation metrics. The available "
                f"evaluation metrics are: {list(eval_metrics.keys())}. Consider "
                "changing `metric_for_best_model`."
            )
        _, value = resolved
        if not is_metric_improved(
            value, self.state.best_metric, self.args.greater_is_better
        ):
            return
        self.state.best_metric = value
        if self.args.save_strategy in {"steps", "epoch", "best"}:
            self.state.best_global_step = global_step

    def _load_best_model(self, ctx: "_TrainingContext") -> None:
        """Restore best-checkpoint weights into the underlying ``nn.Module``.

        Ordering contract (HF parity):

        1. Read the saved state dict from ``state.best_model_checkpoint``.
        2. Mutate ``self._model`` via ``load_state_dict(...)`` — the
           caller verified the model's parameter set matches what was
           saved.
        3. Rebuild ``ctx.trainable_params`` from the freshly mutated
           module so the functional path picks up the loaded weights.

        Mutating the module first means a ``save_model()`` call between
        ``_load_best_model`` and the ``train()`` ``finally`` block sees
        the loaded weights even before ``_restore_params`` runs.

        Raises ``RuntimeError`` when ``load_best_model_at_end=True`` was
        requested but the best-checkpoint contract can't be honored —
        no best checkpoint was recorded (eval never improved) or the
        recorded directory has no weights file.  Soft-failing here
        leaves the user with the last-trained weights silently masquerading
        as "best", which is exactly what the flag is meant to prevent.
        """
        ckpt_dir = self.state.best_model_checkpoint
        if ckpt_dir is None:
            raise RuntimeError(
                "load_best_model_at_end=True but no best checkpoint was recorded "
                "during training (eval never improved on metric_for_best_model="
                f"{self.args.metric_for_best_model!r}).  Either disable "
                "load_best_model_at_end, or verify the eval/metric configuration "
                "produces at least one improving step."
            )
        log.info("Loading best model from %s", ckpt_dir)
        new_state, mutated = self._read_weights_file(ckpt_dir)
        if not new_state and not mutated:
            raise RuntimeError(
                f"load_best_model_at_end=True: best checkpoint recorded at "
                f"{ckpt_dir!r} but no weights file (model.safetensors / "
                "pytorch_model.bin / sharded index) was found there.  The "
                "directory may have been pruned or moved between save and "
                "end-of-train load."
            )

        # Mutate the underlying module so ``save_model()`` and any callback
        # firing on ``on_train_end`` observe the best weights immediately.
        # Sharded loads have already mutated the model in place — skip
        # the redundant ``load_state_dict`` call.  PEFT detection drives
        # ``strict``: adapter dirs ship only adapter weights, full-model
        # dirs ship the full state dict.
        if not mutated:
            strict = not self._is_peft
            self._model.load_state_dict(new_state, strict=strict)

        # Rebuild the functional view from the (now-mutated) module.  Keep
        # ``ctx.trainable_params`` keyed by the same names as before — i.e.
        # every parameter that was trainable at the start of the run.
        for name, param in self._model.named_parameters():
            if name in ctx.trainable_params:
                ctx.trainable_params[name] = param.detach().to(self._device)

    def _read_weights_file(
        self,
        ckpt_dir: str,
    ) -> tuple[dict[str, torch.Tensor], bool]:
        """Load weights from a checkpoint directory.

        Supports four on-disk shapes:

        - Single-file safetensors: ``model.safetensors`` (full model) or
          ``adapter_model.safetensors`` (PEFT adapter).
        - Single-file pickle: ``pytorch_model.bin`` /
          ``adapter_model.bin``.
        - Sharded safetensors with index ``model.safetensors.index.json``.
        - Sharded pickle with index ``pytorch_model.bin.index.json``.

        Returns ``(state_dict, model_already_mutated)``.  In the
        single-file cases ``state_dict`` carries the loaded tensors and
        ``model_already_mutated`` is ``False`` (the caller must call
        ``self._model.load_state_dict(state_dict, ...)`` itself).  In
        the sharded case ``load_sharded_checkpoint`` has already
        mutated ``self._model`` in place, so the state dict comes back
        empty and ``model_already_mutated`` is ``True`` to signal "no
        further ``load_state_dict`` needed".  Returns ``({}, False)``
        when no checkpoint files were found.
        """
        from safetensors.torch import load_file as load_safetensors
        from transformers.modeling_utils import load_sharded_checkpoint
        from transformers.utils import (
            SAFE_WEIGHTS_INDEX_NAME,
            WEIGHTS_INDEX_NAME,
        )

        # Single-file shapes win over sharded indices (HF parity:
        # ``save_pretrained`` writes a single file when the model fits
        # under ``max_shard_size``).
        candidates = [
            os.path.join(ckpt_dir, ckpt.SAFE_WEIGHTS_NAME),
            os.path.join(ckpt_dir, "adapter_model.safetensors"),
            os.path.join(ckpt_dir, ckpt.WEIGHTS_NAME),
            os.path.join(ckpt_dir, "adapter_model.bin"),
        ]
        for path in candidates:
            if not os.path.exists(path):
                continue
            if path.endswith(".safetensors"):
                return load_safetensors(path, device=str(self._device)), False
            # ``weights_only=False``: ``pytorch_model.bin`` is a pickled
            # state-dict that may carry ``torch.dtype`` / ``torch.device``
            # markers (HF historically stamps these into checkpoints) —
            # PyTorch 2.6's safe-load default rejects them.  Pinning the
            # explicit ``False`` keeps the behaviour we tested against.
            return (
                torch.load(path, map_location=self._device, weights_only=False),
                False,
            )

        # Sharded checkpoints: ``load_sharded_checkpoint`` mutates the
        # model in place.  ``strict=False`` mirrors the PEFT-friendly
        # single-file path so partial-key checkpoints still load.
        sharded_indices = (
            os.path.join(ckpt_dir, SAFE_WEIGHTS_INDEX_NAME),
            os.path.join(ckpt_dir, WEIGHTS_INDEX_NAME),
        )
        if any(os.path.exists(p) for p in sharded_indices):
            load_sharded_checkpoint(
                self._model,
                ckpt_dir,
                strict=False,
                prefer_safe=True,
            )
            return {}, True
        return {}, False

    def save_model(
        self,
        output_dir: str | None = None,
    ) -> None:
        """Restore in-memory params into the model and call ``model.save_pretrained``.

        Mirrors ``Trainer.save_model``; safe to call after training finishes.

        Args:
            output_dir: Directory to save to.  Defaults to ``args.output_dir``.
        """
        a = self.args
        target = output_dir or self._effective_output_dir()
        if target is None:
            raise ValueError("save_model requires output_dir (arg or args.output_dir)")
        if self._ctx is not None:
            self._restore_params(self._ctx.trainable_params)
        if _distributed.should_save(a, self._ddp):
            os.makedirs(target, exist_ok=True)
            self._save_model_artifacts(target)
            self._save_training_args(target)
            # Privacy provenance travels with every saved model.  Use the
            # live ``_ctx`` accountant when training is mid-flight (most up
            # to date), otherwise the trainer-level slot populated by the
            # ``_setup_training`` finally block.
            accountant = (
                self._ctx.accounting if self._ctx is not None else self._accountant
            )
            if accountant is not None:
                self._save_accountant(target, accountant)
            else:
                log.info(
                    "save_model called before any training run; "
                    "no accountant to serialise (model only)."
                )
        # Barrier so non-saving ranks don't proceed before the save lands.
        _distributed.barrier(self._ddp)

    def _save_checkpoint(self, ctx: "_TrainingContext", step: int) -> str:
        """Write a complete ``checkpoint-<step>`` directory; returns its path.

        Under DDP, the rank-0 process writes shared artefacts (model weights,
        trainer state, training args, accountant, optimizer, DP runtime),
        every rank writes its own RNG snapshot (per-rank file so each rank
        can resume its own non-DP RNG), and a barrier at the end keeps all
        ranks in lockstep before any continues.
        """
        a = self.args
        output_dir = self._effective_output_dir()
        if output_dir is None:
            raise ValueError("Saving checkpoints requires args.output_dir to be set")
        ckpt_dir = os.path.join(output_dir, f"{ckpt.PREFIX_CHECKPOINT_DIR}-{step}")
        # Rank-0 owns the directory creation + bulk artefacts; every rank
        # restores params (needed for either RNG snapshot writers reading
        # `self._model.state_dict()` shapes consistently in future, and for
        # callbacks below that may inspect params).
        self._restore_params(ctx.trainable_params)
        if _distributed.should_save(a, self._ddp):
            os.makedirs(ckpt_dir, exist_ok=True)
            self._save_model_artifacts(ckpt_dir)

            # HF parity: register ``best_model_checkpoint`` by *looking up*
            # the folder named ``checkpoint-{best_global_step}`` rather than
            # only recognising the best step when it coincides with this
            # save's step.  Without this, an eval boundary that improves the
            # metric at step 100 followed by a save_strategy="steps" boundary
            # at step 200 would never register checkpoint-100 as best — and
            # rotation could delete it because no folder is protected.
            # Resolve *before* writing ``trainer_state.json`` so the file
            # lands once with the final ``best_model_checkpoint`` populated.
            if self.state.best_global_step is not None:
                best_dir = os.path.join(
                    output_dir,
                    f"{ckpt.PREFIX_CHECKPOINT_DIR}-{self.state.best_global_step}",
                )
                if os.path.isdir(best_dir):
                    self.state.best_model_checkpoint = best_dir
                else:
                    log.debug(
                        "best_global_step=%d but no checkpoint-%d/ folder "
                        "exists (best step fell into a non-saved bucket); "
                        "leaving best_model_checkpoint unset",
                        self.state.best_global_step,
                        self.state.best_global_step,
                    )

            self._save_trainer_state(ckpt_dir)
            self._save_training_args(ckpt_dir)
            self._save_accountant(ckpt_dir, ctx.accounting)
            if not a.save_only_model:
                self._save_optimizer(ckpt_dir, ctx)
                self._save_dp_runtime(ckpt_dir, ctx)

        # Per-rank RNG snapshot — every rank, after rank-0 has created the
        # directory.  Barrier guarantees the dir exists before non-zero
        # ranks try to write into it.
        _distributed.barrier(self._ddp)
        if not a.save_only_model:
            self._save_rng_state(ckpt_dir)

        if _distributed.should_save(a, self._ddp):
            # Rotation honours ``save_total_limit`` and protects best when set.
            ckpt.rotate_checkpoints(
                output_dir,
                save_total_limit=a.save_total_limit,
                best_model_checkpoint=self.state.best_model_checkpoint,
            )
            self._control.should_save = False
            log.info("Saved checkpoint to %s", ckpt_dir)
            # Notify callbacks that a checkpoint was just written.
            self._control = self._callback_handler.on_save(
                self.args, self.state, self._control
            )

        # Final barrier so all ranks see post-save state consistently before
        # any continues into the next training step / eval / rotation.
        _distributed.barrier(self._ddp)
        return ckpt_dir

    def _save_model_artifacts(self, output_dir: str) -> None:
        """Save model weights/config plus processing class using HF-compatible names."""
        if hasattr(self._model, "save_pretrained"):
            self._model.save_pretrained(
                output_dir,
                safe_serialization=self.args.save_safetensors,
            )
        else:
            state_dict = {
                name: tensor.detach().cpu()
                for name, tensor in self._model.state_dict().items()
            }
            if self.args.save_safetensors:
                from safetensors.torch import save_file as save_safetensors

                save_safetensors(
                    state_dict,
                    os.path.join(output_dir, ckpt.SAFE_WEIGHTS_NAME),
                    metadata={"format": "pt"},
                )
            else:
                torch.save(state_dict, os.path.join(output_dir, ckpt.WEIGHTS_NAME))
        if self._processing_class is not None:
            self._processing_class.save_pretrained(output_dir)

    def _save_optimizer(self, ckpt_dir: str, ctx: "_TrainingContext") -> None:
        torch.save(
            opaque_state_dict(ctx.opt_state),
            os.path.join(ckpt_dir, ckpt.DP_OPTIMIZER_NAME),
        )

    def _save_dp_runtime(self, ckpt_dir: str, ctx: "_TrainingContext") -> None:
        # ``state_dict`` from the opaque.serialization registry — each
        # sampler family (Poisson here, dp-ftrl variants in subclasses)
        # registers its own serializer pair at module-import time.
        from opaque.serialization import state_dict as opaque_state_dict

        sampler_state = (
            opaque_state_dict(ctx.current_sampler)
            if ctx.current_sampler is not None
            else None
        )

        if ctx.mf is not None:
            _amp = ctx.mf.amplifier_factory(ctx.noise_multiplier)
            mf_n_steps: int | None = int(_amp.n_steps)
            mf_min_sep: int | None = int(_amp.min_sep)
            mf_max_participations: int | None = int(_amp.max_participations)
        else:
            mf_n_steps = mf_min_sep = mf_max_participations = None

        ckpt.save_dp_runtime_state(
            os.path.join(ckpt_dir, ckpt.DP_STATE_NAME),
            clip_state=ctx.clip_state,
            noise_state=ctx.noise_state,
            sampler_state=sampler_state,
            sample_rate=ctx.sample_rate,
            target_delta=ctx.target_delta,
            noise_multiplier=ctx.noise_multiplier,
            expected_steps_per_epoch=ctx.expected_steps_per_epoch,
            expected_batch_size=int(self.args.train_batch_size),
            total_steps=ctx.total_steps,
            mechanism_kind=ctx.mechanism_kind,
            mf_n_steps=mf_n_steps,
            mf_min_sep=mf_min_sep,
            mf_max_participations=mf_max_participations,
        )

    def _save_accountant(self, ckpt_dir: str, accountant: "Accountant") -> None:
        path = os.path.join(ckpt_dir, ckpt.DP_ACCOUNTANT_NAME)
        with open(path, "w") as f:
            json.dump(opaque_state_dict(accountant), f, indent=2)

    def _save_rng_state(self, ckpt_dir: str) -> None:
        """Snapshot this rank's Python/NumPy/torch RNG state to disk.

        Each rank writes its own ``rng_state.pth`` (single-process) or
        ``rng_state_{rank}.pth`` (multi-rank) — every rank carries an
        independent non-DP RNG that affects collator stochasticity, model
        init, eval shuffling, etc.  The DP RNG chain is keyed off
        ``args.seed`` and folded by epoch / iter_count, so it is
        deterministic and does not need a per-rank file.
        """
        torch.save(
            ckpt.snapshot_rng_state(),
            ckpt.rng_state_path(
                ckpt_dir,
                rank=self._ddp.rank,
                world_size=self._ddp.world_size,
            ),
        )

    def _save_trainer_state(self, ckpt_dir: str) -> None:
        """Serialize :class:`DPTrainerState` to ``trainer_state.json``.

        Round-trips through ``DPTrainerState.to_json`` / ``from_json``.
        Callback state is collected via HF's ``ExportableState`` protocol
        (modern shape ``{"args": {...}, "attributes": {...}}`` produced
        by ``cb.state()``); callbacks that don't implement ``state()`` are
        skipped.
        """
        cb_states: dict[str, Any] = {}
        for cb in self._callback_handler.callbacks:
            state_fn = getattr(cb, "state", None)
            if callable(state_fn):
                cb_states[type(cb).__name__] = state_fn()
        self.state.stateful_callbacks = cb_states

        payload = self.state.to_json()
        path = os.path.join(ckpt_dir, ckpt.TRAINER_STATE_NAME)
        with open(path, "w") as f:
            json.dump(payload, f, indent=2, default=str)

    def _save_training_args(self, ckpt_dir: str) -> None:
        # Filename matches HF's ``TRAINING_ARGS_NAME``; HF tooling that
        # ``torch.load(.../training_args.bin)`` accepts the bundled
        # ``TrainingArguments`` because the dataclass is a strict
        # superset of ``TrainingArguments``.
        torch.save(self.args, os.path.join(ckpt_dir, ckpt.TRAINING_ARGS_NAME))

    def _maybe_final_save(self, ctx: "_TrainingContext", global_step: int) -> None:
        """Always emit a final checkpoint when saving is enabled (HF parity).

        Skipped if a checkpoint at this exact step already exists (e.g. an
        epoch-strategy save just fired and we're now at end-of-training).
        """
        if self.args.save_strategy == "no":
            return
        output_dir = self._effective_output_dir()
        if output_dir is None:
            return
        target = os.path.join(output_dir, f"{ckpt.PREFIX_CHECKPOINT_DIR}-{global_step}")
        if os.path.isdir(target):
            return
        self._save_checkpoint(ctx, global_step)

    def _refresh_final_checkpoint_state(self, global_step: int) -> None:
        """Refresh final checkpoint metadata after final logs update callbacks."""
        if self.args.save_strategy == "no":
            return
        output_dir = self._effective_output_dir()
        if output_dir is None:
            return
        target = os.path.join(output_dir, f"{ckpt.PREFIX_CHECKPOINT_DIR}-{global_step}")
        if os.path.isdir(target):
            self._save_trainer_state(target)

    # ------------------------------------------------------------------
    # Resume / load  (Phase 2c)
    # ------------------------------------------------------------------

    def _resolve_resume_path(self, value: str | bool | None) -> str | None:
        """Resolve ``resume_from_checkpoint`` to a concrete directory or ``None``.

        ``True`` is **tolerant**: if a checkpoint exists under
        ``args.output_dir`` it is auto-selected; if no checkpoints exist
        the call falls through to ``None`` (fresh run) with an info-level
        log.  This lets scripts pass ``resume_from_checkpoint=True``
        unconditionally — "resume if you can, else start fresh" — without
        having to probe the filesystem first.  ``output_dir is None``
        still raises since that's a config error, not a missing artefact.
        """
        if value is None or value is False:
            return None
        if value is True:
            output_dir = self._effective_output_dir()
            if output_dir is None:
                raise ValueError(
                    "resume_from_checkpoint=True requires args.output_dir to be set"
                )
            found = ckpt.get_last_checkpoint(output_dir)
            if found is None:
                log.info(
                    "resume_from_checkpoint=True but no checkpoints found under "
                    "%s; starting a fresh run.",
                    output_dir,
                )
                return None
            return found
        if not isinstance(value, str):
            raise TypeError(
                f"resume_from_checkpoint must be str | bool | None, got {type(value).__name__}"
            )
        if not os.path.isdir(value):
            raise FileNotFoundError(
                f"resume_from_checkpoint directory does not exist: {value}"
            )
        return value

    def _load_model_weights(self, ckpt_dir: str) -> None:
        """Load saved weights into ``self._model`` so make_functional starts from them."""
        new_state, mutated = self._read_weights_file(ckpt_dir)
        if not new_state and not mutated:
            log.warning("No weights file in %s; model untouched", ckpt_dir)
            return
        # Sharded loads already mutated ``self._model`` in place — skip
        # the redundant ``load_state_dict`` call.  ``strict`` follows
        # PEFT detection: adapter checkpoints store only adapter
        # parameters (subset → ``strict=False``); full-model checkpoints
        # surface mismatched keys as errors (``strict=True``).
        if not mutated:
            strict = not self._is_peft
            self._model.load_state_dict(new_state, strict=strict)

    def _read_runtime_for_resume(
        self, ckpt_dir: str
    ) -> tuple["ckpt.RuntimeCheckpoint | None", "Accountant | None"]:
        """Pre-load ``dp_state.pt`` and ``accountant.json`` for resume.

        Returns ``(runtime, accountant)``.  ``runtime`` is ``None`` when
        the checkpoint was written with ``save_only_model=True``.
        The runtime file stores flat ``opaque.serialization`` dicts for clip
        and noise state; they are merged in :meth:`_apply_runtime_state`.

        ``accountant.json`` carries the privacy provenance of all prior
        training.  Missing-file policy:

        - **Default** (``args.privacy_resume_without_accountant=False``):
          raise ``FileNotFoundError``.  Resuming without a recorded
          accountant would silently discard the spent privacy budget —
          surface that as a hard failure so it can't go unnoticed.
        - **Opt-in** (``args.privacy_resume_without_accountant=True``):
          install an empty ``Accountant()`` as the prefix and proceed.
          Calibration runs over the remaining steps as if the prior
          run had zero DP cost.  Designed for "warmup on public data,
          then DP-fine-tune" workflows where this is genuinely correct;
          dangerous to enable in any other context.
        """
        runtime_path = os.path.join(ckpt_dir, ckpt.DP_STATE_NAME)
        runtime_payload = (
            ckpt.load_dp_runtime_state(runtime_path)
            if os.path.exists(runtime_path)
            else None
        )

        acct_path = os.path.join(ckpt_dir, ckpt.DP_ACCOUNTANT_NAME)
        if os.path.exists(acct_path):
            with open(acct_path) as f:
                accountant = opaque_from_state_dict(Accountant(), json.load(f))
        elif self.args.privacy_resume_without_accountant:
            log.info(
                "No accountant.json in %s; privacy_resume_without_accountant=True, "
                "treating prior training as zero DP cost and calibrating remaining "
                "steps against an empty accountant.",
                ckpt_dir,
            )
            accountant = Accountant()
        else:
            raise FileNotFoundError(
                f"Cannot resume from {ckpt_dir}: accountant.json is missing. "
                "Resuming without the saved accountant would silently discard "
                "the spent privacy budget.  Either restore accountant.json from "
                "the source of truth, or — only when the resumed checkpoint has "
                "genuinely zero prior DP cost (e.g. warmup on public data) — "
                "pass privacy_resume_without_accountant=True to opt in to "
                "recalibration against an empty accountant."
            )
        return runtime_payload, accountant

    def _read_trainer_state(self, ckpt_dir: str) -> dict[str, Any] | None:
        """Read ``trainer_state.json`` from a checkpoint directory."""
        path = os.path.join(ckpt_dir, ckpt.TRAINER_STATE_NAME)
        if not os.path.exists(path):
            return None
        with open(path) as f:
            return json.load(f)

    def _apply_runtime_state(
        self,
        ctx: "_TrainingContext",
        runtime: "ckpt.RuntimeCheckpoint",
        accountant: "Accountant | None",
        ckpt_dir: str,
    ) -> None:
        """Overwrite ctx fields with values restored from a checkpoint."""
        ctx.clip_state = opaque_from_state_dict(ctx.clip_state, runtime.clip_state)
        ctx.noise_state = opaque_from_state_dict(ctx.noise_state, runtime.noise_state)

        opt_path = os.path.join(ckpt_dir, ckpt.DP_OPTIMIZER_NAME)
        if os.path.exists(opt_path):
            # Load flat serialisation on CPU; tensors move with ``opt.update``.
            opt_sd = torch.load(
                opt_path,
                map_location="cpu",
                weights_only=False,
            )
            ctx.opt_state = opaque_from_state_dict(ctx.opt_state, opt_sd)

        if accountant is not None:
            ctx.accounting = accountant

    def _load_rng_state(self, ckpt_dir: str) -> None:
        """Restore this rank's RNG snapshot from a checkpoint.

        Reads ``rng_state.pth`` (single-process) or ``rng_state_{rank}.pth``
        (multi-rank) — the world-aware helper picks the right path so the
        call site doesn't have to branch.
        """
        path = ckpt.rng_state_path(
            ckpt_dir,
            rank=self._ddp.rank,
            world_size=self._ddp.world_size,
        )
        if not os.path.exists(path):
            return
        # ``weights_only=False``: the snapshot bundles
        # ``random.getstate()`` (a Python tuple) and NumPy's RNG state
        # alongside the torch tensors — the safe-load path can't
        # reconstruct either.  PyTorch 2.6+ flips the default to
        # ``True``; keeping the explicit ``False`` pins the contract.
        snap = torch.load(path, map_location="cpu", weights_only=False)
        ckpt.restore_rng_state(snap)

    def _load_callback_states(self) -> None:
        """Restore each callback's state when ``restore_callback_states_from_checkpoint`` is set.

        Reads from ``self.state.stateful_callbacks`` — the dataclass field
        populated by ``DPTrainerState.from_json`` during resume.  The
        payload uses HF's ``ExportableState`` shape ``{"args": {...},
        "attributes": {...}}``; saved attributes are set back onto the
        live callback instance so its identity is preserved
        (``EarlyStoppingCallback`` parity).

        Drift between the saved set and the live set is logged at
        ``info`` so a user who removed / added a callback between runs
        sees it explicitly instead of silently inheriting the previous
        run's callback state (or not).
        """
        if not self.args.restore_callback_states_from_checkpoint:
            return
        cb_states = self.state.stateful_callbacks or {}
        live_names = {type(cb).__name__ for cb in self._callback_handler.callbacks}
        saved_names = set(cb_states)

        missing_in_live = saved_names - live_names
        missing_in_saved = live_names - saved_names
        if missing_in_live:
            log.info(
                "Resume callback drift: %d saved callback(s) not present in "
                "live trainer (state will not be restored): %s",
                len(missing_in_live),
                sorted(missing_in_live),
            )
        if missing_in_saved:
            log.info(
                "Resume callback drift: %d live callback(s) not present in "
                "saved state (will start with fresh state): %s",
                len(missing_in_saved),
                sorted(missing_in_saved),
            )

        for cb in self._callback_handler.callbacks:
            name = type(cb).__name__
            if name not in cb_states:
                continue
            payload = cb_states[name]
            attrs = payload.get("attributes") or {} if isinstance(payload, dict) else {}
            for attr_key, value in attrs.items():
                setattr(cb, attr_key, value)

    def _warn_on_arg_drift(self, runtime: "ckpt.RuntimeCheckpoint") -> None:
        """Surface drift between the saved checkpoint and current ``args``.

        Fields with ``metadata={"compare_on_resume": True}`` on
        :class:`~opaque.api.transformers.trainer._checkpoint.RuntimeCheckpoint`
        are checked against current values.  Heterogeneous composition
        still produces a correct ε, but a saved-vs-current mismatch is
        worth surfacing.

        Adding a new privacy-relevant field becomes a one-line edit on
        ``RuntimeCheckpoint`` (add ``field(metadata={"compare_on_resume":
        True})``); the drift iteration picks it up automatically as
        long as :meth:`_current_value_for_drift` knows where to read
        the live value from.
        """
        a = self.args
        ctx = self._ctx
        current_by_name = self._current_values_for_drift(a, ctx)
        for f in dataclasses.fields(runtime):
            if not f.metadata.get("compare_on_resume"):
                continue
            saved = getattr(runtime, f.name)
            current = current_by_name.get(f.name)
            if saved is None or current is None:
                continue
            if isinstance(saved, float) and isinstance(current, float):
                if abs(saved - current) / max(abs(saved), 1e-12) > 1e-6:
                    log.warning(
                        "Resume arg drift on %s: saved=%g, current=%g — "
                        "heterogeneous composition still gives a correct ε but "
                        "the saved/current mechanisms differ",
                        f.name,
                        saved,
                        current,
                    )
            elif saved != current:
                log.warning(
                    "Resume arg drift on %s: saved=%r, current=%r",
                    f.name,
                    saved,
                    current,
                )

    def _current_values_for_drift(
        self, a, ctx: "_TrainingContext | None"
    ) -> dict[str, Any]:
        """Resolve current (post-setup) values for the drift-checked fields.

        Splits the lookup off so :meth:`_warn_on_arg_drift` stays a pure
        iteration over field metadata — no per-field if/else.  Returns
        ``None`` for any field whose live value can't be determined yet
        (e.g. ``sample_rate`` before ``ctx`` exists).
        """
        # ``sample_rate`` fallback (when ``ctx`` isn't built yet) must
        # mirror ``_setup_training``'s computation — same numerator
        # (``expected_batch_size``, which is ``world_size *
        # per_device_train_batch_size``) and same denominator (the
        # post-DDP-trim dataset size).  Using ``len(self._train_dataset)``
        # raw would falsely report drift on every DDP resume because
        # the trim hasn't been applied at compare time.  Returns
        # ``None`` (skips the check) when the dataset isn't available.
        if ctx is not None:
            fallback_rate: float | None = ctx.sample_rate
        elif self._train_dataset is None:
            fallback_rate = None
        else:
            fallback_rate = a.train_batch_size / max(
                1, self._effective_train_dataset_size()
            )
        return {
            "sample_rate": fallback_rate,
            "target_delta": (
                ctx.target_delta if ctx is not None else a.privacy_target_delta
            ),
            "noise_multiplier": (
                ctx.noise_multiplier if ctx is not None else a.privacy_noise_multiplier
            ),
            "total_steps": (
                ctx.total_steps if ctx is not None else self._predict_total_steps()
            ),
            "expected_batch_size": int(a.train_batch_size),
            "expected_steps_per_epoch": (
                ctx.expected_steps_per_epoch if ctx is not None else None
            ),
            "mechanism_kind": (
                ctx.mechanism_kind if ctx is not None else a.privacy_noise_mechanism
            ),
        }
