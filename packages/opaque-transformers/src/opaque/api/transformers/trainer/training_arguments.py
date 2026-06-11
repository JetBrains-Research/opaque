"""HF ``transformers.TrainingArguments`` → opaque ``TrainingArguments``.

This module holds two things:

1. The **manifest engine** (:func:`_apply_manifest` + helpers) that every
   converter — HF here, TRL in :mod:`opaque.api.transformers.trl` — drives
   with its own DIRECT/RENAME/TRANSFORM/REJECT/DROP buckets.
2. The **HF manifest** itself, classifying every field on
   ``transformers.TrainingArguments`` into exactly one bucket:

   - **DIRECT** — field name and semantics match opaque; copy as-is.
   - **RENAME** — HF name differs from opaque; rename, preserve value.
   - **TRANSFORM** — multi-field derivation (e.g. ``per_device_train_batch_size``
     + ``gradient_accumulation_steps`` collapse into the opaque logical
     Poisson batch; ``max_grad_norm`` → ``clipping_norm``; Liger → perf kernels).
   - **REJECT_IF_SET** — field exists in HF but has no opaque equivalent; if
     set to anything non-default, raise ``ValueError`` with a per-field
     rationale + suggested alternative.
   - **DROP_WITH_WARN** — field exists in HF but is irrelevant on the
     opaque path; silently drop, emit a ``RuntimeWarning`` if non-default.

The partition is enforced by ``test_compat_manifest_exhaustive.py``: every
field on ``transformers.TrainingArguments`` appears in exactly one bucket.
Drift in upstream HF fails that test.

The public entry point is :meth:`TrainingArguments.from_hf`; this module holds
the machinery it delegates to.
"""

from __future__ import annotations

import dataclasses
import logging
import warnings
from collections.abc import Mapping
from typing import Any, Callable

log = logging.getLogger(__name__)


# ===========================================================================
# Manifest engine — shared by the HF (this module) and TRL converters.
# ===========================================================================


def _normalize_dp_overrides(
    overrides: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate and return a dict of opaque-side DP fields from kwargs.

    At least one of ``privacy_noise_multiplier`` or ``privacy_target_epsilon``
    must be set — opaque's runtime would reject the config otherwise, and
    surfacing the error here gives a clearer message tied to the converter
    call site.
    """
    noise_mult = overrides.get("privacy_noise_multiplier")
    target_eps = overrides.get("privacy_target_epsilon")
    if noise_mult is None and target_eps is None:
        raise ValueError(
            "Converting to an opaque config requires a DP knob: pass either "
            "``privacy_noise_multiplier=<float>`` (fixed-noise mode) or "
            "``privacy_target_epsilon=<float>`` (calibrated-noise mode) as a "
            "keyword argument to the converter."
        )
    # Pass through every override verbatim — opaque's TrainingArguments
    # __post_init__ does the cross-field validation.
    return dict(overrides)


def _is_default(value: Any, default: Any) -> bool:
    """``True`` if ``value`` matches ``default`` for a dataclass field.

    Handles the awkward cases the HF surface raises: dataclass instances
    (like ``AcceleratorConfig``) whose ``__eq__`` may not be defined, and
    pairs where ``default_factory`` constructs a fresh instance that
    isn't ``==`` to the value the user got.
    """
    if isinstance(default, dataclasses.Field):
        if default.default is not dataclasses.MISSING:
            default_value = default.default
        elif default.default_factory is not dataclasses.MISSING:
            default_value = default.default_factory()
        else:
            return False
    else:
        default_value = default

    # Fast path: identity match.
    if value is default_value:
        return True

    # Standard equality.
    try:
        if value == default_value:
            return True
    except Exception:  # pragma: no cover — pathological __eq__
        pass

    # Both dataclass instances? Compare via field dicts (handles
    # ``AcceleratorConfig`` and other HF nested dataclasses).
    if dataclasses.is_dataclass(value) and dataclasses.is_dataclass(default_value):
        try:
            return dataclasses.asdict(value) == dataclasses.asdict(default_value)
        except Exception:  # pragma: no cover
            pass

    return False


def _get_dataclass_field_values(
    obj: Any,
) -> dict[str, Any]:
    """Return a name→value dict for every field of a dataclass instance."""
    if not dataclasses.is_dataclass(obj):
        raise TypeError(
            f"Expected a dataclass instance, got {type(obj).__name__}. "
            "The HF/TRL converters accept dataclass instances only."
        )
    return {f.name: getattr(obj, f.name) for f in dataclasses.fields(obj)}


def _apply_manifest(
    *,
    source_values: Mapping[str, Any],
    source_defaults: Mapping[str, Any],
    direct: frozenset[str],
    rename: Mapping[str, str],
    transform: Mapping[str, Callable[[dict[str, Any]], dict[str, Any]]],
    reject: Mapping[str, Callable[[Any], str | None]],
    drop: Mapping[str, str],
    source_label: str,
    strict: bool,
) -> dict[str, Any]:
    """Apply the bucketed manifest to ``source_values`` and return an opaque dict.

    ``reject`` maps field name → callable that returns an error message
    string if the value is unsupported, or ``None`` if the value is
    benign (e.g. ``packing=False`` is fine, only ``packing=True`` is
    rejected). The callable receives the raw source value.

    ``drop`` maps field name → reason string surfaced in the
    ``RuntimeWarning`` when the field is non-default.

    ``transform`` maps field name → callable that receives the full
    ``source_values`` dict and returns a partial opaque-side dict to
    merge in. Used for multi-field transforms like the batch collapse.
    """
    opaque: dict[str, Any] = {}
    errors: list[str] = []

    for name, value in source_values.items():
        # Layer 1: REJECT_IF_SET — default value is a silent drop; a non-default
        # value runs the rejector, which returns an error message or None
        # (benign, falls through to the next bucket).
        if name in reject:
            if _is_default(value, source_defaults.get(name)):
                continue  # default value → silent drop
            message = reject[name](value)
            if message is not None:
                errors.append(f"  - {source_label}.{name}={value!r}: {message}")
                continue
            # message is None → benign; fall through to other buckets.

        # Layer 2: DROP_WITH_WARN — drop and (optionally) warn.
        if name in drop:
            if not _is_default(value, source_defaults.get(name)):
                _warn_drop(source_label, name, value, drop[name], strict)
            continue

        # Layer 3: TRANSFORM — multi-field derivation. Each transform callable
        # runs once below, regardless of how many source fields it inspects.
        if name in transform:
            continue

        # Layer 4: RENAME — name swap, value preserved.
        if name in rename:
            opaque[rename[name]] = value
            continue

        # Layer 5: DIRECT — copy as-is.
        if name in direct:
            opaque[name] = value
            continue

        # Unbucketed field: surface as a hard error so the canary test can
        # catch upstream additions before they bite a user.
        errors.append(
            f"  - {source_label}.{name}={value!r}: field is not classified by "
            f"the opaque argument manifest (neither DIRECT, RENAME, TRANSFORM, "
            f"REJECT, nor DROP). This usually means upstream HF/TRL added a "
            f"new field on a version the opaque manifest hasn't been updated "
            f"for. File a bug or pin a supported version."
        )

    if errors:
        raise ValueError(
            f"Converting {source_label} to opaque failed:\n" + "\n".join(errors)
        )

    # Run all transforms — they see the full source dict.
    for name, transform_fn in transform.items():
        if name in source_values:
            opaque.update(transform_fn(dict(source_values)))

    return opaque


def _warn_drop(
    source_label: str, name: str, value: Any, reason: str, strict: bool
) -> None:
    """Emit a Python warning for a dropped non-default field."""
    msg = (
        f"opaque: dropping {source_label}.{name}={value!r} — {reason}. "
        f"This field has no opaque equivalent and is being discarded."
    )
    if strict:
        warnings.warn(msg, RuntimeWarning, stacklevel=4)
    # When ``strict=False``, drop silently.


# ===========================================================================
# HF manifest — every ``transformers.TrainingArguments`` field, one bucket.
# ===========================================================================

# ---------------------------------------------------------------------------
# DIRECT — name and semantics match opaque ``TrainingArguments``.
# ---------------------------------------------------------------------------
HF_DIRECT_FIELDS: frozenset[str] = frozenset(
    {
        # Output / scope
        "output_dir",
        "overwrite_output_dir",
        # Batch sizes
        "per_device_eval_batch_size",
        "eval_accumulation_steps",
        "eval_delay",
        # Optimizer / LR schedule
        "learning_rate",
        "weight_decay",
        "adam_beta1",
        "adam_beta2",
        "adam_epsilon",
        "warmup_ratio",
        "warmup_steps",
        "lr_scheduler_kwargs",
        # Training duration
        "num_train_epochs",
        "max_steps",
        # Logging
        "log_level",
        "log_level_replica",
        "log_on_each_node",
        "logging_dir",
        "logging_strategy",
        "logging_first_step",
        "logging_steps",
        # Saving
        "save_strategy",
        "save_steps",
        "save_total_limit",
        "save_safetensors",
        "save_on_each_node",
        "save_only_model",
        "restore_callback_states_from_checkpoint",
        # Reproducibility
        "seed",
        "data_seed",
        "full_determinism",
        # Precision
        "use_cpu",
        "use_mps_device",
        "bf16",
        "bf16_full_eval",
        "tf32",
        # Distributed
        "local_rank",
        "ddp_backend",
        "ddp_timeout",
        # Evaluation
        "eval_strategy",
        "eval_steps",
        "eval_on_start",
        "eval_do_concat_batches",
        "prediction_loss_only",
        "include_for_metrics",
        "average_tokens_across_devices",
        "metric_for_best_model",
        "greater_is_better",
        "load_best_model_at_end",
        "ignore_data_skip",
        # DataLoader
        "dataloader_num_workers",
        "dataloader_persistent_workers",
        "dataloader_pin_memory",
        "dataloader_prefetch_factor",
        "dataloader_drop_last",
        "remove_unused_columns",
        "torch_empty_cache_steps",
        # Labels
        "label_names",
        "label_smoothing_factor",
        # Reporting
        "report_to",
        "disable_tqdm",
        "run_name",
        "project",
        # Optimizer kwargs string / dict
        "optim_args",
        # Hub
        "push_to_hub",
        "hub_model_id",
        "hub_token",
        "hub_private_repo",
        "hub_revision",
        # Compile
        "torch_compile",
        "torch_compile_backend",
        "torch_compile_mode",
        # Misc
        "gradient_checkpointing",
        "gradient_checkpointing_kwargs",
        "skip_memory_metrics",
        "include_tokens_per_second",
        "include_num_input_tokens_seen",
        "debug",
        "resume_from_checkpoint",
    }
)


# ---------------------------------------------------------------------------
# RENAME — HF name → opaque name; value preserved.
# ---------------------------------------------------------------------------
HF_RENAME_MAP: dict[str, str] = {
    # HF deprecated ``per_gpu_*`` long ago in favor of ``per_device_*``;
    # accept the deprecated name and rename silently.
    "per_gpu_train_batch_size": "per_device_train_batch_size",
    "per_gpu_eval_batch_size": "per_device_eval_batch_size",
    # Opaque dropped the ``_type`` suffix on the scheduler kind.
    "lr_scheduler_type": "lr_scheduler",
    # HF's legacy ``push_to_hub_*`` aliases (kept for old-config compat).
    "push_to_hub_model_id": "hub_model_id",
    "push_to_hub_token": "hub_token",
}


# ---------------------------------------------------------------------------
# TRANSFORM — multi-field derivations.
# ---------------------------------------------------------------------------
def _batch_collapse(hf: dict[str, Any]) -> dict[str, Any]:
    """Collapse HF ``(per_device, grad_accum, auto_find_batch_size)`` into the
    opaque logical Poisson batch + vmap chunk.

    HF's effective batch (``per_device × grad_accum``) IS the privacy-relevant
    unit, so it becomes opaque's logical ``per_device_train_batch_size``;
    dropping ``grad_accum`` would under-account the amplification and emit a
    too-optimistic ε. The HF per-device size becomes the opaque vmap chunk
    (``microbatch_size``), and HF's ``auto_find_batch_size`` maps to opaque's
    ``auto_find_microbatch_size`` (shrinks the vmap chunk on OOM, not the
    logical batch — privacy-neutral).
    """
    per_device = int(hf.get("per_device_train_batch_size", 8))
    grad_accum = int(hf.get("gradient_accumulation_steps", 1) or 1)
    out: dict[str, Any] = {
        "per_device_train_batch_size": per_device * grad_accum,
        "auto_find_microbatch_size": bool(hf.get("auto_find_batch_size", False)),
    }
    if grad_accum > 1:
        # Only set the vmap chunk when grad_accum > 1; at grad_accum=1 the
        # logical batch == per_device, so leave microbatch_size at its
        # default (None → vmap over the full batch).
        out["microbatch_size"] = per_device
    return out


_HF_PAGED_OPTIMS = frozenset(
    {
        "adamw_apex_fused",
        "adamw_bnb_8bit",
        "adamw_8bit",
        "paged_adamw_8bit",
        "paged_adamw_32bit",
        "lion_8bit",
        "paged_lion_8bit",
        "paged_lion_32bit",
        "rmsprop_bnb",
        "rmsprop_bnb_8bit",
        "rmsprop_bnb_32bit",
    }
)


def _optim_collapse(hf: dict[str, Any]) -> dict[str, Any]:
    """Collapse HF optim aliases to opaque's ``adamw`` (+ raise on paged variants).

    HF carries half a dozen ``adamw_*`` aliases (``adamw_torch``,
    ``adamw_hf``, ``adamw_torch_fused``) that all resolve to AdamW. Opaque
    has a single ``adamw`` name. Quantized/Apex-fused variants
    (``paged_adamw_*``, ``adamw_8bit``, …) are not in opaque-engine's
    torchopt path and raise.

    Other optimizer families (``sgd``, ``adafactor``, ``lion``, ``rmsprop``,
    …) pass through unchanged — opaque's ``_resolve_optimizer_name`` will
    handle name resolution at trainer-construction time.
    """
    optim_value = hf.get("optim")
    if optim_value is None:
        return {}
    optim_str = str(optim_value).lower()
    # OptimizerNames enums stringify to e.g. ``"OptimizerNames.ADAMW_TORCH"``;
    # use the ``.value`` if present for a clean string.
    optim_str = getattr(optim_value, "value", optim_str)
    optim_str = str(optim_str).lower()

    if optim_str in _HF_PAGED_OPTIMS:
        raise ValueError(
            f"hf_training_arguments.optim={optim_value!r}: Quantized / "
            f"Apex-fused optimizers are not in opaque-engine's torchopt "
            f"path. Use ``optim='adamw'`` (with ``optim_args={{'fused': True}}`` "
            f"if you want the fused CUDA path)."
        )
    if optim_str in {"adamw_torch", "adamw_hf"}:
        return {"optim": "adamw"}
    if optim_str == "adamw_torch_fused":
        return {"optim": "adamw", "optim_args": {"fused": True}}
    # Pass through other optimizer names verbatim (as string, not enum).
    return {"optim": optim_str}


def _max_grad_norm_to_clipping(hf: dict[str, Any]) -> dict[str, Any]:
    """Loosely map HF's global grad-norm clip → opaque ``clipping_norm``.

    Not semantically identical (HF clips the aggregate gradient; opaque
    clips per example for DP), but it's the closest knob and a sensible
    default. A ``clipping_norm`` DP override wins (applied after the manifest).
    """
    v = hf.get("max_grad_norm")
    return {"clipping_norm": float(v)} if v is not None else {}


def _liger_to_perf_kernels(hf: dict[str, Any]) -> dict[str, Any]:
    """Map HF Liger fused kernels → opaque performance kernels."""
    if not hf.get("use_liger_kernel"):
        return {}
    out: dict[str, Any] = {"use_performance_kernels": True}
    cfg = hf.get("liger_kernel_config")
    if cfg:
        out["performance_kernels_config"] = dict(cfg)
    return out


def _adafactor_to_optim(hf: dict[str, Any]) -> dict[str, Any]:
    """Legacy ``adafactor=True`` boolean → opaque ``optim='adafactor'``."""
    return {"optim": "adafactor"} if hf.get("adafactor") else {}


HF_TRANSFORM_MAP: dict[str, Callable[[dict[str, Any]], dict[str, Any]]] = {
    "per_device_train_batch_size": _batch_collapse,
    "gradient_accumulation_steps": _batch_collapse,
    "auto_find_batch_size": _batch_collapse,
    "optim": _optim_collapse,
    "max_grad_norm": _max_grad_norm_to_clipping,
    "use_liger_kernel": _liger_to_perf_kernels,
    "liger_kernel_config": _liger_to_perf_kernels,
    "adafactor": _adafactor_to_optim,
}


# ---------------------------------------------------------------------------
# REJECT_IF_SET — value must be the dataclass default to skip rejection.
# Callable returns an error message string (raised) or None (benign value).
# ---------------------------------------------------------------------------
def _reject_if_truthy(message: str) -> Callable[[Any], str | None]:
    """Helper: reject only if user set a truthy non-default value."""

    def inner(value: Any) -> str | None:
        if value:
            return message
        return None

    return inner


HF_REJECTED_FIELDS: dict[str, Callable[[Any], str | None]] = {
    # Half-precision: opaque is bf16-only.
    "fp16": _reject_if_truthy(
        "DP-SGD requires deterministic numerics; opaque is bf16-only. "
        "Use ``bf16=True`` instead."
    ),
    "fp16_full_eval": _reject_if_truthy(
        "Use ``bf16_full_eval=True``; opaque does not support fp16."
    ),
    "fp16_opt_level": _reject_if_truthy(
        "Apex fp16 opt levels are not supported; opaque is bf16-only."
    ),
    "fp16_backend": _reject_if_truthy(
        "Apex fp16 backends are not supported; opaque is bf16-only."
    ),
    "half_precision_backend": _reject_if_truthy(
        "Apex half-precision backends are not supported; opaque is bf16-only."
    ),
    # Distributed / sharding integrations not on the per-example DP path.
    "fsdp": _reject_if_truthy(
        "FSDP is not supported on the per-example DP-SGD path. Use the "
        "standard DDP backend via ``ddp_backend='nccl'``."
    ),
    "fsdp_config": _reject_if_truthy(
        "FSDP config is not supported; opaque uses per-example DDP only."
    ),
    "fsdp_transformer_layer_cls_to_wrap": _reject_if_truthy(
        "FSDP layer-class wrapping is not supported."
    ),
    "fsdp_min_num_params": _reject_if_truthy("FSDP wrapping policy is not supported."),
    "deepspeed": _reject_if_truthy(
        "DeepSpeed is not supported on the per-example DP-SGD path."
    ),
    # NEFTune embedding noise would interact with the privacy accountant.
    "neftune_noise_alpha": _reject_if_truthy(
        "NEFTune embedding noise is not wired through the DP-SGD path "
        "(it would interact with the privacy accountant)."
    ),
    # Hub auto-push.
    "hub_always_push": _reject_if_truthy(
        "Per-checkpoint auto-push is not supported; use ``push_to_hub=True`` "
        "for the end-of-training push."
    ),
    # ``adafactor``, ``max_grad_norm``, ``use_liger_kernel`` /
    # ``liger_kernel_config``, and ``auto_find_batch_size`` are remapped in
    # HF_TRANSFORM_MAP, not rejected here.
}


# ---------------------------------------------------------------------------
# DROP_WITH_WARN — field exists in HF but is irrelevant on opaque; drop with
# a warning when non-default.
# ---------------------------------------------------------------------------
HF_DROP_FIELDS: dict[str, str] = {
    # HF deprecated runtime-gating booleans; opaque uses explicit
    # ``trainer.train()`` / ``trainer.evaluate()`` calls.
    "do_train": "HF deprecated runtime-gating booleans; call trainer.train() directly.",
    "do_eval": "HF deprecated runtime-gating booleans; call trainer.evaluate() directly.",
    "do_predict": "HF deprecated runtime-gating booleans; call trainer.predict() directly.",
    # TPU support is not on the opaque path.
    "tpu_num_cores": "Opaque has no TPU support.",
    "tpu_metrics_debug": "Removed in HF 4.x; opaque has no TPU support.",
    # XPU (Intel GPU) backend not exposed by opaque.
    # SageMaker compat.
    "mp_parameters": "SageMaker model-parallel parameters are not used by opaque.",
    # Length-based grouping is incompatible with Poisson sampling.
    "length_column_name": (
        "Length-grouped batching is incompatible with Poisson subsampling "
        "and is not used by opaque."
    ),
    "group_by_length": (
        "Length-grouped batching is incompatible with Poisson subsampling "
        "and is not used by opaque."
    ),
    # HF stats output paths opaque does not honor.
    "logging_nan_inf_filter": "Opaque's logging path computes NaN/Inf filtering separately.",
    # DDP knobs opaque doesn't expose.
    "ddp_find_unused_parameters": "Opaque's per-example DDP path doesn't use this knob.",
    "ddp_bucket_cap_mb": "Opaque's per-example DDP path doesn't use bucket sizing.",
    "ddp_broadcast_buffers": "Opaque's per-example DDP path manages buffer sync internally.",
    # HF stats / metric callbacks not in opaque.
    "include_inputs_for_metrics": (
        "Opaque does not feed inputs into ``compute_metrics``; use "
        "``include_for_metrics=['inputs']`` instead."
    ),
    "jit_mode_eval": "JIT mode is not supported on opaque's per-example path.",
    "use_legacy_prediction_loop": (
        "Opaque uses its own prediction loop; the HF legacy flag has no effect."
    ),
    # HF-deprecated runtime flags.
    "no_cuda": "Deprecated in HF; use ``use_cpu`` (opaque's equivalent).",
    "past_index": "Legacy field for seq2seq past-state index; not used.",
    # Private / runtime-computed HF fields.
    "_n_gpu": "Private HF field, runtime-computed; opaque computes its own.",
    # New HF features outside the opaque path.
    "parallelism_config": "HF parallelism config is not used by opaque.",
    "trackio_space_id": "HF trackio tracking is not used by opaque (use report_to=['wandb']).",
    "torchdynamo": "Deprecated; use ``torch_compile=True``.",
    "ray_scope": "Ray Tune integration is not used by opaque.",
    "optim_target_modules": "Galore-target-modules selection is not on the opaque optim path.",
    "batch_eval_metrics": "Streaming-eval metric callback is not used by opaque.",
    "eval_use_gather_object": "HF Accelerate-only eval gather; not used by opaque.",
    # Hub auto-push and deprecated push_to_hub_organization.
    "push_to_hub_organization": "Deprecated HF alias; set ``hub_model_id='org/repo'`` instead.",
    "hub_strategy": (
        "Opaque only supports end-of-training push via ``push_to_hub=True``; "
        "any HF ``hub_strategy`` setting is ignored."
    ),
    # Accelerate-driven config: opaque manages distribution natively.
    "accelerator_config": (
        "Accelerate-driven config is not used; opaque uses ``ddp_backend`` directly."
    ),
}


# ---------------------------------------------------------------------------
# Public API: convert an HF TrainingArguments instance to an opaque dict.
# ---------------------------------------------------------------------------
def _convert_hf_training_arguments(
    hf_args: Any,
    *,
    strict: bool = True,
    **dp_overrides: Any,
) -> dict[str, Any]:
    """Translate an HF ``TrainingArguments`` instance into opaque-side kwargs.

    The DP-override kwargs are merged onto the converted dict last (they
    take precedence over both HF values and HF-derived defaults).
    """
    # Local import so this module stays importable when HF is absent.
    try:
        from transformers import TrainingArguments as HFTrainingArguments
    except ImportError as e:
        raise ImportError(
            "_convert_hf_training_arguments requires the ``transformers`` "
            "package. Install it with ``pip install transformers``."
        ) from e

    if not isinstance(hf_args, HFTrainingArguments):
        raise TypeError(
            f"Expected ``transformers.TrainingArguments`` instance, got "
            f"{type(hf_args).__name__}. To convert a dict, build a "
            "``TrainingArguments(**your_dict)`` first."
        )

    source_values = _get_dataclass_field_values(hf_args)
    # HF's ``__post_init__`` rewrites several field defaults at runtime, so
    # ``dataclasses.fields()`` defaults don't match what an unconfigured user
    # sees. Build a baseline instance and use its field values as the
    # "user didn't set this" defaults.
    import tempfile

    baseline_output_dir = source_values.get("output_dir") or tempfile.mkdtemp(
        prefix="opaque_baseline_"
    )
    baseline = type(hf_args)(output_dir=baseline_output_dir)
    source_defaults = _get_dataclass_field_values(baseline)

    # Wrap REJECT and TRANSFORM signatures into the dispatcher contract.
    converted = _apply_manifest(
        source_values=source_values,
        source_defaults=source_defaults,
        direct=HF_DIRECT_FIELDS,
        rename=HF_RENAME_MAP,
        transform=HF_TRANSFORM_MAP,
        reject=HF_REJECTED_FIELDS,
        drop=HF_DROP_FIELDS,
        source_label="hf_training_arguments",
        strict=strict,
    )

    # Performance kernels default ON in opaque but OFF in HF/TRL; default them
    # OFF here to match upstream (the Liger transform sets True when Liger was
    # on; a name override below can force either way).
    converted.setdefault("use_performance_kernels", False)

    # Overrides win over every converted/derived value (the privacy knobs plus
    # any opaque field overridden by name, e.g. ``use_performance_kernels=True``
    # even though HF had no such field).
    overrides = _normalize_dp_overrides(dp_overrides)
    converted.update(overrides)

    log.info(
        "converted hf_training_arguments → %d opaque fields (%d overridden by name)",
        len(converted),
        len(overrides),
    )
    return converted
