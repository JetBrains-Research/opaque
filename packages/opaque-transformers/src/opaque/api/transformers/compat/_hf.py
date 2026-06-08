"""HF ``transformers.TrainingArguments`` → opaque ``TrainingArguments`` manifest.

Every HF field is classified into exactly one bucket:

- **DIRECT** — field name and semantics match opaque exactly; copy as-is.
- **RENAME** — HF name differs from opaque name; rename, preserve value.
- **TRANSFORM** — multi-field derivation (e.g. ``per_device_train_batch_size``
  + ``gradient_accumulation_steps`` collapse into the opaque logical
  Poisson batch).
- **REJECT_IF_SET** — field exists in HF but has no opaque equivalent; if
  the user set it to anything non-default, raise ``ValueError`` with a
  per-field rationale + suggested alternative.
- **DROP_WITH_WARN** — field exists in HF but is irrelevant on the
  opaque path; silently drop, emit a ``RuntimeWarning`` if non-default.

The partition is enforced by ``test_compat_manifest_exhaustive.py``: every
field on ``transformers.TrainingArguments`` appears in exactly one of these
sets. Drift in upstream HF fails that test.
"""

from __future__ import annotations

from typing import Any, Callable

from ._common import (
    apply_manifest,
    get_dataclass_field_defaults,
    get_dataclass_field_values,
    normalize_dp_overrides,
)


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
    """Collapse HF ``(per_device, grad_accum)`` into opaque logical batch.

    Opaque's ``per_device_train_batch_size`` is the *logical Poisson batch*
    — the privacy-relevant unit. HF's effective batch (``per_device ×
    grad_accum``) IS the privacy-relevant batch. Dropping ``grad_accum``
    on the floor would under-account the sampling amplification and emit
    a wrong (too-optimistic) ε.

    The HF microbatch becomes the opaque vmap chunk.
    """
    per_device = int(hf.get("per_device_train_batch_size", 8))
    grad_accum = int(hf.get("gradient_accumulation_steps", 1) or 1)
    out: dict[str, Any] = {
        "per_device_train_batch_size": per_device * grad_accum,
    }
    if grad_accum > 1:
        # Only set microbatch_size when grad_accum > 1; at grad_accum=1
        # the HF and opaque batch concepts are identical and we leave
        # opaque's ``microbatch_size`` at its default (None → vmap on the
        # full batch).
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


HF_TRANSFORM_MAP: dict[str, Callable[[dict[str, Any]], dict[str, Any]]] = {
    "per_device_train_batch_size": _batch_collapse,
    "gradient_accumulation_steps": _batch_collapse,
    "optim": _optim_collapse,
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


def _reject_max_grad_norm(value: Any) -> str | None:
    # HF's default is 1.0; we treat that as benign at the dispatcher level
    # via ``_is_default``. Any other non-default value here means the
    # user wired up HF's pre-DP gradient norm clipping, which has no
    # analogue on opaque's DP path — opaque's ``clipping_norm`` is the
    # per-example DP clipping bound, not a global pre-step norm clip.
    return (
        "HF's ``max_grad_norm`` is a pre-step global gradient norm clip; "
        "opaque has no equivalent. Per-example DP clipping is controlled by "
        "``clipping_norm`` (passed as a kwarg to ``from_hf``). Drop "
        "``max_grad_norm`` from your HF config."
    )


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
    "fsdp_min_num_params": _reject_if_truthy(
        "FSDP wrapping policy is not supported."
    ),
    "deepspeed": _reject_if_truthy(
        "DeepSpeed is not supported on the per-example DP-SGD path."
    ),
    # NEFTune embedding noise would interact with the privacy accountant.
    "neftune_noise_alpha": _reject_if_truthy(
        "NEFTune embedding noise is not wired through the DP-SGD path "
        "(it would interact with the privacy accountant)."
    ),
    # Optimizer rejections — the ``optim`` field itself is handled in the
    # TRANSFORM (``_optim_collapse``) which raises on paged variants.
    "adafactor": _reject_if_truthy(
        "Use ``optim='adafactor'`` directly (the legacy ``adafactor=True`` "
        "boolean flag is collapsed in opaque)."
    ),
    # Hub auto-push.
    "hub_always_push": _reject_if_truthy(
        "Per-checkpoint auto-push is not supported; use ``push_to_hub=True`` "
        "for the end-of-training push."
    ),
    # Pre-step global gradient norm clip (HF) has no opaque equivalent.
    "max_grad_norm": _reject_max_grad_norm,
    # Liger fused kernels — not on the per-example DP path.
    "use_liger_kernel": _reject_if_truthy(
        "Liger fused kernels are not on the opaque per-example DP-SGD path."
    ),
    "liger_kernel_config": _reject_if_truthy(
        "Liger fused kernels are not on the opaque per-example DP-SGD path."
    ),
    # Auto batch size — different semantics in opaque.
    "auto_find_batch_size": _reject_if_truthy(
        "Opaque uses ``auto_find_microbatch_size`` (different semantics: "
        "the vmap chunk shrinks on OOM, not the logical batch). Either "
        "pass ``auto_find_microbatch_size=True`` via ``opaque_overrides`` "
        "or omit."
    ),
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
def convert_hf_training_arguments(
    hf_args: Any,
    *,
    strict: bool = True,
    **dp_overrides: Any,
) -> dict[str, Any]:
    """Translate an HF ``TrainingArguments`` instance into opaque-side kwargs.

    The DP-override kwargs are merged onto the converted dict last (they
    take precedence over both HF values and HF-derived defaults).
    """
    # Local import — HF is a runtime dep of opaque-transformers but we want
    # the converter module to be importable even if HF is missing (it
    # wouldn't be useful, but it shouldn't error at import time).
    try:
        from transformers import TrainingArguments as HFTrainingArguments
    except ImportError as e:
        raise ImportError(
            "convert_hf_training_arguments requires the ``transformers`` "
            "package. Install it with ``pip install transformers``."
        ) from e

    if not isinstance(hf_args, HFTrainingArguments):
        raise TypeError(
            f"Expected ``transformers.TrainingArguments`` instance, got "
            f"{type(hf_args).__name__}. To convert a dict, build a "
            "``TrainingArguments(**your_dict)`` first."
        )

    source_values = get_dataclass_field_values(hf_args)
    # HF's ``__post_init__`` rewrites several field defaults at runtime
    # (``fsdp: None → []``, ``fsdp_config: None → {'min_num_params': 0, ...}``,
    # ``accelerator_config: None → AcceleratorConfig(...)``, …). The
    # field-level defaults from ``dataclasses.fields()`` are therefore
    # NOT the values the user sees when they don't customize. Construct
    # a baseline instance with the same ``output_dir`` and use its field
    # values as the canonical "user didn't set this" baseline.
    import tempfile

    baseline_output_dir = source_values.get("output_dir") or tempfile.mkdtemp(
        prefix="opaque_compat_baseline_"
    )
    baseline = type(hf_args)(output_dir=baseline_output_dir)
    source_defaults = get_dataclass_field_values(baseline)

    # Wrap REJECT and TRANSFORM signatures into the dispatcher contract.
    converted = apply_manifest(
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

    # Layer DP overrides on top.
    dp_layer = normalize_dp_overrides(dp_overrides)
    converted.update(dp_layer)

    return converted
