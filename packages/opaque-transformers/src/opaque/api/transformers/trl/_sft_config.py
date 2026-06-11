"""``SFTConfig`` — training arguments for :class:`SFTTrainer`.

Mirrors ``trl.SFTConfig`` for the subset meaningful under per-example DP,
extending Opaque's standalone
:class:`~opaque.api.transformers.trainer._config.TrainingArguments`. Fields with
no DP meaning (DeepSpeed/FSDP/Accelerate knobs, VLM args, packing / padding-free)
are absent, so passing them is an unexpected-keyword ``TypeError``; an unknown
``loss_type`` fails at the trainer's dispatch table, not a curated check here.
"""

from __future__ import annotations

import dataclasses
from typing import Any, Callable

from opaque.api.transformers.trainer._config import TrainingArguments
from opaque.api.transformers.trainer.training_arguments import normalize_dp_overrides

from ._convert import (
    _convert_trl_config,
    _import_trl,
    _reject_if_truthy,
    _reject_pad_token,
    _reject_truncation_mode,
)


@dataclasses.dataclass
class SFTConfig(TrainingArguments):
    """Arguments for supervised fine-tuning on :class:`DPTrainer`.

    Adds TRL-parity data-prep / loss fields on top of
    :class:`TrainingArguments`, keeping the TRL field names and defaults so the
    two configs read as analogues.
    """

    # ---- Learning rate override (TRL default differs from HF) ------------
    learning_rate: float = 2e-5

    # ---- Model loading ---------------------------------------------------
    #: Extra kwargs forwarded to ``AutoModelForCausalLM.from_pretrained`` when
    #: ``model`` is passed as a string (e.g. ``torch_dtype``, ``attn_implementation``).
    #: Ignored when ``model`` is an already-instantiated module.
    model_init_kwargs: dict | None = None

    # ---- Data preparation ------------------------------------------------
    #: Name of the column holding raw text on a language-modeling dataset.
    dataset_text_field: str = "text"
    # ``truncation_mode`` is intentionally absent: tokenization keeps the start
    # of the sequence (``keep_start``, TRL's default and forward path); TRL's
    # deprecated ``keep_end`` is not offered, so passing it is a TypeError.
    #: Maximum tokenized sequence length; ``None`` disables truncation.
    max_length: int | None = 1024
    #: Compute the loss only over completion tokens (prompt-completion data).
    #: ``None`` auto-detects from the dataset format at trainer-init time.
    completion_only_loss: bool | None = None
    #: EOS token appended to plain-text examples so the model learns to stop.
    #: When set, this exact token overrides ``tokenizer.eos_token``; when ``None``
    #: the tokenizer's own ``eos_token`` is used, or nothing if it has none.
    eos_token: str | None = None
    #: Pad the collated batch length up to a multiple of this value.
    pad_to_multiple_of: int | None = None
    #: Number of processes for ``datasets.map`` during preprocessing.
    dataset_num_proc: int | None = None
    #: Compute the loss only over assistant turns (conversational data). Uses
    #: the ``{% generation %}``-marked training chat template + the assistant
    #: token mask. Implies completion-only masking for chat data.
    assistant_only_loss: bool = False
    #: Path to a tokenizer dir or Jinja file whose chat template (and special
    #: tokens) is cloned onto ``processing_class`` before tokenization.
    chat_template_path: str | None = None

    # ---- Loss ------------------------------------------------------------
    #: ``"nll"`` (standard CE), ``"dft"`` (Dynamic Fine-Tuning), or the fused,
    #: logits-free ``"chunked_nll"`` (enables the ``fused_linear_cross_entropy``
    #: kernel). Unknown values fail at the trainer's loss-dispatch table.
    loss_type: str = "nll"

    # ``activation_offloading`` is inherited from the base ``TrainingArguments``
    # (shared by SFT and DPO); the base ``DPTrainer`` reads it.

    # ---- Telemetry -------------------------------------------------------
    #: Log the per-step completion-metric telemetry (``entropy``,
    #: ``mean_token_accuracy``, ``logits/*``). When ``False`` these logits-derived
    #: diagnostics are skipped, which also clears the way for the fused,
    #: logits-free loss path (see :class:`SFTTrainer`).
    log_completion_metrics: bool = True

    # ---- TRL base-default overrides (shared base default unchanged for
    # other trainers) ----------------------------------------------------
    #: TRL logs every 10 steps by default (vs. HF's 500).
    logging_steps: float = 10
    #: GC disabled by default: the vmap DP path recomputes activations once per
    #: microbatch, so GC multiplies that cost without memory benefit on models
    #: that fit. Enable explicitly when memory is the bottleneck.
    gradient_checkpointing: bool = False
    #: Opaque-specific (no TRL analogue): enable the model-level Triton kernels
    #: (``rope`` / ``rms_norm`` / ``activation`` / non-fused ``cross_entropy``)
    #: by default — they cut per-sample-gradient memory/compute under the vmap DP
    #: path. CUDA + Triton only; no-op on CPU/MPS. The
    #: ``fused_linear_cross_entropy`` kernel stays opt-in (it returns
    #: ``logits=None``, incompatible with the eager nll/dft loss + telemetry).
    #: Opt out with ``use_performance_kernels=False``.
    use_performance_kernels: bool = True

    def __post_init__(self) -> None:
        # DP-driven: the collator consumes raw dataset columns
        # (``completion_mask`` is folded into ``-100`` labels) that are not
        # ``model.forward`` parameters and would be stripped by column pruning.
        self.remove_unused_columns = False
        # TRL parity: default bf16 on when supported and no precision was set.
        # The base still raises if bf16 was set explicitly on unsupported
        # hardware, so only auto-enable when it's actually available.
        if not self.bf16 and not self.bf16_full_eval:
            import torch

            if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
                self.bf16 = True
        super().__post_init__()

    @classmethod
    def from_trl(
        cls,
        trl_cfg: Any,
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
    ) -> "SFTConfig":
        """Convert a ``trl.SFTConfig`` to an opaque ``SFTConfig``.

        Requires the optional ``trl`` extra:
        ``pip install opaque[trl]``.

        TRL-specific fields (``dataset_text_field``, ``chat_template_path``,
        ``completion_only_loss``, ``assistant_only_loss``, ``loss_type``,
        ``eos_token``, ``max_length``, ``pad_to_multiple_of``,
        ``dataset_num_proc``, ``model_init_kwargs``) are copied directly.

        Fields TRL has that opaque does not implement raise
        ``ValueError``: ``packing``, ``padding_free``, ``eval_packing``,
        ``shuffle_dataset``, ``truncation_mode='keep_end'``, ``pad_token``.

        HF-inherited fields go through the same translation as
        :meth:`TrainingArguments.from_hf` — same DP-knob requirement, same
        ``per_device * gradient_accumulation_steps`` → opaque logical
        batch collapse.

        Raises
        ------
        ImportError
            If the optional ``trl`` dependency is not installed.
        TypeError
            If ``trl_cfg`` is not a ``trl.SFTConfig`` instance.
        ValueError
            If a required DP knob is missing, or any REJECT_IF_SET field
            is set to a non-default value.
        """
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

        converted = convert_trl_sft_config(trl_cfg, strict=strict, **dp_overrides)
        converted.update(opaque_overrides)
        return cls(**converted)


# ===========================================================================
# trl.SFTConfig → opaque SFTConfig manifest + converter
# ===========================================================================
#
# TRL-specific fields only; the HF-inherited subset is delegated to the HF
# manifest by ``_convert_trl_config``. Every TRL ``SFTConfig`` field appears in
# exactly one bucket, enforced by ``test_compat_manifest_exhaustive.py``.

# DIRECT — TRL field name matches opaque, same semantics.
TRL_SFT_DIRECT_FIELDS: frozenset[str] = frozenset(
    {
        "model_init_kwargs",
        "chat_template_path",
        "dataset_text_field",
        "dataset_num_proc",
        "eos_token",
        "max_length",
        "pad_to_multiple_of",
        "completion_only_loss",
        "assistant_only_loss",
        "loss_type",
        # On TRL's SFTConfig/DPOConfig but not on HF base ``TrainingArguments``;
        # opaque's base has the same field with the same semantics.
        "activation_offloading",
    }
)


# RENAME — TRL SFT field name differs from opaque.
TRL_SFT_RENAME_MAP: dict[str, str] = {}


# TRANSFORM — TRL SFT field requires a derivation step.
TRL_SFT_TRANSFORM_MAP: dict[str, Callable[[dict[str, Any]], dict[str, Any]]] = {}


# REJECT_IF_SET — TRL has the field but opaque does not implement it.
TRL_SFT_REJECTED_FIELDS: dict[str, Callable[[Any], str | None]] = {
    "truncation_mode": _reject_truncation_mode,
    "packing": _reject_if_truthy(
        "Sequence packing breaks the fixed per-example batch shape DP-SGD's "
        "per-example vmap requires. Disable ``packing`` or use opaque without "
        "the converter."
    ),
    "padding_free": _reject_if_truthy(
        "``padding_free`` flattens the batch and skips the attention mask "
        "DP-SGD's per-example accounting depends on. Not supported."
    ),
    "eval_packing": _reject_if_truthy(
        "Eval-side sequence packing is not supported for the same reason as "
        "``packing``."
    ),
    "shuffle_dataset": _reject_if_truthy(
        "Opaque's Poisson sampler controls ordering; explicit shuffling "
        "would defeat the DP accounting."
    ),
    "pad_token": _reject_pad_token,
}


# DROP_WITH_WARN — TRL has the field but it's irrelevant on opaque path.
TRL_SFT_DROP_FIELDS: dict[str, str] = {
    "dataset_kwargs": (
        "Opaque does not expose a ``datasets.map`` hook; the collator and "
        "tokenizer handle preprocessing."
    ),
    "packing_strategy": (
        "Only meaningful with ``packing=True``, which opaque does not "
        "support; silently dropped."
    ),
}


def convert_trl_sft_config(
    trl_cfg: Any,
    *,
    strict: bool = True,
    **dp_overrides: Any,
) -> dict[str, Any]:
    """Translate a ``trl.SFTConfig`` instance into opaque ``SFTConfig`` kwargs."""
    trl = _import_trl()
    if not isinstance(trl_cfg, trl.SFTConfig):
        raise TypeError(
            f"Expected ``trl.SFTConfig`` instance, got {type(trl_cfg).__name__}."
        )

    dp_layer = normalize_dp_overrides(dp_overrides)
    return _convert_trl_config(
        trl_cfg,
        trl_direct=TRL_SFT_DIRECT_FIELDS,
        trl_rename=TRL_SFT_RENAME_MAP,
        trl_transform=TRL_SFT_TRANSFORM_MAP,
        trl_reject=TRL_SFT_REJECTED_FIELDS,
        trl_drop=TRL_SFT_DROP_FIELDS,
        source_label="trl_sft_config",
        strict=strict,
        dp_overrides=dp_layer,
    )
