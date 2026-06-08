"""``SFTConfig`` — training arguments for :class:`SFTTrainer`.

Mirrors ``trl.SFTConfig`` (``trl/trainer/sft_config.py``) for the subset that
is meaningful under per-example DP, extending Opaque's standalone
:class:`~opaque.api.transformers.trainer._config.TrainingArguments`.

Per the trainers' design philosophy (``docs/development/sft-dpo-trainers-plan.md``
§3.3): fields that have no DP meaning (DeepSpeed/FSDP/Accelerate knobs, VLM
args, packing / padding-free — for now) are simply **absent** from this
surface, so passing them is a standard unexpected-keyword ``TypeError``; an
unknown ``loss_type`` value fails at the trainer's dispatch table, not via a
curated check here.
"""

from __future__ import annotations

import dataclasses
from typing import Any

from opaque.api.transformers.trainer._config import TrainingArguments


@dataclasses.dataclass
class SFTConfig(TrainingArguments):
    """Arguments for supervised fine-tuning on :class:`DPTrainer`.

    Adds TRL-parity data-prep / loss fields on top of
    :class:`TrainingArguments`. The TRL field names and defaults are kept so the
    two configs read as analogues (``sft_config.py:137-275``).
    """

    # ---- Learning rate override (TRL default differs from HF) ------------
    learning_rate: float = 2e-5  # sft_config.py:137

    # ---- Model loading ---------------------------------------------------
    #: Extra kwargs forwarded to ``AutoModelForCausalLM.from_pretrained`` when
    #: ``model`` is passed as a string (e.g. ``torch_dtype``, ``attn_implementation``).
    #: Ignored when ``model`` is an already-instantiated module. (sft_config.py:145)
    model_init_kwargs: dict | None = None

    # ---- Data preparation ------------------------------------------------
    #: Name of the column holding raw text on a language-modeling dataset.
    dataset_text_field: str = "text"  # sft_config.py:163
    # ``truncation_mode`` is intentionally absent: tokenization keeps the start
    # of the sequence (``keep_start``), which is TRL's default and forward path.
    # TRL deprecated ``keep_end`` (warns, removes it in v2.0.0;
    # sft_config.py:297-300), so there is no DP-meaningful reason to add a knob
    # upstream is dropping — passing it is a standard unexpected-keyword TypeError.
    #: Maximum tokenized sequence length; ``None`` disables truncation.
    max_length: int | None = 1024  # sft_config.py:186
    #: Compute the loss only over completion tokens (prompt-completion data).
    #: ``None`` auto-detects from the dataset format at trainer-init time.
    completion_only_loss: bool | None = None  # sft_config.py:242
    #: EOS token appended to plain-text examples so the model learns to stop.
    #: When set, this exact token is used (and overrides ``tokenizer.eos_token``);
    #: when ``None`` (the default), the tokenizer's own ``eos_token`` is used. If
    #: the tokenizer has no EOS token either, nothing is appended.
    eos_token: str | None = None  # sft_config.py:180
    #: Pad the collated batch length up to a multiple of this value.
    pad_to_multiple_of: int | None = None  # sft_config.py:232
    #: Number of processes for ``datasets.map`` during preprocessing.
    dataset_num_proc: int | None = None  # sft_config.py:176
    #: Compute the loss only over assistant turns (conversational data). Uses
    #: the ``{% generation %}``-marked training chat template + the assistant
    #: token mask. Implies completion-only masking for chat data.
    assistant_only_loss: bool = False  # sft_config.py:254
    #: Path to a tokenizer dir or Jinja file whose chat template (and special
    #: tokens) is cloned onto ``processing_class`` before tokenization.
    chat_template_path: str | None = None  # sft_config.py:152

    # ---- Loss ------------------------------------------------------------
    #: ``"nll"`` (standard CE), ``"dft"`` (Dynamic Fine-Tuning), or the fused,
    #: logits-free ``"chunked_nll"`` (enables the ``fused_linear_cross_entropy``
    #: kernel). Unknown values fail at the trainer's loss-dispatch table (no
    #: curated check here).
    loss_type: str = "nll"  # sft_config.py:264

    # ``activation_offloading`` is inherited from the base ``TrainingArguments``
    # (one field, shared by SFT and DPO); the base ``DPTrainer`` reads it.

    # ---- Telemetry -------------------------------------------------------
    #: Log the per-step completion-metric telemetry (``entropy``,
    #: ``mean_token_accuracy``, ``logits/*``). When ``False`` these logits-derived
    #: diagnostics are skipped, which also clears the way for the fused,
    #: logits-free loss path (see :class:`SFTTrainer`).
    log_completion_metrics: bool = True

    # ---- TRL base-default overrides (override inherited defaults; the shared
    # base default is unchanged for other trainers) ----------------------
    #: TRL logs every 10 steps by default (vs. HF's 500). (sft_config.py)
    logging_steps: float = 10
    #: TRL enables gradient checkpointing by default to fit larger batches.
    gradient_checkpointing: bool = True

    def __post_init__(self) -> None:
        # The one DP-driven override: the collator consumes raw dataset
        # columns (``completion_mask`` is folded into ``-100`` labels), which
        # are not ``model.forward`` parameters and would be stripped by HF-style
        # column pruning. See plan §3.1.
        self.remove_unused_columns = False
        # TRL parity: default to bf16 mixed precision when the hardware supports
        # it and the user hasn't explicitly chosen a precision. The base's
        # ``__post_init__`` still raises if bf16 was *explicitly* requested on
        # unsupported hardware, so only auto-enable when it's actually available.
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
        clipping_norm: float | dict[str, float] = 1.0,
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
        from ..compat._trl import convert_trl_sft_config

        dp_overrides: dict[str, Any] = {
            "privacy_noise_multiplier": privacy_noise_multiplier,
            "privacy_target_epsilon": privacy_target_epsilon,
            "clipping_norm": clipping_norm,
            "privacy_noise_mechanism": privacy_noise_mechanism,
            "privacy_noise_radius": privacy_noise_radius,
            "clipping_mode": clipping_mode,
            "sampling_mode": sampling_mode,
        }
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
