"""``DPOConfig`` — training arguments for :class:`DPOTrainer`.

Mirrors ``trl.DPOConfig`` (``trl/trainer/dpo_config.py:211-304``) for the subset
that is meaningful under per-example DP, extending Opaque's standalone
:class:`~opaque.api.transformers.trainer._config.TrainingArguments`.

Design philosophy (plan §3.3): no bespoke "rejection" code. Batch-coupled
losses (``aot`` / ``aot_unpaired``) have no per-example DP meaning — there is no
``aot`` head in ``opaque-alignment``, so ``loss_type=["aot"]`` fails with an
ordinary ``KeyError`` at the trainer's dispatch table. TR-DPO sync
(``sync_ref_model`` / ``ref_model_mixup_alpha`` / ``ref_model_sync_steps``) *is*
supported (see the fields below); ``padding_free`` is absent from this surface,
so passing it is a standard unexpected-keyword ``TypeError``. TRL's *own*
faithful validations (label-smoothing bounds) are kept.
"""

from __future__ import annotations

import dataclasses
from typing import Any, Callable

import torch

from opaque.api.transformers.trainer._config import TrainingArguments
from opaque.api.transformers.trainer.training_arguments import normalize_dp_overrides

from ._convert import (
    _convert_trl_config,
    _import_trl,
    _reject_if_truthy,
    _reject_pad_token,
    _reject_truncation_mode,
)

# Loss heads that score the policy's own logp(s) — no reference model is
# resolved or precomputed for them. ``self._needs_reference`` is derived from
# the configured ``loss_type`` against this set (a run is reference-free only
# when *every* head is in it), replacing the old public ``reference_free`` flag.
_REFERENCE_FREE_HEADS = frozenset({"sft", "simpo", "cpo", "orpo"})


@dataclasses.dataclass
class DPOConfig(TrainingArguments):
    """Arguments for Direct Preference Optimization on :class:`DPTrainer`."""

    # ---- Learning rate override (TRL default differs from HF) ------------
    learning_rate: float = 1e-6  # dpo_config.py:142

    # ---- TRL base defaults (override the inherited HF/base values) -------
    logging_steps: float = 10  # dpo_config.py / TRL parity
    #: GC disabled by default: vmap recomputes activations per microbatch, so
    #: GC adds recompute overhead without memory benefit on models that fit.
    gradient_checkpointing: bool = False
    #: Opaque-specific (no TRL analogue): enable model-level Triton kernels
    #: (``rope`` / ``rms_norm`` / ``activation`` / non-fused ``cross_entropy``)
    #: by default — numerically non-critical but practically important under the
    #: per-example ``vmap`` DP path (cuts per-sample-gradient memory/compute).
    #: CUDA + Triton only; no-op on CPU/MPS. Opt out with
    #: ``use_performance_kernels=False``. Base default stays ``False``.
    use_performance_kernels: bool = True

    # ---- Model loading ---------------------------------------------------
    #: Extra kwargs forwarded to ``AutoModelForCausalLM.from_pretrained`` when
    #: ``model`` is passed as a string (e.g. ``torch_dtype``, ``attn_implementation``).
    #: Ignored when ``model`` is an already-instantiated module. (dpo_config.py:148)
    model_init_kwargs: dict | None = None

    # ---- Loss ------------------------------------------------------------
    #: One or more loss variants (list ⇒ MPO via ``mpo_combine``). TRL-style
    #: names: ``sigmoid``, ``hinge``, ``ipo``, ``robust``, ``exo_pair``,
    #: ``nca_pair``, ``bco_pair``, ``sppo_hard``, ``apo_zero``, ``apo_down``,
    #: ``discopop``, ``sft``, ``sigmoid_norm``. A bare string is coerced to a
    #: one-element list in ``__post_init__``.
    loss_type: list[str] | str = dataclasses.field(
        default_factory=lambda: ["sigmoid"]
    )  # dpo_config.py:211
    #: Per-loss weights for multi-loss (MPO) combination; defaults to all-ones.
    loss_weights: list[float] | None = None  # dpo_config.py:220
    #: Policy–reference KL penalty strength (τ for IPO).
    beta: float = 0.1  # dpo_config.py:260
    #: Robust-DPO label-flip probability in ``[0, 0.5)``; ε for EXO.
    label_smoothing: float = 0.0  # dpo_config.py:251
    #: f-divergence regulariser: ``reverse_kl`` (default), ``forward_kl``,
    #: ``js_divergence``, ``alpha_divergence``.
    f_divergence_type: str = "reverse_kl"  # dpo_config.py:237
    #: α coefficient for ``alpha_divergence``.
    f_alpha_divergence_coef: float = 0.5  # dpo_config.py:244
    #: LD-DPO verbose-token weight in ``[0, 1]``; ``None`` ⇒ standard DPO.
    ld_alpha: float | None = None  # dpo_config.py:228
    #: WPO length-normalized probability weighting.
    use_weighting: bool = False  # dpo_config.py:268
    #: DiscoPOP temperature.
    discopop_tau: float = 0.05  # dpo_config.py:275
    #: SimPO target reward margin γ subtracted inside the sigmoid (dpo_config.py).
    simpo_gamma: float = 0.5
    #: CPO supervised-NLL regulariser weight on the chosen completion.
    cpo_alpha: float = 1.0
    #: ORPO odds-ratio term weight (maps to TRL ORPO's ``beta``).
    orpo_lambda: float = 1.0

    # ---- Reference model -------------------------------------------------
    #: Batch size for the reference precompute pass; defaults to the train
    #: per-device batch size when ``None``. (There is no ``precompute_ref_log_probs``
    #: toggle: under the per-example DP substrate a *static* reference is always
    #: precomputed — the ``vmap`` loss can only read it as a constant column —
    #: so the toggle would be misleading. A reference-free ``loss_type``
    #: (``simpo``/``cpo``/``orpo``/``sft``) skips it; TR-DPO (``sync_ref_model``)
    #: recomputes per step.)
    precompute_ref_batch_size: int | None = None  # dpo_config.py:201
    #: Disable dropout in policy (and reference) before training.
    disable_dropout: bool = True  # dpo_config.py:155

    # ---- TR-DPO (reference sync, arXiv:2502.18014) -----------------------
    #: Periodically move the reference toward the policy by an EMA step
    #: (recomputed per training step, outside vmap). Requires a reference-using
    #: ``loss_type`` (nothing to sync toward otherwise); full fine-tuning only
    #: (not PEFT, mirroring TRL).
    sync_ref_model: bool = False  # dpo_config.py:287
    #: EMA mixup α: ``ref ← (1 - α)·ref + α·policy``.
    ref_model_mixup_alpha: float = 0.6  # dpo_config.py:296
    #: Apply the EMA update every ``ref_model_sync_steps`` steps.
    ref_model_sync_steps: int = 512  # dpo_config.py:304

    # ---- Data preparation ------------------------------------------------
    # ``truncation_mode`` is intentionally absent: tokenization keeps the start
    # of the sequence (``keep_start``), which is TRL's default and forward path.
    # TRL deprecated ``keep_end`` (warns, removes it in v2.0.0;
    # dpo_config.py:335-338), so there is no DP-meaningful reason to add a knob
    # upstream is dropping — passing it is a standard unexpected-keyword TypeError.
    #: Maximum tokenized sequence length; ``None`` disables truncation.
    max_length: int | None = 1024  # dpo_config.py:165
    #: Pad the collated batch length up to a multiple of this value.
    pad_to_multiple_of: int | None = None  # dpo_config.py:189
    #: Number of processes for ``datasets.map`` during preprocessing.
    dataset_num_proc: int | None = None  # dpo_config.py:161

    # ---- Telemetry -------------------------------------------------------
    #: Log the logits-consuming completion telemetry (``entropy``,
    #: ``mean_token_accuracy``, ``logits/*``) each step. Default ``True`` (the
    #: eager path). When ``False`` these are skipped, which also lets the trainer
    #: select the fused, logits-free log-prob primitives when CUDA is available
    #: and no other logits-consuming feature is active.
    log_completion_metrics: bool = True

    def __post_init__(self) -> None:
        # DP-driven: the preference collator consumes non-``forward`` columns
        # (``chosen_input_ids`` … ``ref_chosen_logps``); keep them (plan §3.1).
        self.remove_unused_columns = False
        # TRL parity: ``loss_type`` is always a list internally.
        if isinstance(self.loss_type, str):
            self.loss_type = [self.loss_type]
        if self.loss_weights is None:
            self.loss_weights = [1.0] * len(self.loss_type)
        super().__post_init__()

        # TRL's own validations (dpo_trainer.py:680-694), not DP-driven rejections.
        if "robust" in self.loss_type and not 0.0 <= self.label_smoothing < 0.5:
            raise ValueError(
                "Robust DPO (loss_type='robust') requires label_smoothing in "
                f"[0.0, 0.5); got {self.label_smoothing}."
            )
        if "exo_pair" in self.loss_type and self.label_smoothing <= 0.0:
            raise ValueError(
                "EXO (loss_type='exo_pair') requires label_smoothing > 0; got "
                f"{self.label_smoothing}."
            )
        if self.loss_weights is not None and len(self.loss_weights) != len(
            self.loss_type
        ):
            raise ValueError(
                "loss_weights must have the same length as loss_type: "
                f"{len(self.loss_weights)} != {len(self.loss_type)}."
            )
        # MPO terms are keyed by loss name; duplicates would silently collapse
        # and change the objective. Fail fast (review feedback).
        if len(set(self.loss_type)) != len(self.loss_type):
            raise ValueError(
                f"loss_type contains duplicates: {self.loss_type}. Each loss "
                "variant may appear at most once in an MPO combination."
            )
        # TR-DPO has nothing to sync toward when every head is reference-free.
        if self.sync_ref_model and all(
            lt in _REFERENCE_FREE_HEADS for lt in self.loss_type
        ):
            names = ", ".join(self.loss_type)
            raise ValueError(
                "TR-DPO (sync_ref_model) requires a reference-using loss_type; "
                f"{names} are reference-free."
            )

        # TRL parity: default bf16 on when the hardware supports it and the user
        # did not opt in/out explicitly. Set after the loss validation and only
        # when supported, so the base's explicit-bf16-on-unsupported-hw raise
        # (already run in super().__post_init__) is not retriggered.
        if (
            not self.bf16
            and torch.cuda.is_available()
            and torch.cuda.is_bf16_supported()
        ):
            self.bf16 = True

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
    ) -> "DPOConfig":
        """Convert a ``trl.DPOConfig`` to an opaque ``DPOConfig``.

        Requires the optional ``trl`` extra:
        ``pip install opaque[trl]``.

        TRL-specific fields (``loss_type``, ``loss_weights``, ``beta``,
        ``label_smoothing``, ``f_divergence_type``, ``ld_alpha``,
        ``use_weighting``, ``simpo_gamma``, ``cpo_alpha``, ``orpo_lambda``,
        ``sync_ref_model``, ``ref_model_mixup_alpha``,
        ``ref_model_sync_steps``, ``precompute_ref_batch_size``,
        ``disable_dropout``, ``max_length``, ``pad_to_multiple_of``,
        ``dataset_num_proc``, ``model_init_kwargs``) are copied directly.

        ``loss_type`` is validated per-element against opaque's implemented
        heads; the TRL 1.x ``aot`` / ``aot_unpaired`` Adversarial Optimal
        Transport family raises ``ValueError`` because opaque does not
        implement them.

        Fields TRL has that opaque does not implement raise
        ``ValueError``: ``padding_free``, ``truncation_mode='keep_end'``,
        ``pad_token``.

        HF-inherited fields go through the same translation as
        :meth:`TrainingArguments.from_hf` — same DP-knob requirement, same
        batch-collapse, same rejection set.

        Raises
        ------
        ImportError
            If the optional ``trl`` dependency is not installed.
        TypeError
            If ``trl_cfg`` is not a ``trl.DPOConfig`` instance.
        ValueError
            If a required DP knob is missing, any REJECT_IF_SET field is
            set to a non-default value, or ``loss_type`` contains an
            opaque-unsupported head.
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

        converted = convert_trl_dpo_config(trl_cfg, strict=strict, **dp_overrides)
        converted.update(opaque_overrides)
        return cls(**converted)


# ===========================================================================
# trl.DPOConfig → opaque DPOConfig manifest + converter
# ===========================================================================
#
# TRL-specific fields only; the HF-inherited subset is delegated to the HF
# manifest by ``_convert_trl_config``. Every TRL ``DPOConfig`` field appears in
# exactly one bucket, enforced by ``test_compat_manifest_exhaustive.py``.

TRL_DPO_DIRECT_FIELDS: frozenset[str] = frozenset(
    {
        "model_init_kwargs",
        "disable_dropout",
        "dataset_num_proc",
        "max_length",
        "pad_to_multiple_of",
        "precompute_ref_batch_size",
        "beta",
        "label_smoothing",
        "loss_weights",
        "f_divergence_type",
        "f_alpha_divergence_coef",
        "ld_alpha",
        "use_weighting",
        "discopop_tau",
        "sync_ref_model",
        "ref_model_mixup_alpha",
        "ref_model_sync_steps",
        # TRL's DPOConfig also exposes activation_offloading (not on HF
        # base TrainingArguments).
        "activation_offloading",
        # TRL 1.x SimPO / CPO / ORPO head-specific tunables. Opaque has
        # them on its own DPOConfig with the same names and semantics.
        "simpo_gamma",
        "cpo_alpha",
        "orpo_lambda",
    }
)


TRL_DPO_RENAME_MAP: dict[str, str] = {}


_OPAQUE_DPO_LOSS_TYPES = frozenset(
    {
        "sigmoid",
        "hinge",
        "ipo",
        "robust",
        "exo_pair",
        "nca_pair",
        "bco_pair",
        "sppo_hard",
        "apo_zero",
        "apo_down",
        "discopop",
        "sft",
        "sigmoid_norm",
        # CPO / ORPO / SimPO are assembled specially in opaque but
        # appear in ``DPOConfig.loss_type`` as accepted values.
        "cpo",
        "orpo",
        "simpo",
    }
)


def _loss_type_transform(trl: dict[str, Any]) -> dict[str, Any]:
    """Validate every entry in ``loss_type`` is a head opaque implements.

    TRL 1.x added Adversarial Optimal Transport heads (``aot``,
    ``aot_unpaired``) that opaque does not implement. Reject those
    explicitly so the user knows opaque's surface is narrower than
    upstream TRL.
    """
    loss_type = trl.get("loss_type")
    if loss_type is None:
        return {}
    # TRL stores loss_type as list[str] in 1.x; coerce singletons.
    values = [loss_type] if isinstance(loss_type, str) else list(loss_type)
    unsupported = [v for v in values if v not in _OPAQUE_DPO_LOSS_TYPES]
    if unsupported:
        raise ValueError(
            f"trl_dpo_config.loss_type contains unsupported heads: "
            f"{sorted(set(unsupported))}. Opaque implements: "
            f"{sorted(_OPAQUE_DPO_LOSS_TYPES)}. The Adversarial Optimal "
            f"Transport family (``aot``, ``aot_unpaired``) added in TRL 1.x "
            f"is not in opaque."
        )
    return {"loss_type": values}


TRL_DPO_TRANSFORM_MAP: dict[str, Callable[[dict[str, Any]], dict[str, Any]]] = {
    "loss_type": _loss_type_transform,
}


TRL_DPO_REJECTED_FIELDS: dict[str, Callable[[Any], str | None]] = {
    "truncation_mode": _reject_truncation_mode,
    "padding_free": _reject_if_truthy(
        "``padding_free`` is not supported for DPO — see the SFT rationale."
    ),
    # NB: ``precompute_ref_log_probs`` is not flagged as REJECT here because
    # opaque's default of always-precompute matches TRL's "True" mode for
    # reference-using heads. A False setting in TRL doesn't translate, but
    # the trainer-side runtime check provides the user-facing error.
    "pad_token": _reject_pad_token,
}


TRL_DPO_DROP_FIELDS: dict[str, str] = {
    "precompute_ref_log_probs": (
        "Opaque always precomputes reference logps under DP for static-"
        "reference heads; this TRL flag is silently honored at its True "
        "mode and ignored otherwise."
    ),
}


def convert_trl_dpo_config(
    trl_cfg: Any,
    *,
    strict: bool = True,
    **dp_overrides: Any,
) -> dict[str, Any]:
    """Translate a ``trl.DPOConfig`` instance into opaque ``DPOConfig`` kwargs."""
    trl = _import_trl()
    if not isinstance(trl_cfg, trl.DPOConfig):
        raise TypeError(
            f"Expected ``trl.DPOConfig`` instance, got {type(trl_cfg).__name__}."
        )

    dp_layer = normalize_dp_overrides(dp_overrides)
    return _convert_trl_config(
        trl_cfg,
        trl_direct=TRL_DPO_DIRECT_FIELDS,
        trl_rename=TRL_DPO_RENAME_MAP,
        trl_transform=TRL_DPO_TRANSFORM_MAP,
        trl_reject=TRL_DPO_REJECTED_FIELDS,
        trl_drop=TRL_DPO_DROP_FIELDS,
        source_label="trl_dpo_config",
        strict=strict,
        dp_overrides=dp_layer,
    )
