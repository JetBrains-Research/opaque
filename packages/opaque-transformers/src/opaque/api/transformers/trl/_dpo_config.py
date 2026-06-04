"""``DPOConfig`` — training arguments for :class:`DPOTrainer`.

Mirrors ``trl.DPOConfig`` (``trl/trainer/dpo_config.py:211-304``) for the subset
that is meaningful under per-example DP, extending Opaque's standalone
:class:`~opaque.api.transformers.trainer._config.TrainingArguments`.

Design philosophy (plan §3.3): no bespoke "rejection" code. Batch-coupled
losses (``aot`` / ``aot_unpaired``) have no per-example DP meaning — there is no
``aot`` head in ``opaque-alignment``, so ``loss_type=["aot"]`` fails with an
ordinary ``KeyError`` at the trainer's dispatch table. TR-DPO sync
(``sync_ref_model`` …) and ``padding_free`` are absent from this surface (they
land in iteration 2), so passing them is a standard unexpected-keyword
``TypeError``. TRL's *own* faithful validations (label-smoothing bounds) are
kept, since iteration 1 mirrors TRL.
"""

from __future__ import annotations

import dataclasses

from opaque.api.transformers.trainer._config import TrainingArguments


@dataclasses.dataclass
class DPOConfig(TrainingArguments):
    """Arguments for Direct Preference Optimization on :class:`DPTrainer`."""

    # ---- Learning rate override (TRL default differs from HF) ------------
    learning_rate: float = 1e-6  # dpo_config.py:142

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
    #: RPO chosen-NLL regulariser weight; ``None`` ⇒ off.
    rpo_alpha: float | None = None
    #: DiscoPOP temperature.
    discopop_tau: float = 0.05  # dpo_config.py:275
    #: Score against the policy's own logp instead of a reference (CPO/ORPO/SimPO).
    reference_free: bool = False

    # ---- Reference model -------------------------------------------------
    #: Batch size for the reference precompute pass; defaults to the train
    #: per-device batch size when ``None``. (There is no ``precompute_ref_log_probs``
    #: toggle: under the per-example DP substrate a *static* reference is always
    #: precomputed — the ``vmap`` loss can only read it as a constant column —
    #: so the toggle would be misleading. ``reference_free`` skips it; TR-DPO
    #: (``sync_ref_model``) recomputes per step.)
    precompute_ref_batch_size: int | None = None  # dpo_config.py:201
    #: Disable dropout in policy (and reference) before training.
    disable_dropout: bool = True  # dpo_config.py:155

    # ---- TR-DPO (reference sync, arXiv:2502.18014) -----------------------
    #: Periodically move the reference toward the policy by an EMA step
    #: (recomputed per training step, outside vmap). Incompatible with
    #: ``reference_free``; full fine-tuning only (not PEFT, mirroring TRL).
    sync_ref_model: bool = False  # dpo_config.py:287
    #: EMA mixup α: ``ref ← (1 - α)·ref + α·policy``.
    ref_model_mixup_alpha: float = 0.6  # dpo_config.py:296
    #: Apply the EMA update every ``ref_model_sync_steps`` steps.
    ref_model_sync_steps: int = 512  # dpo_config.py:304

    # ---- Data preparation ------------------------------------------------
    #: Maximum tokenized sequence length; ``None`` disables truncation.
    max_length: int | None = 1024  # dpo_config.py:165
    #: Pad the collated batch length up to a multiple of this value.
    pad_to_multiple_of: int | None = None  # dpo_config.py:189
    #: Number of processes for ``datasets.map`` during preprocessing.
    dataset_num_proc: int | None = None  # dpo_config.py:161

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

        # TRL's own faithful validations (dpo_trainer.py:680-694). Not
        # DP-driven rejections — kept because iteration 1 mirrors TRL.
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
        if self.sync_ref_model and self.reference_free:
            raise ValueError(
                "sync_ref_model (TR-DPO) is incompatible with reference_free: "
                "there is no reference to sync."
            )
        # DPO eval must route through the per-example loss path: a preference
        # batch has ``chosen_input_ids`` / ``rejected_input_ids`` (no single
        # ``input_ids``), so the standard single-forward prediction path can't
        # run. ``'loss' in include_for_metrics`` selects the per-example path.
        if "loss" not in self.include_for_metrics:
            self.include_for_metrics = [*self.include_for_metrics, "loss"]
