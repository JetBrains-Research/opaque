"""``DPOTrainer`` — Direct Preference Optimization on :class:`DPTrainer`.

Mirrors ``trl.DPOTrainer`` (``trl/trainer/dpo_trainer.py``) in structure and
method names — ``_prepare_dataset`` / ``tokenize_row`` for data prep,
``compute_ref_log_probs`` for the reference pass, ``dpo_loss`` for the loss
dispatch — but routes the training gradient through Opaque's per-example
:meth:`DPTrainer.compute_per_example_loss` hook. TRL's batched
``concatenated_forward`` has no per-example DP meaning, so the two forwards
(chosen + rejected) are folded into the hook rather than kept as a standalone
batched method (plan §2.1a).

The reference policy enters via **precompute** (plan §3.2): a one-shot pass
attaches per-example ``ref_chosen_logps`` / ``ref_rejected_logps`` columns the
collator emits as constants, so the per-example loss reads them without a
second model inside ``vmap``.
"""

from __future__ import annotations

import dataclasses
from typing import Any, Callable

import torch

from opaque.alignment.dpo.collator import preference_collator
from opaque.alignment.dpo.data import extract_prompt
from opaque.alignment.dpo.loss import (
    apo_down_loss,
    apo_zero_loss,
    bco_loss,
    chosen_nll_loss,
    discopop_loss,
    exo_loss,
    f_divergence_remap,
    hinge_loss,
    ipo_loss,
    mpo_combine,
    nca_loss,
    robust_loss,
    sequence_logp,
    sigmoid_loss,
    sppo_loss,
    squarechipo_loss,
)
from opaque.alignment.dpo.reference import (
    compute_ref_logprobs_for_dataset,
    null_ref_context,
)
from opaque.api.transformers.trainer import DPTrainer

from ._dpo_config import DPOConfig

# TRL ``loss_type`` name → ``opaque.alignment.dpo.loss`` head. An unknown value
# (e.g. ``"aot"`` — batch-sort, no per-example DP meaning) raises a standard
# ``KeyError`` at dispatch, not a curated rejection (plan §3.3).
_DPO_HEADS: dict[str, Callable] = {
    "sigmoid": sigmoid_loss,
    "sigmoid_norm": sigmoid_loss,  # length-normalized via _NORM_LOSSES flag
    "hinge": hinge_loss,
    "ipo": ipo_loss,
    "robust": robust_loss,
    "apo_zero": apo_zero_loss,
    "apo_down": apo_down_loss,
    "exo_pair": exo_loss,
    "nca_pair": nca_loss,
    "bco_pair": bco_loss,
    "sppo_hard": sppo_loss,
    "discopop": discopop_loss,
    "squarechipo": squarechipo_loss,
    "sft": chosen_nll_loss,  # special-cased: consumes chosen_logp, not the ratio
}

# Loss variants that score length-normalized log-probs (per-token mean).
_NORM_LOSSES = frozenset({"ipo", "sigmoid_norm"})

_REF_COLUMNS = ("ref_chosen_logps", "ref_rejected_logps")


class DPOTrainer(DPTrainer):
    """DP Direct Preference Optimization trainer."""

    def __init__(
        self,
        model: Any = None,
        ref_model: Any = None,
        args: DPOConfig | None = None,
        data_collator: Callable | None = None,
        train_dataset: Any = None,
        eval_dataset: Any = None,
        processing_class: Any = None,
        compute_metrics: Callable | None = None,
        callbacks: list[Any] | None = None,
        optimizers: tuple[Any | None, Any | None] = (None, None),
        optimizer_cls_and_kwargs: tuple[Any, dict[str, Any]] | None = None,
        preprocess_logits_for_metrics: Callable | None = None,
        peft_config: Any = None,
    ) -> None:
        # ---- args normalization -------------------------------------------
        if args is None:
            args = DPOConfig(output_dir="trainer_output")
        elif not isinstance(args, DPOConfig):
            args = DPOConfig(
                **{f.name: getattr(args, f.name) for f in dataclasses.fields(args)}
            )

        if isinstance(model, str):
            from transformers import AutoModelForCausalLM

            model = AutoModelForCausalLM.from_pretrained(model)
        if model is ref_model and ref_model is not None:
            raise ValueError(
                "`model` and `ref_model` must be different objects (the reference "
                "is a frozen copy of the policy)."
            )

        # ---- capture loss configuration onto self -------------------------
        self._beta = float(args.beta)
        self._loss_type: list[str] = list(args.loss_type)
        self._loss_weights: list[float] = list(args.loss_weights)
        self._label_smoothing = float(args.label_smoothing)
        self._f_divergence_type = args.f_divergence_type
        self._f_alpha_coef = float(args.f_alpha_divergence_coef)
        self._ld_alpha = args.ld_alpha
        self._discopop_tau = float(args.discopop_tau)
        self._rpo_alpha = args.rpo_alpha
        self._reference_free = bool(args.reference_free)
        self._length_normalized = any(lt in _NORM_LOSSES for lt in self._loss_type)
        # Build the head dispatch eagerly so an unknown loss_type fails now.
        self._heads = [_DPO_HEADS[name] for name in self._loss_type]
        if args.use_weighting:
            # WPO is not DP-incompatible — just not wired yet (lands in the DPO
            # breadth phase). Fail with a plain not-implemented signal.
            raise NotImplementedError(
                "use_weighting (WPO) is not implemented yet; it lands in the "
                "DPO-breadth phase."
            )

        # ---- tokenizer ----------------------------------------------------
        processing_class = self._resolve_tokenizer(model, processing_class)
        self._pad_token_id = processing_class.pad_token_id

        # ---- dropout / PEFT ----------------------------------------------
        if args.disable_dropout:
            self._disable_dropout_in_model(model)
        if peft_config is not None:
            from peft import get_peft_model

            model = get_peft_model(model, peft_config)
        self._is_peft = _is_peft_model(model)

        # ---- collator (reused for precompute and training) ---------------
        if data_collator is None:
            data_collator = preference_collator(
                pad_token_id=self._pad_token_id,
                max_length=args.max_length,
                pad_to_multiple_of=args.pad_to_multiple_of,
            )

        # ---- tokenize datasets (before super) -----------------------------
        train_dataset = self._prepare_dataset(train_dataset, processing_class, args)
        if eval_dataset is not None and not isinstance(eval_dataset, dict):
            eval_dataset = self._prepare_dataset(eval_dataset, processing_class, args)

        # ---- reference precompute -----------------------------------------
        if not self._reference_free:
            self._precompute_device = args.device
            name_or_path = getattr(model.config, "_name_or_path", "")
            cache_key = ("dpo", name_or_path, self._length_normalized)
            batch_size = (
                args.precompute_ref_batch_size or args.per_device_train_batch_size
            )
            train_dataset = self._precompute_ref_logps(
                train_dataset,
                model=model,
                ref_model=ref_model,
                collator=data_collator,
                batch_size=batch_size,
                cache_key=cache_key,
                disable_dropout=args.disable_dropout,
            )
            if eval_dataset is not None and not isinstance(eval_dataset, dict):
                eval_dataset = self._precompute_ref_logps(
                    eval_dataset,
                    model=model,
                    ref_model=ref_model,
                    collator=data_collator,
                    batch_size=batch_size,
                    cache_key=cache_key + ("eval",),
                    disable_dropout=args.disable_dropout,
                )

        super().__init__(
            model=model,
            args=args,
            data_collator=data_collator,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=processing_class,
            compute_metrics=compute_metrics,
            callbacks=callbacks,
            optimizers=optimizers,
            optimizer_cls_and_kwargs=optimizer_cls_and_kwargs,
            preprocess_logits_for_metrics=preprocess_logits_for_metrics,
        )

    # ------------------------------------------------------------------
    # Tokenizer / dropout helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _resolve_tokenizer(model: Any, processing_class: Any) -> Any:
        if processing_class is None:
            from transformers import AutoTokenizer

            name = getattr(model.config, "_name_or_path", None)
            if not name:
                raise ValueError(
                    "processing_class is None and the model config has no "
                    "_name_or_path; pass processing_class."
                )
            processing_class = AutoTokenizer.from_pretrained(name)
        if processing_class.pad_token_id is None:
            processing_class.pad_token = processing_class.eos_token
        return processing_class

    @staticmethod
    def _disable_dropout_in_model(model: Any) -> None:
        for module in model.modules():
            if isinstance(module, torch.nn.Dropout):
                module.p = 0.0

    # ------------------------------------------------------------------
    # Dataset preparation (TRL-shaped)
    # ------------------------------------------------------------------
    def _prepare_dataset(
        self, dataset: Any, processing_class: Any, args: DPOConfig
    ) -> Any:
        if dataset is None:
            return None
        column_names = list(getattr(dataset, "column_names", []) or [])
        if "chosen_input_ids" in column_names:
            return dataset  # already tokenized
        # Extract the shared prompt prefix when only (chosen, rejected) are given.
        if "prompt" not in column_names:
            dataset = dataset.map(extract_prompt, num_proc=args.dataset_num_proc)
            column_names = list(dataset.column_names)

        max_length = args.max_length

        def tokenize_row(example: dict) -> dict:
            return self.tokenize_row(example, processing_class, max_length)

        return dataset.map(
            tokenize_row,
            remove_columns=column_names,
            num_proc=args.dataset_num_proc,
            desc="Tokenizing preference dataset",
        )

    def tokenize_row(
        self, example: dict, processing_class: Any, max_length: int | None
    ) -> dict:
        """Tokenize one preference example into the collator's input schema.

        Produces ``chosen_input_ids`` / ``rejected_input_ids`` (prompt +
        completion) and the matching ``*_completion_mask`` (``0`` over prompt,
        ``1`` over completion). Lifted from ``examples/train_dpo.py``.
        """
        prompt = example.get("prompt", [])
        chosen = example["chosen"]
        rejected = example["rejected"]

        def _apply_template(messages: Any) -> list[int]:
            if isinstance(messages, list):
                return processing_class.apply_chat_template(
                    messages, tokenize=True, add_generation_prompt=False
                )
            return processing_class.encode(messages, add_special_tokens=False)

        if isinstance(prompt, list) and prompt:
            prompt_ids = processing_class.apply_chat_template(
                prompt, tokenize=True, add_generation_prompt=True
            )
        elif isinstance(prompt, str) and prompt:
            prompt_ids = processing_class.encode(prompt, add_special_tokens=True)
        else:
            prompt_ids = []

        if isinstance(chosen, list):
            chosen_ids = _apply_template(prompt + chosen)
            rejected_ids = _apply_template(prompt + rejected)
        else:
            if prompt_ids:
                chosen_ids = prompt_ids + processing_class.encode(
                    chosen, add_special_tokens=False
                )
                rejected_ids = prompt_ids + processing_class.encode(
                    rejected, add_special_tokens=False
                )
            else:
                chosen_ids = processing_class.encode(chosen, add_special_tokens=True)
                rejected_ids = processing_class.encode(rejected, add_special_tokens=True)

        prompt_len = len(prompt_ids)
        chosen_cmask = [0] * min(prompt_len, len(chosen_ids)) + [1] * max(
            0, len(chosen_ids) - prompt_len
        )
        rejected_cmask = [0] * min(prompt_len, len(rejected_ids)) + [1] * max(
            0, len(rejected_ids) - prompt_len
        )

        if max_length is not None:
            chosen_ids = chosen_ids[:max_length]
            rejected_ids = rejected_ids[:max_length]
            chosen_cmask = chosen_cmask[:max_length]
            rejected_cmask = rejected_cmask[:max_length]

        return {
            "chosen_input_ids": chosen_ids,
            "rejected_input_ids": rejected_ids,
            "chosen_completion_mask": chosen_cmask,
            "rejected_completion_mask": rejected_cmask,
        }

    # ------------------------------------------------------------------
    # Reference log-probs (precompute; outside vmap)
    # ------------------------------------------------------------------
    def _precompute_ref_logps(
        self,
        dataset: Any,
        *,
        model: Any,
        ref_model: Any,
        collator: Callable,
        batch_size: int,
        cache_key: tuple,
        disable_dropout: bool,
    ) -> Any:
        """Attach ``ref_{chosen,rejected}_logps`` columns via a one-shot pass.

        Resolves the reference per plan §3.2: an explicit ``ref_model``, the
        PEFT base model (adapter disabled via ``null_ref_context``), or an
        auto-loaded copy of the policy.
        """
        device = self._precompute_device
        null_ref = False
        if ref_model is not None:
            ref = ref_model
        elif self._is_peft:
            ref = model  # base model with the adapter disabled at forward time
            null_ref = True
        else:
            from transformers import AutoModelForCausalLM

            ref = AutoModelForCausalLM.from_pretrained(
                getattr(model.config, "_name_or_path")
            )
        if disable_dropout:
            self._disable_dropout_in_model(ref)

        was_training = ref.training
        ref.to(device).eval()

        def ref_callable(batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
            return self.compute_ref_log_probs(batch, ref, device, null_ref=null_ref)

        dataset = compute_ref_logprobs_for_dataset(
            dataset,
            ref=ref_callable,
            collator=collator,
            output_columns=_REF_COLUMNS,
            batch_size=batch_size,
            cache_key=cache_key,
        )

        # Free the reference: training reads the precomputed columns, not the
        # model. For the PEFT null-ref the "ref" *is* the policy — restore it.
        if null_ref:
            if was_training:
                ref.train()
        else:
            ref.to("cpu")
        return dataset

    def compute_ref_log_probs(
        self,
        batch: dict[str, torch.Tensor],
        ref_model: Any,
        device: Any,
        *,
        null_ref: bool,
    ) -> dict[str, torch.Tensor]:
        """Reference chosen/rejected sequence logps for one collated batch."""
        cids = batch["chosen_input_ids"].to(device)
        cmask = batch["chosen_attention_mask"].to(device)
        ccmask = batch["chosen_completion_mask"].to(device)
        rids = batch["rejected_input_ids"].to(device)
        rmask = batch["rejected_attention_mask"].to(device)
        rcmask = batch["rejected_completion_mask"].to(device)

        def forward(ids: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
            return ref_model(input_ids=ids, attention_mask=mask).logits

        if null_ref:
            with null_ref_context(ref_model):
                c_logits = forward(cids, cmask)
                r_logits = forward(rids, rmask)
        else:
            c_logits = forward(cids, cmask)
            r_logits = forward(rids, rmask)

        ref_chosen = sequence_logp(
            c_logits, cids, ccmask, length_normalized=self._length_normalized
        )
        ref_rejected = sequence_logp(
            r_logits, rids, rcmask, length_normalized=self._length_normalized
        )
        return {
            "ref_chosen_logps": ref_chosen.detach().float().cpu(),
            "ref_rejected_logps": ref_rejected.detach().float().cpu(),
        }

    # ------------------------------------------------------------------
    # Loss
    # ------------------------------------------------------------------
    def _head_kwargs(self, name: str) -> dict[str, Any]:
        if name in ("sigmoid", "sigmoid_norm", "robust", "exo_pair"):
            return {"label_smoothing": self._label_smoothing}
        if name == "discopop":
            return {"discopop_tau": self._discopop_tau}
        return {}

    def dpo_loss(
        self,
        chosen_logratio: torch.Tensor,
        rejected_logratio: torch.Tensor,
        *,
        chosen_logp: torch.Tensor,
    ) -> torch.Tensor:
        """Combine the configured loss head(s) into one per-example scalar.

        ``chosen_logratio`` / ``rejected_logratio`` are ``policy_logp -
        ref_logp`` (or the policy logp itself when ``reference_free``). The
        ``sft`` head and RPO regulariser consume ``chosen_logp`` directly.
        """
        if self._f_divergence_type != "reverse_kl":
            chosen_logratio = f_divergence_remap(
                chosen_logratio,
                f_divergence_type=self._f_divergence_type,
                alpha=self._f_alpha_coef,
            )
            rejected_logratio = f_divergence_remap(
                rejected_logratio,
                f_divergence_type=self._f_divergence_type,
                alpha=self._f_alpha_coef,
            )

        parts: dict[str, torch.Tensor] = {}
        weights: dict[str, float] = {}
        for name, weight, head in zip(
            self._loss_type, self._loss_weights, self._heads
        ):
            if name == "sft":
                parts[name] = chosen_nll_loss(chosen_logp)
            else:
                parts[name] = head(
                    chosen_logratio,
                    rejected_logratio,
                    beta=self._beta,
                    **self._head_kwargs(name),
                )
            weights[name] = weight
        if self._rpo_alpha:
            parts["rpo_nll"] = chosen_nll_loss(chosen_logp)
            weights["rpo_nll"] = float(self._rpo_alpha)
        return mpo_combine(parts, weights)

    def compute_per_example_loss(
        self,
        fmodel: Callable[..., Any],
        params: dict[str, Any],
        inputs: dict[str, Any],
        *,
        return_logits: bool = False,
    ) -> Any:
        """One preference pair's DPO loss (vmap-batched by :class:`DPTrainer`).

        Two policy forwards (chosen, rejected) → per-sequence logps → the
        configured head(s). The reference enters as the precomputed constant
        ``ref_*_logps`` (read from ``inputs``), so no second model runs inside
        ``vmap``.
        """
        chosen_out = fmodel(
            params,
            input_ids=inputs["chosen_input_ids"],
            attention_mask=inputs["chosen_attention_mask"],
        )
        rejected_out = fmodel(
            params,
            input_ids=inputs["rejected_input_ids"],
            attention_mask=inputs["rejected_attention_mask"],
        )
        chosen_logp = sequence_logp(
            chosen_out.logits,
            inputs["chosen_input_ids"],
            inputs["chosen_completion_mask"],
            ld_alpha=self._ld_alpha,
            length_normalized=self._length_normalized,
        )
        rejected_logp = sequence_logp(
            rejected_out.logits,
            inputs["rejected_input_ids"],
            inputs["rejected_completion_mask"],
            ld_alpha=self._ld_alpha,
            length_normalized=self._length_normalized,
        )

        if self._reference_free:
            chosen_lr, rejected_lr = chosen_logp, rejected_logp
        else:
            chosen_lr = chosen_logp - inputs["ref_chosen_logps"]
            rejected_lr = rejected_logp - inputs["ref_rejected_logps"]

        loss = self.dpo_loss(chosen_lr, rejected_lr, chosen_logp=chosen_logp)
        if return_logits:
            return loss, chosen_out.logits
        return loss


def _is_peft_model(model: Any) -> bool:
    """Best-effort PEFT detection (mirrors DPTrainer's helper)."""
    try:
        from peft import PeftModel

        return isinstance(model, PeftModel)
    except Exception:
        return False
