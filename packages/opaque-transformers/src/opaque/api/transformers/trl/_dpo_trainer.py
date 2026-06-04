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
second model inside ``vmap``. TR-DPO (``sync_ref_model``) instead recomputes the
reference logps each step from an EMA reference, via the
:meth:`DPTrainer._augment_inputs` pre-``vmap`` hook.
"""

from __future__ import annotations

import copy
import dataclasses
import uuid
from collections.abc import Mapping
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
    wpo_weights,
)
from opaque.alignment.dpo.metric import reward_metrics
from opaque.alignment.dpo.reference import (
    compute_ref_logprobs_for_dataset,
    ema_update_reference,
    null_ref_context,
)
from opaque.api.transformers.trainer import DPTrainer

# Single source of truth for PEFT detection (handles PeftModel + PeftMixedModel).
from opaque.api.transformers.trainer._dp_trainer import _is_peft_model

from ._dpo_config import DPOConfig

# TRL ``loss_type`` name → ``opaque.alignment.dpo.loss`` head. An unknown value
# (e.g. ``"aot"`` — batch-sort, no per-example DP meaning) raises a standard
# ``KeyError`` at dispatch, not a curated rejection (plan §3.3).
_DPO_HEADS: dict[str, Callable] = {
    "sigmoid": sigmoid_loss,
    "sigmoid_norm": sigmoid_loss,  # length-normalized log-ratio (see dpo_loss)
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

# Loss variants that score the *length-normalized* log-ratio (per-token mean).
# Normalization is applied per head in ``dpo_loss`` so an MPO list may mix
# normalized and summed variants (the reference is always precomputed summed).
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
        self._use_weighting = bool(args.use_weighting)
        # Per-head normalization (mixed MPO supported); the reference is always
        # precomputed summed, and normalized log-ratios are derived in dpo_loss.
        self._any_norm = any(lt in _NORM_LOSSES for lt in self._loss_type)
        # Emit per-example DPO reward telemetry through the clipped-grad aux
        # channel so ``rewards/*`` are logged every training step (TR-parity),
        # computed in the same clipped forward (live weights, DDP-synced).
        self._emit_loss_aux = True
        # Build the head dispatch eagerly so an unknown loss_type fails now.
        self._heads = [_DPO_HEADS[name] for name in self._loss_type]

        # ---- TR-DPO ----
        self._sync_ref_model = bool(args.sync_ref_model)
        self._ref_mixup_alpha = float(args.ref_model_mixup_alpha)
        self._ref_sync_steps = int(args.ref_model_sync_steps)
        self._tr_ref: Any = None  # EMA reference module (kept on device)

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
        if self._sync_ref_model and self._is_peft:
            raise ValueError(
                "sync_ref_model (TR-DPO) requires full fine-tuning, not PEFT "
                "(the EMA reference tracks the full policy)."
            )

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

        # ---- reference logps (precompute, or seed for TR-DPO) -------------
        # Cache fingerprint: include max_length (the collator truncates to it at
        # precompute time, so different lengths must not alias) and a stable
        # model id. For in-memory models (empty _name_or_path) use a per-run
        # nonce so distinct models never collide on the cache (review feedback).
        self._precompute_device = args.device
        name_or_path = getattr(model.config, "_name_or_path", "") or ""
        model_id = name_or_path or f"inmemory-{uuid.uuid4().hex}"
        cache_key = ("dpo", model_id, args.max_length)
        batch_size = args.precompute_ref_batch_size or args.per_device_train_batch_size

        if self._sync_ref_model:
            train_dataset, eval_dataset = self._setup_tr_dpo(
                model,
                ref_model,
                train_dataset,
                eval_dataset,
                collator=data_collator,
                batch_size=batch_size,
                disable_dropout=args.disable_dropout,
            )
        elif not self._reference_free:
            train_dataset = self._precompute_ref_logps(
                train_dataset,
                model=model,
                ref_model=ref_model,
                model_id=model_id,
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
                    model_id=model_id,
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
                rejected_ids = processing_class.encode(
                    rejected, add_special_tokens=True
                )

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
    # Reference log-probs (precompute / TR-DPO; outside vmap)
    # ------------------------------------------------------------------
    def _precompute_ref_logps(
        self,
        dataset: Any,
        *,
        model: Any,
        ref_model: Any,
        model_id: str,
        collator: Callable,
        batch_size: int,
        cache_key: tuple,
        disable_dropout: bool,
    ) -> Any:
        """Attach ``ref_{chosen,rejected}_logps`` columns via a one-shot pass.

        Resolves the reference per plan §3.2: an explicit ``ref_model``, the
        PEFT base model (adapter disabled via ``null_ref_context``), or an
        auto-loaded copy of the policy. A user-supplied ``ref_model`` is left in
        the device/mode it started in (review feedback); only an auto-loaded
        copy is freed to CPU.
        """
        null_ref = False
        owns_ref = False  # auto-loaded by us → safe to drop / move to CPU
        if ref_model is not None:
            ref = ref_model
        elif self._is_peft:
            ref = model  # base model with the adapter disabled at forward time
            null_ref = True
        else:
            from transformers import AutoModelForCausalLM

            if not model_id or model_id.startswith("inmemory-"):
                raise ValueError(
                    "No reference available: pass ref_model=, use a PEFT policy, "
                    "set reference_free=True, or load the policy from a path so a "
                    "reference copy can be auto-loaded."
                )
            ref = AutoModelForCausalLM.from_pretrained(model_id)
            owns_ref = True

        orig_device = next(ref.parameters()).device
        was_training = ref.training
        if disable_dropout:
            self._disable_dropout_in_model(ref)
        ref.to(self._precompute_device).eval()

        def ref_callable(batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
            return self.compute_ref_log_probs(batch, ref, null_ref=null_ref)

        dataset = compute_ref_logprobs_for_dataset(
            dataset,
            ref=ref_callable,
            collator=collator,
            output_columns=_REF_COLUMNS,
            batch_size=batch_size,
            cache_key=cache_key,
        )

        # Restore caller-visible state. Training reads the cached columns, not
        # the model, so we don't keep the reference around.
        if null_ref:
            if was_training:
                ref.train()
        elif owns_ref:
            ref.to("cpu")
        else:
            ref.to(orig_device)  # explicit user ref_model: leave as we found it
            if was_training:
                ref.train()
        return dataset

    def _setup_tr_dpo(
        self,
        model: Any,
        ref_model: Any,
        train_dataset: Any,
        eval_dataset: Any,
        *,
        collator: Callable,
        batch_size: int,
        disable_dropout: bool,
    ) -> tuple[Any, Any]:
        """Build the EMA reference (kept on device) and seed the ref columns.

        TR-DPO recomputes the reference logps each step (``_augment_inputs``), so
        the seed values don't matter for correctness — they only populate the
        columns so ``batch_keys`` includes them. The seed is therefore not cached
        (a per-run nonce in the cache key).
        """
        ref = ref_model if ref_model is not None else copy.deepcopy(model)
        if disable_dropout:
            self._disable_dropout_in_model(ref)
        ref.to(self._precompute_device).eval()
        for p in ref.parameters():
            p.requires_grad_(False)
        self._tr_ref = ref

        nonce = ("trdpo-seed", uuid.uuid4().hex)

        def seed(dataset: Any, suffix: tuple) -> Any:
            if dataset is None or isinstance(dataset, dict):
                return dataset
            return compute_ref_logprobs_for_dataset(
                dataset,
                ref=lambda b: self.compute_ref_log_probs(b, ref, null_ref=False),
                collator=collator,
                output_columns=_REF_COLUMNS,
                batch_size=batch_size,
                cache_key=nonce + suffix,
            )

        return seed(train_dataset, ()), seed(eval_dataset, ("eval",))

    def compute_ref_log_probs(
        self,
        batch: dict[str, torch.Tensor],
        ref_model: Any,
        *,
        null_ref: bool,
        to_cpu: bool = True,
    ) -> dict[str, torch.Tensor]:
        """Reference chosen/rejected *summed* sequence logps for one batch.

        Always summed (``length_normalized=False``); ``dpo_loss`` derives any
        length-normalized log-ratio by dividing by the completion length, so a
        single precompute serves both normalized and summed losses. With
        ``to_cpu=True`` (precompute) returns CPU float tensors; with
        ``to_cpu=False`` (the per-step TR-DPO path) returns device tensors.
        """
        device = next(ref_model.parameters()).device
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

        ref_chosen = sequence_logp(c_logits, cids, ccmask)
        ref_rejected = sequence_logp(r_logits, rids, rcmask)
        if to_cpu:
            return {
                "ref_chosen_logps": ref_chosen.detach().float().cpu(),
                "ref_rejected_logps": ref_rejected.detach().float().cpu(),
            }
        return {
            "ref_chosen_logps": ref_chosen.detach(),
            "ref_rejected_logps": ref_rejected.detach(),
        }

    # ------------------------------------------------------------------
    # TR-DPO: per-step reference refresh + EMA update (outside vmap)
    # ------------------------------------------------------------------
    def _augment_inputs(
        self, inputs: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        if not self._sync_ref_model or self._tr_ref is None:
            return inputs
        step = int(self.state.global_step)
        if step > 0 and step % self._ref_sync_steps == 0:
            self._ema_update_reference()
        with torch.no_grad():
            refs = self.compute_ref_log_probs(
                inputs, self._tr_ref, null_ref=False, to_cpu=False
            )
        device = inputs["chosen_input_ids"].device
        return {
            **inputs,
            "ref_chosen_logps": refs["ref_chosen_logps"].to(device),
            "ref_rejected_logps": refs["ref_rejected_logps"].to(device),
        }

    def _ema_update_reference(self) -> None:
        """``ref ← (1-α)·ref + α·policy`` over the live functional policy params."""
        ctx = self._ctx
        if ctx is None:
            return
        policy = {**ctx.frozen_params, **ctx.trainable_params}
        ref_named = dict(self._tr_ref.named_parameters())
        keys = [k for k in ref_named if k in policy]
        with torch.no_grad():
            updated = ema_update_reference(
                {k: ref_named[k].detach() for k in keys},
                {k: policy[k].detach().to(ref_named[k].device) for k in keys},
                self._ref_mixup_alpha,
            )
            for k in keys:
                ref_named[k].data.copy_(updated[k])

    # ------------------------------------------------------------------
    # Loss
    # ------------------------------------------------------------------
    def _head_kwargs(self, name: str) -> dict[str, Any]:
        if name in ("sigmoid", "sigmoid_norm", "robust", "exo_pair"):
            return {"label_smoothing": self._label_smoothing}
        if name == "discopop":
            return {"discopop_tau": self._discopop_tau}
        return {}

    @staticmethod
    def _completion_len(completion_mask: torch.Tensor) -> torch.Tensor:
        """Shifted completion-token count (matches ``sequence_logp``'s divisor)."""
        return (completion_mask[..., 1:] != 0).sum(-1).clamp(min=1)

    @staticmethod
    def _ld_shared_prefix(
        chosen_completion_mask: torch.Tensor,
        rejected_completion_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Per-side LD-DPO shared-prefix length (shifted): prompt + shared span.

        Tokens up to ``shared`` (the shorter of the two completions) keep weight
        ``1``; the verbose tail is damped by ``ld_alpha`` inside ``sequence_logp``.
        """
        c = chosen_completion_mask[..., 1:]
        r = rejected_completion_mask[..., 1:]
        shared = torch.minimum((c != 0).sum(-1), (r != 0).sum(-1))
        c_start = torch.argmax((c != 0).to(torch.int32), dim=-1)
        r_start = torch.argmax((r != 0).to(torch.int32), dim=-1)
        return c_start + shared, r_start + shared

    def dpo_loss(
        self,
        chosen_logratio: torch.Tensor,
        rejected_logratio: torch.Tensor,
        *,
        chosen_logp: torch.Tensor,
        chosen_completion_mask: torch.Tensor,
        rejected_completion_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Combine the configured loss head(s) into one per-example scalar.

        ``chosen_logratio`` / ``rejected_logratio`` are the *summed*
        ``policy_logp - ref_logp`` (or the policy logp itself when
        ``reference_free``). Length-normalized heads (``ipo`` / ``sigmoid_norm``)
        use the log-ratio divided by the completion length — so an MPO list may
        mix normalized and summed variants. The ``sft`` head and RPO regulariser
        consume ``chosen_logp`` directly.
        """
        chosen_norm = rejected_norm = None
        if self._any_norm:
            chosen_norm = chosen_logratio / self._completion_len(chosen_completion_mask)
            rejected_norm = rejected_logratio / self._completion_len(
                rejected_completion_mask
            )

        parts: dict[str, torch.Tensor] = {}
        weights: dict[str, float] = {}
        for name, weight, head in zip(self._loss_type, self._loss_weights, self._heads):
            if name == "sft":
                parts[name] = chosen_nll_loss(chosen_logp)
                weights[name] = weight
                continue
            if name in _NORM_LOSSES:
                clr, rlr = chosen_norm, rejected_norm
            else:
                clr, rlr = chosen_logratio, rejected_logratio
            if self._f_divergence_type != "reverse_kl":
                clr = f_divergence_remap(
                    clr,
                    f_divergence_type=self._f_divergence_type,
                    alpha=self._f_alpha_coef,
                )
                rlr = f_divergence_remap(
                    rlr,
                    f_divergence_type=self._f_divergence_type,
                    alpha=self._f_alpha_coef,
                )
            parts[name] = head(clr, rlr, beta=self._beta, **self._head_kwargs(name))
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
        return_aux: bool = False,
    ) -> Any:
        """One preference pair's DPO loss (vmap-batched by :class:`DPTrainer`).

        Two policy forwards (chosen, rejected) → per-sequence summed logps → the
        configured head(s). The reference enters as the constant ``ref_*_logps``
        (precomputed, or TR-DPO's per-step values injected by
        :meth:`_augment_inputs`), so no second model runs inside ``vmap``.

        With ``return_aux=True`` also returns the per-example ``rewards/*``
        telemetry dict (the clipped-grad aux channel), which ``training_step``
        means + logs each step; the eval reward pass reuses the same producer.
        """
        c_cmask = inputs["chosen_completion_mask"]
        r_cmask = inputs["rejected_completion_mask"]

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

        c_lp_kwargs: dict[str, Any] = {}
        r_lp_kwargs: dict[str, Any] = {}
        if self._ld_alpha is not None:
            c_sp, r_sp = self._ld_shared_prefix(c_cmask, r_cmask)
            c_lp_kwargs = {"ld_alpha": self._ld_alpha, "shared_prefix_len": c_sp}
            r_lp_kwargs = {"ld_alpha": self._ld_alpha, "shared_prefix_len": r_sp}

        chosen_logp = sequence_logp(
            chosen_out.logits, inputs["chosen_input_ids"], c_cmask, **c_lp_kwargs
        )
        rejected_logp = sequence_logp(
            rejected_out.logits, inputs["rejected_input_ids"], r_cmask, **r_lp_kwargs
        )

        if self._reference_free:
            chosen_lr, rejected_lr = chosen_logp, rejected_logp
        else:
            chosen_lr = chosen_logp - inputs["ref_chosen_logps"]
            rejected_lr = rejected_logp - inputs["ref_rejected_logps"]

        loss = self.dpo_loss(
            chosen_lr,
            rejected_lr,
            chosen_logp=chosen_logp,
            chosen_completion_mask=c_cmask,
            rejected_completion_mask=r_cmask,
        )

        # WPO (arXiv:2406.11827): reweight by the policy's detached average
        # completion probability on each side. The weight is per-example and
        # carries no gradient (``wpo_weights`` detaches), so per-example DP is
        # preserved.
        if self._use_weighting:
            weight = self._wpo_weight(
                chosen_out.logits, inputs["chosen_input_ids"], c_cmask
            ) * self._wpo_weight(
                rejected_out.logits, inputs["rejected_input_ids"], r_cmask
            )
            loss = loss * weight

        if return_aux:
            return loss, self._reward_aux(chosen_lr, rejected_lr)
        if return_logits:
            return loss, chosen_out.logits
        return loss

    def _reward_aux(
        self, chosen_logratio: torch.Tensor, rejected_logratio: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        """Per-example ``rewards/*`` components (detached telemetry).

        These are the per-example quantities ``reward_metrics`` averages; the
        trainer means them across the (DDP-synced) batch when logging.
        """
        beta = self._beta
        return {
            "rewards/chosen": (beta * chosen_logratio).detach(),
            "rewards/rejected": (beta * rejected_logratio).detach(),
            "rewards/accuracies": (chosen_logratio > rejected_logratio)
            .to(chosen_logratio.dtype)
            .detach(),
            "rewards/margins": (beta * (chosen_logratio - rejected_logratio)).detach(),
        }

    @staticmethod
    def _wpo_weight(
        logits: torch.Tensor,
        input_ids: torch.Tensor,
        completion_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Detached WPO weight for one side (causal-LM shift + completion mask)."""
        shifted_logits = logits[..., :-1, :]
        shifted_ids = input_ids[..., 1:]
        shifted_mask = completion_mask[..., 1:]
        # Per-token logp of the realised next token (public log_softmax + gather;
        # equivalent to selective_log_softmax, kept on the public boundary).
        per_token_logps = (
            torch.log_softmax(shifted_logits, dim=-1)
            .gather(-1, shifted_ids.unsqueeze(-1))
            .squeeze(-1)
        )
        return wpo_weights(per_token_logps, shifted_mask)

    def get_batch_loss_metrics(
        self,
        chosen_logp: torch.Tensor,
        rejected_logp: torch.Tensor,
        ref_chosen_logps: torch.Tensor | None = None,
        ref_rejected_logps: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Reward telemetry for a batch of logps (eager; for eval logging).

        Mirrors ``trl.DPOTrainer.get_batch_loss_metrics``' reward block: returns
        chosen/rejected rewards, their margin, and the preference accuracy via
        :func:`opaque.alignment.dpo.metric.reward_metrics`. Detached — telemetry
        only, never released.
        """
        if self._reference_free or ref_chosen_logps is None:
            chosen_lr, rejected_lr = chosen_logp, rejected_logp
        else:
            chosen_lr = chosen_logp - ref_chosen_logps
            rejected_lr = rejected_logp - ref_rejected_logps
        return reward_metrics(chosen_lr, rejected_lr, beta=self._beta)

    # ------------------------------------------------------------------
    # Evaluation (DPO-shaped: own functional loss + reward pass)
    # ------------------------------------------------------------------
    def evaluate(
        self,
        eval_dataset: Any = None,
        ignore_keys: list[str] | None = None,
        metric_key_prefix: str = "eval",
    ) -> dict[str, float]:
        """Functional DPO eval — ``{prefix}_loss`` + ``{prefix}_rewards/*``.

        A preference batch carries ``chosen_input_ids`` / ``rejected_input_ids``
        (no single ``input_ids`` / ``labels``), so the inherited LM-shaped
        ``prediction_step`` can't run it. This computes the loss and reward
        telemetry directly through :meth:`compute_per_example_loss`
        (``return_aux``) using the **same** functional context the training step
        uses (``self._ctx``) — so in-training eval scores the *live* policy
        weights, not the stale ``self.model`` — then logs + fires ``on_evaluate``
        via :meth:`DPTrainer._after_evaluate`.
        """
        dataset = eval_dataset if eval_dataset is not None else self._eval_dataset
        if dataset is None:
            raise ValueError("DPOTrainer.evaluate() requires an eval_dataset.")
        # Multi-dataset eval: recurse per split with a namespaced prefix.
        if isinstance(dataset, Mapping):
            merged: dict[str, float] = {}
            for name, sub in dataset.items():
                merged.update(
                    self.evaluate(
                        eval_dataset=sub,
                        ignore_keys=ignore_keys,
                        metric_key_prefix=f"{metric_key_prefix}_{name}",
                    )
                )
            return merged

        metrics = self._dpo_eval_metrics(dataset, metric_key_prefix)
        self._after_evaluate(metrics)
        return metrics

    def _dpo_eval_metrics(
        self, dataset: Any, metric_key_prefix: str
    ) -> dict[str, float]:
        """Mean loss + ``rewards/*`` over ``dataset`` (functional, live weights)."""
        from opaque.distributed import gather_for_metrics
        from opaque.functional import make_functional

        ctx = self._ctx
        if ctx is not None:
            fmodel, frozen, keys = ctx.fmodel, ctx.frozen_params, ctx.batch_keys
            trainable = ctx.trainable_params
        else:
            fmodel, trainable, frozen = make_functional(
                self._model, partition_trainable=True
            )
            keys = self._discover_batch_keys()
        params = {**frozen, **trainable}

        def per_example(p, *batch_args):
            inputs = dict(zip(keys, batch_args))
            return self.compute_per_example_loss(fmodel, p, inputs, return_aux=True)

        vmapped = torch.vmap(per_example, in_dims=(None,) + (0,) * len(keys))

        amp_dtype = self._amp_dtype
        device_type = self._device.type
        loader = self.get_eval_dataloader(dataset)
        losses: list[torch.Tensor] = []
        collected: dict[str, list[torch.Tensor]] = {}
        was_training = self._model.training
        if was_training:
            self._model.eval()
        try:
            with torch.no_grad():
                for batch in loader:
                    batch = self._prepare_input(batch)
                    batch_args = tuple(batch[k] for k in keys)
                    if amp_dtype is not None:
                        with torch.autocast(device_type=device_type, dtype=amp_dtype):
                            loss, aux = vmapped(params, *batch_args)
                    else:
                        loss, aux = vmapped(params, *batch_args)
                    losses.append(loss.detach())
                    for name, value in aux.items():
                        collected.setdefault(name, []).append(value.detach())
        finally:
            if was_training:
                self._model.train()

        metrics: dict[str, float] = {}
        if losses:
            all_loss = gather_for_metrics(torch.cat(losses))
            metrics[f"{metric_key_prefix}_loss"] = float(all_loss.float().mean().item())
        for name, chunks in collected.items():
            gathered = gather_for_metrics(torch.cat(chunks))
            metrics[f"{metric_key_prefix}_{name}"] = float(
                gathered.float().mean().item()
            )
        return metrics
