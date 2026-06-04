"""``SFTTrainer`` — supervised fine-tuning on :class:`DPTrainer`.

Mirrors ``trl.SFTTrainer`` (``trl/trainer/sft_trainer.py``) in structure and
method names — ``_prepare_dataset`` / ``tokenize_row`` for data prep, a
language-modeling collator, and an ``nll``/``dft`` loss dispatch — but routes
the training gradient through Opaque's per-example
:meth:`DPTrainer.compute_per_example_loss` hook (see plan §2.1a).

The per-example loss math and the collator are the merged ``opaque-alignment``
primitives; this class is the orchestration layer.
"""

from __future__ import annotations

from typing import Any, Callable

from opaque.alignment.data import (
    apply_chat_template_with_mask,
    clone_chat_template,
    get_training_chat_template,
)
from opaque.alignment.metric import entropy_from_logits, mean_token_accuracy
from opaque.alignment.sft.collator import language_modeling_collator
from opaque.alignment.sft.loss import dft_loss, nll_loss
from opaque.api.transformers.trainer import DPTrainer

from ._sft_config import SFTConfig

# Loss dispatch (TRL ``loss_type`` → ``opaque.alignment.sft.loss`` head). An
# unknown ``loss_type`` raises ``KeyError`` here — the "standard unknown value"
# behavior, no curated rejection (plan §3.3). ``chunked_nll`` is handled
# separately (the model's fused linear-CE forward computes the loss logits-free).
_SFT_LOSSES: dict[str, Callable] = {"nll": nll_loss, "dft": dft_loss}

# Columns that may carry chat-format conversations.
_CHAT_COLUMNS = ("messages", "conversations", "chat")


def _detect_chat_column(row: dict) -> str | None:
    """Return the chat-message column in *row*, or ``None`` for plain text."""
    for col in _CHAT_COLUMNS:
        value = row.get(col)
        if (
            isinstance(value, list)
            and value
            and isinstance(value[0], dict)
            and "role" in value[0]
            and "content" in value[0]
        ):
            return col
    return None


class SFTTrainer(DPTrainer):
    """DP supervised fine-tuning trainer.

    Args mirror ``trl.SFTTrainer.__init__`` for the supported subset; the model
    forward and gradient go through the DP per-example path.
    """

    def __init__(
        self,
        model: Any = None,
        args: SFTConfig | None = None,
        data_collator: Callable | None = None,
        train_dataset: Any = None,
        eval_dataset: Any = None,
        processing_class: Any = None,
        compute_loss_func: Callable | None = None,
        compute_metrics: Callable | None = None,
        callbacks: list[Any] | None = None,
        optimizers: tuple[Any | None, Any | None] = (None, None),
        optimizer_cls_and_kwargs: tuple[Any, dict[str, Any]] | None = None,
        preprocess_logits_for_metrics: Callable | None = None,
        peft_config: Any = None,
        formatting_func: Callable[[dict], str] | None = None,
    ) -> None:
        # ---- args normalization (accept a plain TrainingArguments) --------
        if args is None:
            args = SFTConfig(output_dir="trainer_output")
        elif not isinstance(args, SFTConfig):
            import dataclasses

            args = SFTConfig(
                **{f.name: getattr(args, f.name) for f in dataclasses.fields(args)}
            )

        # ---- model (string ⇒ load) ----------------------------------------
        if isinstance(model, str):
            from transformers import AutoModelForCausalLM

            model = AutoModelForCausalLM.from_pretrained(
                model, **(args.model_init_kwargs or {})
            )

        # ---- tokenizer / processing_class ---------------------------------
        processing_class = self._resolve_tokenizer(model, processing_class, args)

        # ---- chat template clone (resizes embeddings) ---------------------
        added_tokens: list[int] = []
        if args.chat_template_path is not None:
            model, processing_class, added_tokens = clone_chat_template(
                model, processing_class, args.chat_template_path
            )

        # ---- PEFT ----------------------------------------------------------
        if peft_config is not None:
            from peft import get_peft_model

            # Newly cloned-in special tokens have randomly-initialised embedding
            # rows a frozen base would never learn. Mark exactly those rows
            # trainable and (with a warning) keep the lm_head in modules_to_save
            # so the model can learn to emit them — mirrors trl.SFTTrainer
            # (sft_trainer.py:1064-1084).
            if added_tokens:
                self._mark_added_tokens_trainable(peft_config, added_tokens)
            model = get_peft_model(model, peft_config)

        # ---- activation offloading alias ----------------------------------
        if args.activation_offloading:
            args.cpu_offload_activations = True

        # ---- custom-loss guard --------------------------------------------
        # A custom compute_loss_func is only meaningful on the standard ``nll``
        # path, where this trainer has the model logits to hand it. ``dft``
        # computes its own token-weighted loss (TRL parity, sft_trainer.py:1267),
        # and ``chunked_nll`` is logits-free (the fused kernel reduces the loss
        # inside the forward), so neither can route a custom loss.
        if args.loss_type in ("dft", "chunked_nll") and compute_loss_func is not None:
            raise ValueError(
                f"loss_type={args.loss_type!r} computes its own loss; pass "
                "loss_type='nll' to use a custom compute_loss_func."
            )
        # Resolve the loss path. ``chunked_nll`` lets the model compute its own
        # loss via the fused linear-CE kernel (logits-free on CUDA, eager
        # fallback elsewhere); ``nll`` / ``dft`` dispatch to an alignment head.
        # An unknown loss_type fails with a standard KeyError (plan §3.3).
        self._loss_type: str = args.loss_type
        if args.loss_type == "chunked_nll":
            self._loss_fn: Callable | None = None
            cfg = dict(args.performance_kernels_config or {})
            cfg["fused_linear_cross_entropy"] = True
            args.performance_kernels_config = cfg
        else:
            self._loss_fn = _SFT_LOSSES[args.loss_type]

        # ---- dataset preprocessing (before super().__init__) --------------
        self._formatting_func = formatting_func
        completion_only = self._resolve_completion_only(train_dataset, args)
        self._completion_only_loss = completion_only

        train_dataset = self._prepare_dataset(
            train_dataset, processing_class, args, "train"
        )
        if eval_dataset is not None and not isinstance(eval_dataset, dict):
            eval_dataset = self._prepare_dataset(
                eval_dataset, processing_class, args, "eval"
            )

        # ---- collator ------------------------------------------------------
        if data_collator is None:
            data_collator = language_modeling_collator(
                pad_token_id=processing_class.pad_token_id,
                max_length=args.max_length,
                completion_only_loss=completion_only,
                pad_to_multiple_of=args.pad_to_multiple_of,
            )

        super().__init__(
            model=model,
            args=args,
            data_collator=data_collator,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=processing_class,
            compute_loss_func=compute_loss_func,
            compute_metrics=compute_metrics,
            callbacks=callbacks,
            optimizers=optimizers,
            optimizer_cls_and_kwargs=optimizer_cls_and_kwargs,
            preprocess_logits_for_metrics=preprocess_logits_for_metrics,
        )

    # ------------------------------------------------------------------
    # PEFT: trainability of cloned-in token embeddings
    # ------------------------------------------------------------------
    @staticmethod
    def _mark_added_tokens_trainable(peft_config: Any, added_tokens: list[int]) -> None:
        """Make newly cloned-in token embeddings trainable under PEFT.

        Points ``peft_config.trainable_token_indices['embed_tokens']`` at the new
        token ids and ensures ``lm_head`` is in ``modules_to_save`` (warning when
        it has to be added), mirroring ``trl.SFTTrainer`` (sft_trainer.py:1064-1084).
        A no-op when ``added_tokens`` is empty.
        """
        if not added_tokens:
            return
        tti = getattr(peft_config, "trainable_token_indices", None)
        if tti is None:
            peft_config.trainable_token_indices = {"embed_tokens": list(added_tokens)}
        elif "embed_tokens" not in tti:
            tti["embed_tokens"] = list(added_tokens)
        else:
            tti["embed_tokens"] = list(tti["embed_tokens"]) + list(added_tokens)

        mts = getattr(peft_config, "modules_to_save", None)
        if mts is None or "lm_head" not in mts:
            import warnings

            warnings.warn(
                "New tokens were added to the chat template but 'lm_head' is not "
                "in the PEFT config's modules_to_save; adding it so the model can "
                "learn to generate them. Pass modules_to_save=['lm_head'] to "
                "silence this.",
                stacklevel=2,
            )
            if mts is None:
                peft_config.modules_to_save = ["lm_head"]
            else:
                mts.append("lm_head")

    # ------------------------------------------------------------------
    # Tokenizer / format resolution
    # ------------------------------------------------------------------
    @staticmethod
    def _resolve_tokenizer(model: Any, processing_class: Any, args: SFTConfig) -> Any:
        if processing_class is None:
            from transformers import AutoTokenizer

            name = getattr(model.config, "_name_or_path", None)
            if not name:
                raise ValueError(
                    "processing_class is None and the model config has no "
                    "_name_or_path to load a tokenizer from; pass processing_class."
                )
            processing_class = AutoTokenizer.from_pretrained(name)
        if args.eos_token is not None:
            processing_class.eos_token = args.eos_token
        if processing_class.pad_token_id is None:
            processing_class.pad_token = processing_class.eos_token
        return processing_class

    def _resolve_completion_only(self, dataset: Any, args: SFTConfig) -> bool:
        """Auto-detect completion-only loss when ``args`` leaves it ``None``.

        TRL parity (sft_trainer.py:1160-1173): ``True`` for prompt-completion or
        chat datasets (an assistant/completion mask exists), else ``False``.
        """
        if args.assistant_only_loss:
            return True
        if args.completion_only_loss is not None:
            return bool(args.completion_only_loss)
        if dataset is None or len(dataset) == 0:
            return False
        row = dataset[0]
        if "completion_mask" in row:
            return True
        if "prompt" in row and "completion" in row:
            return True
        if _detect_chat_column(row) is not None:
            return True
        return False

    # ------------------------------------------------------------------
    # Dataset preparation (TRL-shaped)
    # ------------------------------------------------------------------
    def _prepare_dataset(
        self, dataset: Any, processing_class: Any, args: SFTConfig, dataset_name: str
    ) -> Any:
        """Tokenize *dataset* into ``input_ids`` (+ ``completion_mask``).

        Mirrors ``trl.SFTTrainer._prepare_dataset``: skip already-tokenized
        datasets, optionally apply a ``formatting_func``, then dispatch by
        format (plain text / prompt-completion / chat).
        """
        if dataset is None:
            return None
        column_names = list(getattr(dataset, "column_names", []) or [])
        if "input_ids" in column_names:
            return dataset  # already tokenized

        if self._formatting_func is not None:
            dataset = dataset.map(
                lambda ex: {args.dataset_text_field: self._formatting_func(ex)},
                num_proc=args.dataset_num_proc,
            )
            column_names = list(dataset.column_names)

        row = dataset[0]
        chat_col = _detect_chat_column(row)

        # Chat data with a completion/assistant mask needs the generation-marker
        # template installed so ``apply_chat_template_with_mask`` can recover it.
        if chat_col is not None and self._completion_only_loss:
            processing_class.chat_template = get_training_chat_template(
                processing_class
            )

        def tokenize_row(example: dict) -> dict:
            return self.tokenize_row(example, processing_class, args, chat_col=chat_col)

        return dataset.map(
            tokenize_row,
            remove_columns=column_names,
            num_proc=args.dataset_num_proc,
            desc=f"Tokenizing {dataset_name} dataset",
        )

    def tokenize_row(
        self,
        example: dict,
        processing_class: Any,
        args: SFTConfig,
        *,
        chat_col: str | None,
    ) -> dict:
        """Tokenize one SFT example into model-ready columns."""
        max_length = args.max_length

        if chat_col is not None:
            encoded = apply_chat_template_with_mask(
                processing_class,
                example[chat_col],
                max_length=max_length,
                truncation=max_length is not None,
            )
            ids = encoded["input_ids"]
            cmask = encoded["completion_mask"]
            if max_length is not None:
                ids, cmask = ids[:max_length], cmask[:max_length]
            return {"input_ids": ids, "completion_mask": cmask}

        if "prompt" in example and "completion" in example:
            prompt_ids = processing_class(example["prompt"], add_special_tokens=True)[
                "input_ids"
            ]
            full_ids = (
                prompt_ids
                + processing_class(example["completion"], add_special_tokens=False)[
                    "input_ids"
                ]
            )
            cmask = [0] * len(prompt_ids) + [1] * (len(full_ids) - len(prompt_ids))
            if max_length is not None:
                full_ids, cmask = full_ids[:max_length], cmask[:max_length]
            return {"input_ids": full_ids, "completion_mask": cmask}

        # Plain text field. Append EOS whenever the tokenizer has one (TRL
        # parity) so the model learns to stop — not only when ``args.eos_token``
        # was explicitly overridden.
        text = example[args.dataset_text_field]
        eos = processing_class.eos_token
        if eos is not None and not text.endswith(eos):
            text = text + eos
        ids = processing_class(
            text, truncation=max_length is not None, max_length=max_length
        )["input_ids"]
        return {"input_ids": ids}

    # ------------------------------------------------------------------
    # Loss (DP per-example hook)
    # ------------------------------------------------------------------
    def compute_per_example_loss(
        self,
        fmodel: Callable[..., Any],
        params: dict[str, Any],
        inputs: dict[str, Any],
        *,
        return_logits: bool = False,
    ) -> Any:
        """One example's SFT loss (vmap-batched by :class:`DPTrainer`).

        The collator already folds ``completion_mask`` into ``-100`` labels, so
        this is a single forward + the configured head (``nll`` / ``dft``); both
        heads use a DP-safe per-example divisor. A custom ``compute_loss_func``
        (only valid on the ``nll`` path; ``dft`` / ``chunked_nll`` are guarded at
        init) is honoured here per ``DPTrainer``'s contract
        ``compute_loss_func(outputs, labels) -> scalar`` — it runs inside vmap,
        so it must be a pure per-example tensor op (no ``num_items_in_batch`` /
        batch coupling). Per-example telemetry lives in
        :meth:`compute_per_example_loss_and_metrics`.
        """
        if self._loss_type == "chunked_nll":
            # The model computes the (logits-free, fused) NLL when given labels.
            out = fmodel(
                params,
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                labels=inputs["labels"],
            )
            loss = out["loss"]
            logits = out.get("logits")  # None on the fused path
        else:
            out = fmodel(
                params,
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
            )
            logits = out.logits
            if self._compute_loss_func is not None:
                loss = self._compute_loss_func(out, inputs["labels"])
            else:
                loss = self._loss_fn(out.logits, inputs["labels"])
        if return_logits:
            return loss, logits
        return loss

    def compute_per_example_loss_and_metrics(
        self,
        fmodel: Callable[..., Any],
        params: dict[str, Any],
        inputs: dict[str, Any],
    ) -> tuple[Any, dict[str, Any]]:
        """One example's SFT ``(loss, telemetry)`` (vmap-batched).

        Adds the per-example diagnostics TRL logs each step — ``entropy`` and
        ``mean_token_accuracy`` over the supervised (non-``-100``) tokens —
        riding the clipped-grad aux channel (and the symmetric eval aux). They
        are computed from the model logits, **independent of the loss head**, so
        a custom ``compute_loss_func`` that returns only a scalar still gets
        telemetry (the aux fail-safe). When logits are unavailable (the
        logits-free ``chunked_nll`` path) the telemetry dict is empty and only
        the loss is logged — the harness handles an empty aux dict.
        """
        loss, logits = self.compute_per_example_loss(
            fmodel, params, inputs, return_logits=True
        )
        if logits is None:
            return loss, {}
        labels = inputs["labels"]
        mask = labels != -100
        return loss, {
            "entropy": entropy_from_logits(logits, mask),
            "mean_token_accuracy": mean_token_accuracy(logits, labels, mask),
        }
