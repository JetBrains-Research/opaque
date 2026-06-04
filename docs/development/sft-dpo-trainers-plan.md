# SFT & DPO Trainers on `DPTrainer` — Implementation Plan

**Status:** Planning. Scoped to two class trainers — `SFTTrainer` and
`DPOTrainer` — built on the in-house `DPTrainer` and consuming the
already-merged `opaque-alignment` primitives (PR #251).

**Author context:** Written against `main` at `909ed54` (opaque-alignment in
place) on branch `claude/modest-gates-WpC4d`. Supersedes the pre-merge draft
`docs/development/trl-trainers-plan.md` (which also covered building the
`opaque-alignment` package and a `KTOTrainer`; both are out of scope here).

**Reference TRL:** `huggingface/trl` `main` (cloned fresh for this analysis).
File:line references below point at `trl/trainer/sft_trainer.py`,
`sft_config.py`, `dpo_trainer.py`, `dpo_config.py`, `data_utils.py`,
`utils.py`, `callbacks.py`.

---

## Table of contents

1. [What already exists vs. what's missing](#1-what-already-exists-vs-whats-missing)
2. [Architecture: the single extension point](#2-architecture-the-single-extension-point)
3. [Cross-cutting design decisions](#3-cross-cutting-design-decisions)
4. [`SFTConfig` / `SFTTrainer`](#4-sftconfig--sfttrainer)
5. [`DPOConfig` / `DPOTrainer`](#5-dpoconfig--dpotrainer)
6. [Core `DPTrainer` changes required](#6-core-dptrainer-changes-required)
7. [Loss-type coverage matrix](#7-loss-type-coverage-matrix)
8. [Phasing](#8-phasing)
9. [Test strategy](#9-test-strategy)
10. [Examples & Cadence configs](#10-examples--cadence-configs)
11. [Risks, non-goals, and deferred features](#11-risks-non-goals-and-deferred-features)

---

## 1. What already exists vs. what's missing

The hard parts — the DP-correct loss math, collators, reference-logp handling,
and a proven end-to-end functional pipeline — are **already merged**. This plan
is mostly an *orchestration* layer.

### 1.1 Already built (consume, don't rebuild)

**`opaque-alignment` functional primitives** (method-first layout mirroring
`opaque.dpsgd` / `opaque.dpftrl`):

| Concern | Public symbol | Location |
|---|---|---|
| SFT loss | `nll_loss(logits, labels)`, `dft_loss(logits, labels)` | `opaque.alignment.sft.loss` |
| SFT loss (fused, logits never materialised) | `fused_nll_loss(hidden, lm_head_weight, labels)`, `fused_dft_loss(...)` | `opaque.alignment.sft.loss` |
| SFT collator | `language_modeling_collator(pad_token_id, max_length, *, completion_only_loss, pad_to_multiple_of) -> LMBatch` | `opaque.alignment.sft.collator` |
| Per-sequence logp | `sequence_logp(logits, input_ids, completion_mask, *, ld_alpha, shared_prefix_len, length_normalized)`, `fused_sequence_logp(...)` | `opaque.alignment.dpo.loss` |
| DPO heads (14) | `sigmoid_loss`, `hinge_loss`, `robust_loss`, `ipo_loss`, `discopop_loss`, `chosen_nll_loss`, `squarechipo_loss`, `apo_zero_loss`, `apo_down_loss`, `exo_loss`, `nca_loss`, `bco_loss`, `sppo_loss`, `simpo_loss`, `odds_ratio_loss` | `opaque.alignment.dpo.loss` |
| DPO combinators | `f_divergence_remap`, `f_divergence_logits`, `mpo_combine`, `wpo_weights`, `ld_dpo_split` | `opaque.alignment.dpo.loss` |
| DPO collator | `preference_collator(pad_token_id, max_length, *, pad_to_multiple_of)` | `opaque.alignment.dpo.collator` |
| Reference logp | `compute_ref_logprobs_for_dataset(dataset, ref, collator, output_columns, *, batch_size, cache_key, cache_dir)`, `null_ref_context`, `with_disabled_adapter`, `ema_update_reference` | `opaque.alignment.dpo.reference` |
| DPO reward telemetry | `reward_metrics(chosen_logratio, rejected_logratio, *, beta)` | `opaque.alignment.dpo.metric` |
| Preference prompt extraction | `extract_prompt` | `opaque.alignment.dpo.data` |
| Chat-template data prep | `clone_chat_template`, `get_training_chat_template`, `apply_chat_template_with_mask` | `opaque.alignment.data` |
| Token metrics | `entropy_from_logits`, `mean_token_accuracy` | `opaque.alignment.metric` |

**The DP-correctness story is baked into the primitives, not the trainer.**
`dft_loss` already replaces TRL's batch-level `num_items_in_batch` divisor
(`sft_trainer.py:815-829`) with a per-example divisor
`mask.sum(-1).clamp(min=1)` (`_dft.py:48-106`); `nll_loss` is a per-example
mean. The preference collator already uses the per-pair `(B, …)` layout (keys
`chosen_input_ids`, `chosen_attention_mask`, `chosen_completion_mask`,
`rejected_*`, optional `ref_chosen_logps`/`ref_rejected_logps`) rather than
TRL's `(2B, L)` concatenation (`dpo_trainer.py:90-211`), so each pair is a
self-contained privacy unit.

**Proven functional pipelines.** `examples/train_sft.py` and
`examples/train_dpo.py` already run the full manual DP-SGD loop
(`make_functional` → `vmap(grad(...))` → `clipped_grad` → `gaussian_noise` →
`adamw` → accounting) over these primitives. They are the **reference
implementation the trainers must reproduce numerically.** The DPO per-example
closure (`examples/train_dpo.py:528-551`) is the exact body the trainer's
override will wrap:

```python
def per_example_loss(trainable_params, chosen_ids, chosen_mask, chosen_cmask,
                     rejected_ids, rejected_mask, rejected_cmask,
                     ref_chosen_logps, ref_rejected_logps):
    merged = {**frozen, **trainable_params}
    chosen_out   = fmodel(merged, input_ids=chosen_ids,   attention_mask=chosen_mask)
    rejected_out = fmodel(merged, input_ids=rejected_ids, attention_mask=rejected_mask)
    chosen_logp   = sequence_logp(chosen_out.logits,   chosen_ids,   chosen_cmask)
    rejected_logp = sequence_logp(rejected_out.logits, rejected_ids, rejected_cmask)
    return _DPO_LOSSES[loss_type](chosen_logp - ref_chosen_logps,
                                  rejected_logp - ref_rejected_logps, beta=beta)
```

### 1.2 Missing (this plan's deliverables)

1. `SFTConfig` / `DPOConfig` dataclasses (extend `opaque.api.transformers.trainer._config.TrainingArguments`).
2. `SFTTrainer` / `DPOTrainer` — thin `DPTrainer` subclasses overriding `compute_per_example_loss` and doing dataset/ref preprocessing in `__init__`.
3. A small set of `DPTrainer` core touch-ups (§6).
4. `examples/train_sft_trainer.py` / `examples/train_dpo_trainer.py` (mirror `train_causal_lm.py` ↔ `train_causal_lm_trainer.py`).
5. Matching `.cadence/configs/train_{sft,dpo}_trainer (qwen_alignment).yaml`.
6. Trainer-level integration tests + TRL numeric-parity + DP-purity tests.
7. User-guide docs.

---

## 2. Architecture: the single extension point

`DPTrainer` was designed for exactly this. Its docstring names the hook
explicitly: *"`compute_per_example_loss()` — DP-correct override hook; the
single extension point for SFT / DPO / KTO subclasses"*
(`_dp_trainer.py:259-260`).

### 2.1 How the per-example loss flows

```
collator(dataset[0])  ──▶  _discover_batch_keys()           # tensor keys, ordered  (_dp_trainer.py:3123)
                                   │
                                   ▼
_build_per_example_loss(fmodel, frozen, batch_keys)         # (_dp_trainer.py:3002)
    def per_example_loss(trainable, *batch_args):
        merged = {**frozen, **trainable}
        inputs = dict(zip(batch_keys, batch_args))          # name-keyed
        return self.compute_per_example_loss(fmodel, merged, inputs, ...)
                                   │
                                   ▼
vmap(per_example_loss, in_dims=(None,)+(0,)*K) ─▶ grad ─▶ clipped_grad(normalize_by=expected_batch_size)
                                                                      # (_dp_trainer.py:3628-3676)
```

Three consequences for the trainers:

- **Override `compute_per_example_loss(self, fmodel, params, inputs, *, return_logits=False)`** (`_dp_trainer.py:1955`). `inputs` is a *name-keyed* dict of one example's tensors (batch dim already stripped by vmap). The subclass reads `inputs["chosen_input_ids"]`, etc. — order-independent, so the collator's key order doesn't matter.
- **`params` arrives pre-merged** (`frozen | trainable`); call `fmodel(params, input_ids=..., attention_mask=...)`. DPO calls it twice (chosen, rejected); SFT once.
- **Never divide by a cross-example batch quantity.** The DP-correct divisor is `expected_batch_size`, applied *outside* the per-example closure by `clipped_grad(normalize_by=...)` (`_dp_trainer.py:3654/3663/3673`). The per-example loss returns a per-example scalar; the batch `.mean()` emerges from the clipped-gradient sum ÷ `expected_batch_size`. This is the structural reason the alignment heads return per-example scalars and never `.mean()` internally.

### 2.2 Where the code lives (api/façade discipline)

`opaque-transformers` owns `opaque.api.transformers.*` (impl) and re-exports
through `opaque.transformers.*` (façade). The façade-discipline contract test
(`tests/contracts/test_facade_discipline.py`) enforces that façade modules are
pure re-exports. New surface:

```
packages/opaque-transformers/src/opaque/
├── api/transformers/trl/                  ← IMPLEMENTATION (new)
│   ├── __init__.py                        # SFTTrainer, SFTConfig, DPOTrainer, DPOConfig
│   ├── _sft_config.py
│   ├── _sft_trainer.py
│   ├── _dpo_config.py
│   └── _dpo_trainer.py
└── transformers/trl/                      ← FAÇADE (new)
    └── __init__.py                        # from opaque.api.transformers.trl import …
```

Public import surface: `from opaque.transformers.trl import SFTTrainer,
SFTConfig, DPOTrainer, DPOConfig`. (Name `trl` keeps parity-of-intent obvious;
an alternative is `opaque.transformers.alignment` — see §11 open questions.)

### 2.3 Dependency wiring

Add `opaque-alignment` to `packages/opaque-transformers/pyproject.toml`
`dependencies`. Direction is legal: `opaque-alignment` is forbidden from
importing `opaque.api.transformers`
(`tests/contracts/test_dependency_direction.py`), and `opaque-transformers`
sits above it — `transformers → alignment` is a clean forward edge. No new
forbidden-import entry is needed (the contract only constrains *lower* wheels).

---

## 3. Cross-cutting design decisions

### 3.1 `remove_unused_columns = False` is mandatory for both trainers

`DPTrainer` defaults `remove_unused_columns=True` (`_config.py:345`) and prunes
dataset columns to the model's `forward` signature via
`_set_signature_columns_if_needed` (`_dp_trainer.py:2899`) /
`_remove_unused_columns` (`_dp_trainer.py:2927`). Both trainers emit columns
that are **not** `model.forward` parameters:

- SFT: `completion_mask` (consumed by the collator to build `-100` labels), `assistant_masks`.
- DPO: `chosen_input_ids`, `rejected_input_ids`, `chosen_completion_mask`, … and `ref_chosen_logps` / `ref_rejected_logps`.

If pruning runs, `_discover_batch_keys` (which calls
`_prepare_dataset_and_collator`, `_dp_trainer.py:3135`) would strip these
before the collator sees them. **Both configs force
`remove_unused_columns=False` in `__post_init__`** (TRL relies on the same
posture by consuming raw columns inside custom collators). Document it; warn if
a user sets it `True`.

### 3.2 Reference model: precompute is the universal DP-clean path

TRL has four ref paths (`dpo_trainer.py:762-823`): explicit `ref_model`, PEFT
`null_ref_context`, `precompute_ref_log_probs`, and auto-load. Under
per-example `vmap(grad(...))`, an on-the-fly reference forward cannot run
*inside* the closure (it either runs a second `nn.Module` or mutates PEFT
adapter flags, neither of which composes with `functional_call`/`vmap`).

**Decision:** the static-reference cases all collapse to **precompute**:

| TRL path | Opaque realization |
|---|---|
| explicit `ref_model` | `compute_ref_logprobs_for_dataset(dataset, ref=ref_model, collator=preference_collator(...), output_columns=("ref_chosen_logps","ref_rejected_logps"))` in `__init__`, before `super().__init__`. |
| PEFT base-as-reference (`null_ref_context`) | same call with `ref` = the policy run under `with null_ref_context(model): …` (adapter disabled). |
| auto-load | load a second copy via `AutoModelForCausalLM.from_pretrained(...)`, treat as explicit. |
| `precompute_ref_log_probs=True` | this *is* the default; the flag becomes a no-op/always-on for static refs. |

The cache is content-addressed (`cache_key=`, `cache_dir=`) and per-example;
the resulting columns feed the collator, which emits them as constant
`(B,)` tensors consumed by the loss as the log-ratio baseline
`policy_logp − ref_logp`. **DP semantics:** ref logps depend on
(private example *i* × public ref params); per-example clipping covers the
policy gradient that reads them as constants. The cache file is private — never
commit, log, or release it.

**Reference-free** (`reference_free=True`, CPO/ORPO/SimPO): skip precompute; the
head takes the policy logp directly (`sigmoid_loss(chosen_logp, rejected_logp,
…)` with no ref subtraction, or `odds_ratio_loss` / `simpo_loss`).

**TR-DPO** (`sync_ref_model`, ref changes every N steps) genuinely needs
on-the-fly refs → deferred to a later phase that adds a pre-vmap batch hook
(§6.4) or a callback re-running precompute on a cadence. Incompatible with
precompute and PEFT in TRL too (`dpo_config.py:287`).

### 3.3 Reject batch-coupled losses

`aot` and `aot_unpaired` sort log-ratios *across the batch*
(`dpo_trainer.py` loss block) before applying the sigmoid. Cross-example
ordering makes the per-example gradient depend on other examples → breaks
per-example DP composition. **Reject at config validation** with a documented
error. (Mirrors the deferred plan's stance; `opaque-alignment` already ships no
`aot` head.)

### 3.4 Loss-type list = MPO for free

Both TRL configs accept `loss_type` as a list with `loss_weights`
(`dpo_config.py:211,220`). Map directly onto `mpo_combine({name: head(...)},
{name: weight})`. A single-element list is the scalar case.

### 3.5 Optimizer / kernels / offload already covered

- **DP-AdamW** (the headline alignment recipe, arXiv:2505.08849) is `optim="adamw"` + `optim_args="noise_bias_correction=True"` — already in `opaque-optimizers`. No new optimizer.
- **Fused chunked CE** (TRL's `chunked_nll`, `sft_trainer.py:123-232`) → `fused_nll_loss` / `fused_dft_loss` / `fused_sequence_logp` (already vmap-safe, never materialise `(B,T,V)`).
- **Activation offloading** (`SFTConfig.activation_offloading`, `sft_config.py:275`) ≈ `DPTrainer`'s existing CPU-offload arg; reuse, optionally alias the name for parity.

### 3.6 Eval & metrics

`compute_per_example_loss(..., return_logits=True)` is also the per-example eval
hook (`_dp_trainer.py:2002-2009`, `_get_eval_per_example_loss_fn:3077`).

- **SFT eval** returns `(loss, logits)` directly; token accuracy / entropy via `mean_token_accuracy` / `entropy_from_logits` through `compute_metrics` or a thin eval override.
- **DPO eval** has no single "logits" (two forwards). Options: (a) return `(loss, chosen_logits)` to satisfy the signature and compute rewards in a custom eval pass that calls `reward_metrics(chosen_logratio, rejected_logratio, beta)`; (b) accumulate reward/accuracy telemetry inside the loss override via a side-channel. Pick (a) — keep the override pure, log rewards in an `evaluation_loop`/`compute_metrics` extension. Flag as the one non-trivial eval design item.

---

## 4. `SFTConfig` / `SFTTrainer`

### 4.1 `SFTConfig` (extends `TrainingArguments`)

Add TRL-parity fields (defaults from `sft_config.py`). Grouped:

**Data prep**

| Field | Default | Source | Phase |
|---|---|---|---|
| `dataset_text_field: str` | `"text"` | `sft_config.py:163` | 1 |
| `max_length: int \| None` | `1024` | `:186` | 1 |
| `completion_only_loss: bool \| None` | `None` (auto-detect prompt+completion) | `:242` | 1 |
| `eos_token: str \| None` | `None` | `:180` | 1 |
| `pad_to_multiple_of: int \| None` | `None` | `:232` | 1 |
| `assistant_only_loss: bool` | `False` | `:254` | 2 (uses `apply_chat_template_with_mask` + `get_training_chat_template`) |
| `chat_template_path: str \| None` | `None` | `:152` | 2 (uses `clone_chat_template`) |
| `packing` / `packing_strategy` / `padding_free` | `False`/`"bfd"`/`False` | `:206-222` | deferred (§11) |

**Loss / training**

| Field | Default | Notes |
|---|---|---|
| `loss_type: str` | `"nll"` | `"nll"` → `nll_loss`; `"dft"` → `dft_loss`; `"chunked_nll"` → `fused_nll_loss` (phase 2). |
| `activation_offloading: bool` | `False` | map to existing DPTrainer offload. |
| `learning_rate` override | `2e-5` | `sft_config.py:137`. |

`__post_init__`: force `remove_unused_columns=False`; validate `loss_type ∈
{nll, dft, chunked_nll}`; reject `packing`/`padding_free` until their phase.

### 4.2 `SFTTrainer.__init__`

Thin wrapper. Mirrors TRL's flow (`sft_trainer.py:932-1359`) minus
Accelerate/DeepSpeed/FSDP/VLM:

1. Normalize `args` (accept `TrainingArguments` → upcast to `SFTConfig`).
2. Resolve tokenizer / `eos_token`; (phase 2) chat-template install via `clone_chat_template`.
3. **Preprocess dataset** (before `super().__init__`): tokenize `dataset_text_field`, append EOS, detect/auto-set `completion_only_loss`, produce `input_ids` (+ `completion_mask` / `assistant_masks`). Reuse the tokenization helpers from `examples/train_sft.py`; promote any shared bits into `opaque.alignment.sft` if they aren't already.
4. **Build collator** if none supplied: `language_modeling_collator(pad_token_id=tok.pad_token_id, max_length=args.max_length, completion_only_loss=…, pad_to_multiple_of=…)`.
5. Store `self._loss_type`; call `super().__init__(model, args=…, data_collator=…, train_dataset=…, processing_class=…, …)`.

### 4.3 `compute_per_example_loss` override

The collator already folds `completion_mask` into `-100` labels (`LMBatch`),
so the override is one forward + one head:

```python
def compute_per_example_loss(self, fmodel, params, inputs, *, return_logits=False):
    out = fmodel(params, input_ids=inputs["input_ids"],
                         attention_mask=inputs["attention_mask"])
    labels = inputs["labels"]
    if self._loss_type == "dft":
        loss = dft_loss(out.logits, labels)
    else:  # "nll"
        loss = nll_loss(out.logits, labels)
    return (loss, out.logits) if return_logits else loss
```

`chunked_nll` swaps the logits path for `fused_nll_loss(out.hidden_states,
lm_head_weight, labels)` (needs the model to expose hidden states / lm_head
weight in the functional call — phase 2 wiring).

**DP-correctness:** `labels` is per-example data; both heads use per-example
divisors. No `num_items_in_batch`. ✔

---

## 5. `DPOConfig` / `DPOTrainer`

### 5.1 `DPOConfig` (extends `TrainingArguments`)

| Field | Default | Source | Maps to |
|---|---|---|---|
| `loss_type: list[str]` | `["sigmoid"]` | `dpo_config.py:211` | head dispatch (§7) |
| `loss_weights: list[float] \| None` | `None` | `:220` | `mpo_combine` weights |
| `beta: float` | `0.1` | `:260` | `head(..., beta=)` |
| `label_smoothing: float` | `0.0` | `:251` | `sigmoid_loss/robust_loss/exo_loss(... label_smoothing=)` |
| `f_divergence_type: str` | `"reverse_kl"` | `:237` | `f_divergence_remap` |
| `f_alpha_divergence_coef: float` | `0.5` | `:244` | α-divergence coef |
| `ld_alpha: float \| None` | `None` | `:228` | `sequence_logp(ld_alpha=)` / `ld_dpo_split` |
| `use_weighting: bool` | `False` | `:268` | `wpo_weights` |
| `rpo_alpha: float \| None` | `None` | TRL RPO | blend `chosen_nll_loss` |
| `discopop_tau: float` | `0.05` | `:275` | `discopop_loss(tau=)` |
| `reference_free: bool` | `False` | TRL | skip ref subtraction |
| `precompute_ref_log_probs: bool` | `False`→effectively always | `:193` | precompute (§3.2) |
| `precompute_ref_batch_size: int \| None` | `None` | `:201` | `compute_ref_logprobs_for_dataset(batch_size=)` |
| `max_length` / `truncation_mode` / `pad_to_multiple_of` | `1024`/`keep_start`/`None` | `:165-189` | tokenize + collator |
| `sync_ref_model` / `ref_model_mixup_alpha` / `ref_model_sync_steps` | `False`/`0.6`/`512` | `:287-304` | TR-DPO (deferred, §3.2) |
| `disable_dropout: bool` | `True` | `:155` | dropout off on policy+ref |
| `learning_rate` override | `1e-6` | `:142` | — |

`__post_init__`: force `remove_unused_columns=False`; **reject `aot` /
`aot_unpaired`** (§3.3); reject `sync_ref_model` until its phase; validate
`label_smoothing ∈ [0,0.5)` for robust, `> 0` for exo (`dpo_trainer.py:680-694`);
default `loss_weights` to `[1.0]*len(loss_type)`.

### 5.2 `DPOTrainer.__init__`

Mirrors `dpo_trainer.py:501-851` minus Accelerate/DeepSpeed/FSDP/VLM:

1. Normalize `args`; capture `beta`, `loss_types`, `loss_weights`, divergence/weighting/ld flags onto `self`.
2. Resolve tokenizer / `pad_token`; disable dropout on policy (+ref) if `disable_dropout`.
3. PEFT: if `peft_config`, `get_peft_model`; QLoRA bf16 promotion (`dpo_trainer.py:611-614`). The "ref adapter clone" (`:591-599`) is only needed for on-the-fly PEFT ref — under precompute we instead run the base model with `null_ref_context`.
4. **Tokenize dataset** (before `super().__init__`): `extract_prompt` if absent (`data_utils.py`); append EOS; tokenize prompt/chosen/rejected; assemble the collator's input schema — `chosen_input_ids` (= prompt+chosen ids), `chosen_completion_mask` (0 on prompt, 1 on completion), `rejected_*`. (This is the schema `preference_collator` reads — `_preference.py:152-158` — and what `examples/train_dpo.py`'s tokenizer already emits.)
5. **Resolve reference & precompute** (unless `reference_free`): build `ref` callable per §3.2; `dataset = compute_ref_logprobs_for_dataset(dataset, ref, collator=preference_collator(...), output_columns=("ref_chosen_logps","ref_rejected_logps"), batch_size=precompute_ref_batch_size or per_device_batch, cache_key=("dpo", fingerprint))`. Same for eval dataset.
6. Build `preference_collator(pad_token_id, max_length, pad_to_multiple_of)` if none supplied.
7. `super().__init__(model, args, data_collator=…, train_dataset=…, eval_dataset=…, processing_class=…)`.

### 5.3 `compute_per_example_loss` override

Wraps the proven closure (`examples/train_dpo.py:528-551`), generalized over
the loss-type list, f-divergence, WPO, LD-DPO, RPO:

```python
def compute_per_example_loss(self, fmodel, params, inputs, *, return_logits=False):
    chosen_out   = fmodel(params, input_ids=inputs["chosen_input_ids"],
                                  attention_mask=inputs["chosen_attention_mask"])
    rejected_out = fmodel(params, input_ids=inputs["rejected_input_ids"],
                                  attention_mask=inputs["rejected_attention_mask"])

    norm = "norm" in self._loss_types or "ipo" in self._loss_types
    chosen_logp   = sequence_logp(chosen_out.logits, inputs["chosen_input_ids"],
                                  inputs["chosen_completion_mask"],
                                  ld_alpha=self._ld_alpha, length_normalized=norm)
    rejected_logp = sequence_logp(rejected_out.logits, inputs["rejected_input_ids"],
                                  inputs["rejected_completion_mask"],
                                  ld_alpha=self._ld_alpha, length_normalized=norm)

    if self._reference_free:
        chosen_lr, rejected_lr = chosen_logp, rejected_logp
    else:
        chosen_lr   = chosen_logp   - inputs["ref_chosen_logps"]
        rejected_lr = rejected_logp - inputs["ref_rejected_logps"]

    if self._f_divergence_type != "reverse_kl":
        chosen_lr, rejected_lr = f_divergence_remap(chosen_lr, rejected_lr,
                                                    kind=self._f_divergence_type,
                                                    alpha=self._f_alpha_coef)

    parts = {name: _HEAD[name](chosen_lr, rejected_lr, beta=self._beta, **kw(name))
             for name in self._loss_types}
    if self._rpo_alpha:
        parts["rpo_nll"] = chosen_nll_loss(chosen_logp)
    loss = mpo_combine(parts, self._weights)

    if self._use_weighting:
        loss = loss * wpo_weights(chosen_out.logits, inputs["chosen_input_ids"],
                                  inputs["chosen_completion_mask"]) * \
                      wpo_weights(rejected_out.logits, inputs["rejected_input_ids"],
                                  inputs["rejected_completion_mask"])

    return (loss, chosen_out.logits) if return_logits else loss
```

(`_HEAD` maps config strings → the `opaque.alignment.dpo.loss` functions;
`kw(name)` injects per-head extras like `label_smoothing`, `tau`. Exact WPO
signature to be confirmed against `_wpo.py` during implementation.)

**DP-correctness:** every quantity is per-example or a public constant; ref
logps enter as constant per-example tensors; no cross-batch coupling (the
rejected variants are pre-filtered). ✔

---

## 6. Core `DPTrainer` changes required

The hook exists; the changes are small and additive.

1. **Dependency + façade scaffolding** (§2.2, §2.3): add `opaque-alignment`; create `opaque.api.transformers.trl` + `opaque.transformers.trl`; export from `opaque.transformers` `__all__` if desired.

2. **Config subclassing ergonomics.** `TrainingArguments` is a standalone `@dataclass` (`_config.py:199`). Confirm `SFTConfig`/`DPOConfig` can extend it cleanly (dataclass field ordering: new fields need defaults, which they have). If `__post_init__` exists on the base, the subclass must `super().__post_init__()` then apply its overrides (force `remove_unused_columns=False`, validations).

3. **Signature-column escape for non-forward keys.** Even with `remove_unused_columns=False`, confirm `_discover_batch_keys` (`:3123`) tolerates extra tensor keys (it keeps *all* tensor keys — good) and that vmap over `ref_chosen_logps` `(B,)` → scalar is fine (it is). No code change expected; add a regression test.

4. **(Deferred phase) Pre-vmap batch-augmentation hook.** For on-the-fly references (TR-DPO) and any future per-step ref forward, add an overridable `def _augment_batch(self, inputs) -> inputs` called in `training_step` *before* `batch_args` are gathered (`_dp_trainer.py:1749`), plus make `batch_keys` include augmented keys. Out of scope for the first SFT+DPO landing; precompute covers static refs.

5. **(Optional) DPO eval rewards.** A hook in `evaluation_loop`/`prediction_step` to call `reward_metrics` and log chosen/rejected rewards + accuracy (§3.6). Could live entirely in `DPOTrainer` via `compute_metrics` without touching core.

---

## 7. Loss-type coverage matrix

### 7.1 SFT (`SFTConfig.loss_type`)

| TRL value | Opaque mapping | Phase |
|---|---|---|
| `nll` | `nll_loss` | 1 |
| `dft` | `dft_loss` (DP-safe divisor already built) | 1 |
| `chunked_nll` | `fused_nll_loss` / `fused_dft_loss` | 2 |

### 7.2 DPO (`DPOConfig.loss_type`)

| TRL value | Opaque head | Phase |
|---|---|---|
| `sigmoid` | `sigmoid_loss` | 1 |
| `hinge` | `hinge_loss` | 1 |
| `ipo` | `ipo_loss` (+ `length_normalized=True`) | 1 |
| `robust` | `robust_loss` | 1 |
| `sigmoid_norm` | `sigmoid_loss` on length-normalized logp | 1 |
| `apo_zero` / `apo_down` | `apo_zero_loss` / `apo_down_loss` | 1 |
| `exo_pair` | `exo_loss` | 2 |
| `nca_pair` | `nca_loss` | 2 |
| `bco_pair` | `bco_loss` | 2 |
| `sppo_hard` | `sppo_loss` | 2 |
| `discopop` | `discopop_loss` (`tau=discopop_tau`) | 2 |
| `sft` | `chosen_nll_loss` (CE on chosen completion) | 2 |
| — (SquareχPO, optimal-rate DP-DPO) | `squarechipo_loss` | 2 |
| `aot`, `aot_unpaired` | **REJECTED** (batch sort) | — |

**Cross-features:** MPO (`loss_weights`) → `mpo_combine` (1); f-divergence →
`f_divergence_remap` (2); WPO (`use_weighting`) → `wpo_weights` (2); LD-DPO
(`ld_alpha`) → `sequence_logp(ld_alpha=)`/`ld_dpo_split` (2); RPO (`rpo_alpha`)
→ `chosen_nll_loss` blend (2); reference-free / CPO / ORPO / SimPO →
`odds_ratio_loss` / `simpo_loss` / reference-free `sigmoid_loss` (2).

---

## 8. Phasing

Each phase is independently shippable, gated by tests, and lands on a sub-branch
off this one.

- **Phase 0 — scaffolding.** Add dependency; create `trl` impl + façade packages; `SFTConfig`/`DPOConfig` skeletons (fields + `__post_init__` validation only); contract tests (façade discipline, dependency direction) green. No trainer behavior yet.
- **Phase 1 — `SFTTrainer` (nll/dft).** Dataset tokenize + `language_modeling_collator` wiring; `compute_per_example_loss` override; completion-only loss; eval token-accuracy/entropy via `compute_metrics`. Numeric parity vs `examples/train_sft.py` and vs TRL at σ=0/C=∞. Example + cadence config.
- **Phase 2 — `DPOTrainer` (core heads + precompute).** Tokenize; `compute_ref_logprobs_for_dataset` precompute (explicit ref + PEFT null-ref); `preference_collator`; override with sigmoid/hinge/ipo/robust/apo/sigmoid_norm; MPO; reward metrics in eval. Parity vs `examples/train_dpo.py` and TRL. Example + cadence config.
- **Phase 3 — DPO breadth.** Remaining heads (exo/nca/bco/sppo/discopop/sft/squarechipo); f-divergence; WPO; LD-DPO; RPO; reference-free/CPO/ORPO/SimPO.
- **Phase 4 — SFT breadth.** `chunked_nll` (fused), `assistant_only_loss` (chat-template mask), `chat_template_path` cloning.
- **Phase 5 — on-the-fly references.** `_augment_batch` core hook (§6.4) + TR-DPO `ema_update_reference` callback.
- **Phase 6 — polish.** Packing/padding-free decision (§11), docs, final parity sweep.

---

## 9. Test strategy

Leverage existing unit coverage; add trainer-level tests.

1. **Unit (already present).** `packages/opaque-alignment/tests/{sft,dpo}/...` cover every loss/collator/reference primitive. No new unit work for the math.
2. **Trainer smoke / contract.** Mirror `tests/opaque_transformers/test_trainer_contract.py`: construct each trainer on a tiny model + synthetic dataset, run a few steps, assert state/metrics shape, checkpoint round-trip.
3. **TRL numeric parity (σ=0, C=∞).** With clipping off and zero noise, `compute_per_example_loss` mean over a fixed batch must match TRL's `compute_loss` to `1e-3` for each loss_type. Build a small fixture that runs TRL's loss on the same logits.
4. **Functional-example parity.** Trainer per-step loss must match the manual loops in `examples/train_{sft,dpo}.py` (same seed, same data) — these are the ground truth.
5. **DP-purity (NaN injection).** Replace one example's tensors with NaN; assert only that row's per-example gradient is affected (no cross-example leakage). One test per loss family. This is the structural DP-correctness guard.
6. **Reference precompute.** Verify cache keying (`cache_key`), that ref columns are populated for every row, and that `reference_free` skips them.
7. **Config validation.** `aot`/`aot_unpaired` rejected; `remove_unused_columns` forced False; label-smoothing bounds.
8. **DDP** (later): one short-run parity test under the existing `tests/distributed/` harness.

---

## 10. Examples & Cadence configs

Mirror the existing `train_causal_lm.py` (functional) ↔
`train_causal_lm_trainer.py` (`DPTrainer`) split. The functional siblings
already exist:

| Method | Functional (exists) | Trainer (new) |
|---|---|---|
| Causal LM | `examples/train_causal_lm.py` | `examples/train_causal_lm_trainer.py` |
| SFT | `examples/train_sft.py` | **`examples/train_sft_trainer.py`** |
| DPO | `examples/train_dpo.py` | **`examples/train_dpo_trainer.py`** |

The trainer examples are short: build `SFTConfig`/`DPOConfig`, load model +
LoRA, build dataset, instantiate the trainer, `.train()`. New cadence presets
`.cadence/configs/train_{sft,dpo}_trainer (qwen_alignment).yaml` clone the
existing `train_{sft,dpo} (qwen_alignment).yaml` and swap the entrypoint.

---

## 11. Risks, non-goals, and deferred features

**Open questions (decide before Phase 0):**

- **Namespace:** `opaque.transformers.trl` vs `opaque.transformers.alignment`. `trl` signals parity-of-intent; `alignment` avoids implying a TRL runtime dependency (there is none). *Recommendation: `opaque.transformers.trl`* (matches the pre-merge `trl-trainers-plan.md` §6).
- **Class names:** TRL-parity `SFTTrainer`/`DPOTrainer`/`SFTConfig`/`DPOConfig` (used throughout this doc) vs the `Opaque`-prefixed `OpaqueSFTTrainer`/`OpaqueSFTConfig` that the pre-merge `trl-trainers-plan.md` (§11.2–11.3, §12.2) proposed to avoid name collision when both TRL and Opaque are importable. *Decide before Phase 0; the prefix is the safer default if users may `import trl` in the same process.*
- **`DPOConfig.loss_type` length-normalized naming:** confirm the exact `simpo`/`ipo`/`sigmoid_norm` flag that toggles `sequence_logp(length_normalized=True)` vs `_norm` suffix handling.
- **DPO eval logits contract:** returning `chosen_out.logits` to satisfy `return_logits=True` is a convention, not a true "prediction"; confirm it doesn't confuse `compute_metrics` consumers.

**Deferred (explicit phases above):** `chunked_nll`, `assistant_only_loss`,
`chat_template_path`, all advanced DPO heads/features, TR-DPO, packing /
padding-free.

**Non-goals:** VLM/multimodal trainers and collators; DeepSpeed / FSDP /
Accelerate (Opaque has its own DDP layer — TRL `self.accelerator.*` and
`is_deepspeed_enabled`/`is_fsdp_enabled` guards are dropped); Liger as a runtime
dep (kernels reimplemented in `opaque-patches`); `aot`/`aot_unpaired` (DP-incompatible);
DP-PPO / trajectory-level DP; `KTOTrainer` (reuses the same primitives — natural
next trainer, but out of this plan's scope).

---

### Appendix — key file:line anchors

- Hook: `_dp_trainer.py:1955` (`compute_per_example_loss`), `:3002` (`_build_per_example_loss`), `:3123` (`_discover_batch_keys`), `:3628` (`_create_grad_fn`, `normalize_by`), `:2899`/`:2927` (signature columns), `:267` (`__init__`).
- Config: `_config.py:199` (`TrainingArguments`), `:345` (`remove_unused_columns`).
- Façade: `opaque/transformers/__init__.py`, `opaque/transformers/trainer/__init__.py`.
- Alignment API: §1.1 table.
- Functional refs: `examples/train_sft.py`, `examples/train_dpo.py:528-551`.
- TRL: `sft_trainer.py:932-1359` / `:815-829` (dft) / `:123-232` (chunked); `dpo_trainer.py:501-851` / `:90-211` (collator) / loss block; `dpo_config.py:211-304`; `sft_config.py:137-275`.
