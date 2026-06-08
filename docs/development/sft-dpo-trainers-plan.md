# SFT & DPO Trainers on `DPTrainer` — Implementation Plan

**Status:** In progress. Scoped to two class trainers — `SFTTrainer` and
`DPOTrainer` — built on the in-house `DPTrainer` and consuming the
already-merged `opaque-alignment` primitives (PR #251).

**Implementation status (iteration 1 — complete):** landed on
`claude/modest-gates-WpC4d` (PR #253) — `opaque.transformers.trl.{SFTConfig,
SFTTrainer, DPOConfig, DPOTrainer}` with the `opaque-alignment` dependency wired
in, `examples/train_{sft,dpo}_trainer.py` + matching cadence presets, and a
hermetic test suite (28 tests: smoke, config-behavior, **DP-purity**
perturbation per-example independence, **numeric parity** of the vmapped
per-example loss vs a direct eager computation, plus TR-DPO / chunked_nll /
mixed-norm MPO). Contract tests green; runs on CPU, CUDA, and MPS CI.

- **SFT:** `nll` / `dft` / `chunked_nll` (fused logits-free CE via the
  `fused_linear_cross_entropy` patch); completion-only; `assistant_only_loss`;
  `chat_template_path` (clone + embedding resize); `activation_offloading` alias.
- **DPO:** precompute reference (explicit / PEFT null-ref / auto-load) +
  reference-free; heads sigmoid / hinge / ipo / robust / apo* / sigmoid_norm /
  exo / nca / bco / sppo / discopop / squarechipo / sft; **MPO with
  per-head mixed normalization**; f-divergence remap; **LD-DPO** (ld_alpha
  with a per-pair `shared_prefix_len`); **WPO** (`use_weighting`); **TR-DPO**
  (`sync_ref_model` — per-step EMA reference via the new
  `DPTrainer._augment_inputs` pre-vmap hook); full TRL telemetry (`rewards/*`,
  `logps/*`, `logits/*`, `entropy`, `mean_token_accuracy`) for both train and
  eval via the generic `(loss, aux)` seam. (See the parity-completion pass below
  for the 2026-06-04 follow-up: `rpo_alpha` dropped to match current TRL, SFT
  telemetry + `compute_loss_func`, `model_init_kwargs`, PEFT added-token
  trainability.)

**Reward telemetry — via a generic `(loss, aux)` harness seam.** `DPTrainer`
exposes a single rich seam, `compute_per_example_loss_and_metrics(fmodel,
params, inputs) -> (loss, telemetry_dict)`, defaulting to `(compute_per_example_loss(...), {})`
— so the simple trainers (SFT/causal-LM) override only the plain
`compute_per_example_loss` and are untouched, while a trainer that overrides the
seam auto-enables the aux path (detected by override; no flag). **Training:** the
per-example telemetry rides the clipped-grad `loss_aux` channel
(`_build_per_example_loss(with_metrics=…)` → `_create_grad_fn(has_aux=…)` →
`clipped_grad(has_aux=True)`), is DDP-summed by the existing `sync(aux)`, meaned
in `training_step`, and logged each step — same forward, live weights, and the
default (no-aux) path is **bit-identical** (gated off; reproducibility verified).
**Eval:** symmetric — `DPOTrainer` overrides the designed `prediction_step` seam
to produce per-example `(loss, rewards)` through the same functional context
(`self._ctx`, live weights), publishing rewards on `self._pending_eval_aux`;
`evaluation_loop` aggregates them into `eval_rewards/*` alongside `eval_loss`,
reusing the whole eval harness (memory/speed metrics, multi-dataset recursion,
`_after_evaluate`, accounting barrier). DPO supplies only content; the harness
does carry/aggregate/log for both train and eval — reusable for KTO. (This also
fixes DPO eval end-to-end: a preference batch — `chosen_input_ids` /
`rejected_input_ids`, no `labels` — couldn't run the inherited LM-shaped
prediction path at all.) TRL-as-test-dep numeric parity stays out: the heads are
unit-tested in `opaque-alignment` and the trainer is parity-tested against a
direct computation.

### Iteration 1 — parity-completion pass (2026-06-04)

A follow-up audit of the iteration-1 trainers against current TRL `main`
surfaced a handful of remaining gaps. Decisions (all kept inside iteration 1 —
faithful-TRL-baseline — not the iteration-2 redesign):

- **Verification posture.** No `trl` test dependency. Parity is established by
  re-derivation (heads unit-tested in `opaque-alignment`) plus a hand-checked
  HF-golden comparison, consistent with the iteration-1 stance above.
- **`rpo_alpha` — dropped.** It is **not** part of current TRL (`dpo_config.py`
  has no such field; it was removed upstream), so the config field
  (`_dpo_config.py`), the captured `self._rpo_alpha`, and the `dpo_loss` RPO
  block are gone. RPO-style chosen-NLL blending remains available explicitly via
  MPO (`loss_type=["sigmoid", "sft"]`).
- **Full DPO logged-metric parity.** `_reward_aux` now emits the complete TRL
  set — `rewards/*` **plus** `logps/{chosen,rejected}`, `logits/{chosen,rejected}`
  (masked-mean completion logit, `_masked_mean_logit`, vmap-safe), `entropy`, and
  `mean_token_accuracy` — all detached, riding the **same** clipped-grad
  `loss_aux` channel in training and the symmetric `_pending_eval_aux` at eval.
  No second forward: they are read off the chosen/rejected forwards already run.
  (TRL also tracks a cumulative `num_tokens` counter; that is a running session
  total, orthogonal to the per-example-mean aux channel, and is left to a
  separate trainer counter — deferred.)
- **SFT telemetry seam.** `SFTTrainer` now overrides
  `compute_per_example_loss_and_metrics` to emit `entropy` + `mean_token_accuracy`
  over the supervised (`!= -100`) tokens, computed from the model logits and so
  **independent of the loss head**. When logits are unavailable (the CUDA fused
  `chunked_nll` path) the aux dict is empty (fail-safe). The loss value and its
  gradient are unchanged — telemetry is detached.
- **`compute_loss_func` — supported per-example (was silently ignored).** On the
  `nll` path `SFTTrainer.compute_per_example_loss` now honours
  `compute_loss_func(outputs, labels) -> scalar` (the existing `DPTrainer`
  contract; runs inside `vmap`, so it must be a pure per-example op with no
  `num_items_in_batch` / batch coupling). `dft` / `chunked_nll` reject a custom
  loss at construction (they reduce their own loss; no logits to hand it). The
  **aux fail-safe** the reviewer asked for: telemetry is computed from logits, so
  a custom loss returning only a scalar still yields telemetry — the seam never
  depends on the loss fn providing aux.
- **`model_init_kwargs` added** to both configs and threaded into the
  string-`model` load path (`AutoModelForCausalLM.from_pretrained(model,
  **model_init_kwargs)`), matching TRL.
- **PEFT added-token trainability.** `clone_chat_template` now returns
  `(model, tokenizer, added_token_ids)`; `SFTTrainer` marks exactly those new
  embedding rows trainable (`peft_config.trainable_token_indices['embed_tokens']`)
  and keeps `lm_head` in `modules_to_save` (with a warning) before
  `get_peft_model`, mirroring `trl.SFTTrainer` (`sft_trainer.py:1064-1084`) — a
  frozen base would never learn an embedding for a token absent at pre-training.
- **`truncation_mode='keep_end'` — left cut.** Confirmed deprecated upstream
  (TRL warns and removes it in v2.0.0; `dpo_config.py:335-338`,
  `sft_config.py:297-300`). Tokenization keeps `keep_start` (TRL's default and
  forward path); the knob stays absent and is documented as such in both configs.
- **Stricter validations — kept.** Where Opaque raises and TRL warns/falls back
  (`f_alpha_divergence_coef == 1`; `robust` `label_smoothing` outside `[0, 0.5)`)
  we keep the fail-fast raise (`_f_divergence.py`, `_dpo_config.py`).
- **Deferred to iteration 2:** `IterableDataset` / streaming datasets, and the
  cumulative `num_tokens` counter.

### Iteration 2 — usage-driven refinement (2026-06-08)

Driven by a four-agent **usage** audit (examples/cadence, API ergonomics vs TRL,
docs, and the `opaque-alignment` functional layer) plus a maintainer review. The
implementation is correct but *ahead of its surface*: features are undocumented,
unexercised by examples, and a few primitives the trainers need aren't exposed.
This section records the resolved questions and the agreed work.

#### 2.1 Resolved questions (investigation)

- **Base `TrainingArguments` is already public and inherited.** It is re-exported
  from `opaque.transformers` (`transformers/__init__.py:23`) *and*
  `opaque.transformers.trainer`, and `SFTConfig`/`DPOConfig` extend it — so every
  base knob (`privacy_*`, clipping, sampling, lr/schedule, eval/save) is settable
  directly on the trainer configs. The audit's "P0: re-export the base config"
  is **withdrawn** — it is a *documentation* gap (users don't know the inherited
  fields exist), not an access gap. Action: the trainer docs (§2.4) point at the
  inherited surface; no façade re-export needed.
- **The DPO "late `ValueError`"** (`_dpo_trainer.py:396-401`) fires inside
  `__init__`'s `_precompute_ref_logps` (after dataset tokenization) when
  `ref_model=None`, the policy is not PEFT, and the model was passed **in-memory**
  (empty `_name_or_path` ⇒ no path to auto-load a reference from). Static
  precompute itself is intrinsic to per-example DP (the `vmap` loss can only read
  the reference as a constant column). Action: **hoist** the no-reference check to
  the top of `__init__` (before tokenization) so it fails fast, and reconsider
  whether the magic auto-load fallback should exist at all vs. requiring explicit
  `ref_model=`/`reference_free=True` (opaque-native: no magic).
  - **Implemented:** the no-reference check is **hoisted** to the top of
    `__init__` (fails fast before tokenize/precompute); the auto-load fallback is
    **kept** (now threading `model_init_kwargs`). The message points at the
    reference-free `loss_type`s instead of the removed `reference_free` flag.
- **`model_init_kwargs` is reverted.** It is net-new in `e9ccca8` (not a
  resurfaced feature — the only other history hits are the package add #251 and
  these plan docs), only parameterises the pre-existing string-`model` load path
  (`from_pretrained(**kwargs)`, itself from `d826668`), and is unused by any
  example. opaque deliberately cleaned the HPO model-init surface and the native
  flow is "pass an instantiated model". Action: **drop the `model_init_kwargs`
  field** from both configs and the `**` in the load path; keep the bare
  `from_pretrained(model)` string convenience.
  - **Implemented (revised — maintainer review):** *kept*, not dropped.
    `model_init_kwargs` gives TRL parity for the string-`model` load and is now
    also **threaded into the reference load** (the auto-loaded copy *and* a
    string `ref_model`), so policy and reference instantiate consistently
    (matching TRL, which reuses one dict for both). The "pass an instantiated
    model" flow is unaffected.
- **CPO / ORPO / SimPO fit as reference-free `DPOTrainer` heads, not new
  trainers.** Current TRL keeps them as *separate experimental* trainers
  (`trl/experimental/{cpo,orpo}/`); SimPO is a CPO loss variant; all are
  reference-free. `opaque-alignment` already exports `simpo_loss` and
  `odds_ratio_loss`, and `examples/train_dpo.py` already treats `simpo`/`cpo`/
  `orpo` as reference-free loss types (`_REFERENCE_FREE`, `train_dpo.py:209`).
  Action: surface them as `loss_type` values on `DPOTrainer` under
  `reference_free=True` (add `simpo`, `orpo`→`odds_ratio_loss`, and CPO as the
  reference-free `["sigmoid","sft"]` MPO blend) plus the few extra hyperparams
  (`simpo_gamma`, `cpo_alpha`, `orpo_lambda`). One DP preference trainer, many
  heads — matches opaque's functional philosophy and avoids TRL's trainer sprawl.
  - **Implemented (revised):** the public `reference_free` flag was **removed**
    entirely (not used as the surface). Reference-need is **derived from
    `loss_type`** via `_REFERENCE_FREE_HEADS = {sft, simpo, cpo, orpo}` — `sft`
    already proved the per-head pattern. `simpo` is a registry head; `cpo`/`orpo`
    are trainer-composed reference-free objectives (reusing `sigmoid_loss` /
    `odds_ratio_loss` / `chosen_nll_loss`, not new exported heads). `dpo_loss`
    takes both the log-ratio pair and the policy-logp pair so mixed MPO lists stay
    coherent. Flat params `simpo_gamma`/`cpo_alpha`/`orpo_lambda` as planned.
- **`aot` / unsupported `loss_type` keeps the bare `KeyError`** (no curated
  message, no shim). Confirmed maintainer decision — overrides the audit's
  "friendlier error" suggestion. Unsupported = the head does not exist.

#### 2.2 Quick-win fixes (correctness / parity — land in PR #253 or a fast follow-up)

- **Entropy telemetry off-by-one.** `entropy_from_logits` does not shift while
  `mean_token_accuracy` does, so the logged `entropy` mask is misaligned by one
  vs the next-token distribution. Fix in the primitive (`metric/_token.py`): shift
  `logits[...,:-1,:]` / `mask[...,1:]` like `mean_token_accuracy`, update both
  callers (`_sft_trainer.py`, `_dpo_trainer.py`) to pass the full-length mask, and
  re-check `examples/train_sft.py`'s usage.
- **`activation_offloading` everywhere, no back-compat.** Rename the base config
  field `cpu_offload_activations` → `activation_offloading` *directly* (drop the
  one-release deprecated-alias plan of §8.3/§4.12), remove `SFTConfig`'s separate
  `activation_offloading` field and the `_sft_trainer.py:120` alias shim. Both SFT
  and DPO then inherit the single field — closes the DPOConfig gap by construction.
- **Match TRL base defaults** on the trainer configs: `logging_steps=10`,
  `gradient_checkpointing=True`. `bf16`: TRL defaults it on when `fp16` is unset,
  but opaque raises on non-Ampere/CPU — so default it **conditionally** (on only
  when `torch.cuda.is_bf16_supported()`), not unconditionally; document the
  divergence otherwise.
- **Fix the doc prose drifts:** `docs/reference/alignment.md:106`
  (`clone_chat_template` now returns the 3-tuple `(model, tokenizer,
  added_token_ids)`) and `docs/alignment/index.md:41` ("14 heads" → 15 exported).
- **Unify DPO loss names + functional telemetry.** Rename
  `examples/train_dpo.py`'s `_DPO_LOSSES` keys to the TRL-canonical/trainer set
  (`exo_pair`, `nca_pair`, `bco_pair`, `sft`, `sigmoid_norm`) so a `--loss-type`
  copies cleanly between the functional and trainer styles; update `train_dpo.py`
  to emit the full telemetry set (it still imports only the legacy
  `reward_metrics`).

#### 2.3 Base / `DPTrainer` surface changes

- **Drop `gradient_accumulation_steps` entirely** from the `DPTrainer` /
  `TrainingArguments` surface (currently a read-only property pinned to `1`,
  `_config.py:1048`). It has no meaning under the Poisson per-example substrate;
  passing it should be an unknown-keyword `TypeError`, not a silently-honoured-as-1
  property. (Base-config change — coordinate with any DPTrainer callers.)
  - **Implemented (revised):** it is *already* a non-settable read-only property
    shim — a deliberate HF-utility compat (`transformers.modelcard`'s
    `extract_hyperparameters_from_trainer` and reporting callbacks read it off the
    args), so passing it as a kwarg **already raises `TypeError`** and it is not a
    field. Fully deleting the property would break those HF utilities for no
    user-visible gain, so it is kept; the actual work was removing the stale
    "usable knob" mentions from `docs/user-guide/huggingface/training-arguments.md`.
- **`per_device_train_batch_size` keeps its Poisson-expected-batch semantics** —
  by design, and not identical across samplers. No code change; document the
  meaning at the field level so a porting user isn't misled.

#### 2.4 Coverage work (examples, cadence, docs)

- **Examples & cadence.** The trainer scripts (`examples/train_{sft,dpo}_trainer.py`
  + their cadence configs) should exercise the trainer-level features that have
  zero coverage: `compute_loss_func`, `chunked_nll`, the PEFT added-token path
  (`chat_template_path` + `peft_config=` together), TR-DPO, MPO/`loss_weights`,
  WPO, LD-DPO, f-divergence, `assistant_only_loss`, and the new telemetry. Add a
  **mix** of trainer SFT/DPO examples + matching cadence presets. For features
  that live in opaque *itself* (the functional path), fold them into
  `examples/train_{sft,dpo}.py` where it makes sense.
- **Fused-path auto-activation.** The fused alignment primitives
  (`fused_sequence_logp`, `fused_nll_loss`, `fused_dft_loss`) should be selected
  **automatically when possible** (CUDA available *and* no logits-consuming
  feature is active), mirroring how `DPTrainer` resolves kernels — so
  `train_sft`/`train_dpo` and the trainers use the memory-efficient path by
  default "when no extra is assumed". **Tension to resolve:** the telemetry added
  in iteration-1 (`entropy`, `mean_token_accuracy`, `logits/*`) needs logits, so
  it must become **gated** (e.g. a `log_completion_metrics` toggle, default on)
  and the trainer falls back to the eager path when telemetry/`return_logits` is
  requested, fused otherwise.
- **User-facing docs (matching existing style/quality).** The TRL trainers have
  no guide, no reference autodoc, no `mkdocs.yml` nav entry, no README mention.
  Add: (a) an "SFT/DPO trainers" guide under `docs/alignment/` in the walkthrough
  style of `docs/alignment/sft.md`; (b) a structured reference section for
  `opaque.transformers.trl` in the prose style of `docs/reference/transformers.md`
  (which hand-documents `DPTrainer`/`TrainingArguments` by section) plus the
  inherited-base-config pointer from §2.1; (c) `mkdocs.yml` nav entries; (d) an
  `opaque-alignment` row + SFT/DPO mention in `README.md`'s package table.

**Author context:** Written against `main` at `909ed54` (opaque-alignment in
place) on branch `claude/modest-gates-WpC4d`. Supersedes the pre-merge draft
`docs/development/trl-trainers-plan.md` (which also covered building the
`opaque-alignment` package and a `KTOTrainer`; both are out of scope here).

**Reference TRL:** `huggingface/trl` `main` (cloned fresh for this analysis).
File:line references below point at `trl/trainer/sft_trainer.py`,
`sft_config.py`, `dpo_trainer.py`, `dpo_config.py`, `data_utils.py`,
`utils.py`, `callbacks.py`.

### Iteration model (read this first)

This is a **two-iteration** effort, and iteration 1 is deliberately *not* the
opaque-idiomatic end state:

- **Iteration 1 — faithful TRL baseline.** Mirror TRL's `SFTTrainer` /
  `DPOTrainer` as closely as the DP substrate allows: same **class structure,
  method names, config field names, and per-method responsibilities**
  (`_prepare_dataset`, `tokenize_row`, `concatenated_forward`, `dpo_loss`,
  reward telemetry, `compute_loss`, `DataCollatorForPreference`, …).
  The goal is a **close, diffable baseline** against upstream TRL, so a reviewer
  can read the two side by side and see exactly what changed and why. The DP
  loss/collator math is the merged `opaque-alignment` primitives (which already
  mirror TRL's formulas); the trainer just wires them through TRL-shaped methods.
- **Iteration 2 — reconsider & redesign (separate effort).** Once the baseline
  runs and matches TRL numerically, **stop**. Step back and decide, method by
  method, what is *auxiliary* (HF/Accelerate plumbing that DP doesn't need),
  what is *missing* relative to opaque / opaque-alignment (per-example DP
  semantics, accounting, mechanism-agnosticism), and redesign toward an
  opaque-native shape. This doc plans iteration 1 and only *flags* iteration-2
  candidates; it does not pre-commit the redesign.

### Design philosophy: a replica that is its own thing

`DPTrainer` is an HF-`Trainer` replica but deliberately **not a full-surface
mimic** — it cuts parameters and values that are incompatible with, or
meaningless under, per-example DP. The trainers inherit that stance:

- **No bespoke "rejection" code.** We do **not** write validation that raises
  hand-authored "X is rejected because DP" errors. Anything we don't support is
  simply **absent from the config surface** (so passing it is a standard
  unexpected-keyword `TypeError`) or **not wired into a dispatch table** (so an
  unsupported *value* like `loss_type="aot"` fails with an ordinary
  `KeyError`/lookup error). Unsupported = unknown, handled by standard Python /
  dataclass errors — not a curated rejection list.
- **Cut, don't stub.** Incompatible TRL features (DeepSpeed/FSDP/Accelerate,
  VLM, batch-coupled losses like `aot`/`aot_unpaired`, on-the-fly TR-DPO sync,
  padding-free) are left out of the config dataclass entirely rather than
  accepted-and-ignored. Their absence *is* the contract.

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

### 2.1a TRL method-name mirroring (iteration 1)

Mirror TRL's method decomposition **where each method carries real
responsibility** under DP, so the classes read as close analogues of upstream
and diff cleanly. The guiding rule (per design feedback): *don't manufacture a
structure that makes no sense under DP just to look like TRL.* Opaque and
`DPTrainer` differ in logic and approach — where TRL's batched shape is
meaningless for per-example DP, adapt or fold it into the hook rather than
keeping a vestigial parity shell.

| TRL method (`*_trainer.py`) | Iteration-1 analogue | Faithful? |
|---|---|---|
| `_prepare_dataset` | same name, same job (extract prompt, add EOS, tokenize → columns) | ✅ direct |
| `tokenize_row` / `_tokenize` | same | ✅ direct |
| `DataCollatorForLanguageModeling` / `DataCollatorForPreference` | same class names, wrapping `language_modeling_collator` / `preference_collator` | ⚠️ wraps opaque primitive (layout differs — §3.3) |
| `compute_ref_log_probs` / precompute | same name; runs `compute_ref_logprobs_for_dataset` | ⚠️ precompute-only (§3.2) |
| `dpo_loss(loss_type, …)` / SFT `dft_loss` | same name; enumerates the supported `loss_type`s and dispatches to the matching `opaque.alignment.dpo.loss` head | ✅ structure preserved |
| `get_batch_loss_metrics` (TRL's reward block) | the per-example `rewards/*` ride the generic `(loss, aux)` seam — logged each train step (clipped-grad aux) and aggregated in the eval loop (`eval_rewards/*`); no standalone method | ➖ adapted (seam, not a method) |
| `concatenated_forward` | **folded into `compute_per_example_loss`**, not reproduced as a standalone batched method. A batched `(2B,L)` forward that never drives the gradient would be dead weight under per-example DP. Its responsibility (two forwards → `{chosen_logps, rejected_logps}`) lives in the hook. | ➖ adapted (no vestigial shell) |
| `compute_loss` | the training gradient goes through `compute_per_example_loss` + `vmap`; we do **not** add a parallel batched `compute_loss` for grads. Any eval-only loss is read off the same per-example hook (`_get_eval_per_example_loss_fn`). | ➖ adapted |

So the only methods that change shape are exactly the ones whose TRL form is a
batched forward — and those are *adapted into the hook*, not kept as no-op
parity. Everything upstream of the forward (data prep, tokenization, collation,
ref precompute) and downstream of it (loss dispatch, reward/metric assembly)
keeps TRL's names and responsibilities verbatim.

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

### 3.3 Unsupported losses/params are *absent*, not rejected

Per the design philosophy (top of doc), we do not author rejection branches.
Batch-coupled losses (`aot`, `aot_unpaired`) sort log-ratios *across the batch*
before the sigmoid (`dpo_trainer.py` loss block), so the per-example gradient
would depend on other examples — meaningless under per-example DP. Rather than
"reject" them:

- `opaque-alignment` already ships **no** `aot` head, so the `dpo_loss`
  dispatch table simply has no `"aot"` key. `loss_type=["aot"]` fails with an
  ordinary `KeyError`/lookup error at dispatch — standard "unknown value", no
  curated message.
- Likewise, config fields with no DP meaning (DeepSpeed/FSDP/Accelerate knobs,
  VLM args, `sync_ref_model` for now, `padding_free` for now) are **omitted
  from the `SFTConfig`/`DPOConfig` dataclasses**, so passing them is a standard
  unexpected-keyword `TypeError`. Their absence is the contract.

The collator layout also already diverges at the primitive layer (separate
`chosen_*`/`rejected_*` keys vs TRL's `(2B, L)` concat); iteration 1 keeps the
TRL class name `DataCollatorForPreference` but wraps `preference_collator`.
Whether to converge or keep the divergence is an iteration-2 question.

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

`__post_init__`: force `remove_unused_columns=False` (the one DP-driven
override). No `packing`/`padding_free` fields exist on the dataclass at this
phase, so passing them is a standard unexpected-keyword `TypeError`; an unknown
`loss_type` value fails at the `nll`/`dft`/`chunked_nll` dispatch table, not via
a curated check.

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

`__post_init__`: force `remove_unused_columns=False`; default `loss_weights` to
`[1.0]*len(loss_type)`; keep TRL's *own* faithful validations (`label_smoothing
∈ [0,0.5)` for robust, `> 0` for exo — `dpo_trainer.py:680-694`). No DP-driven
rejections: `aot`/`aot_unpaired` have no dispatch key and fail at lookup (§3.3);
`sync_ref_model` is absent from the dataclass until its iteration-2 phase, so it
fails as a standard unexpected-keyword `TypeError`.

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
| `aot`, `aot_unpaired` | *no dispatch key* — `loss_type=["aot"]` fails with a standard `KeyError` (batch-sort, no DP meaning; not rejected by bespoke code) | — |

**Cross-features:** MPO (`loss_weights`) → `mpo_combine` (1); f-divergence →
`f_divergence_remap` (2); WPO (`use_weighting`) → `wpo_weights` (2); LD-DPO
(`ld_alpha`) → `sequence_logp(ld_alpha=)`/`ld_dpo_split` (2); RPO (`rpo_alpha`)
→ `chosen_nll_loss` blend (2); reference-free / CPO / ORPO / SimPO →
`odds_ratio_loss` / `simpo_loss` / reference-free `sigmoid_loss` (2).

---

## 8. Phasing

**Iteration 1 (Phases 0–4) = faithful TRL baseline.** Build the two trainers as
close structural ports of TRL, wiring the merged `opaque-alignment` primitives
through TRL-named methods (§2.1a). Each phase is independently shippable, gated
by tests, and lands on a sub-branch off this one.

- **Phase 0 — scaffolding. ✅ done.** Added `opaque-alignment` dependency + workspace source; created `trl` impl + façade packages; `SFTConfig`/`DPOConfig` dataclasses carrying the *supported* TRL fields only (incompatible fields simply absent — §3.3); `__post_init__` forces `remove_unused_columns=False` and defaults `loss_weights`. Contract tests green.
- **Phase 1 — `SFTTrainer` (nll/dft). ✅ done.** TRL-shaped `_prepare_dataset`/`tokenize_row` + language-modeling collator; `compute_per_example_loss` override; completion-only loss; activation-offloading alias. Example + cadence config. *Remaining:* eval token-accuracy/entropy logging; TRL numeric parity at σ=0/C=∞.
- **Phase 2 — `DPOTrainer` (core heads + precompute). ✅ done.** TRL-shaped `_prepare_dataset`/`tokenize_row`/`compute_ref_log_probs`/`dpo_loss`; the two-forwards step is folded into `compute_per_example_loss`, not a standalone batched method (§2.1a). Precompute reference (explicit ref + PEFT null-ref + auto-load); reference-free; sigmoid/hinge/ipo/robust/apo/sigmoid_norm + MPO. Example + cadence config. *Remaining:* reward-metric eval logging; TRL numeric parity.
- **Phase 3 — DPO breadth. ✅ done.** exo/nca/bco/sppo/discopop/sft/squarechipo heads, f-divergence remap, RPO, **LD-DPO** (ld_alpha + per-pair `shared_prefix_len`), reference-free, **WPO** (`use_weighting`), **mixed-normalization MPO** (per-head normalization; reference precomputed summed), and **TR-DPO** (`sync_ref_model`, via the new `DPTrainer._augment_inputs` pre-vmap hook + per-step EMA reference); `rewards/*` telemetry (train + eval) via the generic `(loss, aux)` seam. *Out of scope:* ORPO/SimPO/CPO are separate TRL trainer classes, not DPO `loss_type`s.
- **Phase 4 — SFT breadth. ✅ done.** `assistant_only_loss` (chat-template mask), `chat_template_path` (clone + embedding resize), and **`chunked_nll`** (model's fused logits-free CE via the `fused_linear_cross_entropy` patch) wired.

**🛑 Iteration-1 checkpoint — stop and reconsider.** With both trainers running
and matching TRL, pause. Audit method by method: which TRL methods are
*auxiliary* HF/Accelerate plumbing the DP path never exercises (candidates for
deletion); which TRL-named methods we kept only for diff-against-upstream and
could now rename/merge into the opaque-native flow; what opaque/opaque-alignment
offers that TRL has no concept of (accounting hooks, mechanism-agnostic
DP-SGD↔DP-FTRL swap, per-example DP telemetry). The output of this checkpoint is
the **iteration-2 redesign brief** — not planned here.

**Iteration 2 (post-checkpoint) = opaque-native redesign.** Driven by the brief
above. Likely items (not committed): rename/prune the TRL-shaped methods kept for
the baseline diff; converge or
formalize the collator-layout divergence; on-the-fly references (`_augment_batch`
core hook §6.4 + TR-DPO `ema_update_reference`); packing/padding-free decision
(§11); prune auxiliary config/methods.

---

## 9. Test strategy

Leverage existing unit coverage; add trainer-level tests.

1. **Unit (already present).** `packages/opaque-alignment/tests/{sft,dpo}/...` cover every loss/collator/reference primitive. No new unit work for the math.
2. **Trainer smoke / contract.** Mirror `tests/opaque_transformers/test_trainer_contract.py`: construct each trainer on a tiny model + synthetic dataset, run a few steps, assert state/metrics shape, checkpoint round-trip.
3. **TRL numeric parity (σ=0, C=∞).** With clipping off and zero noise, `compute_per_example_loss` mean over a fixed batch must match TRL's `compute_loss` to `1e-3` for each loss_type. Build a small fixture that runs TRL's loss on the same logits.
4. **Functional-example parity.** Trainer per-step loss must match the manual loops in `examples/train_{sft,dpo}.py` (same seed, same data) — these are the ground truth.
5. **DP-purity (NaN injection).** Replace one example's tensors with NaN; assert only that row's per-example gradient is affected (no cross-example leakage). One test per loss family. This is the structural DP-correctness guard.
6. **Reference precompute.** Verify cache keying (`cache_key`), that ref columns are populated for every row, and that `reference_free` skips them.
7. **Unsupported-arg behavior.** Assert `loss_type=["aot"]` raises a standard `KeyError` at dispatch (no bespoke rejection); assert an omitted field like `DPOConfig(sync_ref_model=True)` raises the standard dataclass `TypeError`; assert `remove_unused_columns` is forced `False`.
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
- **Class names:** since iteration 1 mirrors TRL, use the **TRL-parity names** `SFTTrainer`/`DPOTrainer`/`SFTConfig`/`DPOConfig` (recognizability beats collision-avoidance for a baseline; they live under the distinct `opaque.transformers.trl` namespace, so `from opaque.transformers.trl import SFTTrainer` never clashes with `from trl import SFTTrainer` at the symbol level — only if both are `import *`-ed). The `Opaque`-prefixed alternative from the pre-merge `trl-trainers-plan.md` (§11.2–11.3) is an iteration-2 reconsideration item, not a Phase-0 blocker.
- **`DPOConfig.loss_type` length-normalized naming:** confirm the exact `simpo`/`ipo`/`sigmoid_norm` flag that toggles `sequence_logp(length_normalized=True)` vs `_norm` suffix handling.
- **DPO eval logits contract:** returning `chosen_out.logits` to satisfy `return_logits=True` is a convention, not a true "prediction"; confirm it doesn't confuse `compute_metrics` consumers.

**Deferred (explicit phases above):** `chunked_nll`, `assistant_only_loss`,
`chat_template_path`, all advanced DPO heads/features, TR-DPO, packing /
padding-free.

**Non-goals:** VLM/multimodal trainers and collators; DeepSpeed / FSDP /
Accelerate (Opaque has its own DDP layer — TRL `self.accelerator.*` and
`is_deepspeed_enabled`/`is_fsdp_enabled` guards are dropped); Liger as a runtime
dep (kernels reimplemented in `opaque-patches`); `aot`/`aot_unpaired` (no
dispatch key — they have no per-example DP meaning; §3.3); DP-PPO /
trajectory-level DP; `KTOTrainer` (reuses the same primitives — natural next
trainer, but out of this plan's scope).

---

### Appendix — key file:line anchors

- Hook: `_dp_trainer.py:1955` (`compute_per_example_loss`), `:3002` (`_build_per_example_loss`), `:3123` (`_discover_batch_keys`), `:3628` (`_create_grad_fn`, `normalize_by`), `:2899`/`:2927` (signature columns), `:267` (`__init__`).
- Config: `_config.py:199` (`TrainingArguments`), `:345` (`remove_unused_columns`).
- Façade: `opaque/transformers/__init__.py`, `opaque/transformers/trainer/__init__.py`.
- Alignment API: §1.1 table.
- Functional refs: `examples/train_sft.py`, `examples/train_dpo.py:528-551`.
- TRL: `sft_trainer.py:932-1359` / `:815-829` (dft) / `:123-232` (chunked); `dpo_trainer.py:501-851` / `:90-211` (collator) / loss block; `dpo_config.py:211-304`; `sft_config.py:137-275`.
