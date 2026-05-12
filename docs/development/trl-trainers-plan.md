# TRL Trainers on Opaque — Implementation Plan

**Status:** Planning. Slow phased implementation against a moving main.

**Scope:**
- A new package `opaque-alignment` providing functional, mechanism-agnostic primitives for DP-safe preference learning (losses, collators, dataset transforms, reference-model helpers, alignment metrics).
- TRL-style class trainers `opaque.transformers.trl.{SFTTrainer, DPOTrainer, KTOTrainer}` plus matching configs, built on top of `DPTrainer` and consuming `opaque-alignment` primitives.
- Functional examples (`examples/train_sft.py`, `train_dpo.py`, `train_kto.py`) as siblings of trainer-based examples (`train_*_trainer.py`), mirroring the existing `train_causal_lm.py` ↔ `train_causal_lm_trainer.py` pattern.

**Branches:**
- Planning (this doc): `claude/add-trl-trainers-plan-nB07O`.
- Implementation phases: per-phase sub-branches off main once each phase is approved.
- Coordinating dependency: the full `DPTrainer` implementation that TRL subclasses inherit from currently lives on `evgri243/hf-trainer-continuation`. Phase 0 of this plan blocks on (or co-merges with) that branch.

---

## Table of contents

1. [Goals and non-goals](#1-goals-and-non-goals)
2. [Repository context: post-refactor layout and the api/façade pattern](#2-repository-context-post-refactor-layout-and-the-apifaçade-pattern)
3. [Architectural philosophy: two-layer split (`opaque-alignment` + `opaque.transformers.trl`)](#3-architectural-philosophy-two-layer-split-opaque-alignment--opaquetransformerstrl)
4. [Cross-cutting design decisions](#4-cross-cutting-design-decisions)
5. [`opaque-alignment` package design](#5-opaque-alignment-package-design)
6. [`opaque.transformers.trl` package design](#6-opaquetransformerstrl-package-design)
7. [Phase −1 — Kernel parity pass (`opaque-patches`)](#7-phase-1--kernel-parity-pass-opaque-patches)
8. [Phase 0 — DPTrainer foundational changes](#8-phase-0--dptrainer-foundational-changes)
9. [Phase 0.25 — `opaque-alignment` package skeleton](#9-phase-025--opaque-alignment-package-skeleton)
10. [Phase 0.5 — `opaque.distributed` extensions](#10-phase-05--opaquedistributed-extensions)
11. [Phase 1 — SFTTrainer + alignment primitives](#11-phase-1--sfttrainer--alignment-primitives)
12. [Phase 2 — DPOTrainer + alignment primitives](#12-phase-2--dpotrainer--alignment-primitives)
13. [Phase 3 — KTOTrainer + alignment primitives](#13-phase-3--ktotrainer--alignment-primitives)
14. [Phase 4 — Advanced data pipeline](#14-phase-4--advanced-data-pipeline)
15. [Phase 5 — Polish, examples, parity tests, docs](#15-phase-5--polish-examples-parity-tests-docs)
16. [Roadmap beyond this plan](#16-roadmap-beyond-this-plan)
17. [Risk register](#17-risk-register)
18. [DP correctness checklist (used at every loss port)](#18-dp-correctness-checklist-used-at-every-loss-port)
19. [Test strategy](#19-test-strategy)
20. [References](#20-references)
21. [Glossary](#21-glossary)

---

## 1. Goals and non-goals

### 1.1 Goals

- **New package `opaque-alignment`** with functional, mechanism-agnostic primitives for preference learning. Pure functions (no hidden state, no subclassing), composable with `vmap(grad(...))` per Opaque's design philosophy. Mirrors the role `opaque-engine` plays for DP-SGD/DP-FTRL primitives.
- **Three TRL-style trainers** under `opaque.transformers.trl`: `SFTTrainer`, `DPOTrainer`, `KTOTrainer`. Each thin orchestration over `DPTrainer` + `opaque-alignment` primitives.
- **Matching configs** `SFTConfig`, `DPOConfig`, `KTOConfig` extending `DPTrainingArguments` with TRL-parity fields.
- **Every TRL loss variant** that is mathematically compatible with per-example DP-SGD lands in the appropriate phase. Variants that structurally violate per-example DP (`aot`, `aot_pair` — cross-example batch sort) are deliberately rejected with a documented reason.
- **Every TRL advanced feature** that is compatible lands in some phase by the end of the plan. Features deliberately skipped (vision-language models) are documented in §16.
- **Strong DP-correctness story**: every per-example loss closure passes the checklist in §18. No cross-example data-dependent quantities inside the per-example loss (in particular, no `num_items_in_batch`-style global divisors — see §4.4).
- **TRL numeric parity at `σ=0, C=∞`** to within `1e-3` on representative batches.
- **Two example styles per method**: a functional `train_<method>.py` using primitives directly, and a `train_<method>_trainer.py` using the class wrapper.

### 1.2 Non-goals (this plan)

- **Vision / multimodal (VLM) trainers and collators** — out of scope.
- **DeepSpeed, FSDP, Accelerate-driven multi-device** — out of architectural reach for per-example DP-SGD. Opaque has its own DDP layer.
- **An `opaque.transformers.trl.X.accelerator` shim** — explicitly rejected; we expand `opaque.distributed` with functional primitives instead.
- **Liger as a runtime dependency** — Opaque reimplements equivalent kernels in `opaque-patches` (see [Phase −1](#7-phase-1--kernel-parity-pass-opaque-patches)).
- **TRL DPO/KTO loss variants whose math sorts or aggregates across the batch** (`aot`, `aot_pair`): rejected at init.
- **Mid-training mutation** of the model's PEFT adapter set, embedding size, or tokenizer vocabulary (must happen before trainer construction).
- **DP-PPO / DP policy-gradient** (arXiv:2501.19080) — trajectory-level DP is a fundamentally different design.

---

## 2. Repository context: post-refactor layout and the api/façade pattern

Main was substantially refactored. Plans written against the pre-refactor `opaque-core` layout (including earlier drafts of this doc) must be updated for two structural changes.

### 2.1 Package split

`opaque-core` was retired and replaced by three packages:

| Old distribution | New distribution(s) | What lives where |
|---|---|---|
| `opaque-core` (clipping, functional, scheduling, distributed, optimizers, serialization, random, profiling) | `opaque-base`, `opaque-engine`, `opaque-optimizers` | `opaque-base`: serialization. `opaque-engine`: clipping, functional, scheduling, distributed, random, profiling. `opaque-optimizers`: functional DP-aware optimizers. |

Current distribution inventory:

- `opaque-base` — serialization, base contracts.
- `opaque-engine` — clipping, functional, scheduling, random, profiling, distributed.
- `opaque-optimizers` — functional optimizers (`adamw`, `adam`, `sgd`, `lion`, `ademamix`, `adafactor`, `adagrad`, `adadelta`, `radam`, `rmsprop`, `schedule_free`).
- `opaque-dpsgd` — DP-SGD-specific noise, samplers, accounting integration.
- `opaque-dpftrl` — DP-FTRL mechanisms (matrix factorization, BLT, BSR, BiSR, band-MF, λ-CGD), specialized samplers.
- `opaque-patches` — HF Transformers compat patches + Triton kernels with vmap rules (the "Liger/Unsloth equivalent" reimplemented natively).
- `opaque-transformers` — HF Trainer compatibility layer (`DPTrainer`). Currently an empty skeleton on main; the full implementation lives on `evgri243/hf-trainer-continuation`.
- `opaque-accounting` — PLD privacy accounting (Rust via PyO3).
- `opaque-auditing` — empirical privacy auditing.

After this plan: `opaque-alignment` — functional primitives for preference learning.

### 2.2 The api/façade pattern

Every distribution now follows a strict separation of implementation namespace and public façade namespace:

```
packages/<dist>/src/opaque/
├── api/<dist-suffix>/                 ← IMPLEMENTATION namespace (owned by this dist)
│   └── <module>/
│       ├── __init__.py
│       ├── _private_impl.py
│       └── public_types.py
└── <module>/                          ← PUBLIC FAÇADE namespace (composes via PEP 420)
    └── __init__.py                    ← `from opaque.api.<dist-suffix>.<module> import …`
```

Verified across the refactored packages:

| Distribution | Owned impl namespace | Public façade(s) |
|---|---|---|
| `opaque-base` | `opaque.api.base.*` | `opaque.serialization` |
| `opaque-engine` | `opaque.api.engine.*` | `opaque.clipping`, `opaque.functional`, `opaque.scheduling`, `opaque.random`, `opaque.profiling`, `opaque.distributed` |
| `opaque-optimizers` | `opaque.api.optimizers.*` | `opaque.optimizers` |
| `opaque-transformers` | `opaque.api.transformers.*` | `opaque.transformers` |

Concrete example (`opaque-engine`):

```python
# packages/opaque-engine/src/opaque/clipping/__init__.py  — FAÇADE
from opaque.api.engine.clipping._auto import auto_clipped_grad
from opaque.api.engine.clipping._clipped_grad import clipped_grad
from opaque.api.engine.clipping._per_group import per_group
import opaque.api.engine.clipping._distributed  # noqa: F401  (side-effect: registers sync handlers)
```

Properties:

- No `__init__.py` at `opaque/` or `opaque/api/` (PEP 420 namespace) — multiple distributions compose without collision.
- Implementation modules can be reorganized within `opaque.api.<dist>.*` without breaking user imports, as long as the façade keeps re-exporting the public names.
- Sub-distributions get a stable "owned implementation" namespace where they can freely add internals.

### 2.3 Consequences for this plan

- All references to "lives in `opaque-core`" must be re-resolved to the correct successor.
- `opaque-alignment` adopts the same api/façade pattern.
- The `DPTrainer` code I read at `packages/opaque-transformers/src/opaque/api/transformers/trainer/_dp_trainer.py` (4656 LOC) does not currently exist on main; it lives on `evgri243/hf-trainer-continuation` and must merge before Phase 0 of this plan can land.

---

## 3. Architectural philosophy: two-layer split (`opaque-alignment` + `opaque.transformers.trl`)

### 3.1 The pattern Opaque already uses

```
DP primitives (functional, composable)              Class wrapper (orchestration)
──────────────────────────────────────────          ─────────────────────────────────
opaque-engine   (clipping, functional, …)       →   opaque.transformers.DPTrainer
opaque-dpsgd    (noise, samplers, accounting)
opaque-optimizers (adamw, sgd, …)

examples/train_causal_lm.py        (functional)     examples/train_causal_lm_trainer.py  (DPTrainer)
```

### 3.2 The same pattern, lifted one layer

```
Alignment primitives (functional, composable)        Class wrapper (orchestration)
──────────────────────────────────────────────       ──────────────────────────────────────────────
opaque-alignment  (losses, collators, data,       →  opaque.transformers.trl.{SFT,DPO,KTO}Trainer
                   reference, logprob, metrics,
                   kernels)

examples/train_dpo.py          (functional)          examples/train_dpo_trainer.py    (DPOTrainer)
examples/train_sft.py          (functional)          examples/train_sft_trainer.py    (SFTTrainer)
examples/train_kto.py          (functional)          examples/train_kto_trainer.py    (KTOTrainer)
```

### 3.3 Why this split

- **Researcher access**: comparing DP-DPO loss variants doesn't require subclassing `DPOTrainer`. Import `from opaque.alignment.losses.dpo import squarechipo, sigmoid`, write a 50-line training loop.
- **Mechanism-agnosticism**: `opaque-alignment` depends only on `opaque-engine` + `opaque-base` + `opaque-patches`. It works with DP-SGD via `opaque-dpsgd` *or* with DP-FTRL via `opaque-dpftrl`. The mechanism is picked by the caller (trainer or example), not by the library.
- **Thin trainers**: `DPOTrainer.compute_loss` becomes ~30 lines of orchestration; the loss math lives in one file with citable line numbers.
- **Reuse for future trainers**: `RewardTrainer`, `ORPOTrainer`, `CPOTrainer`, `SimPOTrainer`, `GRPOTrainer` all reuse the same primitives — no copy-paste of loss math between trainer subclasses.
- **Testing**: loss functions get unit-tested standalone; trainers get integration tests on top.

### 3.4 What lives where

| Concern | Lives in |
|---|---|
| Per-example loss math (DPO, KTO, IPO, SquareχPO, …) | `opaque.api.alignment.losses` |
| `selective_log_softmax`, `sequence_logp`, completion-mask handling | `opaque.api.alignment.logprob` |
| Preference data collators | `opaque.api.alignment.collators` |
| Dataset transforms (prompt extraction, packing, chat templates, KTO rotation) | `opaque.api.alignment.data` |
| Reference-logp precompute, PEFT-disable adapter context, TR-DPO EMA | `opaque.api.alignment.reference` |
| Reward metrics, KL estimators, token accuracy | `opaque.api.alignment.metrics` |
| Fused chunked-preference kernels (the Liger memory trick) | `opaque.api.patches.kernels` (base) + dispatcher in `opaque.api.alignment.losses` |
| Trainer classes, configs, signature columns, log() override | `opaque.api.transformers.trl` |
| DPTrainer generic features (compute_loss redirect, autocast ctx) | `opaque.api.transformers.trainer` (DPTrainer itself) |
| Cross-rank reductions for metrics / precompute | `opaque.api.engine.distributed` (functional primitives, not Accelerator-shaped) |

---

## 4. Cross-cutting design decisions

### 4.1 Reference-model strategy (DPO/KTO) — all four TRL paths supported

TRL has three explicit paths plus an auto-load fallback. All four are DP-safe and all four land in this plan. The strategy is selected at trainer-init time based on user-provided arguments.

| Path | Trigger | Implementation |
|---|---|---|
| **Explicit `ref_model=`** | User passes `ref_model` | Stored, moved to device, `disable_dropout_in_model`. Per-batch ref forward under `torch.no_grad()` **outside** vmap, in `_prepare_inputs`. Logps passed as constant tensor args into the vmap closure. |
| **PEFT `null_ref_context`** | `ref_model is None`, `is_peft_model(model)`, no precompute | Per-batch ref forward outside vmap, wrapped in `with self.null_ref_context(): ...`. Toggles PEFT adapter flag (instance attribute, not part of `state_dict()` — not captured by `functional_call`). Works cleanly because we toggle outside the vmap region. |
| **Precompute** | `args.precompute_ref_log_probs=True` | One-shot pass via `opaque.alignment.reference.compute_ref_logprobs_for_dataset(...)` before training. Adds dataset columns (`ref_chosen_logps`, `ref_rejected_logps` for DPO; `reference_logps`, `reference_KL_logps` for KTO). Collator emits them. Vmap closure reads them as plain tensor args. Cached to `.npz` keyed on `(dataset._fingerprint, hash_module(ref_or_policy))`. |
| **Auto-load** | All three above are absent and model is not PEFT | Load a second copy via `AutoModelForCausalLM.from_pretrained(get_config_model_id(self.model.config))`, treat as explicit `ref_model`. |

**DP semantics.** In all four paths, ref logps enter the per-example loss as constant tensors that depend only on (training_data_i × ref_params). Reference parameters are public; training data is private. The privacy unit is "all computations dependent on example i, including its ref logp." Per-example clipping covers the policy gradient that depends on these constants. The cache file is itself private (do not commit, do not log, do not release).

### 4.2 KTO sampler and KL term — TRL-faithful

TRL's KTO rotation happens at **dataset-prep time**, not at sampler time (`kto_trainer.py:609-623`):

1. `dataset.map(_get_kl_dataset, batched=True, batch_size=args.per_device_train_batch_size)` rotates `completion_ids` one position right within each map-batch.
2. The rotated dataset is column-renamed `completion_ids → KL_completion_ids`.
3. `concatenate_datasets([original, kl], axis=1)` produces a dataset where every row carries both `completion_ids` and `KL_completion_ids` (the rotated partner).
4. TRL uses a `SequentialSampler` to preserve the within-batch pairing.

Under Opaque's Poisson sampler, the rotation is already baked into the row, so any sampler that preserves row identity (Poisson does) works. The constraint is realized batch size ≥ 2 (TRL refuses at `<= 1` because rotation by 1 of a 1-element batch is identity, which collapses KL into the chosen log-ratio).

**Adaptation for Poisson:** when a realized batch arrives with size ≤ 1, the KL term falls back to 0 (equivalent to running `apo_zero_unpaired` for that step). Privacy unchanged because the per-example loss remains independent. This is a documented degradation but does not require changing TRL's algorithm.

**Self-pair degeneracy guard:** add a runtime assertion in `_prepare_dataset` that the rotation produced a non-identity permutation, since dataset chunks of size 1 under `dataset.map(batch_size=N)` could produce identity rotations.

### 4.3 DPO collator layout — `(B, 2, L)`

TRL uses `(2B, L)` with `[chosen..., rejected...]` concatenated along the batch dim, then `chunk(2, dim=0)` (`dpo_trainer.py:174,1189`). The reason is purely single-forward efficiency.

For DP-SGD, the privacy unit is the **pair** (chosen_i, rejected_i). The cleanest fit:

- Collator emits `(B, ...)` tensors with `chosen_input_ids`, `chosen_attention_mask`, `chosen_completion_mask`, `rejected_input_ids`, `rejected_attention_mask`, `rejected_completion_mask` as separate keys.
- Optional `ref_chosen_logps: (B,)`, `ref_rejected_logps: (B,)` when precomputed.
- The per-example loss closure runs **two forwards per pair inside vmap**: `fmodel(merged, chosen_*)` and `fmodel(merged, rejected_*)`. Vmap auto-batches over the pair index.

**Properties:** zero numeric difference vs `(2B, L)` (attention is per-row, no batch-norm in LLMs); slightly more kernel launches per step but irrelevant for per-pair DP-SGD where we have to per-example-clip anyway.

### 4.4 `num_items_in_batch` is private — DP-correct normalization rules

TRL's `num_items_in_batch` is the sum of `(labels != -100)` across the global accumulated batch. Under DP-SGD this is a data-dependent batch-level quantity. Any loss that divides by it embeds private cross-example information in the gradient magnitude.

**Per-example DP-SGD requires:** the per-example loss `L_i` must depend only on example `i`'s data. If `L_i = f_i / N` where `N = Σ_j (private function of example j)`, then `∇L_i = ∇f_i / N` carries data from `j ≠ i` into example `i`'s gradient. Breaks per-example independence → breaks DP accounting.

**Audit rule for every loss port:** every divisor in the per-example loss closure must be one of:

1. **Per-example computation** of example `i`'s data alone (e.g., `(labels_i != -100).sum()` for that example).
2. **`expected_batch_size`** (public hyperparameter; Opaque already uses this for `clipped_grad(..., normalize_by=...)` at `_dp_trainer.py:3689`).
3. **A public constant** from `args` (e.g., `args.max_length`, `args.beta`).
4. **Dropped entirely** and the normalization left to `clipped_grad`'s `normalize_by=expected_batch_size`.

**Loss-by-loss DP-correctness in TRL idioms:**

| Loss / formula | TRL idiom | DP-correct rewrite |
|---|---|---|
| SFT NLL via `model(**inputs).loss` | per-example mean over non-ignored tokens (HF default) | OK — per-example mean is per-example data |
| SFT DFT (`sft_trainer.py:788-802`) | `(per_token_loss * mask).sum() / num_items_in_batch` | Use per-example `mask.sum()` instead |
| SFT MoE aux loss | `coef * aux_loss` (constant coefficient) | OK |
| DPO sigmoid / IPO / hinge / robust / APO / etc. | per-pair scalar then `.mean()` over batch | OK — the batch `.mean()` happens *after* vmap'd per-example clipping; we drop it and rely on `normalize_by=expected_batch_size` |
| DPO IPO normalized | `(chosen_avg − rejected_avg)` where `avg = logp / completion_len` | OK — `completion_len` is per-example, public ratio |
| WPO weights | per-example `exp(mean_logps)` under `no_grad` | OK |
| LD-DPO | per-example `shared_logp + α·tail_logp` decomposition | OK |
| KTO chosen/rejected losses | per-example `1 - sigmoid(β·(...))` | OK |
| KTO `kl = (KL_logps - ref_KL_logps).mean()` over batch | **PROBLEMATIC** — this is a batch-level mean | Inside the vmap'd per-example closure, this is `(KL_logp_i - ref_KL_logp_i)` per example. The "cross-batch mean" of TRL emerges naturally from the per-example clipped gradient sum. **Verify during port.** |
| `cat(desirable_weight * chosen, undesirable_weight * rejected).nanmean()` (KTO) | `.nanmean()` over batch | The per-example loss is `desirable_weight * chosen_i OR undesirable_weight * rejected_i` (one or the other based on label). Drop the `.mean()`; rely on `normalize_by=expected_batch_size`. |
| `aot` / `aot_pair` (DPO) | sorts `logratios` across batch, applies sigmoid to sorted delta | **REJECTED** — sort-across-batch fundamentally breaks per-example DP |

**Validation:** every loss closure that lands gets a "DP-purity" unit test: replace one example's data with NaN; verify only that row's gradient is affected. See §19.5.

### 4.5 PEFT integration

- **Adapter clone for reference policy** (TRL pattern at `dpo_trainer.py:592-600`): `model.add_adapter("ref", model.peft_config["default"])` then copy `.default.` LoRA params into `.ref.`. Done *before* trainer construction; the resulting "ref" adapter is frozen.
- **`null_ref_context`** (`opaque.alignment.reference._adapter`): `@contextmanager` that:
  - If `model` is PEFT and has `"ref"` adapter: `model.set_adapter("ref")`, restore on exit.
  - Else if `model` is PEFT and only `"default"`: `model.disable_adapter()`.
  - Else: no-op.
- **Functional-call interaction:** PEFT's `_disable_adapters` / `active_adapter` flags are **instance attributes**, not in `state_dict()`. `functional_call(model, captured_params, ...)` does not capture these flags. Therefore toggling adapters **outside** the vmap region (in `_prepare_inputs` or `compute_loss`'s eager branch) is safe; toggling them **inside** a vmap'd `functional_call` is undefined and forbidden.
- **`make_functional` + PEFT:** trainable params = LoRA adapter tensors (`requires_grad=True`); frozen params = base model weights. Verified compatible by existing SFT path.
- **QLoRA bf16 promotion:** mirror TRL — after `get_peft_model`, cast `param.requires_grad` params to bf16 if model is 4-bit/8-bit loaded.
- **`merge_and_unload`:** must run *before* trainer construction (mutates base weights). Document.

### 4.6 Loss-type coverage

**SFT loss types** (per `OpaqueSFTConfig.loss_type`):

| Loss type | Phase | DP-correctness work |
|---|---|---|
| `nll` (default HF CE via `model(**inputs).loss`) | 1 | none (per-example mean) |
| `dft` (token-weighted detached CE) | 1 | rewrite `/ num_items_in_batch` to per-example `/ mask.sum()` |
| `chunked_nll` (logits never materialized) | 4 | implement via Opaque's existing `opaque_linear_cross_entropy_loss` (already vmap-safe; never materializes `(B, T, V)`) — cleaner than TRL's monkey-patch |

**DPO loss types** (per `OpaqueDPOConfig.loss_type` — list, supports MPO):

| Loss type | Phase | DP-correctness work |
|---|---|---|
| `sigmoid` (vanilla DPO) | 2 | none |
| `ipo` | 2 | use per-example `completion_len` |
| `hinge` | 2 | none |
| `robust` (label-smoothed) | 2 | none |
| `apo_zero` | 2 | none |
| `apo_down` | 2 | none |
| `exo_pair` (EXO) | 2 | none |
| `nca_pair` (NCA) | 2 | none |
| `bco_pair` (BCO) | 2 | none |
| `sppo_hard` (SPPO) | 2 | none |
| `discopop` (DiscoPOP) | 2 | none |
| `sft` (CE on chosen completion) | 2 | per-example |
| `sigmoid_norm` (length-normalized) | 2 | use per-example `completion_len` |
| **`squarechipo`** (arXiv:2505.21395, first optimal-rate DP-DPO) | 2 | none |
| `aot`, `aot_pair`, `aot_unpaired` | — | **REJECTED**: sort across batch |

**KTO loss types** (per `OpaqueKTOConfig.loss_type`):

| Loss type | Phase | DP-correctness work |
|---|---|---|
| `kto` | 3 | verify per-example KL term |
| `apo_zero_unpaired` (no KL term) | 3 | none |

**Cross-DPO features** (all Phase 2):

- f-divergence variants (`reverse_kl` default, `forward_kl`, `js_divergence`, `alpha_divergence`)
- MPO (`loss_type=list`, `loss_weights=list`) — trivial loop
- WPO (`use_weighting=True`) — incompatible with `aot*` (which we reject anyway)
- LD-DPO (`ld_alpha`) — per-example shared/tail decomposition
- TR-DPO (`SyncRefModelCallback`) — `TrainerCallback` firing `on_step_end` doing `ref ← (1−α)·ref + α·policy` EMA. Incompatible with PEFT and `precompute_ref_log_probs` (TRL precedent).

### 4.7 Optimizer integration — DP-AdamW already implemented

`opaque.optimizers.adamw(noise_bias_correction=True)` implements the DP-AdamW recipe from arXiv:2505.08849 verbatim. The v-moment correction `ṽ_t = max(v_t - (1 - β₂^t)σ², 0)` is the formula at `packages/opaque-optimizers/src/opaque/api/optimizers/_adam.py` (per the audit). Headline TRL recipe is `optim="adamw"` + `optim_args="noise_bias_correction=True"` in trainer configs. No new optimizer needed.

### 4.8 Kernel optimization — already covered

`opaque-patches` provides vmap-safe two-level (`Opaque_Foo / _FooBackward`) autograd.Function wrappers for:

- `opaque_cross_entropy_loss` — fused softmax+CE
- `opaque_linear_cross_entropy_loss` — fused linear+CE; never materializes `(B, T, V)`; with DP-aware `dC` skip when weight is frozen
- `opaque_swiglu`, `opaque_geglu_exact`, `opaque_geglu_approx`
- `opaque_rope`, `opaque_rope_qk`, `opaque_slow_rope`
- `opaque_rms_norm`, `opaque_fused_add_rms_norm`
- `opaque_lora_w`, `opaque_lora_qkv`, `opaque_lora_mlp`

Architectural coverage: Llama, Gemma/2/3, Qwen2/3, Mistral/Ministral, Olmo2/3, Phi3, GLM4, Granite, Cohere/2, SmolLM3, Exaone4 (`packages/opaque-patches/src/opaque/patches/transformers/models/`).

**The remaining gap is Liger's chunked preference loss pattern** (`LigerFusedLinearPreferenceBase`) — the alignment-specific chunked loop that avoids materializing per-chunk logits during DPO/KTO/ORPO/CPO/SimPO. Phase −1 addresses this.

### 4.9 No `self.accelerator` shim

Opaque deliberately doesn't use Accelerator. Rather than mimic its API, we **expand `opaque.distributed`** with the small set of functional primitives TRL needs (`gather_for_metrics`, `is_main_process`, `wait_for_everyone`). At port time, TRL idioms `self.accelerator.X(...)` get rewritten to `opaque.distributed.X(...)`. TRL guards like `if self.is_deepspeed_enabled` / `is_fsdp_enabled` get deleted (always False under DP).

### 4.10 No mutable dataset setters

Subclass `__init__` preprocesses the dataset (tokenize + ref-logp precompute) **before** calling `super().__init__()`, passing the preprocessed dataset in. No need to expose post-init setters on `train_dataset` / `eval_dataset`. Verified: only `args`, `processing_class`, `ref_model`, and `data_collator` are needed for preprocessing — all settable before super.

### 4.11 No `model_accepts_loss_kwargs` flag

The flag exists in HF Trainer to gate `num_items_in_batch` injection for grad-accum loss scaling. Under Opaque, `gradient_accumulation_steps` is reinterpreted as a Poisson sample-rate scaler; HF's grad-accum scaling path doesn't run. Flag is irrelevant.

### 4.12 Rename: `cpu_offload_activations` → `activation_offloading`

Existing arg at `_config.py:305` (wired at `_dp_trainer.py:1268-1270` via `torch.autograd.graph.save_on_cpu(pin_memory=True)`) is the same feature as TRL's `args.activation_offloading`. Rename for parity; keep `cpu_offload_activations` as a deprecated alias for one release.

---

## 5. `opaque-alignment` package design

### 5.1 Module layout (api/façade pattern)

```
packages/opaque-alignment/
├── pyproject.toml
├── README.md
└── src/opaque/
    ├── api/
    │   └── alignment/                          ← IMPLEMENTATION
    │       ├── __init__.py
    │       ├── losses/
    │       │   ├── __init__.py
    │       │   ├── _dpo.py                     # sigmoid, ipo, hinge, robust,
    │       │   │                               #   apo_zero, apo_down, exo_pair,
    │       │   │                               #   nca_pair, bco_pair, sppo_hard,
    │       │   │                               #   discopop, sft, sigmoid_norm,
    │       │   │                               #   squarechipo + LOSSES dict
    │       │   ├── _kto.py                     # kto, apo_zero_unpaired + LOSSES dict
    │       │   ├── _f_divergence.py            # reverse_kl, forward_kl, js, alpha
    │       │   ├── _mpo.py                     # combinator for multi-loss
    │       │   ├── _wpo.py                     # WPO per-example weight fn
    │       │   ├── _ld_dpo.py                  # shared/tail logp decomposition
    │       │   └── _fused.py                   # kernel-accelerated dispatchers
    │       │                                   #   wrapping opaque-patches base kernels
    │       ├── collators/
    │       │   ├── __init__.py
    │       │   ├── _language_modeling.py       # SFT collator (+ completion_mask, assistant_mask)
    │       │   ├── _preference.py              # DPO collator with (B, 2, L) layout
    │       │   └── _unpaired_preference.py     # KTO collator
    │       ├── data/
    │       │   ├── __init__.py
    │       │   ├── _prompt.py                  # extract_prompt
    │       │   ├── _packing.py                 # _pack_bfd, _pack_wrapped, _pack_bfd_split
    │       │   ├── _chat_template.py           # clone_chat_template, get_training_chat_template
    │       │   └── _kto_rotation.py            # _get_kl_dataset + concatenate_datasets glue
    │       ├── reference/
    │       │   ├── __init__.py
    │       │   ├── _precompute.py              # compute_ref_logprobs_for_dataset (cached)
    │       │   ├── _adapter.py                 # null_ref_context, with_disabled_adapter
    │       │   └── _sync.py                    # EMA update for TR-DPO
    │       ├── logprob.py                      # selective_log_softmax, sequence_logp, get_batch_logps
    │       └── metrics.py                      # reward metrics, KL estimator helpers
    └── alignment/                              ← PUBLIC FAÇADE
        ├── __init__.py                         # headline re-exports
        ├── losses/__init__.py                  # re-exports from opaque.api.alignment.losses
        ├── collators/__init__.py
        ├── data/__init__.py
        ├── reference/__init__.py
        ├── logprob.py                          # re-exports
        └── metrics.py                          # re-exports
```

### 5.2 Dependency pin (mechanism-agnostic)

```toml
# packages/opaque-alignment/pyproject.toml
[project]
name = "opaque-alignment"
dynamic = ["version"]
description = "Functional primitives for DP-safe preference learning (DPO, KTO, SFT)"
requires-python = ">=3.11,<3.13"
dependencies = [
    "torch>=2.10.0",
    "transformers>=4.57.0,<5",
    "datasets>=2.0.0",
    "peft>=0.18.0",
    "opaque-engine",         # clipping, functional, distributed primitives
    "opaque-base",           # serialization (for ref-logp cache state)
    "opaque-patches",        # fused preference kernels
    # NO opaque-dpsgd, NO opaque-dpftrl, NO opaque-optimizers
    # — mechanism + optimizer are chosen by the caller (trainer or example)
]

[tool.setuptools.packages.find]
where = ["src"]
include = ["opaque.alignment*", "opaque.api.alignment*"]
namespaces = true
```

### 5.3 Top-level public surface

```python
# opaque/alignment/__init__.py
from opaque.alignment.logprob import (
    sequence_logp, selective_log_softmax, get_batch_logps,
)
from opaque.alignment.losses import (
    DPO_LOSSES, KTO_LOSSES,
    f_divergence_remap, mpo_combine, wpo_weights, ld_dpo_split,
)
from opaque.alignment.collators import (
    DataCollatorForLanguageModeling,
    DataCollatorForPreference,
    DataCollatorForUnpairedPreference,
)
from opaque.alignment.data import (
    extract_prompt, pack_bfd, pack_wrapped, pack_bfd_split,
    clone_chat_template, get_training_chat_template,
    rotate_kto_completions,
)
from opaque.alignment.reference import (
    compute_ref_logprobs_for_dataset, null_ref_context, ema_update_reference,
)
from opaque.alignment.metrics import reward_metrics, kl_estimator
```

### 5.4 Functional examples (parallel to `train_causal_lm.py`)

The `examples/train_dpo.py` skeleton (full file in Phase 2):

```python
# Setup
model, tokenizer = load_model_and_tokenizer(...)
ref_model = load_model(...)
fmodel, trainable, frozen = make_functional(model, partition_trainable=True)

# Preprocess + precompute ref logps (functional, cached)
from opaque.alignment import (
    DataCollatorForPreference, compute_ref_logprobs_for_dataset, sequence_logp,
)
from opaque.alignment.losses import dpo_sigmoid

dataset = preprocess_preference(raw_dataset, tokenizer, max_length=1024)
dataset = compute_ref_logprobs_for_dataset(
    dataset, ref_model, collator=DataCollatorForPreference(...),
    cache_key=("dpo", "ref"),
    output_columns=("ref_chosen_logps", "ref_rejected_logps"),
)

# Per-example loss (uses primitives directly)
def per_example_loss(
    trainable_params,
    chosen_input_ids, chosen_attention_mask, chosen_completion_mask,
    rejected_input_ids, rejected_attention_mask, rejected_completion_mask,
    ref_chosen_logps, ref_rejected_logps,
):
    merged = {**frozen, **trainable_params}
    chosen_out = fmodel(merged, input_ids=chosen_input_ids,
                                 attention_mask=chosen_attention_mask)
    rejected_out = fmodel(merged, input_ids=rejected_input_ids,
                                   attention_mask=rejected_attention_mask)
    chosen_logp = sequence_logp(chosen_out.logits, chosen_input_ids, chosen_completion_mask)
    rejected_logp = sequence_logp(rejected_out.logits, rejected_input_ids, rejected_completion_mask)
    delta = (chosen_logp - ref_chosen_logps) - (rejected_logp - ref_rejected_logps)
    return dpo_sigmoid(beta=0.1, delta=delta)

# DP-SGD glue (mechanism choice happens here — could swap to DP-FTRL)
from opaque.clipping import clipped_grad
from opaque.dpsgd.noise import gaussian_noise          # ← user picks mechanism
from opaque.dpsgd.sampling import OpaqueEpochPoissonBatchSampler
from opaque.optimizers import adamw

grad_fn, clip_state = clipped_grad(
    per_example_loss, ..., normalize_by=expected_batch_size,
)
noise_fn, noise_state = gaussian_noise(...)
opt = adamw(noise_bias_correction=True)
opt_state = opt.init(trainable)

# Loop (identical pattern to train_causal_lm.py)
for batch in dataloader:
    (grads, aux), clip_state = grad_fn(trainable, *batch_args, state=clip_state)
    noised, noise_state = noise_fn(grads, noise_state)
    updates, opt_state = opt.update(noised, opt_state)
    trainable = apply_updates(trainable, updates)
```

### 5.5 Why mechanism-agnostic matters

A researcher running DP-FTRL DPO replaces three imports:

```python
# DP-FTRL variant — everything else unchanged
from opaque.dpftrl.matrix_factorization import band_mf
from opaque.dpftrl.sampling import b_min_sep_sampler
# …
noise_fn, noise_state = band_mf(...)
```

The alignment primitives (`dpo_sigmoid`, `sequence_logp`, `DataCollatorForPreference`, `compute_ref_logprobs_for_dataset`) don't care which mechanism is plugged in. This is the same separation `opaque-engine` already provides for DP-SGD primitives.

---

## 6. `opaque.transformers.trl` package design

### 6.1 Module layout

Lives inside `opaque-transformers` distribution, following the api/façade pattern:

```
packages/opaque-transformers/src/opaque/
├── api/
│   └── transformers/
│       ├── trainer/                      ← (existing, lives on hf-trainer-continuation)
│       │   └── _dp_trainer.py            ←   DPTrainer
│       └── trl/                          ← NEW
│           ├── __init__.py
│           ├── _sft_trainer.py
│           ├── _sft_config.py
│           ├── _dpo_trainer.py
│           ├── _dpo_config.py
│           ├── _kto_trainer.py
│           ├── _kto_config.py
│           ├── _rlhf_mixin.py            ← shared trainer plumbing
│           └── _callbacks.py             ← SyncRefModelCallback (TR-DPO)
└── transformers/
    ├── __init__.py                       ← existing façade (DPTrainer re-export)
    └── trl/                              ← NEW FAÇADE
        └── __init__.py                   ← from opaque.api.transformers.trl import …
```

### 6.2 Trainer responsibilities

Trainers are *thin orchestration*. Their job:

1. **Resolve args** to the appropriate config dataclass.
2. **Load model** from str via `AutoModelForCausalLM.from_pretrained` if needed; PEFT wrap; QLoRA bf16 promotion.
3. **Load processing class** (tokenizer / processor); default pad_token.
4. **Pick a default data collator** from `opaque.alignment.collators`.
5. **Preprocess dataset** via `opaque.alignment.data` helpers + own `_prepare_dataset` method.
6. **Precompute reference logps** (if requested) via `opaque.alignment.reference.compute_ref_logprobs_for_dataset`.
7. **Implement `compute_loss`** by orchestrating primitives from `opaque.alignment.losses` and `opaque.alignment.logprob`.
8. **Override `prediction_step`** for DPO/KTO (force `compute_loss` at eval to log reward metrics).
9. **Override `log`** to drain `self._metrics["train"|"eval"]` into the logs dict (the `_metrics` accumulator lives on the `RLHFMixin`).
10. **Override `_default_signature_columns`** with the appropriate fixed list.

### 6.3 Dependency pin

```toml
# packages/opaque-transformers/pyproject.toml — added pin
dependencies = [
    # ... existing pins ...
    "opaque-alignment",                  # new — primitives layer
]
```

### 6.4 Inheritance

```python
class OpaqueSFTTrainer(DPTrainer, RLHFMixin):
    def __init__(self, model, args=None, ...):
        # 1-7 above
        super().__init__(...)

    def _default_signature_columns(self):
        return ["input_ids", "labels", "attention_mask", "completion_mask", "assistant_masks"]

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        # Orchestrates opaque.alignment.losses.sft.* and opaque.alignment.logprob.*
        ...
```

Same pattern for DPO and KTO.

---

## 7. Phase −1 — Kernel parity pass (`opaque-patches`)

**Goal:** audit `opaque-patches`'s kernel coverage against current Liger and Unsloth, port any missing kernels or kwargs *natively* (not by depending on Liger/Unsloth at runtime).

### 7.1 Audit deliverables

1. **Comparison table** — every Liger kernel × every Unsloth kernel × every `opaque-patches` kernel, with status `(present|missing|partial)`.
2. **Per-architecture coverage matrix** — which models have full kernel routing (RoPE + RMSNorm + SwiGLU + LCE) in opaque-patches vs in Liger.
3. **Kwarg parity** — flag any per-kernel kwarg present in Liger/Unsloth but missing in opaque-patches (e.g., `softcap`, `logit_scaling`, `label_smoothing`, `lse_square_scale`/z-loss, `accum_dtype`).

### 7.2 Headline target: chunked preference loss

Liger's `LigerFusedLinearPreferenceBase` (pure PyTorch, nested `torch.func.grad_and_value` inside `autograd.Function`) is the one alignment-specific Liger trick not yet in `opaque-patches`. The win is ~80% peak-memory reduction for DPO/CPO/ORPO/SimPO by chunking the logits computation.

**Port plan:** reconstruct natively as `Opaque_FusedLinearPreference / _FusedLinearPreferenceBackward` in `packages/opaque-patches/src/opaque/api/patches/kernels/fused_linear_preference.py`, following the existing two-level pattern with explicit `vmap` rules. The base kernel is algorithm-agnostic; per-algorithm dispatchers (`opaque_fused_linear_dpo_loss`, `opaque_fused_linear_kto_loss`, etc.) wrap it and live in `opaque.api.alignment.losses._fused`.

### 7.3 Effort

**M (1–2 days)** for the audit + comparison table.
**L (3–5 days)** for the chunked preference base + first algorithm dispatcher (DPO).
**M per additional algorithm** (KTO, ORPO, CPO, SimPO) once the base is in place.

Phase −1 runs in parallel with Phase 0 and can land independently.

---

## 8. Phase 0 — DPTrainer foundational changes

**Prerequisite:** merge `evgri243/hf-trainer-continuation` into main so DPTrainer exists in the new repo layout.

### 8.1 Training-time `compute_loss` redirect — load-bearing

Today `_build_per_example_loss` (`_dp_trainer.py:3037` on hf-trainer-continuation) hard-codes `model(**inputs).loss` extraction. Subclass `compute_loss` overrides are silently ignored at training time.

**Fix:** refactor so the per-example loss closure delegates to `self.compute_loss(fmodel, kwargs, return_outputs=False, num_items_in_batch=None)`. Default `compute_loss` keeps its current `outputs.loss` behavior. Subclasses now override `compute_loss` once and it takes effect in both training and eval.

**Constraints made explicit in DPTrainer docs:**

- `compute_loss` runs *inside* `vmap(grad(...))` at train time. No `nn.Module` state mutation, no `torch.no_grad()` on captured module instance attributes, no `.item()` on dynamic shapes.
- At eval time (`self._ctx is None`), runs on bound `nn.Module`; full Python latitude.
- `model` arg is `fmodel` (callable taking merged params dict) during training, `self._model` during eval. Subclasses switch behavior via `self._ctx is not None`.

### 8.2 Smaller hooks

| Hook | Purpose |
|---|---|
| `compute_loss_context_manager()` | Returns `torch.autocast(...)` when `self._amp_dtype` set, else `nullcontext()`. HF parity. Used by DPO `prediction_step`. |
| `_default_signature_columns() -> list[str]` | Subclass override hook (vs monkey-patching `_set_signature_columns_if_needed`). Empty by default. |
| `_extra_forward_kwargs: set[str]` | Allowlist of kwarg names that bypass `remove_unused_columns` filtering. For TRL Liger flags (`skip_logits`, `return_token_accuracy`, `use_token_scaling`). |
| `_default_collator` factory hook | Subclass-overridable factory for a default collator when user doesn't pass one. |

### 8.3 Rename

`cpu_offload_activations` → `activation_offloading` (keep old name as deprecated alias for one release).

### 8.4 Phase 0 deliverables

- Patches to `packages/opaque-transformers/src/opaque/api/transformers/trainer/_dp_trainer.py` and `_config.py`.
- New tests in `tests/opaque_transformers/test_dp_trainer_subclass_compute_loss.py`:
  - Construct DPTrainer with a `compute_loss` override.
  - Confirm the override drives training (not just eval).
  - Confirm `_default_signature_columns` is honored.
  - Confirm `_extra_forward_kwargs` bypasses column filtering.
- No new public API in `opaque.transformers.trl` yet. Phase 0 is internal-facing.

**Effort: M (3–4 days).**

---

## 9. Phase 0.25 — `opaque-alignment` package skeleton

### 9.1 Deliverables

- New directory `packages/opaque-alignment/` with `pyproject.toml`, `README.md`, `src/opaque/api/alignment/`, `src/opaque/alignment/`.
- Empty `__init__.py` everywhere (the façade re-exports come in Phase 1–3 as primitives land).
- `pyproject.toml` per §5.2 (mechanism-agnostic deps).
- Register in workspace `Cargo.toml` / `uv.lock`.
- Add `opaque-alignment` to the all-packages tests / lints / smoke imports.
- CI matrix entry for `pytest packages/opaque-alignment/tests/`.
- README documenting the api/façade pattern, functional-primitives philosophy, mechanism-agnostic posture, and pointer to `examples/train_dpo.py`.

### 9.2 Smoke test

```python
# packages/opaque-alignment/tests/test_import.py
def test_namespace_composes():
    import opaque.alignment  # noqa: F401
    import opaque.api.alignment  # noqa: F401
```

**Effort: S (≤ 1 day).**

---

## 10. Phase 0.5 — `opaque.distributed` extensions

### 10.1 New functional primitives

Add to `packages/opaque-engine/src/opaque/api/engine/distributed/`:

| Function | Purpose |
|---|---|
| `gather_for_metrics(tensor: Tensor) -> Tensor` | All-gather across ranks; in DP context, duplicates from sampling aren't an issue. |
| `is_main_process() -> bool` | Rank-0 check. Wraps existing `DDPState`. |
| `wait_for_everyone() -> None` | Existing `barrier()` exposed at module level. |
| `num_processes() -> int` | World size. |
| `process_index() -> int` | Local rank. |

These are pure module functions, not class methods. Re-exported at `opaque.distributed.*`. TRL idioms get rewritten at port time:

```python
# TRL source                              # Opaque rewrite
self.accelerator.gather_for_metrics(x) →  gather_for_metrics(x)
self.accelerator.is_main_process       →  is_main_process()
self.accelerator.wait_for_everyone()   →  wait_for_everyone()
```

### 10.2 Effort

**S (1–2 days)** including tests.

Phase 0.5 can run in parallel with Phase 0 and Phase 0.25.

---

## 11. Phase 1 — SFTTrainer + alignment primitives

### 11.1 `opaque-alignment` primitives landing in Phase 1

- `opaque/api/alignment/logprob.py` — `selective_log_softmax`, `sequence_logp`, `get_batch_logps`.
- `opaque/api/alignment/losses/_sft.py` — `nll_loss` (per-example, mean over non-ignored tokens) and `dft_loss` (per-example, detached softmax weighting; DP-corrected divisor).
- `opaque/api/alignment/collators/_language_modeling.py` — `DataCollatorForLanguageModeling` with `completion_mask` and `assistant_masks` support.
- `opaque/api/alignment/data/_prompt.py` — `extract_prompt`.
- `opaque/api/alignment/data/_chat_template.py` — `apply_chat_template` glue + `get_training_chat_template` for `{% generation %}`-tagged templates. **No** `clone_chat_template` with embedding resize yet (Phase 4).
- `opaque/api/alignment/metrics.py` — `entropy_from_logits`, `mean_token_accuracy`, `num_tokens` aggregator.

### 11.2 `OpaqueSFTConfig` (`_sft_config.py`)

Extends `DPTrainingArguments`:

| Field | Default | Phase |
|---|---|---|
| `dataset_text_field` | `"text"` | 1 |
| `dataset_kwargs` | `None` | 1 |
| `dataset_num_proc` | `None` | 1 |
| `max_length` | `1024` | 1 |
| `truncation_mode` | `"keep_start"` | 1 |
| `eos_token` | `None` | 1 |
| `loss_type` | `"nll"` | 1 (`"nll"`, `"dft"`); 4 (`"chunked_nll"`) |
| `completion_only_loss` | `None` (auto) | 1 |
| `assistant_only_loss` | `False` | 1 (consume `{% generation %}` mask only; template *insertion* deferred to 4) |
| `chat_template_path` | `None` | 4 |
| `formatting_func` | `None` | 1 |
| `model_init_kwargs` | `None` | 1 |
| `activation_offloading` | `False` | 4 |
| `packing` | `False` | 4 |
| `packing_strategy` | `"bfd"` | 4 |
| `padding_free` | `False` | 4 |
| `pad_to_multiple_of` | `None` | 1 |

### 11.3 `OpaqueSFTTrainer` (`_sft_trainer.py`)

- `__init__`: model load, tokenizer load, optional PEFT wrap, QLoRA bf16, `_prepare_dataset` (uses `extract_prompt` + tokenize + completion-mask + optional assistant-mask), default collator = `DataCollatorForLanguageModeling`, `super().__init__()`.
- `_default_signature_columns`: `["input_ids", "labels", "attention_mask", "completion_mask", "assistant_masks"]`.
- `compute_loss`: dispatches on `loss_type`:
  - `"nll"`: `nll_loss(model, inputs)` from `opaque.alignment.losses.sft`. At train (model = fmodel) returns scalar per example; at eval (model = bound) returns batch mean.
  - `"dft"`: `dft_loss(model, inputs)` with per-example token-count divisor.
- Eval-time path: computes `entropy_from_logits`, `mean_token_accuracy`, `num_tokens`; pushes into `self._metrics["eval"]`.
- `log` override (from `RLHFMixin`): drains `_metrics[mode]` and merges into logs.

### 11.4 Examples

- `examples/train_sft.py` — functional, hand-rolled loop over alignment primitives.
- `examples/train_sft_trainer.py` — class-based via `OpaqueSFTTrainer`.

### 11.5 Phase 1 tests

- Loss-fn unit tests for `nll_loss`, `dft_loss` (vs hand-computed reference).
- Closure test: vmap'd per-example SFT loss over 4 examples → finite gradients.
- Trainer contract test: 2 training steps, eval, save_pretrained.
- TRL parity test: `OpaqueSFTTrainer(σ=0, C=∞)` vs `trl.SFTTrainer` on identical batch → loss within `1e-3`.

**Effort: M (4–5 days).**

---

## 12. Phase 2 — DPOTrainer + alignment primitives

The heaviest phase. All advanced DPO features land here (except chunked preference kernel, which is Phase −1).

### 12.1 `opaque-alignment` primitives landing in Phase 2

- `opaque/api/alignment/losses/_dpo.py` — all 13 DP-safe variants + `LOSSES` dict:
  - `sigmoid`, `ipo`, `hinge`, `robust`, `apo_zero`, `apo_down`, `exo_pair`, `nca_pair`, `bco_pair`, `sppo_hard`, `discopop`, `sft`, `sigmoid_norm`, `squarechipo`.
- `opaque/api/alignment/losses/_f_divergence.py` — `reverse_kl`, `forward_kl`, `js_divergence`, `alpha_divergence`.
- `opaque/api/alignment/losses/_mpo.py` — `mpo_combine(losses, weights)` combinator.
- `opaque/api/alignment/losses/_wpo.py` — `wpo_weights(...)` per-example weight fn (under `no_grad`).
- `opaque/api/alignment/losses/_ld_dpo.py` — `ld_dpo_split(per_token_logps, mask, alpha)`.
- `opaque/api/alignment/collators/_preference.py` — `DataCollatorForPreference` with `(B, 2, L)` layout per §4.3.
- `opaque/api/alignment/reference/_precompute.py` — `compute_ref_logprobs_for_dataset(dataset, ref_model, collator, ..., cache_key, output_columns)` with `.npz` caching.
- `opaque/api/alignment/reference/_adapter.py` — `null_ref_context(model)` context manager.
- `opaque/api/alignment/reference/_sync.py` — `ema_update_reference(ref_params, policy_params, alpha)` functional EMA.
- `opaque/api/alignment/metrics.py` — `reward_metrics(chosen_logratios, rejected_logratios, beta) -> dict`.

### 12.2 `OpaqueDPOConfig`

| Field | Default |
|---|---|
| `beta` | `0.1` |
| `loss_type` | `["sigmoid"]` (list, supports MPO) |
| `loss_weights` | `None` (equal weights) |
| `label_smoothing` | `0.0` |
| `ld_alpha` | `None` |
| `f_divergence_type` | `"reverse_kl"` |
| `f_alpha_divergence_coef` | `0.5` |
| `use_weighting` | `False` (WPO) |
| `discopop_tau` | `0.05` |
| `precompute_ref_log_probs` | `False` |
| `precompute_ref_batch_size` | `None` |
| `disable_dropout` | `True` |
| `sync_ref_model` | `False` (TR-DPO) |
| `ref_model_mixup_alpha` | `0.6` |
| `ref_model_sync_steps` | `512` |
| `max_length` | `1024` |
| `max_prompt_length` | `512` |
| `truncation_mode` | `"keep_start"` |
| `pad_to_multiple_of` | `None` |
| `dataset_num_proc` | `None` |

### 12.3 `OpaqueDPOTrainer`

- `__init__`: model + ref_model resolution per §4.1 (four paths), tokenizer, PEFT wrap (with QLoRA bf16 + ref-adapter clone if applicable), `_prepare_dataset`, optional `_precompute_ref_logps` via `opaque.alignment.reference.compute_ref_logprobs_for_dataset`. Reject `aot`/`aot_pair` at init.
- `_prepare_inputs` override: live-ref path computes ref logps under `torch.no_grad()` with `null_ref_context` *outside* vmap, injects into inputs.
- `compute_loss`: orchestrates `sequence_logp` × 2 (chosen + rejected forwards), `ld_dpo_split` if `ld_alpha`, `f_divergence_remap`, dispatch through `DPO_LOSSES` × `loss_weights`, optional `wpo_weights` multiplication.
- `prediction_step` override: force `compute_loss` at eval to log reward metrics (no labels otherwise).
- TR-DPO: `SyncRefModelCallback` registered when `sync_ref_model=True`. Uses `ema_update_reference` on the captured ref-params dict, not on a bound module.

### 12.4 Examples

- `examples/train_dpo.py` — functional with `dpo_sigmoid` directly.
- `examples/train_dpo_trainer.py` — class-based via `OpaqueDPOTrainer`.

### 12.5 Phase 2 tests

- Loss-fn unit tests for all 14 variants + 4 f-divergence remaps + MPO combinator.
- Closure test: vmap'd per-example DPO loss across 4 pairs → finite gradients.
- DP-purity test: NaN-one-example → only that row's grad affected.
- Trainer contract test (one per ref-model path): construct, train 2 steps, eval, save.
- TR-DPO callback test: ref params change after `ref_model_sync_steps`.
- TRL parity test on all loss variants at `σ=0, C=∞`.

**Effort: L (6–8 days).**

---

## 13. Phase 3 — KTOTrainer + alignment primitives

### 13.1 `opaque-alignment` primitives landing in Phase 3

- `opaque/api/alignment/losses/_kto.py` — `kto`, `apo_zero_unpaired` + `LOSSES` dict.
- `opaque/api/alignment/collators/_unpaired_preference.py` — `DataCollatorForUnpairedPreference`.
- `opaque/api/alignment/data/_kto_rotation.py` — `rotate_kto_completions(dataset, batch_size, seed)` = `_get_kl_dataset` + `concatenate_datasets(axis=1)`.
- Extension to `compute_ref_logprobs_for_dataset` to emit `reference_KL_logps` when KL is enabled.

### 13.2 `OpaqueKTOConfig`

| Field | Default |
|---|---|
| `beta` | `0.1` |
| `loss_type` | `"kto"` |
| `desirable_weight` | `1.0` |
| `undesirable_weight` | `1.0` |
| `precompute_ref_log_probs` | `False` |
| `precompute_ref_batch_size` | `None` |
| `disable_dropout` | `True` |
| `max_length` | `1024` |
| `model_init_kwargs` | `None` |
| `dataset_num_proc` | `None` |

### 13.3 `OpaqueKTOTrainer`

- `_prepare_dataset`: extract prompt → `unpair_preference_dataset` if needed → tokenize → if `calculate_KL`: `rotate_kto_completions(...)` → assert non-identity rotation.
- `compute_loss`:
  1. Forward `completion_input_ids` → `completion_logps` via `sequence_logp`.
  2. Forward `KL_completion_input_ids` under `torch.no_grad()` → `KL_logps`.
  3. Read ref logps from `inputs` (precomputed) or compute live (via `_prepare_inputs` + `null_ref_context`).
  4. `chosen_logratios` / `rejected_logratios` split by `label`.
  5. Compute KL per-example (not batch-mean — that emerges from the per-example gradient sum after vmap).
  6. Dispatch through `KTO_LOSSES`.
  7. Apply `desirable_weight` / `undesirable_weight`.
  8. Per-example loss returned.
- Skip-KL fallback when realized batch size ≤ 1 (Poisson edge case): set `kl=0` for that step.

### 13.4 Examples

- `examples/train_kto.py` — functional.
- `examples/train_kto_trainer.py` — class-based.

### 13.5 Phase 3 tests

- Loss-fn unit tests for `kto`, `apo_zero_unpaired`.
- Rotation correctness test (assert non-identity permutation after `dataset.map`).
- Trainer contract test with mixed labels.
- TRL parity at `σ=0, C=∞`.

**Effort: M–L (4–6 days).**

---

## 14. Phase 4 — Advanced data pipeline

These are workstreams independent of loss math; separated to keep Phases 1–3 reviewable.

### 14.1 SFT packing — `bfd`, `bfd_split`, `wrapped`

- `opaque.alignment.data._packing`: port `_pack_bfd` (segment-tree best-fit, ~90 LOC), `_pack_wrapped` (~15 LOC), `_pack_bfd_split` (~25 LOC).
- Generates `seq_lengths` column for `DataCollatorForLanguageModeling.get_position_ids_from_packed_seq_lengths`.
- **Attention requirements:** packing produces `position_ids` with per-doc restarts. The cross-doc attention blocking requires either:
  - **FlashAttention2** via `cu_seq_lens` — fastest, but requires `flash-attn` dep we don't have.
  - **FlexAttention** — PyTorch 2.5+ native, no extra dep, supports arbitrary score mods.
  - **SDPA with explicit block-diagonal 4D mask** — slowest but always works; defeats flash speed.
- **Open question:** does `torch.nn.functional.flex_attention` compose with `torch.func.vmap`? Phase 4 includes a fixture test. If yes, FlexAttention is the path. If no, fall back to SDPA-with-mask and accept the perf hit.

### 14.2 SFT `padding_free`

- Flatten batch into one row; rely on `position_ids` restart for doc separation.
- Same attention requirement as packing.

### 14.3 SFT `chunked_nll` via Opaque LCE

**Better than TRL's implementation.** TRL monkey-patches `model.forward` and gradient-checkpoints per-chunk. Opaque already has `opaque_linear_cross_entropy_loss` (`packages/opaque-patches/.../linear_cross_entropy.py`) which never materializes `(B, T, V)` and is vmap-safe via the `Opaque_LinearCrossEntropyLoss / _LinearCEBackward` two-level pattern, including a DP-aware `dC` skip when the weight is frozen.

**Implementation:** wire `loss_type="chunked_nll"` in `OpaqueSFTTrainer` to extract hidden states (skip `lm_head` in forward, possibly via the existing `_extra_forward_kwargs` allowlist or a small kernel adapter), then call `opaque_linear_cross_entropy_loss(hidden, lm_head.weight, labels, ...)`. No monkey-patching, no checkpoint trickery, no Liger dependency.

### 14.4 SFT `clone_chat_template` with embedding resize

- Port `clone_chat_template` from TRL (~210 LOC).
- Mutations (`add_tokens`, `resize_token_embeddings`, `pad_to_multiple_of=64`) happen **before** trainer construction. Document.
- PEFT case: mutate `peft_config.trainable_token_indices` + `modules_to_save = ["lm_head"]`.

### 14.5 `activation_offloading`

- Rename `cpu_offload_activations` → `activation_offloading` at DPTrainer level (Phase 0).
- Verify existing ctx-manager composes with `functional_call` under vmap (Phase 4 fixture).

### 14.6 Phase 4 tests

- Packing density vs target (BFD test on synthetic data).
- FlexAttention + vmap composition fixture.
- `chunked_nll` peak-memory test (< `(B, T, V)` materialization).
- `clone_chat_template` round-trip on a tokenizer.

**Effort: L (5–7 days).**

---

## 15. Phase 5 — Polish, examples, parity tests, docs

### 15.1 Examples

Six total (two per method):

- `examples/train_sft.py`, `train_sft_trainer.py` — code dataset (`bigcode/the-stack-smol` or `JetBrains/KExercises`).
- `examples/train_dpo.py`, `train_dpo_trainer.py` — code-leaning preference set (e.g., `argilla/distilabel-intel-orca-dpo-pairs`).
- `examples/train_kto.py`, `train_kto_trainer.py` — binary-labeled set.

### 15.2 TRL parity test (the strong correctness gate)

For each trainer:

1. Load TRL + Opaque with identical model, tokenizer, dataset, optimizer state.
2. Disable DP in Opaque (`privacy_noise_multiplier=0.0`, `clipping_norm=float("inf")`).
3. Single forward + backward pass on each.
4. Assert per-batch loss matches within `1e-3`.
5. Assert reward metrics (DPO) match within `1e-4`.

### 15.3 DP regression

50-step run per trainer at `ε=10`, fixed seeds. Snapshot final loss; track for drift across PRs.

### 15.4 Docs

- `docs/alignment/index.md` — package overview, functional philosophy, mechanism-agnostic posture.
- `docs/alignment/losses.md` — per-loss reference (formula, paper, DP notes).
- `docs/alignment/collators.md`.
- `docs/alignment/recipes.md` — pointers to functional examples + scripted recipes (decoupled DP-RLHF lives here as a notebook).
- `docs/trainers/sft.md`, `dpo.md`, `kto.md` — class-API docs, supported features, deferred features with paper-cited justification, ref-model path matrix.

**Effort: M (3–4 days).**

---

## 16. Roadmap beyond this plan

Sibling workstreams that become natural under the `opaque-alignment` + `opaque.transformers.trl` split:

| Item | Effort | Rationale |
|---|---|---|
| **`OpaqueRewardTrainer` (DP RM)** | M | Pairwise BT loss on `AutoModelForSequenceClassification`. Direct `DPTrainer` subclass using `opaque.alignment.losses` (add a `_reward.py` with the BT formula). Prerequisite for decoupled DP-RLHF (arXiv:2603.22563). |
| **Decoupled DP-RLHF recipe** | M | Notebook chaining `OpaqueRewardTrainer` → vanilla `trl.PPOTrainer` (non-DP actor). Lives in `opaque.alignment.recipes`. |
| **`OpaqueORPOTrainer`, `OpaqueCPOTrainer`, `OpaqueSimPOTrainer`** | M each | Different loss heads; thin trainer + new loss module. Pattern established by DPO. |
| **`OpaqueGRPOTrainer`** | L | Trajectory-level — needs `old_logps` / `ref_logps` plumbing similar to DPO precompute. Worth designing once on top of completed infrastructure. |
| **Liger native chunked-preference-loss port (full algorithm coverage)** | M (after Phase −1's base + DPO) | Add KTO, ORPO, CPO, SimPO dispatchers on the same `Opaque_FusedLinearPreference` base. |
| **DP-PPO (arXiv:2501.19080)** | L | Trust-region-coupled noise budget; fundamentally different architecture. Defer until demand. |
| **VLM trainers** | L+ | Out of scope this plan; revisit when vision-language models become a Opaque priority. |
| **Alignment-specific eval harnesses** | M | `opaque.alignment.eval.{reward_bench, alpaca_eval, kl_drift}`. Pure functions over `(policy_model, ref_model, dataset)`. Notebook-friendly. |
| **Recipe DSL** | L | `@register_recipe("sft+dpo")` for reproducible paper recipes (SquareχPO defaults, DP-AdamW + DPO, decoupled DP-RLHF). |

---

## 17. Risk register

| ID | Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|---|
| R1 | Some HF/TRL protocols still missing post-Phase-5 | M | L | Maintain a parity table in `docs/trainers/`. Implement over time, gated only by "doesn't violate DP." |
| R2 | `_discover_batch_keys` ordering instability across collator runs | L | H | Add explicit determinism check (collator on 2 rows, assert key-tuple equality). |
| R3 | PEFT mid-training mutations (e.g., `merge_and_unload`) | L | M | User-action only; document constraint. Outside vmap is safe per TRL mechanics finding. |
| R4 | `vmap(grad(...))` over `selective_log_softmax` produces wrong shapes for short sequences | M | H | Unit-test loss closure standalone vs TRL eager on 4–8 synthetic examples. |
| R5 | Reference / policy vocab mismatch silently corrupts loss | L | H | Validate `policy.config.vocab_size == ref_model.config.vocab_size` at init; raise. |
| R6 | KTO variance under Poisson with rare small batches | L | M | Skip KL term when realized batch ≤ 1 (per §4.2). Document. |
| R7 | DP-purity violation slips through (data-dependent cross-example divisor) | M | H | Mandatory DP-purity test (§19.5) for every loss closure. NaN one example, assert only that row's grad changes. |
| R8 | DP-AdamW formula drift vs paper | L | L | `opaque.optimizers.adamw(noise_bias_correction=True)` already matches arXiv:2505.08849. Treated as audit done. |
| R9 | `compute_loss` redirect (Phase 0.1) breaks existing DPTrainer subclasses | L | H | Keep `_build_per_example_loss` as an override surface; default delegates to `compute_loss`. Existing tests unchanged. |
| R10 | FlexAttention + vmap composition broken | M | M | Phase 4 fixture test. Fallback: SDPA + 4D block mask (slower). |
| R11 | DPTrainer hf-trainer-continuation merge slips | M | H | Plan Phase 0 in parallel; the changes are small enough that they can be re-applied on top of whatever lands. |
| R12 | Chunked preference kernel port (Phase −1) takes longer than estimated | M | L | Optional optimization; SFT/DPO/KTO ship without it via standard `opaque_linear_cross_entropy_loss`. |

---

## 18. DP correctness checklist (used at every loss port)

Apply to every per-example loss closure before it lands in `opaque.alignment.losses.*`:

- [ ] **No divisor that aggregates across the batch.** Specifically, no `num_items_in_batch`, no `batch_size`, no global token count. Allowed: per-example `mask.sum()`, `args.max_length`, `expected_batch_size`, or drop and let `clipped_grad` normalize.
- [ ] **No data-dependent computation that crosses examples.** Sorts, ranks, top-k, percentile — all forbidden (this is the `aot` family's failure mode).
- [ ] **No PEFT adapter toggling inside the closure.** Allowed only in `_prepare_inputs` pre-vmap.
- [ ] **No `torch.no_grad()` on module state.** Allowed: `torch.no_grad()` on tensor-only computations inside the closure (e.g., detached metric assembly).
- [ ] **No `.item()` on dynamic-shape tensors.** Breaks vmap.
- [ ] **No in-place mutation of input tensors.** Forbidden under vmap.
- [ ] **NaN-injection test passes.** Replace one example's input with NaN; only that example's gradient is affected (verifiable post-clipping).
- [ ] **`σ=0, C=∞` matches TRL eager.** Numeric within `1e-3` on synthetic batches.

---

## 19. Test strategy

### 19.1 Unit tests (per loss function in `opaque.api.alignment.losses`)

~10 tests per loss variant against hand-computed reference. Pure, no model.

### 19.2 Closure tests

Build `_build_per_example_loss`, call on a 4-example synthetic batch, verify shape, finite non-zero gradient w.r.t. `trainable_params`. No full training loop.

### 19.3 Trainer contract tests (mirror existing `test_trainer_contract.py` pattern)

Per trainer: construct, 2 training steps on tiny model + dataset, evaluate, save/load.

### 19.4 TRL parity test (per trainer per loss type)

Disable DP on Opaque side (`σ=0, C=∞`); compare per-batch loss to TRL within `1e-3` on the same first batch.

### 19.5 DP-purity test (per loss closure)

Replace one example's data with NaN; verify only that row's gradient is affected (after clipping). Catches accidental cross-example divisors.

### 19.6 DP regression (per trainer)

50 steps at `ε=10`, fixed seeds; snapshot final loss; track for drift.

---

## 20. References

### Opaque codebase (post-refactor, current main)

- `packages/opaque-engine/src/opaque/api/engine/clipping/_clipped_grad.py` — `clipped_grad`, the per-example DP primitive.
- `packages/opaque-engine/src/opaque/api/engine/functional/` — `make_functional`, `with_batch_dim`.
- `packages/opaque-engine/src/opaque/api/engine/distributed/` — DDP plumbing (extension target for Phase 0.5).
- `packages/opaque-optimizers/src/opaque/api/optimizers/_adam.py` — DP-AdamW (`noise_bias_correction=True`).
- `packages/opaque-patches/src/opaque/api/patches/kernels/linear_cross_entropy.py` — fused linear CE with vmap rules (the chunked-NLL replacement).
- `packages/opaque-patches/src/opaque/api/patches/kernels/{swiglu,geglu,rope_embedding,rms_norm,lora,fused_add_rms_norm}.py` — kernel coverage.

### Opaque codebase (hf-trainer-continuation branch — DPTrainer)

- `packages/opaque-transformers/src/opaque/api/transformers/trainer/_dp_trainer.py:3037` — `_build_per_example_loss` (Phase 0.1 redirect target).
- `packages/opaque-transformers/src/opaque/api/transformers/trainer/_dp_trainer.py:2054` — `compute_loss` (currently eval-only).
- `packages/opaque-transformers/src/opaque/api/transformers/trainer/_dp_trainer.py:3689` — `clipped_grad(..., normalize_by=expected_batch_size)`.
- `packages/opaque-transformers/src/opaque/api/transformers/trainer/_config.py:305` — `cpu_offload_activations` (rename target).

### TRL (analyzed at `/tmp/trl_src`, v1.5.0.dev0)

- `trl/trainer/sft_trainer.py:104-339,480-489,788-802,1302-1303` — chunked_nll, padding_free, DFT, activation offloading.
- `trl/trainer/dpo_trainer.py:763-797,799-816,976-994,1000-1084,1086-1474,1167,1189,1224-1252,1257-1402,1389-1400,1502-1521,1075-1188` — ref-model paths, TR-DPO, signature columns, precompute, full compute_loss + loss dispatch + WPO + LD-DPO.
- `trl/trainer/dpo_trainer.py:152-211` — `DataCollatorForPreference`.
- `trl/experimental/kto/kto_trainer.py:83-90,354-358,609-623,653-661,711-810,875-887,1173-1178` — KTO rotation, KL guard, null_ref_context, get_batch_logps, kto_loss math, sampler.
- `trl/trainer/utils.py:1056-1093` — `use_adapter` context manager.
- `trl/data_utils.py:686-789` — packing (BFD + wrapped).
- `trl/chat_template_utils.py:28-119` — `clone_chat_template`.

### HF Transformers (analyzed at `/tmp/transformers_src`, 5.8.0.dev0)

- `src/transformers/trainer.py:362,1870,1941,2883,3039` — `Trainer.__init__`, `training_step`, `compute_loss`, `prediction_step`, `_save_checkpoint`.

### Liger / Unsloth (analyzed at `/tmp/liger`, `/tmp/unsloth`, `/tmp/unsloth_zoo`)

- `liger_kernel/chunked_loss/fused_linear_preference.py` — chunked preference base (~80% memory reduction; Phase −1 port target).
- `liger_kernel/chunked_loss/{dpo,kto,cpo,orpo,simpo,grpo}_loss.py` — per-algorithm losses.
- `liger_kernel/transformers/monkey_patch.py:3411` — `_apply_liger_kernel` per-architecture dispatch (already mirrored in `opaque-patches`).
- `unsloth/models/dpo.py` — empty stub (no `FastDPOTrainer` exists; their "fast DPO" is TRL+Liger underneath).
- `unsloth_zoo/rl_replacements.py:549` — `UnslothEfficientGRPO` (not adoptable; incompatible with DP-SGD).

### DP-alignment papers (status against this plan)

- **arXiv:2505.21395** — SquareχPO (first optimal-rate DP-DPO). Lands as `loss_type="squarechipo"` in Phase 2.
- **arXiv:2505.08849** — DP-AdamW (+15% over prior baselines at ε∈[2,5]). Already implemented as `opaque.optimizers.adamw(noise_bias_correction=True)`.
- **arXiv:2603.22563** — Decoupled DP-RLHF. Roadmap recipe; built on `OpaqueRewardTrainer` (post-plan).
- **arXiv:2501.19080** — DP-PolicyGradient. Out of scope; trajectory-level DP is a different design.
- **arXiv:2510.21060** — Sample-complexity theory. Informational; cite in docs.

---

## 21. Glossary

- **DP-SGD** — Differentially Private Stochastic Gradient Descent. Per-example clip then aggregate, then add calibrated Gaussian noise. The standard "private SGD" mechanism.
- **DP-FTRL** — DP Follow-The-Regularized-Leader. Uses matrix factorization to correlate noise across steps; tighter privacy for the same utility under streaming / fixed-pass training. Opaque ships both mechanisms; `opaque-alignment` is mechanism-agnostic.
- **api/façade pattern** — Opaque's package layout where each distribution `<dist>` owns the implementation namespace `opaque.api.<dist-suffix>.*` and exposes a thin re-exporting façade at `opaque.<module>.*`.
- **functional model** — A model exposed as a callable taking `(params_dict, *args, **kwargs)`; obtained via `opaque.functional.make_functional(model, partition_trainable=True)`. Composes with `torch.func.vmap`/`grad`.
- **per-example clipping** — Bounding the L2 norm of each example's gradient to `C` before aggregation; gives a constant L2 sensitivity used by privacy accounting. Realized via `opaque.clipping.clipped_grad`.
- **`null_ref_context`** — Context manager that disables PEFT adapters (or activates a `"ref"` adapter) so a single model can serve as both policy and reference. Toggles instance attributes (`_disable_adapters`, `active_adapter`), not `state_dict()` entries. Safe outside vmap, forbidden inside vmap.
- **MPO** — Multi-loss DPO; `loss_type` is a list of variant names with corresponding `loss_weights`.
- **TR-DPO** — Trust-region DPO; reference parameters EMA-track the policy parameters at fixed step intervals.
- **WPO** — Weighted Preference Optimization; per-pair weight derived from the policy's marginal logp.
- **LD-DPO** — Length-Decoupled DPO; decomposes per-token logps into a shared prefix and a tail, weights the tail by `ld_alpha`.
- **SquareχPO** — Square chi-squared Preference Optimization; loss `0.5·(σ(βΔ) − 1)²`. First optimal-rate DP-DPO.
- **DP-purity** — Property of a per-example loss closure: its output for example `i` depends only on example `i`'s data. Required for per-example clipping to give a correct privacy bound.
