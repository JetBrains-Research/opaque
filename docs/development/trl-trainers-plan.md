# TRL Trainers on Opaque — Implementation Plan

**Status:** Planning. Phased implementation; each phase is structured as a Claude Code dynamic-workflow run (parallel work units + adversarial validation — see §6.5 and `opaque-alignment-plan.md` §10.0).

**Scope:**
- A new package `opaque-alignment` providing functional, mechanism-agnostic primitives for DP-safe preference learning (losses, collators, dataset transforms, reference-model helpers, alignment metrics).
- TRL-style class trainers `opaque.transformers.trl.{SFTTrainer, DPOTrainer, KTOTrainer}` plus matching configs, built on top of `DPTrainer` and consuming `opaque-alignment` primitives.
- Functional examples (`examples/train_sft.py`, `train_dpo.py`, `train_kto.py`) as siblings of trainer-based examples (`train_*_trainer.py`), mirroring the existing `train_causal_lm.py` ↔ `train_causal_lm_trainer.py` pattern.

**Branches:**
- Planning (this doc): `claude/add-trl-trainers-plan-nB07O`, rebased onto `feat/dptrainer-main-integration` which has both the post-refactor layout and the full `DPTrainer` implementation integrated.
- Implementation phases: per-phase sub-branches off `feat/dptrainer-main-integration` (or its successor on main) once each phase is approved.

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

## 2. Repository context: package layout and the api/façade pattern

The implementer needs two facts about the repository: how the foundation is split across packages, and the api/façade module pattern every package follows.

### 2.1 Package inventory

The DP foundation is split across three packages — `opaque-base` (serialization), `opaque-engine` (clipping, functional, scheduling, distributed, random, profiling), and `opaque-optimizers` (functional DP-aware optimizers). The full distribution inventory:

- `opaque-base` — serialization, base contracts.
- `opaque-engine` — clipping, functional, scheduling, random, profiling, distributed.
- `opaque-optimizers` — functional optimizers (`adamw`, `adam`, `sgd`, `lion`, `ademamix`, `adafactor`, `adagrad`, `adadelta`, `radam`, `rmsprop`, `schedule_free`).
- `opaque-dpsgd` — DP-SGD-specific noise, samplers, accounting integration.
- `opaque-dpftrl` — DP-FTRL mechanisms (matrix factorization, BLT, BSR, BiSR, band-MF, λ-CGD), specialized samplers.
- `opaque-patches` — HF Transformers compat patches + Triton kernels with vmap rules (the "Liger/Unsloth equivalent" reimplemented natively).
- `opaque-transformers` — HF Trainer compatibility layer (`DPTrainer`). Full implementation present on `feat/dptrainer-main-integration` (the current base of this plan branch).
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

### 2.3 Where the implementation sits

- `opaque-alignment` follows the same api/façade pattern.
- The `DPTrainer` implementation lives on `feat/dptrainer-main-integration` (`_dp_trainer.py` is 4814 LOC, `_config.py` 1295 LOC); this plan's branch builds on it.

### 2.4 `_runtime_bootstrap.py`

`packages/opaque-transformers/src/opaque/api/transformers/_runtime_bootstrap.py` (63 lines) exposes:

- `apply_transformers_runtime_compat_patches()` — installs vmap-safe HF runtime shims (masking, collator, checkpoint).
- `is_patched() -> bool`, `is_vmap_patched() -> bool` — patch-state introspection.
- `patch_all()` — idempotent public entry point.

Called from `DPTrainer.__init__` at `_dp_trainer.py:416`. TRL trainers inherit this bootstrapping for free. Public functions are re-exported through `opaque.transformers.__init__` so user code can verify patch state or pre-apply.

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

- **Researcher access**: comparing DP-DPO loss variants doesn't require subclassing `DPOTrainer`. Import `from opaque.alignment.loss.dpo import DPO_LOSSES; DPO_LOSSES["squarechipo"]`, write a 50-line training loop.
- **Mechanism-agnosticism**: `opaque-alignment` depends only on `opaque-engine` + `opaque-base` + `opaque-patches`. It works with DP-SGD via `opaque-dpsgd` *or* with DP-FTRL via `opaque-dpftrl`. The mechanism is picked by the caller (trainer or example), not by the library.
- **Thin trainers**: `DPOTrainer.compute_per_example_loss` becomes ~30 lines of orchestration; the loss math lives in one file with citable line numbers.
- **Reuse for future trainers**: `RewardTrainer`, `ORPOTrainer`, `CPOTrainer`, `SimPOTrainer`, `GRPOTrainer` all reuse the same primitives — no copy-paste of loss math between trainer subclasses.
- **Testing**: loss functions get unit-tested standalone; trainers get integration tests on top.

### 3.4 What lives where

| Concern | Lives in |
|---|---|
| Per-example loss math (DPO, KTO, IPO, SquareχPO, …) | `opaque.api.alignment.loss.{dpo,kto,sft}` |
| `selective_log_softmax`, `sequence_logp`, completion-mask handling | `opaque.api.alignment.logprob` |
| Preference data collators | `opaque.api.alignment.collator` (factory functions) |
| Dataset transforms (prompt extraction, packing, chat templates, KTO rotation) | `opaque.api.alignment.data` |
| Reference-logp precompute, PEFT-disable adapter context, TR-DPO EMA | `opaque.api.alignment.reference` |
| Reward metrics, KL estimators, token accuracy | `opaque.api.alignment.metric` |
| Fused chunked-preference kernels (the Liger memory trick) | `opaque.api.alignment.kernel` (self-contained pure-PyTorch; no `opaque-patches` dep needed for the base) |
| Trainer classes, configs, signature columns, log() override | `opaque.api.transformers.trl` |
| DPTrainer generic features (subclass hooks: signature columns, extra forward kwargs, autocast ctx) | `opaque.api.transformers.trainer` (DPTrainer itself) |
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

**KL term — Tier 2 detached batch-mean.** The `kl` scalar is computed by the trainer in `_prepare_inputs`:

1. Per-microbatch policy KL forward through `fmodel` under `torch.no_grad()`, returning `policy_KL_logps: (B,)`.
2. Subtract precomputed `reference_KL_logps: (B,)` from the dataset columns.
3. Reduce: `kl = (policy_KL_logps - reference_KL_logps).mean().detach().clamp(min=0)` — scalar.
4. Broadcast into the vmap'd `compute_per_example_loss` as a `vmap`-`None` argument; each example sees the same `kl`.

This faithfully ports TRL `kto_trainer.py:882-884` and matches the KTO paper's stop-gradient requirement (Eq. 8). DP-correctness is per `opaque-alignment-plan.md` §3.3 Tier 2: detached aggregate with `O(1/n)` leverage, unreleased private information consumed inside the per-example loss. See also §9 of that doc for the DDP-aware extension (v2).

**Adaptation for Poisson:** when a realized batch arrives with size ≤ 1, the trainer passes `kl=0` (degenerates to `apo_zero_unpaired` math for that step). Privacy unchanged because the per-example loss remains bounded.

**Self-pair degeneracy guard:** add a runtime assertion in `_prepare_dataset` that the rotation produced a non-identity permutation, since dataset chunks of size 1 under `dataset.map(batch_size=N)` could produce identity rotations.

### 4.3 DPO collator layout — separate chosen/rejected keys

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
2. **`expected_batch_size`** (public hyperparameter; Opaque already uses this for `clipped_grad(..., normalize_by=...)` at `_dp_trainer.py:3654,3663,3673`).
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
| KTO `kl = (KL_logps - ref_KL_logps).mean().detach().clamp(min=0)` over batch | **Tier 2 — detached batch-mean aggregate, valid under DP-SGD** | Faithful to TRL `kto_trainer.py:882-884` and KTO paper Eq. (8) ("we do not backpropagate through z_0"). The kl scalar is computed **outside the vmap** in `_prepare_inputs` (under `no_grad` for the policy KL forward; ref KL precomputed). The trainer broadcasts it into each per-example closure as a `vmap`-`None` argument. DP-correct because (a) `.detach()` is enforced, (b) `O(1/n)` leverage per Kumar et al. arXiv:2310.03104. See `opaque-alignment-plan.md` §3.3 Tier 2 + §7.2. |
| `cat(desirable_weight * chosen, undesirable_weight * rejected).nanmean()` (KTO) | `.nanmean()` over batch | The per-example loss is `desirable_weight * chosen_i OR undesirable_weight * rejected_i` (one or the other based on label). Drop the `.mean()`; rely on `normalize_by=expected_batch_size`. |
| `aot` / `aot_pair` (DPO) | sorts `logratios` across batch, applies sigmoid to sorted delta | **Tier 3 — rejected.** Sort has `O(1)` leverage per swap; no published DP-safe variant. See `opaque-alignment-plan.md` §3.3 Tier 3. |

**Validation:** every loss closure that lands gets a "DP-purity" unit test: replace one example's data with NaN; verify only that row's gradient is affected. See §19.5.

### 4.5 PEFT integration

- **Adapter clone for reference policy** (TRL pattern at `dpo_trainer.py:592-600`): `model.add_adapter("ref", model.peft_config["default"])` then copy `.default.` LoRA params into `.ref.`. Done *before* trainer construction; the resulting "ref" adapter is frozen.
- **`null_ref_context`** (`opaque.alignment.reference._adapter`): `@contextmanager` that:
  - If `model` is PEFT and has `"ref"` adapter: `model.set_adapter("ref")`, restore on exit.
  - Else if `model` is PEFT and only `"default"`: `model.disable_adapter()`.
  - Else: no-op.
- **Functional-call interaction:** PEFT's `_disable_adapters` / `active_adapter` flags are **instance attributes**, not in `state_dict()`. `functional_call(model, captured_params, ...)` does not capture these flags. Therefore toggling adapters **outside** the vmap region (in `_prepare_inputs`, before the vmap'd `compute_per_example_loss`) is safe; toggling them **inside** a vmap'd `functional_call` is undefined and forbidden.
- **`make_functional` + PEFT:** trainable params = LoRA adapter tensors (`requires_grad=True`); frozen params = base model weights. Verified compatible by existing SFT path.
- **QLoRA bf16 promotion:** mirror TRL — after `get_peft_model`, cast `param.requires_grad` params to bf16 if model is 4-bit/8-bit loaded.
- **`merge_and_unload`:** must run *before* trainer construction (mutates base weights). Document.

### 4.6 Loss-type coverage

**SFT loss types** (per `OpaqueSFTConfig.loss_type`):

| Loss type | Phase | DP-correctness work |
|---|---|---|
| `nll` (default HF CE via `model(**inputs).loss`) | 1 | none (per-example mean) |
| `dft` (token-weighted detached CE) | 1 | rewrite `/ num_items_in_batch` to per-example `/ mask.sum()` |
| `chunked_nll` (math-equivalent to `nll`; logits never materialized) | 1 (alias) + 4 (kernel) | Registered as `SFT_LOSSES["chunked_nll"] = nll_loss` in Phase ε; the kernel-level chunked LCE path lands in Phase 4 via `opaque_linear_cross_entropy_loss` (already vmap-safe; never materializes `(B, T, V)`) — cleaner than TRL's monkey-patch. Per the audit, `chunked_nll` is purely a memory optimization, not a different loss. |

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

`opaque.optimizers.adamw(noise_bias_correction=True)` implements the DP-AdamW recipe from arXiv:2505.08849 verbatim. The v-moment correction `ṽ_t = max(v_t - (1 - β₂^t)σ², 0)` is at `packages/opaque-optimizers/src/opaque/api/optimizers/_adam.py:117` (kwarg) + `:191-203` (standard branch) + `:205-245` (BC branch — verified vs paper). Headline TRL recipe is `optim="adamw"` + `optim_args="noise_bias_correction=True"` in trainer configs. No new optimizer needed.

### 4.8 Kernel optimization — already covered

`opaque-patches` provides vmap-safe two-level (`Opaque_Foo / _FooBackward`) autograd.Function wrappers for:

- `opaque_cross_entropy_loss` — fused softmax+CE
- `opaque_linear_cross_entropy_loss` — fused linear+CE; never materializes `(B, T, V)`; with DP-aware `dC` skip when weight is frozen
- `opaque_swiglu`, `opaque_geglu_exact`, `opaque_geglu_approx`
- `opaque_rope`, `opaque_rope_qk`, `opaque_slow_rope`
- `opaque_rms_norm`, `opaque_fused_add_rms_norm`
- `opaque_lora_w`, `opaque_lora_qkv`, `opaque_lora_mlp`

Architectural coverage: Llama, Gemma/2/3, Qwen2/3, Mistral/Ministral, Olmo2/3, Phi3, GLM4, Granite, Cohere/2, SmolLM3, Exaone4 (`packages/opaque-patches/src/opaque/api/patches/transformers/models/`).

Kernels are gated by the **`use_performance_kernels` / `performance_kernels_config`** config fields (`_config.py:389,398`) — **not** TRL's removed `use_liger_kernel` / `liger_kernel_config`. HF compat patches are gated separately by `use_compat_patches` (`_config.py:406`, default `True`). TRL trainers inherit all three from `DPTrainingArguments`.

**The remaining gap is Liger's chunked preference loss pattern** (`LigerFusedLinearPreferenceBase`) — the alignment-specific chunked loop that avoids materializing per-chunk logits during DPO/KTO/ORPO/CPO/SimPO. Phase −1 addresses this.

### 4.9 No `self.accelerator` shim

Opaque deliberately doesn't use Accelerator. Rather than mimic its API, we **expand `opaque.distributed`** with the small set of functional primitives TRL needs (`gather_for_metrics`, `is_main_process`, `wait_for_everyone`). At port time, TRL idioms `self.accelerator.X(...)` get rewritten to `opaque.distributed.X(...)`. TRL guards like `if self.is_deepspeed_enabled` / `is_fsdp_enabled` get deleted (always False under DP).

### 4.10 No mutable dataset setters

Subclass `__init__` preprocesses the dataset (tokenize + ref-logp precompute) **before** calling `super().__init__()`, passing the preprocessed dataset in. No need to expose post-init setters on `train_dataset` / `eval_dataset`. Verified: only `args`, `processing_class`, `ref_model`, and `data_collator` are needed for preprocessing — all settable before super.

### 4.11 No `model_accepts_loss_kwargs` flag

The flag exists in HF Trainer to gate `num_items_in_batch` injection for grad-accum loss scaling. Under Opaque, `gradient_accumulation_steps` is reinterpreted as a Poisson sample-rate scaler; HF's grad-accum scaling path doesn't run. Flag is irrelevant.

### 4.12 Rename: `cpu_offload_activations` → `activation_offloading`

Existing arg at `_config.py:422` (wired via `torch.autograd.graph.save_on_cpu`) is the same feature as TRL's `args.activation_offloading`. Rename for parity; keep `cpu_offload_activations` as a deprecated alias for one release.

---

## 5. `opaque-alignment` package design

**The full spec lives in `docs/development/opaque-alignment-plan.md`.** That doc owns: module layout (api/façade with `loss/{dpo,kto,sft}/` sub-concerns), dependency pin (mechanism-agnostic; no `opaque-patches` core dep), public API surface, per-module spec, two-tier DP-purity invariant (Tier 1 / Tier 2 / rejected), DDP-aware loss design, per-phase plan (α through ι), test strategy, and functional examples.

Key cross-references this plan uses:
- §3.3 of opaque-alignment-plan — DP-purity tiers and the divisor rule.
- §7.1–§7.10 — per-module signatures and DP-correctness notes.
- §7.10 — alignment kernel catalog (lives inside `opaque-alignment.kernel`, not `opaque-patches`).
- §8.1 — Tier-1 vs Tier-2 caller responsibilities (relevant to KTO Phase 3).
- §9 — DDP-aware loss design (`LossAggregateSpec.cross_rank=True`); v2 follow-on.

The remaining sections of this plan reference `opaque.alignment.loss.{dpo,kto,sft}` and `opaque.alignment.{logprob,collator,data,reference,metric,kernel}` symbols by their public façade path. Layout details (which `_*.py` file holds which function) are not load-bearing here.

---

## 6. `opaque.transformers.trl` package design

### 6.1 Module layout

Lives inside `opaque-transformers` distribution, following the api/façade pattern:

```
packages/opaque-transformers/src/opaque/
├── api/
│   └── transformers/
│       ├── trainer/                      ← (existing, on feat/dptrainer-main-integration)
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
4. **Pick a default data collator** from `opaque.alignment.collator` (factory functions returning callables — `preference_collator(...)`, etc.).
5. **Preprocess dataset** via `opaque.alignment.data` helpers + own `_prepare_dataset` method.
6. **Precompute reference logps** (if requested) via `opaque.alignment.reference.compute_ref_logprobs_for_dataset`.
7. **Implement `compute_per_example_loss(fmodel, params, inputs, *, return_logits=False)`** — the single vmap-safe override hook calling into `opaque.alignment.loss.{dpo,kto,sft}.*` + `opaque.alignment.logprob.*`. Covers both training (vmap'd → clip → noise) and per-example eval by construction.
8. **Override `prediction_step`** for DPO/KTO to surface reward metrics at eval (the eval loop already drives `compute_per_example_loss`; the override formats the metric payload).
9. **Override `log`** to drain `self._metrics["train"|"eval"]` into the logs dict (the `_metrics` accumulator lives on the `RLHFMixin`).
10. **Override `_default_signature_columns`** with the appropriate fixed list (keeps preference columns through `_remove_unused_columns`).

### 6.3 Dependency pin

```toml
# packages/opaque-transformers/pyproject.toml — added pin
dependencies = [
    # ... existing pins ...
    "opaque-alignment",                  # new — primitives layer
]
```

### 6.4 Inheritance + single-hook pattern

```python
class OpaqueDPOTrainer(DPTrainer, RLHFMixin):
    def __init__(self, model, ref_model=None, args=None, ...):
        # 1-6 above
        super().__init__(...)

    def _default_signature_columns(self):
        return ["chosen_input_ids", "chosen_attention_mask", "chosen_completion_mask",
                "rejected_input_ids", "rejected_attention_mask", "rejected_completion_mask",
                "ref_chosen_logps", "ref_rejected_logps"]

    def compute_per_example_loss(self, fmodel, params, inputs, *, return_logits=False):
        # THE SINGLE override. vmap-safe; covers training and per-example eval.
        # Forwards chosen + rejected through fmodel(params, **kwargs), computes
        # sequence_logp for each, subtracts the (precomputed or live) ref logps,
        # dispatches through opaque.alignment.loss.dpo.DPO_LOSSES[self.args.loss_type[0]].
        # When return_logits=True (eval), also returns the logits payload so the
        # trainer's eval hook can compute reward metrics.
        ...
```

SFT and KTO follow the same single-hook pattern. The loss math lives in `opaque.alignment.loss.{dpo,kto,sft}.*`; the trainer hook only orchestrates. Reward / token-accuracy / KL metrics are computed by the trainer (from the `return_logits=True` payload) and accumulated into `self._metrics`, not inside `compute_per_example_loss`.

### 6.5 Execution model (Claude Code dynamic workflows)

Each phase below carries a **work-unit DAG** so it can be run as a Claude Code dynamic workflow (one phase = one workflow run, with human sign-off between phases). The shared legend — work-unit schema, the 16-concurrent / 1000-total runtime limits, shared-file ownership via a terminal wire-up unit, worktree-per-unit isolation, and the adversarial-reviewer convergence pattern — is defined once in **`opaque-alignment-plan.md` §10.0**; it applies verbatim here. Phase branches are named `phase/trl-<n>`.

One additional gate profile (beyond the six in §10.0) covers the trainer classes:

- **`trainer`** (the TRL trainer classes + configs): construct + 2 training steps + eval + save (contract test) · TRL parity at σ=0,C=∞ within 1e-3 · **GPU smoke train** (2 steps, 2-layer model) · **ε-budget regression** (50-step run at ε=10; snapshot final loss) · end-to-end DP-purity (NaN one example in a real batch; only that row's grad moves) · ruff + format.

---

## 7. Phase −1 — Kernel parity pass (`opaque-patches`)

**Goal:** audit `opaque-patches`'s kernel coverage against current Liger and Unsloth, port any missing kernels or kwargs *natively* (not by depending on Liger/Unsloth at runtime).

### 7.1 Audit deliverables

1. **Comparison table** — every Liger kernel × every Unsloth kernel × every `opaque-patches` kernel, with status `(present|missing|partial)`.
2. **Per-architecture coverage matrix** — which models have full kernel routing (RoPE + RMSNorm + SwiGLU + LCE) in opaque-patches vs in Liger.
3. **Kwarg parity** — flag any per-kernel kwarg present in Liger/Unsloth but missing in opaque-patches (e.g., `softcap`, `logit_scaling`, `label_smoothing`, `lse_square_scale`/z-loss, `accum_dtype`).

### 7.2 Headline target: chunked preference loss

Liger's `LigerFusedLinearPreferenceBase` (pure PyTorch, nested `torch.func.grad_and_value` inside `autograd.Function`) is the one alignment-specific Liger trick not yet in `opaque-patches`. The win is ~80% peak-memory reduction for DPO/CPO/ORPO/SimPO by chunking the logits computation.

**Port plan:** reconstruct natively as `Opaque_FusedLinearPreference / _FusedLinearPreferenceBackward` in `packages/opaque-alignment/src/opaque/api/alignment/kernel/_fused_linear_preference.py` (per `opaque-alignment-plan.md` §7.10 — alignment-specific kernels live inside alignment, not opaque-patches). Two-level pattern with explicit `vmap` rules. Per-algorithm dispatchers (`opaque_fused_linear_dpo_loss`, `opaque_fused_linear_kto_loss`) wrap it in `opaque.api.alignment.kernel._dpo_dispatch` / `_kto_dispatch`.

### 7.3 Effort

**M (1–2 days)** for the audit + comparison table.
**L (3–5 days)** for the chunked preference base + first algorithm dispatcher (DPO).
**M per additional algorithm** (KTO, ORPO, CPO, SimPO) once the base is in place.

Phase −1 runs in parallel with Phase 0 and can land independently.

---

## 8. Phase 0 — DPTrainer foundational changes

**Prerequisite:** none — `feat/dptrainer-main-integration` already carries the integrated DPTrainer. This phase adds small hooks to DPTrainer to make TRL-style subclassing ergonomic.

### 8.1 The unified `compute_per_example_loss` hook

DPTrainer exposes a **single** vmap-safe override hook that covers both training and per-example eval:

```python
def compute_per_example_loss(
    self,
    fmodel: Callable[..., Any],
    params: dict[str, Tensor],
    inputs: dict[str, Tensor],
    *,
    return_logits: bool = False,
) -> Tensor | tuple[Tensor, Any]:
    """The unified DP-correct override hook (_dp_trainer.py:1955).

    The trainer wraps it with vmap for training (then grad → clip →
    noise) AND for per-example eval (when 'loss' in include_for_metrics).
    Subclasses (SFT, DPO, KTO) override this one method — the same
    override point covers training and eval semantics by construction.
    """
```

| Symbol | Lives at | Role |
|---|---|---|
| `compute_per_example_loss(self, fmodel, params, inputs, *, return_logits)` | `_dp_trainer.py:1955` | **The single override hook.** vmap-batched by the caller for both training and per-example eval. Subclasses override this. |
| `_build_per_example_loss(self, fmodel, frozen_params, batch_keys)` | `_dp_trainer.py:3002` | **Internal builder** — wraps `compute_per_example_loss` into the training vmap closure (calls it at `:3047-3053`), returns `(loss_fn, batch_argnums)` for `clipped_grad`. Subclasses normally do NOT override this. |
| per-example eval path | `_dp_trainer.py:3115-3120` | Also wraps `compute_per_example_loss` under vmap when `'loss' in include_for_metrics`. The batched eval fast-path (`:2233-2253`) uses bound-module forward directly. |

There is exactly one place to put the loss math. The same `compute_per_example_loss` override is vmap'd for training (→ clip → noise) and for per-example eval, so training and eval can never drift apart. The constraints are uniform: the hook is always vmap-safe (no `nn.Module` state mutation, no `.item()` on dynamic shapes, no in-place input mutation, no Python control flow on tensor values).

**TRL trainer override pattern (used in Phases 1–3):**

```python
class OpaqueDPOTrainer(DPTrainer, RLHFMixin):
    def compute_per_example_loss(self, fmodel, params, inputs, *, return_logits=False):
        # The SINGLE override. vmap-safe. Calls opaque.alignment.loss.dpo.DPO_LOSSES[...].
        # Returns scalar per example (training); the eval path returns the same,
        # optionally with logits via return_logits=True for metric collection.
        ...
```

Trainers compute reward / token-accuracy / KL metrics from the `return_logits=True` payload, accumulating into `self._metrics["eval"]` inside the trainer's eval hook (not inside `compute_per_example_loss` itself).

**`num_items_in_batch` is HF-parity passthrough only.** Accepted in `training_step` (`_dp_trainer.py:1725`) but **never used inside the DP path** — the per-example vmap path scales by `expected_batch_size` via `clipped_grad`, and the eval path uses per-real-token weighting (§8.1a). Subclasses must not introduce divisors that depend on it (DP-purity rule §4.4).

### 8.1a Per-real-token eval-loss weighting (new behavior)

The eval loop (`_dp_trainer.py:2406-2432`) now weights per-example losses by **real-token count** (tokens where `labels != -100`), not by batch size — making eval loss padding-invariant and aligned with the manual-loop reference. Consequence for TRL trainers: the eval weighting is applied **by the trainer's eval loop**, not inside `compute_per_example_loss`. Subclasses return the raw per-example loss; the loop handles token-weighted reduction. No subclass action required, but parity tests should assert padding-invariance.

### 8.2 Smaller hooks to add

| Hook | Purpose |
|---|---|
| `_default_signature_columns() -> list[str]` | Subclass override hook (vs. monkey-patching the whole `_set_signature_columns_if_needed` at `_dp_trainer.py:2899`). Empty by default; subclasses return their fixed list (e.g. DPO's `chosen_input_ids`, `ref_chosen_logps`, …). **This is the load-bearing hook** — it tells `_remove_unused_columns` (`:2927`) to keep the preference columns that aren't in `model.forward`'s signature. |
| `compute_loss_context_manager()` | Returns `torch.autocast(...)` when `self._amp_dtype` set (initialized at `_dp_trainer.py:423`), else `nullcontext()`. HF parity. Used by DPO/KTO `prediction_step`. |
| `_default_collator()` factory hook | Subclass-overridable factory for a default collator when user doesn't pass one. |

**No `_extra_forward_kwargs` allowlist.** TRL passes non-`forward()`-signature kwargs (`skip_logits`, `return_token_accuracy`, `use_token_scaling`) into the model — those are TRL's *Liger* metric flags. Opaque does not surface Liger directly (the kernel layer is `use_performance_kernels`), and the unified `compute_per_example_loss(fmodel, params, inputs)` hook hands the subclass full control over exactly what gets forwarded to `fmodel`. The subclass transforms/filters `inputs` itself before the forward, so a trainer-level allowlist is unnecessary. `_default_signature_columns` (column retention) is the only hook needed here.

### 8.3 Rename

`cpu_offload_activations` → `activation_offloading`. Currently at `_config.py:422`. Keep old name as deprecated alias for one release.

### 8.4 Phase 0 deliverables

- Patches to `_dp_trainer.py` (add hooks above) and `_config.py` (rename).
- New tests in `tests/opaque_transformers/test_dp_trainer_subclass_hooks.py`:
  - Confirm `_default_signature_columns` is honored (preference columns survive `_remove_unused_columns`).
  - Confirm a subclass `compute_per_example_loss` override drives **both** training and per-example eval.
  - Confirm `compute_loss_context_manager` returns the right context.
  - Confirm `activation_offloading` works; `cpu_offload_activations` still works with deprecation warning.
- No new public API in `opaque.transformers.trl` yet. Phase 0 is internal-facing.

**Effort: S (1–2 days)** — small: the unified `compute_per_example_loss` hook already exists; Phase 0 only adds the signature-columns hook, the autocast-context shim, the collator factory hook, and the config rename.

**Work units (DAG).** The hooks touch the shared `_dp_trainer.py` / `_config.py`, so units serialize through one agent to avoid intra-file conflict; the test file is a parallel add. Max concurrency 2.

| Unit | Produces | Deps | Gate |
|---|---|---|---|
| 0.1 | `_dp_trainer.py` hooks (`_default_signature_columns`, `compute_loss_context_manager`, `_default_collator`) + `_config.py` rename (`cpu_offload_activations`→`activation_offloading` alias) | — | `infra` (+ GPU smoke, training-path touch) |
| 0.2 | `tests/opaque_transformers/test_dp_trainer_subclass_hooks.py` | 0.1 | `infra` |

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

> **Note:** Phase 0.25 is the same work as `opaque-alignment-plan.md` Phase α (unit α.1). Run it once from either plan; do not duplicate.

**Work units (DAG).** Single atomic unit (= α.1).

| Unit | Produces | Deps | Gate |
|---|---|---|---|
| 0.25.1 | `packages/opaque-alignment/` skeleton + tooling registration (≡ α.1) | — | `infra` |

---

## 10. Phase 0.5 — `opaque.distributed` extensions

### 10.1 New functional primitives

Current `opaque.distributed` surface (audited on `feat/dptrainer-main-integration`):
- ✅ Present: `is_distributed`, `get_rank`, `get_world_size`, `all_reduce`, `sum_gradients`, `sum_gradients_`, `sync`, `reduce_pytree`, `reduce_pytree_`, `local_shard`.
- ❌ Missing: `gather_for_metrics`, `is_main_process`, `wait_for_everyone`, `num_processes`, `process_index`.

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

**Work units (DAG).** Single unit (the five functions share `opaque.api.engine.distributed` and its `__init__.py`).

| Unit | Produces | Deps | Gate |
|---|---|---|---|
| 0.5.1 | `opaque/api/engine/distributed/` additions (`gather_for_metrics`, `is_main_process`, `wait_for_everyone`, `num_processes`, `process_index`) + façade re-exports + tests | — | `infra` |

---

## 11. Phase 1 — SFTTrainer + alignment primitives

### 11.1 `opaque-alignment` primitives landing in Phase 1

- `opaque/api/alignment/logprob/` — `selective_log_softmax`, `sequence_logp`, `get_batch_logps`.
- `opaque/api/alignment/loss/sft/` — `nll_loss` (per-example, mean over non-ignored tokens) and `dft_loss` (per-example, detached softmax weighting; DP-corrected divisor) + `SFT_LOSSES` registry (`chunked_nll` aliased to `nll`).
- `opaque/api/alignment/collator/_language_modeling.py` — `language_modeling_collator(...)` factory returning a callable, with `completion_mask` and `assistant_masks` support.
- `opaque/api/alignment/data/_prompt.py` — `extract_prompt`.
- `opaque/api/alignment/data/_chat_template.py` — `apply_chat_template` glue + `get_training_chat_template` for `{% generation %}`-tagged templates. **No** `clone_chat_template` with embedding resize yet (Phase 4).
- `opaque/api/alignment/metric/` — `entropy_from_logits`, `mean_token_accuracy`, `num_tokens` aggregator.

### 11.2 `OpaqueSFTConfig` (`_sft_config.py`)

Extends `DPTrainingArguments`, inheriting all DP + HF-parity fields — notably `privacy_noise_multiplier`, `privacy_target_epsilon` (**default `None`** on the integration branch; one of NM or target-ε must be set explicitly, and `0.0` selects non-private mode), `privacy_target_delta`, `clipping_norm` (accepts `math.inf` to disable, non-private only), `clipping_mode` (auto-resolves `adaptive→fixed` under MF mechanisms), `sampling_mode` (default `"auto"`, resolves per mechanism), `microbatch_size` + `auto_find_microbatch_size` (the primary vmap-chunk knobs), `use_performance_kernels`, `use_compat_patches`, `cpu_offload_activations` (→ `activation_offloading`). The TRL-specific additions below sit on top of those:

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

- `__init__`: model load, tokenizer load, optional PEFT wrap, QLoRA bf16, `_prepare_dataset` (uses `extract_prompt` + tokenize + completion-mask + optional assistant-mask), default collator = `language_modeling_collator(...)`, `super().__init__()`.
- `_default_signature_columns`: `["input_ids", "labels", "attention_mask", "completion_mask", "assistant_masks"]`.
- `compute_per_example_loss` (single hook): dispatches on `loss_type` to `opaque.alignment.loss.sft.{nll,dft}(...)` per example. Both `nll` and `chunked_nll` route to the same `nll_loss` function (math-equivalent; `chunked_nll` selects a memory-efficient kernel path in Phase 4). `dft` uses a **per-example token-count divisor** (`mask.sum()` on the example, not `num_items_in_batch`) per the DP-correct divisor rule (`opaque-alignment-plan.md` §3.3 + §8.2). With `return_logits=True` (eval), the trainer computes `entropy_from_logits`, `mean_token_accuracy`, `num_tokens` from the payload and pushes into `self._metrics["eval"]`.
- `log` override (from `RLHFMixin`): drains `_metrics[mode]` and merges into logs.

### 11.4 Examples

- `examples/train_sft.py` — functional, hand-rolled loop over alignment primitives.
- `examples/train_sft_trainer.py` — class-based via `OpaqueSFTTrainer`.

### 11.5 Phase 1 tests

- Loss-fn unit tests for `nll_loss`, `dft_loss` (vs hand-computed reference).
- Closure test: vmap'd per-example SFT loss over 4 examples → finite gradients.
- Trainer contract test: 2 training steps, eval, save_pretrained.
- TRL parity test: `OpaqueSFTTrainer(σ=0, C=∞)` vs `trl.SFTTrainer` on identical batch → loss within `1e-3`.

**Work units (DAG).** Consumes alignment ε.W + ε.X (SFT loss + collator). Config and trainer serialize (config feeds trainer); examples + trainer-example parallel. Max concurrency 2. Deps cross into the alignment workflow.

| Unit | Produces | Deps | Gate |
|---|---|---|---|
| 1.1 | `_sft_config.py` (`OpaqueSFTConfig`) | 0.1, ε.W | `infra` |
| 1.2 | `_sft_trainer.py` (`OpaqueSFTTrainer`) + `_rlhf_mixin.py` + contract test | 1.1, ε.W, β.W, 0.5.1 | `trainer` |
| 1.3 | `examples/train_sft_trainer.py` | 1.2 | `example` |
| 1.W | `opaque/api/transformers/trl/__init__.py` + façade `transformers/trl/__init__.py` | 1.1, 1.2 | `infra` |

(The functional `examples/train_sft.py` is alignment unit ε.X — not duplicated here.)

**Effort: M (4–5 days).**

---

## 12. Phase 2 — DPOTrainer + alignment primitives

The heaviest phase. All advanced DPO features land here (except chunked preference kernel, which is Phase −1).

### 12.1 `opaque-alignment` primitives landing in Phase 2

- `opaque/api/alignment/loss/dpo/` — all 14 Tier-1 DP-safe variants + `DPO_LOSSES` registry + `DPO_SPEC`:
  - `sigmoid`, `ipo`, `hinge`, `robust`, `apo_zero`, `apo_down`, `exo_pair`, `nca_pair`, `bco_pair`, `sppo_hard`, `discopop`, `sft`, `sigmoid_norm`, `squarechipo`.
  - DPO-specific helpers live **inside** `loss/dpo/`: `_f_divergence.py` (`reverse_kl`, `forward_kl`, `js_divergence`, `alpha_divergence`), `_mpo.py` (`mpo_combine`), `_wpo.py` (`wpo_weights`, under `no_grad`), `_ld_dpo.py` (`ld_dpo_split`).
- `opaque/api/alignment/collator/_preference.py` — `preference_collator(...)` factory returning a callable; emits separate `chosen_*` / `rejected_*` keys per §4.3.
- `opaque/api/alignment/reference/_precompute.py` — `compute_ref_logprobs_for_dataset(dataset, ref, collator, output_columns, ..., cache_key)` with `.npz` caching.
- `opaque/api/alignment/reference/_adapter.py` — `null_ref_context(model)` context manager (dispatches per RefSpec; `opaque-alignment-plan.md` §7.8).
- `opaque/api/alignment/reference/_sync.py` — `ema_update_reference(ref_params, policy_params, alpha)` functional EMA.
- `opaque/api/alignment/metric/` — `reward_metrics(chosen_logratios, rejected_logratios, beta) -> dict`.

### 12.2 `OpaqueDPOConfig`

Extends `DPTrainingArguments` (inherits the DP + HF-parity fields enumerated in §11.2). TRL-specific additions:

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
- `compute_per_example_loss` (single hook): forwards `(chosen_*, rejected_*)` through `fmodel` twice per pair, computes `sequence_logp` for each, applies `ld_dpo_split` if `ld_alpha`, applies `f_divergence_remap`, dispatches through `DPO_LOSSES` × `loss_weights`, optional `wpo_weights` multiplication. All math via `opaque.alignment.loss.dpo.*`. With `return_logits=True`, returns the logits payload so the trainer can compute reward metrics (`rewards/chosen`, `rewards/rejected`, `rewards/accuracies`, `rewards/margins`) into `self._metrics["eval"]`.
- `prediction_step` override: surfaces reward metrics at eval (no labels otherwise).
- TR-DPO: `SyncRefModelCallback` registered when `sync_ref_model=True`. Uses `ema_update_reference` on the captured ref-params dict, not on a bound module.

### 12.4 Examples

- `examples/train_dpo.py` — functional with `dpo_sigmoid` directly.
- `examples/train_dpo_trainer.py` — class-based via `OpaqueDPOTrainer`.

### 12.5 Phase 2 tests

- Loss-fn unit tests for all 14 variants + 4 f-divergence remaps + MPO combinator.
- Closure test: vmap'd per-example DPO loss across 4 pairs → finite gradients.
- DP-purity **Tier-1 NaN-injection** test: NaN-one-example → only that row's grad affected (all DPO variants are Tier 1).
- Tier-3 rejection test: `loss_type="aot"` raises at init with the `DPSpec.rejection_reason` string.
- Trainer contract test (one per ref-model path): construct, train 2 steps, eval, save.
- TR-DPO callback test: ref params change after `ref_model_sync_steps`.
- TRL parity test on all loss variants at `σ=0, C=∞`.

**Work units (DAG).** Consumes alignment γ.W (DPO losses) + ζ.W (reference). Config → trainer → callback/example. Max concurrency 3.

| Unit | Produces | Deps | Gate |
|---|---|---|---|
| 2.1 | `_dpo_config.py` (`OpaqueDPOConfig`) | 0.1, γ.W | `infra` |
| 2.2 | `_dpo_trainer.py` (`OpaqueDPOTrainer`: `__init__` four ref paths, `_prepare_inputs`, `compute_per_example_loss`, `prediction_step`) + contract tests (one per ref path) | 2.1, γ.W, ζ.W, β.W, 0.5.1 | `trainer` |
| 2.3 | `_callbacks.py` (`SyncRefModelCallback`, TR-DPO) + callback test | 2.2, ζ.W | `infra` (+ GPU smoke) |
| 2.4 | `examples/train_dpo_trainer.py` | 2.2 | `example` |
| 2.W | trl façade `__all__` updates for DPO exports | 2.1, 2.2, 2.3 | `infra` |

(Functional `examples/train_dpo.py` is alignment unit γ.X.)

**Effort: L (6–8 days).**

---

## 13. Phase 3 — KTOTrainer + alignment primitives

### 13.1 `opaque-alignment` primitives landing in Phase 3

- `opaque/api/alignment/loss/kto/` — `kto_loss` (Tier 2, detached `kl` parameter), `apo_zero_unpaired` (Tier 1) + `KTO_LOSSES` registry + `KTO_SPEC` (marks `kto` Tier 2, `cross_batch_aggregate="kl_mean"`, `aggregate_leverage="O(1/n)"`).
- `opaque/api/alignment/collator/_unpaired_preference.py` — `unpaired_preference_collator(...)` factory returning a callable.
- `opaque/api/alignment/data/_kto_rotation.py` — `rotate_kto_completions(dataset, batch_size, seed)` = `_get_kl_dataset` + `concatenate_datasets(axis=1)`.
- Extension to `compute_ref_logprobs_for_dataset` to emit `reference_KL_logps` when KL is enabled.

### 13.2 `OpaqueKTOConfig`

Extends `DPTrainingArguments` (inherits the DP + HF-parity fields enumerated in §11.2). TRL-specific additions:

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
- `_prepare_inputs` override (the Tier-2 aggregate hook): for each microbatch,
  1. Run policy KL forward: `policy_KL_logps = self._model(**KL_kwargs).logp_via_sequence_logp(...)` under `torch.no_grad()` on bound module — **outside the vmap region**.
  2. Compute the aggregate: `kl = (policy_KL_logps - inputs["reference_KL_logps"]).mean().detach().clamp(min=0)` — scalar.
  3. If realized batch size ≤ 1: set `kl=0` (falls back to `apo_zero_unpaired` math for that step).
  4. Inject as `inputs["kl"]` (a `(1,)` scalar tensor broadcast to all examples via vmap-`None`).
- `compute_per_example_loss` (single hook):
  1. Forwards `completion_input_ids` through `fmodel` → per-example `completion_logp` via `sequence_logp`.
  2. Reads ref logps from `inputs["reference_logps"]` (precomputed at trainer init).
  3. Computes `chosen_logratios` / `rejected_logratios` per example, branching on `label` via `torch.where` (no Python `if`).
  4. Dispatches through `KTO_LOSSES[self.args.loss_type]`, passing the detached scalar `inputs["kl"]` and `desirable_weight` / `undesirable_weight`.
  5. Returns per-example scalar (plus logits payload when `return_logits=True` for eval metric collection into `self._metrics["eval"]`).
- DP audit at construction time: `KTO_SPEC[self.args.loss_type]` is read; if `tier == 2`, the trainer asserts `_prepare_inputs` registers the appropriate aggregate per `LossAggregateSpec`. v2 will extend this with `cross_rank=True` handling (`opaque-alignment-plan.md` §9.4).

### 13.4 Examples

- `examples/train_kto.py` — functional.
- `examples/train_kto_trainer.py` — class-based.

### 13.5 Phase 3 tests

- Loss-fn unit tests for `kto`, `apo_zero_unpaired`.
- Rotation correctness test (assert non-identity permutation after `dataset.map`).
- **Tier-2 aggregate-detach audit** for `kto`: trace autograd graph from `loss` backward; assert `kl` has no path to model parameters; assert leverage is `O(1/n)` via aggregate-swap (`opaque-alignment-plan.md` §11.4).
- Trainer contract test with mixed labels: confirm `_prepare_inputs` computes `kl` once per microbatch and broadcasts via vmap-`None`; confirm Poisson batch-size-1 falls back to `kl=0`.
- TRL parity at `σ=0, C=∞`: compare per-example loss to TRL within `1e-3` on identical batches.

**Work units (DAG).** Consumes alignment δ.W (KTO loss + rotation) + ζ.W (reference). Max concurrency 2.

| Unit | Produces | Deps | Gate |
|---|---|---|---|
| 3.1 | `_kto_config.py` (`OpaqueKTOConfig`) | 0.1, δ.W | `infra` |
| 3.2 | `_kto_trainer.py` (`OpaqueKTOTrainer`: `_prepare_dataset`, Tier-2 `_prepare_inputs` KL hook, `compute_per_example_loss`) + contract test (mixed labels, batch-1 fallback) + Tier-2 audit | 3.1, δ.W, ζ.W, β.W, 0.5.1 | `trainer` |
| 3.3 | `examples/train_kto_trainer.py` | 3.2 | `example` |
| 3.W | trl façade `__all__` updates for KTO exports | 3.1, 3.2 | `infra` |

(Functional `examples/train_kto.py` is alignment unit δ.X.)

**Effort: M–L (4–6 days).**

---

## 14. Phase 4 — Advanced data pipeline

These are workstreams independent of loss math; separated to keep Phases 1–3 reviewable.

### 14.1 SFT packing — `bfd`, `bfd_split`, `wrapped`

- `opaque.alignment.data._packing`: port `_pack_bfd` (segment-tree best-fit, ~90 LOC), `_pack_wrapped` (~15 LOC), `_pack_bfd_split` (~25 LOC).
- Generates `seq_lengths` column consumed by the `language_modeling_collator` callable's packed-sequence position-id derivation.
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

**Implementation:** wire `loss_type="chunked_nll"` in `OpaqueSFTTrainer`'s `compute_per_example_loss` to extract hidden states (request `output_hidden_states` / skip `lm_head` via the kwargs the subclass passes to `fmodel`), then call `opaque_linear_cross_entropy_loss(hidden, lm_head.weight, labels, ...)`. The unified hook gives the subclass full control over the forward, so no trainer-level allowlist is needed. No monkey-patching, no checkpoint trickery, no Liger dependency.

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

**Work units (DAG).** Packing/template are alignment θ (units θ.0/θ.1/θ.2); the trainer-side wiring (config flags + `chunked_nll` kernel path + `activation_offloading`) is here. Max concurrency 3.

| Unit | Produces | Deps | Gate |
|---|---|---|---|
| 4.1 | SFT config flag wiring (`packing`, `padding_free`, `chat_template_path`, `activation_offloading`) + `_sft_trainer.py` packing/padding-free dispatch | 1.2, θ.W | `trainer` |
| 4.2 | `chunked_nll` kernel path via `opaque_linear_cross_entropy_loss` in `_sft_trainer.py` + peak-memory test | 1.2 | `trainer` (+ GPU memory) |
| 4.3 | `clone_chat_template` trainer hookup (pre-`make_functional` embedding resize) + round-trip test | 1.2, θ.2 | `trainer` |
| 4.W | doc note on FlexAttention decision (from θ.0) + config validation | 4.1–4.3 | `infra` |

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
2. Disable DP in Opaque via the canonical non-private mode: `privacy_noise_multiplier=0.0` **and** `clipping_norm=math.inf`. (Integration-branch validation: `clipping_norm=math.inf` is rejected unless `noise_multiplier=0.0`, and `noise_multiplier=0.0` is rejected if `privacy_target_epsilon` is set — so leave `privacy_target_epsilon=None` for the parity run.)
3. Single forward + backward pass on each.
4. Assert per-batch loss matches within `1e-3`. Note eval loss uses per-real-token weighting (§8.1a), so the parity assertion is padding-invariant.
5. Assert reward metrics (DPO) match within `1e-4`.

### 15.3 DP regression

50-step run per trainer at `ε=10`, fixed seeds. Snapshot final loss; track for drift across PRs.

### 15.4 Docs

- `docs/alignment/index.md` — package overview, functional philosophy, mechanism-agnostic posture.
- `docs/alignment/loss.md` — per-loss reference (formula, paper, DP notes).
- `docs/alignment/collator.md`.
- `docs/alignment/recipes.md` — pointers to functional examples + scripted recipes (decoupled DP-RLHF lives here as a notebook).
- `docs/trainers/sft.md`, `dpo.md`, `kto.md` — class-API docs, supported features, deferred features with paper-cited justification, ref-model path matrix.

**Work units (DAG).** The parity + DP-regression gates run as **dedicated adversarial-validation units** — the workflow's convergence pattern at the whole-trainer level. Max concurrency 5.

| Unit | Produces | Deps | Gate |
|---|---|---|---|
| 5.1 | TRL parity test suite (per trainer × per loss variant at σ=0,C=∞) | 1.W, 2.W, 3.W | `trainer` |
| 5.2 | DP-regression suite (50-step ε=10 snapshots, drift tracking) | 1.W, 2.W, 3.W | `example` |
| 5.3 | `docs/trainers/{sft,dpo,kto}.md` | 1.W, 2.W, 3.W | `infra` |
| 5.4 | `docs/alignment/recipes.md` decoupled-DP-RLHF notebook scaffold | ζ.W | `infra` |
| 5.W | mkdocs nav + final full-suite green (CPU+MPS+GPU smoke) | 5.1–5.4, ι.W | `trainer` |

**Effort: M (3–4 days).**

---

## 16. Roadmap beyond this plan

Sibling workstreams that become natural under the `opaque-alignment` + `opaque.transformers.trl` split:

| Item | Effort | Rationale |
|---|---|---|
| **`OpaqueRewardTrainer` (DP RM)** | M | Pairwise BT loss on `AutoModelForSequenceClassification`. Direct `DPTrainer` subclass using `opaque.alignment.loss.reward` (new `loss/reward/` sub-concern with the BT formula). Prerequisite for decoupled DP-RLHF (arXiv:2603.22563). |
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
| R10 | FlexAttention + vmap composition broken | M | M | Phase 4 fixture test. Fallback: SDPA + 4D block mask (slower). |
| R11 | `feat/dptrainer-main-integration` rebases or evolves before this plan lands | L | M | Plan branch is rebased on top; rebase again if base moves. Phase 0 changes are small and easy to re-apply. |
| R12 | Chunked preference kernel port (Phase −1) takes longer than estimated | M | L | Optional optimization; SFT/DPO/KTO ship without it via standard `opaque_linear_cross_entropy_loss`. |

---

## 18. DP correctness checklist (used at every loss port)

Apply to every per-example loss closure before it lands in `opaque.alignment.loss.*`. (This checklist is the Tier-1 subset; Tier-2 losses additionally pass the aggregate-detach audit in `opaque-alignment-plan.md` §11.4.)

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

### 19.1 Unit tests (per loss function in `opaque.api.alignment.loss.{dpo,kto,sft}`)

~10 tests per loss variant against hand-computed reference. Pure, no model.

### 19.2 Closure tests

Build the subclass `compute_per_example_loss`, vmap it over a 4-example synthetic batch, verify shape, finite non-zero gradient w.r.t. `trainable_params`. No full training loop.

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

### Opaque DPTrainer (`feat/dptrainer-main-integration`)

Line numbers verified on `feat/dptrainer-main-integration` (`_dp_trainer.py` 4814 LOC, `_config.py` 1295 LOC):

- `_dp_trainer.py:250` — `class DPTrainer`.
- `_dp_trainer.py:1955` — `compute_per_example_loss(self, fmodel, params, inputs, *, return_logits)` (**the single override hook**; unified DP-correct training + per-example eval).
- `_dp_trainer.py:3002` — `_build_per_example_loss(self, fmodel, frozen_params, batch_keys)` (internal builder; wraps `compute_per_example_loss` for the training vmap closure at `:3047-3053`).
- `_dp_trainer.py:3115-3120` — per-example eval path (also wraps `compute_per_example_loss`).
- `_dp_trainer.py:2075` — `prediction_step(...)`.
- `_dp_trainer.py:1725` — `training_step(model, inputs, num_items_in_batch)` (`num_items_in_batch` accepted but unused in DP path).
- `_dp_trainer.py:2288` — `evaluation_loop(...)`; `:2406-2432` — per-real-token eval-loss weighting.
- `_dp_trainer.py:2883` — `_prepare_input(...)`.
- `_dp_trainer.py:2899` — `_set_signature_columns_if_needed()`.
- `_dp_trainer.py:2927` — `_remove_unused_columns(...)`.
- `_dp_trainer.py:3123` — `_discover_batch_keys()`.
- `_dp_trainer.py:2986` — `_prepare_dataset_and_collator(...)`.
- `_dp_trainer.py:3154` — `get_train_dataloader()`.
- `_dp_trainer.py:3297` — `get_eval_dataloader(eval_dataset)`.
- `_dp_trainer.py:3424` — `create_optimizer()`.
- `_dp_trainer.py:3465` — `create_scheduler(num_training_steps)`.
- `_dp_trainer.py:3479` — `log(logs, start_time)`.
- `_dp_trainer.py:3510` — `_maybe_log_save_evaluate(...)`.
- `_dp_trainer.py:3654,3663,3673` — three call sites of `clipped_grad / adaptive_clipped_grad / auto_clipped_grad(..., normalize_by=expected_batch_size)`.
- `_dp_trainer.py:423` — `_amp_dtype` initialization.
- `_dp_trainer.py:416` — `_runtime_bootstrap` invocation (`_opaque_rt.apply_transformers_runtime_compat_patches()`; see §2.4).
- `_config.py:240` — `microbatch_size`; `:244` — `auto_find_microbatch_size`; `:389` — `use_performance_kernels`; `:398` — `performance_kernels_config`; `:406` — `use_compat_patches`; `:422` — `cpu_offload_activations` (rename target).

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
