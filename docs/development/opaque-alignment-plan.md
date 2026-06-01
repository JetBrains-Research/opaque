# opaque-alignment — Package Plan

**Status:** Planning. Sibling doc to `trl-trainers-plan.md`; the two evolve together but address different audiences.

**Scope:** A new distribution `opaque-alignment` that ships **functional, mechanism-agnostic primitives for DP-safe preference learning**: pure per-example loss functions (DPO, KTO, SFT family), logprob helpers, preference collators, dataset transforms (prompt extraction, packing, chat templates, KTO rotation), reference-model precompute helpers, and alignment-specific metrics. Built on `opaque-engine` and `opaque-patches`; consumed by both functional training scripts (`examples/train_dpo.py`-style) and the TRL-style class trainers in `opaque.transformers.trl`.

**Branches:**
- Planning (this doc): `claude/add-trl-trainers-plan-nB07O`, rebased onto `feat/dptrainer-main-integration`.
- Implementation phases: per-phase sub-branches.

**Read alongside:** `docs/development/trl-trainers-plan.md` — covers the class trainers built on top of these primitives.

---

## Table of contents

1. [Goals and non-goals](#1-goals-and-non-goals)
2. [Relationship to `opaque.transformers.trl`](#2-relationship-to-opaquetransformerstrl)
3. [Architectural philosophy](#3-architectural-philosophy)
4. [Module layout (api/façade pattern)](#4-module-layout-apifaçade-pattern)
5. [Dependency pin](#5-dependency-pin)
6. [Public API surface](#6-public-api-surface)
7. [Per-module spec](#7-per-module-spec)
8. [Phase plan](#8-phase-plan)
9. [Test strategy](#9-test-strategy)
10. [Cross-package interactions](#10-cross-package-interactions)
11. [Functional examples](#11-functional-examples)
12. [Roadmap beyond the TRL trainer port](#12-roadmap-beyond-the-trl-trainer-port)
13. [Risk register](#13-risk-register)
14. [References](#14-references)

---

## 1. Goals and non-goals

### 1.1 Goals

- **Standalone usability.** A researcher can build a DP-DPO training script using only `opaque-alignment` + the mechanism of their choice (DP-SGD or DP-FTRL) + an optimizer from `opaque.optimizers`. **No requirement to subclass `DPTrainer`.**
- **Mechanism-agnostic.** The package depends only on `opaque-engine` (clipping, functional, distributed), `opaque-base` (serialization), and `opaque-patches` (fused kernels). It does NOT depend on `opaque-dpsgd`, `opaque-dpftrl`, or `opaque-optimizers`. The mechanism + optimizer are chosen at the call site.
- **DP-purity invariant.** Every per-example loss function passes the DP-purity checklist (see §3.3): no cross-example divisors, no data-dependent batch-level aggregations, vmap-safe ops only.
- **TRL-faithful loss math.** Every DPO / KTO / SFT loss variant that is mathematically compatible with per-example DP-SGD lands at full numeric parity with TRL eager (within `1e-3` at `σ=0, C=∞`).
- **First-class functional examples.** `examples/train_sft.py`, `train_dpo.py`, `train_kto.py` ship as siblings of the existing `train_causal_lm.py`, demonstrating the "primitives → training loop" pattern.
- **Citable home for DP-alignment paper recipes.** SquareχPO (arXiv:2505.21395) lands as `dpo_squarechipo` in the loss registry. Future paper recipes (decoupled DP-RLHF, etc.) live in a `recipes/` subnamespace.
- **Reusable across trainer families.** Future RewardTrainer, ORPOTrainer, CPOTrainer, SimPOTrainer, GRPOTrainer (under `opaque.transformers.trl` or a recipe DSL) all consume the same primitives — no copy-paste of loss math between trainer subclasses.

### 1.2 Non-goals

- **No trainer classes.** Trainers live in `opaque.transformers.trl` (see `trl-trainers-plan.md`). This package only ships primitives.
- **No `accelerate` dependency.** Reductions go through `opaque.distributed` functional primitives.
- **No DeepSpeed / FSDP / Accelerate integration.** Out of architectural reach for per-example DP-SGD.
- **No VLM** (vision-language models). Out of scope.
- **No `aot` / `aot_pair` loss variants.** Sort-across-batch breaks per-example DP. Rejected at function level (not even exposed in the loss registry).
- **No bundled mechanism choice.** Users explicitly import `opaque.dpsgd.*` or `opaque.dpftrl.*` at call site.
- **No `nn.Module` subclasses.** Everything functional. PyTorch's `nn.Module` lifecycle is incompatible with `vmap(grad(...))` and conflicts with our "no hidden state" principle.
- **No DP-PPO actor** (arXiv:2501.19080). Trajectory-level DP is a separate design.

---

## 2. Relationship to `opaque.transformers.trl`

```
                ┌────────────────────────────────────────────────────────────┐
                │  opaque.transformers.trl                                   │
                │  ─────────────────────────                                 │
                │  SFTTrainer, DPOTrainer, KTOTrainer                        │
                │  configs, signature columns, log() drain, prediction_step  │
                │  (~30-line override of compute_per_example_loss — one hook) │
                └─────────────────────┬──────────────────────────────────────┘
                                      │ consumes
                                      ▼
                ┌────────────────────────────────────────────────────────────┐
                │  opaque-alignment                                          │
                │  ────────────────                                          │
                │  Pure per-example loss fns: dpo_sigmoid, ipo, hinge,       │
                │    robust, apo_*, exo_pair, nca_pair, bco_pair, sppo_hard, │
                │    discopop, squarechipo, kto, apo_zero_unpaired, nll, dft │
                │  logprob: sequence_logp, selective_log_softmax             │
                │  collators: DataCollatorFor{LanguageModeling,Preference,…} │
                │  data: extract_prompt, pack_bfd, rotate_kto_completions,…  │
                │  reference: compute_ref_logprobs_for_dataset, null_ref_*   │
                │  metrics: reward_metrics, kl_estimator                     │
                │  kernels: opaque_fused_linear_dpo_loss, …                  │
                └─────┬────────────────┬───────────────────────────────┬─────┘
                      │                │                               │
       ┌──────────────┘                │                               └──────────┐
       │ depends on                    │ depends on                               │ depends on
       ▼                               ▼                                          ▼
┌─────────────────────┐    ┌─────────────────────┐                ┌───────────────────────┐
│  opaque-engine      │    │  opaque-patches     │                │  opaque-base          │
│  (clipping,         │    │  (fused kernels +   │                │  (serialization for   │
│   functional,       │    │   chunked          │                │   ref-logp cache)     │
│   distributed)      │    │   preference kernel)│                │                       │
└─────────────────────┘    └─────────────────────┘                └───────────────────────┘
```

`opaque.transformers.trl` is one consumer. The *other* consumer is the **functional examples** — `train_dpo.py`, `train_kto.py`, `train_sft.py` — which import from `opaque-alignment` directly with no trainer in sight. These are equal first-class deliverables.

Per `trl-trainers-plan.md` §6/§8.1, the TRL trainers override a **single** unified hook in `DPTrainer`: `compute_per_example_loss(fmodel, params, inputs, *, return_logits)`. The trainer wraps it with `vmap` for both training (→ grad → clip → noise) and per-example eval. The hook orchestrates `opaque-alignment` primitives; that single override point is what makes the primitive layer load-bearing.

---

## 3. Architectural philosophy

### 3.1 Functional, no hidden state

Every public symbol in `opaque-alignment` is a pure function (or a context-manager utility for unavoidable side-effecting things like PEFT adapter toggles). No `nn.Module` subclasses. No global state. No hooks.

This mirrors `opaque-engine`'s design (`AGENTS.md`: "every component uses explicit state — no hooks, no subclassing, no hidden mutation"). The reasoning is the same: `vmap(grad(...))` composes cleanly only over pure functions; hidden state breaks per-example independence.

### 3.2 Mechanism-agnostic

Dependencies are pinned to **substrate** packages only (`opaque-engine`, `opaque-base`, `opaque-patches`), not to **mechanism** packages (`opaque-dpsgd`, `opaque-dpftrl`). A DPO loss does not care whether you're going to add Gaussian noise via DP-SGD or matrix-factorized noise via DP-FTRL — it's the caller's choice at clip + noise time.

Practical consequence: a researcher running DP-FTRL DPO writes the same loss closure as DP-SGD DPO. Only the mechanism imports change at the bottom of the training loop. See §11 for the worked example.

### 3.3 DP-purity invariant

**Every per-example loss function in `opaque-alignment` must satisfy:** the output for example `i` depends only on example `i`'s data. No cross-example dependencies, no batch-level aggregations baked into the loss.

This is the per-example DP-SGD precondition. The invariant is enforced by a mandatory **NaN-injection contract test** for every public loss function (§9.3): replace one example's input with NaN; verify only that example's gradient changes after clipping.

Specific rules (the full checklist lives in `trl-trainers-plan.md` §18; here we restate the ones that bite hardest):

- **No `num_items_in_batch`-style divisor.** Allowed divisors: per-example token counts (`mask.sum()` on the example), `args.max_length`, fixed beta constants, or simply omit the divisor and let `clipped_grad(normalize_by=expected_batch_size)` do the aggregation.
- **No sort-across-batch.** This is the `aot` / `aot_pair` failure mode. Such losses are not exposed.
- **No `.item()` on dynamic-shape tensors.** Breaks vmap.
- **No in-place mutation of inputs.** Forbidden under vmap.

### 3.4 Vmap-safety contract

Every loss + logprob + collator-output-consuming primitive must be safe to call under `torch.func.vmap(torch.func.grad(...))` over the trainable params. Specifically:

- Pure tensor operations only.
- No `nn.Module` state mutation inside the function body.
- No Python control flow on tensor values (`torch.where` instead of `if`).
- No `torch.no_grad()` on module instance attributes (PEFT adapter flags etc.).

Primitives that genuinely need to break vmap-safety (PEFT adapter toggles, ref-model precompute over a dataset) are documented as "**outside vmap only**" and labeled in the module docstring. These are typically the `reference/` helpers.

### 3.5 Loss registry pattern

DPO and KTO each have many variants (14 + 2). Rather than 16 separate trainer attributes, each loss family exports a registry dict:

```python
# opaque.alignment.losses.dpo (façade)
LOSSES: dict[str, Callable] = {
    "sigmoid":     dpo_sigmoid,
    "ipo":         dpo_ipo,
    "hinge":       dpo_hinge,
    "robust":      dpo_robust,
    "apo_zero":    dpo_apo_zero,
    "apo_down":    dpo_apo_down,
    "exo_pair":    dpo_exo_pair,
    "nca_pair":    dpo_nca_pair,
    "bco_pair":    dpo_bco_pair,
    "sppo_hard":   dpo_sppo_hard,
    "discopop":    dpo_discopop,
    "sft":         dpo_sft,
    "sigmoid_norm":dpo_sigmoid_norm,
    "squarechipo": dpo_squarechipo,
}
```

Trainer configs and recipe files reference loss types by string. The registry is the single source of truth.

---

## 4. Module layout (api/façade pattern)

Matches Opaque's established convention (`opaque-engine`, `opaque-optimizers`, etc.):

```
packages/opaque-alignment/
├── pyproject.toml
├── README.md
└── src/opaque/
    ├── api/
    │   └── alignment/                                ← IMPLEMENTATION namespace
    │       ├── __init__.py
    │       ├── losses/
    │       │   ├── __init__.py
    │       │   ├── _dpo.py                           # 14 DPO variants + LOSSES dict
    │       │   ├── _kto.py                           # kto, apo_zero_unpaired + LOSSES dict
    │       │   ├── _sft.py                           # nll, dft + LOSSES dict
    │       │   ├── _f_divergence.py                  # reverse_kl, forward_kl, js, alpha
    │       │   ├── _mpo.py                           # combinator for multi-loss
    │       │   ├── _wpo.py                           # per-example weight fn
    │       │   ├── _ld_dpo.py                        # shared/tail logp decomposition
    │       │   └── _fused.py                         # kernel-accelerated dispatchers
    │       ├── collators/
    │       │   ├── __init__.py
    │       │   ├── _language_modeling.py             # SFT collator + completion/assistant masks
    │       │   ├── _preference.py                    # DPO collator with (B, 2, L) layout
    │       │   └── _unpaired_preference.py           # KTO collator
    │       ├── data/
    │       │   ├── __init__.py
    │       │   ├── _prompt.py                        # extract_prompt
    │       │   ├── _packing.py                       # _pack_bfd, _pack_wrapped, _pack_bfd_split
    │       │   ├── _chat_template.py                 # clone_chat_template, get_training_chat_template
    │       │   └── _kto_rotation.py                  # _get_kl_dataset + concatenate_datasets glue
    │       ├── reference/
    │       │   ├── __init__.py
    │       │   ├── _precompute.py                    # compute_ref_logprobs_for_dataset (cached)
    │       │   ├── _adapter.py                       # null_ref_context, with_disabled_adapter
    │       │   └── _sync.py                          # ema_update_reference (TR-DPO core)
    │       ├── logprob.py                            # selective_log_softmax, sequence_logp, get_batch_logps
    │       └── metrics.py                            # reward_metrics, kl_estimator
    └── alignment/                                    ← PUBLIC FAÇADE namespace
        ├── __init__.py                               # headline re-exports
        ├── losses/__init__.py                        # re-exports from opaque.api.alignment.losses
        ├── collators/__init__.py
        ├── data/__init__.py
        ├── reference/__init__.py
        ├── logprob.py
        └── metrics.py
```

Pattern conforms to `opaque-engine`'s layout (private `_*.py` files under concern dirs; public re-export through `__init__.py`; api/façade split with PEP 420 namespacing).

---

## 5. Dependency pin

```toml
# packages/opaque-alignment/pyproject.toml
[project]
name = "opaque-alignment"
dynamic = ["version"]
description = "Functional primitives for DP-safe preference learning (DPO, KTO, SFT)"
requires-python = ">=3.11,<3.13"
license = { text = "Apache-2.0" }

dependencies = [
    "torch>=2.10.0",
    "transformers>=4.57.0,<5",
    "datasets>=2.0.0",
    "peft>=0.18.0",            # for null_ref_context / disable_adapter
    "opaque-engine",           # clipping, functional, distributed primitives
    "opaque-base",             # serialization (for ref-logp cache state)
    "opaque-patches",          # fused preference kernels (after Phase η)
    # NO opaque-dpsgd, NO opaque-dpftrl, NO opaque-optimizers
    # — mechanism + optimizer are chosen by the caller (trainer or example)
]

[project.optional-dependencies]
# Optional acceleration hint; users pick at install time
kernels = ["triton"]            # only if they want the chunked-preference kernel

[tool.setuptools.packages.find]
where = ["src"]
include = ["opaque.alignment*", "opaque.api.alignment*"]
namespaces = true
```

CI matrix entry: `pytest packages/opaque-alignment/tests/`.

---

## 6. Public API surface

```python
# opaque/alignment/__init__.py — headline re-exports
from opaque.alignment.logprob import (
    sequence_logp, selective_log_softmax, get_batch_logps,
)
from opaque.alignment.losses import (
    DPO_LOSSES, KTO_LOSSES, SFT_LOSSES,
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
from opaque.alignment.metrics import (
    reward_metrics, kl_estimator,
)
```

Stable from Phase β onward; populated incrementally as phases land.

---

## 7. Per-module spec

### 7.1 `losses/_dpo.py`

Every DPO variant has the signature:

```python
def dpo_<variant>(
    chosen_logratio: Tensor,     # scalar per example
    rejected_logratio: Tensor,   # scalar per example
    *,
    beta: float,
    **kwargs,
) -> Tensor:                     # scalar per example
    ...
```

Inputs are **post-reference-subtraction** log-ratios (the trainer or example caller computes `chosen_logp - ref_chosen_logp` upstream and passes the result). This keeps the loss functions pure and detached from the ref-model strategy.

Variants and formulas (`Δ = chosen_logratio - rejected_logratio`):

| Variant | Formula | Source |
|---|---|---|
| `sigmoid` | `-logσ(βΔ)` | Rafailov et al. 2023 |
| `hinge` | `relu(1 − βΔ)` | Liu et al. 2023 |
| `ipo` | `((chosen_avg − rejected_avg) − 1/(2β))²` where avg = logp / completion_len | Azar et al. 2024 |
| `robust` | `(−(1−ε)logσ(βΔ) + ε·logσ(−βΔ)) / (1−2ε)` | label-smoothed Rafailov |
| `apo_zero` | `(1 − σ(β·chosen_lr)) + σ(β·rejected_lr)` | Anchored Preference Optimization |
| `apo_down` | `σ(β·chosen_lr) + (1 − σ(βΔ))` | APO down variant |
| `exo_pair` | `qw·(log qw − log(1−ε)) + ql·(log ql − log ε)`, `q = σ(±β·Δ)` | EXO |
| `nca_pair` | `−logσ(β·chosen_lr) − 0.5(logσ(−β·chosen_lr) + logσ(−β·rejected_lr))` | NCA |
| `bco_pair` | `−logσ(β·chosen_lr) − logσ(−β·rejected_lr)` | BCO |
| `sppo_hard` | `(chosen_lr − 0.5/β)² + (rejected_lr + 0.5/β)²` | SPPO |
| `discopop` | logistic/exp blend at temperature `τ` | DiscoPOP |
| `sft` | CE on chosen completion (broadcast scalar) | DPO + SFT regularizer for MPO |
| `sigmoid_norm` | sigmoid loss using length-normalized log-ratios | length-normalized DPO |
| **`squarechipo`** | `0.5·(σ(βΔ) − 1)²` | arXiv:2505.21395, first optimal-rate DP-DPO |

Each ≤ 5 lines of pure torch. Module exports `LOSSES` dict keyed by name.

**Explicitly not exposed:** `aot`, `aot_pair`, `aot_unpaired` — these sort across the batch and violate DP-purity (§3.3). The module docstring documents this rejection so users grepping for them find the rationale.

### 7.2 `losses/_kto.py`

Signature differs from DPO (KTO is unpaired):

```python
def kto_<variant>(
    chosen_logratio: Tensor,         # scalar per example, present if label=True
    rejected_logratio: Tensor,       # scalar per example, present if label=False
    label: Tensor,                   # bool scalar per example
    *,
    beta: float,
    kl: Tensor | None = None,        # scalar per example
    desirable_weight: float = 1.0,
    undesirable_weight: float = 1.0,
) -> Tensor:                         # scalar per example
    ...
```

Two variants:

| Variant | Formula |
|---|---|
| `kto` | If `label`: `desirable_weight · (1 − σ(β·(chosen_lr − kl)))`. Else: `undesirable_weight · (1 − σ(β·(kl − rejected_lr)))`. Branching via `torch.where`. |
| `apo_zero_unpaired` | If `label`: `1 − σ(β·chosen_lr)`. Else: `σ(β·rejected_lr)`. No `kl` term. |

**KL note:** the `kl` input is computed by the trainer/example as `(KL_logp − ref_KL_logp)` per example. Under DP-SGD vmap, this is per-example by construction (each vmap'd item carries its own `kl` scalar). The "cross-batch mean" interpretation in TRL emerges naturally from the per-example gradient sum after vmap. Under Poisson-batch-size ≤ 1, callers pass `kl=0` (mathematically equivalent to `apo_zero_unpaired` for that step).

### 7.3 `losses/_sft.py`

Two variants:

| Variant | Formula | DP correctness |
|---|---|---|
| `nll` | Standard CausalLM CE — typically `model(**inputs).loss` returns per-example mean over non-ignored tokens | OK as-is — per-example mean is per-example data |
| `dft` | `(per_token_loss * mask).sum() / mask.sum()` per example | DP-corrected vs TRL: divisor is per-example `mask.sum()`, not `num_items_in_batch` |

`dft` rewrite vs TRL (`sft_trainer.py:788-802`): TRL divides by `num_items_in_batch` (batch-level token count). Here we divide by the per-example `mask.sum()`. Mathematically identical to TRL under the per-example mean interpretation; DP-safe under per-example clipping.

### 7.4 `losses/_f_divergence.py`

```python
def f_divergence_remap(
    name: Literal["reverse_kl", "forward_kl", "js_divergence", "alpha_divergence"],
    chosen_logratio: Tensor,
    rejected_logratio: Tensor,
    *,
    alpha_coef: float = 0.5,
) -> tuple[Tensor, Tensor]:                 # (chosen_score, rejected_score)
```

Remaps log-ratios into "scores" that DPO losses then consume:

| Variant | Remap |
|---|---|
| `reverse_kl` (default) | identity |
| `forward_kl` | `-exp(-logratio)` |
| `js_divergence` | `logsigmoid(logratio)` |
| `alpha_divergence` | `exp((α-1)·logratio) / (α-1)` with bf16/fp16 clamp |

The remapped scores feed into any DPO variant's `(chosen_logratio, rejected_logratio)` inputs.

### 7.5 `losses/_mpo.py`

```python
def mpo_combine(
    losses: dict[str, Tensor],          # {variant_name: per-example loss}
    weights: dict[str, float],          # {variant_name: weight}
) -> Tensor:                            # weighted sum, per example
    return sum(weights[k] * losses[k] for k in losses)
```

Trivial combinator for `loss_type=list` in TRL DPO.

### 7.6 `losses/_wpo.py`

```python
def wpo_weights(
    per_token_logps: Tensor,            # (B, T)
    logits: Tensor,                     # (B, T, V) — under no_grad upstream
    completion_mask: Tensor,            # (B, T)
) -> Tensor:                            # (B,) per-example weight
```

Computes WPO's per-pair marginal-logp-based weighting from arXiv:2406.11827. Caller passes `under_no_grad` inputs (the weight is detached from the gradient by design).

Incompatible with `dpo_aot*` (already rejected). Incompatible with vmap if `logits` is materialized inside the closure with autograd attached — caller must `detach()` upstream.

### 7.7 `losses/_ld_dpo.py`

```python
def ld_dpo_split(
    per_token_logps: Tensor,            # (B, T)
    completion_mask: Tensor,            # (B, T)
    shared_prefix_len: Tensor,          # (B,) — per-example shared length
    alpha: float,
) -> Tensor:                            # (B,) per-example sum
```

LD-DPO from arXiv:2409.10524: decompose per-token logps into shared-prefix vs tail, weight tail by `alpha`. Used by the trainer/example *before* computing log-ratios.

### 7.8 `losses/_fused.py`

Kernel-accelerated dispatchers wrapping `Opaque_FusedLinearPreference` (lives in `opaque-patches` after Phase η):

```python
def opaque_fused_linear_dpo_loss(
    hidden_states: Tensor,              # (B, 2, T, H) — chosen+rejected hidden
    lm_head_weight: Tensor,
    target_ids: Tensor,                 # (B, 2, T)
    completion_mask: Tensor,
    ref_chosen_logp: Tensor,            # (B,)
    ref_rejected_logp: Tensor,          # (B,)
    *,
    beta: float,
    loss_type: str = "sigmoid",
) -> Tensor:                            # (B,) per-pair loss
```

Memory benefit: ~80% peak-memory reduction vs eager forward (Liger README claim). Same per-pair output semantics as eager loss; opt-in via `config.use_fused_preference_kernel=True` in trainer or by direct import in functional examples.

### 7.9 `logprob.py`

```python
def selective_log_softmax(
    logits: Tensor,                     # (..., T, V)
    indices: Tensor,                    # (..., T)
) -> Tensor:                            # (..., T)
    """vmap-safe: gathers log p_i = log_softmax(logits)[indices[i]]."""

def sequence_logp(
    logits: Tensor,                     # (..., T, V)
    input_ids: Tensor,                  # (..., T)
    completion_mask: Tensor,            # (..., T)
    *,
    ld_alpha: float | None = None,
    shared_prefix_len: Tensor | None = None,
) -> Tensor:                            # scalar per sequence
    """Per-token log_softmax → gather → mask → sum (or LD-decomposed sum)."""

def get_batch_logps(
    logits: Tensor,                     # (B, T, V)
    labels: Tensor,                     # (B, T)
    *,
    average_log_prob: bool = False,
) -> Tensor:                            # (B,)
    """KTO-style helper with shift + ignore_index handling."""
```

Public; matches `opaque-engine`'s convention of putting reusable tensor helpers at top of the concern dir.

### 7.10 `collators/`

Three pure callables (no state):

| Collator | Output keys |
|---|---|
| `DataCollatorForLanguageModeling(pad_token_id, max_length, completion_only_loss, padding_free, pad_to_multiple_of)` | `input_ids: (B, L)`, `labels: (B, L)` (-100 masked), `attention_mask: (B, L)`, optional `completion_mask`, `assistant_masks`, `seq_lengths` (when packed) |
| `DataCollatorForPreference(pad_token_id, max_length, pad_to_multiple_of)` | `chosen_input_ids: (B, L)`, `chosen_attention_mask`, `chosen_completion_mask`, `rejected_input_ids: (B, L)`, `rejected_attention_mask`, `rejected_completion_mask`, optional `ref_chosen_logps: (B,)`, `ref_rejected_logps: (B,)` |
| `DataCollatorForUnpairedPreference(pad_token_id, max_length, calculate_KL)` | `completion_input_ids: (B, L)`, `completion_attention_mask`, `completion_labels`, optional `KL_completion_input_ids: (B, L)`, `KL_completion_attention_mask`, `KL_completion_labels`, `label: list[bool]`, optional `reference_logps`, `reference_KL_logps` |

**Note on `DataCollatorForPreference` layout `(B, ...)` vs TRL's `(2B, L)`:** see `trl-trainers-plan.md` §4.3 for the full rationale. Briefly: DP unit-of-privacy is the pair; the `(B, ...)` layout aligns vmap with the pair index.

### 7.11 `data/`

| Module | Function |
|---|---|
| `_prompt.py` | `extract_prompt(example) -> dict` — same heuristic TRL uses (longest common prefix of chosen+rejected for preference; just `prompt+completion` for unpaired) |
| `_packing.py` | `pack_bfd(dataset, max_length)`, `pack_bfd_split(dataset, max_length)`, `pack_wrapped(dataset, max_length)` — port of `trl/data_utils.py:686-789`. Generates `seq_lengths` column. Requires FlexAttention or SDPA-with-block-mask downstream (Phase θ documents). |
| `_chat_template.py` | `clone_chat_template(model, tokenizer, source_template_path)` — port of `trl/chat_template_utils.py:28-119`. Mutates model embedding via `resize_token_embeddings`; document that this MUST happen before any `make_functional` snapshot. `get_training_chat_template(tokenizer)` — inserts `{% generation %}` markers for `assistant_only_loss`. |
| `_kto_rotation.py` | `rotate_kto_completions(dataset, batch_size, seed)` — `dataset.map(_get_kl_dataset, batched=True, batch_size=...)` + column rename + `concatenate_datasets(..., axis=1)`. Runtime assertion that rotation is non-identity. |

### 7.12 `reference/`

| Module | Function |
|---|---|
| `_precompute.py` | `compute_ref_logprobs_for_dataset(dataset, ref_model_or_callable, collator, output_columns, *, batch_size=None, cache_key=())` — one-shot pass under `torch.no_grad()` + autocast; iterate via `DataLoader(collator)`; gather across ranks via `opaque.distributed.gather_for_metrics` (after Phase 0.5 lands per `trl-trainers-plan.md` §10); cache to `.npz` keyed on `(Hasher.hash(dataset._fingerprint), hash_module(ref), cache_key)`; return `dataset.add_column(name, ...)` for each output column. |
| `_adapter.py` | `null_ref_context(model)` — `@contextmanager`: if PEFT with `"ref"` adapter, swap and restore; elif PEFT with only `"default"`, `model.disable_adapter()`; else no-op. **Labeled "outside vmap only"** — toggles module instance attributes. |
| `_sync.py` | `ema_update_reference(ref_params, policy_params, alpha)` — functional `ref = (1-α)·ref + α·policy` over a pytree. TR-DPO core; used by `SyncRefModelCallback` in the trainer layer. |

### 7.13 `metrics.py`

```python
def reward_metrics(
    chosen_logratio: Tensor,            # (B,)
    rejected_logratio: Tensor,          # (B,)
    *,
    beta: float,
) -> dict[str, Tensor]:
    """Returns rewards/chosen, rewards/rejected, rewards/accuracies, rewards/margins."""

def kl_estimator(
    policy_logp: Tensor,                # (B,)
    ref_logp: Tensor,                   # (B,)
    *,
    detach: bool = True,
    clamp_min: float = 0.0,
) -> Tensor:                            # scalar
    """Empirical KL estimator with detach + clamp."""
```

Pure functions consumed at log-time by the trainer's `log()` override (per `trl-trainers-plan.md` §6.2) or directly by functional examples for diagnostics.

---

## 8. Phase plan

Phases use Greek letters to keep them separable from the trainer plan's Roman numerals. They sequence loosely: α before everything; β foundation primitives unlock γ/δ/ε in parallel; ζ before any TRL trainer Phase 2/3; η optional optimization; θ + ι finish.

### Phase α — Package skeleton (S, ≤ 1 day)

- Create `packages/opaque-alignment/` with `pyproject.toml` (§5), `README.md`, `src/opaque/api/alignment/`, `src/opaque/alignment/`.
- Empty `__init__.py` everywhere.
- Smoke test (`tests/test_import.py`): both `import opaque.alignment` and `import opaque.api.alignment` succeed.
- Register in workspace tooling: `uv.lock`, all-packages CI matrix entry, ruff/pyright config inheritance.
- README documenting the api/façade pattern, functional-primitives philosophy, mechanism-agnostic posture, link to `examples/train_dpo.py` (will be empty until Phase γ).

### Phase β — Foundation primitives (M, 3–4 days)

Independent of any specific loss family. Unlocks everything downstream.

- `logprob.py`: `selective_log_softmax`, `sequence_logp` (with optional `ld_alpha` decomposition stub), `get_batch_logps`.
- `collators/_language_modeling.py`: `DataCollatorForLanguageModeling` text-only path (VLM out of scope; packing deferred to Phase θ).
- `collators/_preference.py`: `DataCollatorForPreference` in `(B, ...)` layout.
- `collators/_unpaired_preference.py`: `DataCollatorForUnpairedPreference` text-only path.
- `data/_prompt.py`: `extract_prompt`.
- `metrics.py`: `reward_metrics`, `kl_estimator`.

Tests:
- Unit tests for `selective_log_softmax`, `sequence_logp` vs hand-computed reference.
- Collator determinism: same input → same output (used by `_discover_batch_keys` upstream).
- DP-purity test for `sequence_logp`: NaN-one-example → only that row's logp is NaN.

### Phase γ — DPO loss family + functional example (M, 4–5 days)

- `losses/_dpo.py`: all 14 variants + `LOSSES` registry.
- `losses/_f_divergence.py`: 4 variants.
- `losses/_mpo.py`: combinator.
- `losses/_wpo.py`: per-example weight fn.
- `losses/_ld_dpo.py`: shared/tail split.
- `examples/train_dpo.py`: functional, hand-rolled DPO training loop (DP-SGD mechanism, see §11). Uses precomputed ref logps for simplicity.

Tests:
- Per-variant unit test vs TRL eager on synthetic batches at `σ=0, C=∞` → within `1e-3`.
- Vmap closure test on a 4-pair synthetic batch → finite per-pair gradients.
- DP-purity NaN-injection test on every variant.
- MPO combinator: `loss_type=["sigmoid", "sft"]` with equal weights → matches hand combination.

### Phase δ — KTO loss + rotation + functional example (M, 3 days)

- `losses/_kto.py`: `kto`, `apo_zero_unpaired` + `LOSSES` registry.
- `data/_kto_rotation.py`: `rotate_kto_completions`.
- Extension to `collators/_unpaired_preference.py` to emit `KL_*` keys when present.
- `examples/train_kto.py`: functional.

Tests:
- Per-variant unit test vs TRL eager (KTO trainer test at `experimental/kto/`).
- Rotation determinism: same seed → same rotation; non-identity assertion.
- Batch-size-0/1 Poisson edge case: `kl=0` produces sensible behavior.
- DP-purity NaN-injection.

### Phase ε — SFT loss family + functional example (S, 2 days)

- `losses/_sft.py`: `nll`, `dft` (with DP-corrected divisor) + `LOSSES` registry.
- `examples/train_sft.py`: functional, code dataset.

Tests:
- `dft` vs hand-computed reference matching TRL's formula but with per-example divisor.
- DP-purity NaN-injection.

### Phase ζ — Reference handling (M, 3 days)

Blocks DPO/KTO TRL trainer phases (per `trl-trainers-plan.md` §11/§12/§13) that need precomputed ref logps.

- `reference/_precompute.py`: `compute_ref_logprobs_for_dataset` with `.npz` caching via `opaque.serialization`.
- `reference/_adapter.py`: `null_ref_context`.
- `reference/_sync.py`: `ema_update_reference` (functional EMA over a params pytree).

Tests:
- Precompute round-trip: cache miss → compute + save → cache hit on re-call.
- `null_ref_context` with PEFT (mocked adapter); state restored on exit.
- `ema_update_reference` on a small pytree; values match expected formula.

### Phase η — Chunked fused-linear preference kernel (L, 3–5 days)

**Coordinates with `trl-trainers-plan.md` Phase −1.** Two sub-deliverables:

- `opaque-patches`: new `Opaque_FusedLinearPreference / _FusedLinearPreferenceBackward` autograd.Function pair under `packages/opaque-patches/src/opaque/api/patches/kernels/fused_linear_preference.py`. Two-level vmap-safe pattern matching existing kernels (LCE, SwiGLU, RoPE, RMSNorm).
- `opaque-alignment`: `losses/_fused.py` dispatchers `opaque_fused_linear_dpo_loss`, `opaque_fused_linear_kto_loss`, … wrapping the base kernel.

Tests:
- Eager-vs-fused numeric parity within `1e-4` on synthetic batches.
- Memory benchmark: peak memory < `(B, T, V)` materialization.
- Vmap-safety: kernel composes with `vmap(grad(...))` on a 4-pair batch.

Optional optimization; not blocking for Phase ι.

### Phase θ — Advanced data pipeline (M, 4–5 days)

Deferred because the value depends on attention-backend support and the math itself is independent of preference learning.

- `data/_packing.py`: `pack_bfd`, `pack_wrapped`, `pack_bfd_split`.
- `data/_chat_template.py`: `clone_chat_template`, `get_training_chat_template`.

Tests:
- Packing density vs target on synthetic variable-length data.
- BFD correctness: every produced bin has total length ≤ `max_length`.
- Chat-template round-trip on a tokenizer fixture.

**Attention-backend question:** packing produces `position_ids` with per-doc restarts. Cross-doc blocking requires FlashAttention2 OR FlexAttention OR an explicit 4D mask via SDPA. Phase θ includes a fixture test for FlexAttention + vmap composition; if it works, FlexAttention is the documented path. If not, SDPA-with-mask is the fallback (slower but correct).

### Phase ι — Docs and recipe scaffolding (S, 2 days)

- `docs/alignment/index.md` — package overview, functional-primitives philosophy, mechanism-agnostic posture, links to functional examples + trainer examples.
- `docs/alignment/losses.md` — per-loss reference (formula, paper, DP notes).
- `docs/alignment/collators.md` — per-collator output schema reference.
- `docs/alignment/reference.md` — the four ref-model paths (mirrors `trl-trainers-plan.md` §4.1 from the primitive-layer perspective).
- `docs/alignment/recipes.md` — placeholder for future recipe documentation (decoupled DP-RLHF, etc.).

### Phase totals

| Phase | Effort | Cumulative |
|---|---|---|
| α — skeleton | S | 0.5d |
| β — foundation | M | 4d |
| γ — DPO | M | 8d |
| δ — KTO | M | 11d |
| ε — SFT | S | 13d |
| ζ — reference | M | 16d |
| η — fused kernel | L | 21d (optional) |
| θ — advanced data | M | 25d |
| ι — docs | S | 26d |

**Total: ~5 working weeks** for the full package surface (excluding the optional Phase η kernel). The TRL trainers can start consuming primitives from end of Phase γ onward.

---

## 9. Test strategy

### 9.1 Loss-fn unit tests

For every loss variant in `losses/_*.py`:
- ≥3 hand-computed reference cases (small β, large β, edge case like `chosen=rejected`).
- Vs TRL eager on a 4-example synthetic batch at `σ=0, C=∞` → within `1e-3`.

### 9.2 Vmap-safety contract test

A meta-test that walks every public symbol in `opaque.alignment.losses.*` + `opaque.alignment.logprob.*` and verifies:
- The function survives `torch.func.vmap(torch.func.grad(...))(params, batch_args)` on a 4-example synthetic batch.
- The resulting per-example gradient is finite.

If a new symbol is added without satisfying the contract, this test fails. Acts as a guardrail.

### 9.3 DP-purity NaN-injection test

A meta-test for every public per-example loss function:
1. Build a batch of 4 examples.
2. Replace example #2's input with NaN.
3. Compute per-example gradients via `vmap(grad(loss_fn))`.
4. Assert that examples #0, #1, #3 have finite gradients; example #2 has NaN/inf.

This catches accidental cross-example divisors (the `num_items_in_batch` failure mode).

### 9.4 TRL eager parity

Parameterized test per loss variant per supported config:
- Build TRL trainer with the variant; run one forward + backward on a synthetic batch.
- Build `opaque-alignment` equivalent (functional, no DP-SGD); run on the same batch.
- Assert per-example loss values match within `1e-3`.

This is the "TRL faithfulness" guarantee.

### 9.5 Mechanism-agnostic integration test

Same loss closure run end-to-end under two mechanism choices:
- DP-SGD: `clipped_grad` + `gaussian_noise` + `OpaqueEpochPoissonBatchSampler`.
- DP-FTRL: `clipped_grad` + `band_mf` noise + `b_min_sep_sampler`.

Assert both complete one training step without errors. Not a numeric-correctness test (the mechanisms produce different noise distributions); a smoke test that the package contract holds under mechanism substitution.

### 9.6 Cache round-trip test (Phase ζ)

`compute_ref_logprobs_for_dataset` called twice on same dataset → second call is a cache hit (verified by mocking the forward pass).

---

## 10. Cross-package interactions

### 10.1 With `opaque-engine`

`opaque-alignment` consumes:
- `opaque.clipping.clipped_grad` — at the call site (trainer or example), not inside primitives.
- `opaque.functional.make_functional` — at the call site, not inside primitives.
- `opaque.distributed.gather_for_metrics`, `is_main_process` — used by `compute_ref_logprobs_for_dataset` (Phase ζ). These are added in `trl-trainers-plan.md` Phase 0.5; `opaque-alignment` Phase ζ blocks on that.
- `opaque.distributed.barrier` — used by precompute to synchronize across ranks before writing cache.
- `opaque.serialization.{state_dict, from_state_dict}` (via `opaque-base`) — for precompute cache state.

### 10.2 With `opaque-patches`

- After Phase η: `losses/_fused.py` consumes `Opaque_FusedLinearPreference` from `opaque-patches`.
- Phase η has a coordinated `opaque-patches` deliverable that adds this kernel.
- All other kernel usage is implicit through HF model forwards (`use_performance_kernels=True` routes through opaque-patches; transparent to the loss layer).

### 10.3 With `opaque.transformers.trl`

`opaque-alignment` is consumed by trainer classes per `trl-trainers-plan.md` §6.2. The trainers' single `compute_per_example_loss` override orchestrates `opaque-alignment` primitives (vmap'd for both training and eval). The dependency edge runs `opaque-transformers → opaque-alignment`; never the reverse.

### 10.4 With `opaque-dpsgd` / `opaque-dpftrl`

**No dependency from `opaque-alignment`.** The mechanism is chosen at the call site:

```python
# Functional example chooses DP-SGD
from opaque.dpsgd.noise import gaussian_noise
from opaque.dpsgd.sampling import OpaqueEpochPoissonBatchSampler

# Or DP-FTRL
from opaque.dpftrl.noise import band_mf
from opaque.dpftrl.sampling import b_min_sep_sampler
```

The `opaque-alignment` primitives (`dpo_sigmoid`, `sequence_logp`, etc.) are unchanged across mechanism choices.

### 10.5 With `opaque-optimizers`

No dependency. Optimizer is chosen at the call site (`from opaque.optimizers import adamw`). DP-AdamW per arXiv:2505.08849 is `adamw(noise_bias_correction=True)`; no opaque-alignment-side work needed.

---

## 11. Functional examples

Each example is a sibling of an existing functional script. Three deliverables:

### `examples/train_sft.py` (Phase ε)

Pattern: equivalent of `train_causal_lm.py` but using `opaque-alignment.losses.sft.nll`:

```python
from opaque.alignment import DataCollatorForLanguageModeling
from opaque.alignment.losses import SFT_LOSSES
from opaque.alignment.logprob import sequence_logp
from opaque.clipping import clipped_grad
from opaque.functional import make_functional
from opaque.dpsgd.noise import gaussian_noise
from opaque.dpsgd.sampling import OpaqueEpochPoissonBatchSampler
from opaque.optimizers import adamw

fmodel, trainable, frozen = make_functional(model, partition_trainable=True)

def per_example_loss(trainable_params, input_ids, attention_mask, labels):
    merged = {**frozen, **trainable_params}
    out = fmodel(merged, input_ids=input_ids, attention_mask=attention_mask)
    return SFT_LOSSES["nll"](out.logits, labels)

grad_fn, clip_state = clipped_grad(per_example_loss, normalize_by=expected_batch_size, ...)
noise_fn, noise_state = gaussian_noise(...)
opt = adamw(noise_bias_correction=True)
opt_state = opt.init(trainable)

for batch in dataloader:
    (grads, aux), clip_state = grad_fn(trainable, *batch_args, state=clip_state)
    noised, noise_state = noise_fn(grads, noise_state)
    updates, opt_state = opt.update(noised, opt_state)
    trainable = apply_updates(trainable, updates)
```

### `examples/train_dpo.py` (Phase γ)

Same shape, with reference-logp precompute + DPO sigmoid loss:

```python
from opaque.alignment import (
    DataCollatorForPreference, compute_ref_logprobs_for_dataset, sequence_logp,
)
from opaque.alignment.losses import DPO_LOSSES

dataset = preprocess_preference(raw, tokenizer, max_length=1024)
dataset = compute_ref_logprobs_for_dataset(
    dataset, ref_model, collator=DataCollatorForPreference(...),
    cache_key=("dpo", model_name),
    output_columns=("ref_chosen_logps", "ref_rejected_logps"),
)

def per_example_loss(trainable_params,
                     chosen_ids, chosen_mask, chosen_completion_mask,
                     rejected_ids, rejected_mask, rejected_completion_mask,
                     ref_chosen_logps, ref_rejected_logps):
    merged = {**frozen, **trainable_params}
    chosen_out = fmodel(merged, input_ids=chosen_ids, attention_mask=chosen_mask)
    rejected_out = fmodel(merged, input_ids=rejected_ids, attention_mask=rejected_mask)
    chosen_logp = sequence_logp(chosen_out.logits, chosen_ids, chosen_completion_mask)
    rejected_logp = sequence_logp(rejected_out.logits, rejected_ids, rejected_completion_mask)
    chosen_logratio = chosen_logp - ref_chosen_logps
    rejected_logratio = rejected_logp - ref_rejected_logps
    return DPO_LOSSES["sigmoid"](chosen_logratio, rejected_logratio, beta=0.1)
```

**Mechanism swap demo (in-comment example):**

```python
# Swap DP-SGD → DP-FTRL: only the noise + sampler imports change.
# Loss closure and optimizer are unchanged.
# from opaque.dpftrl.noise import band_mf
# from opaque.dpftrl.sampling import b_min_sep_sampler
```

### `examples/train_kto.py` (Phase δ)

Same shape with `KTO_LOSSES["kto"]`, KTO collator, and `rotate_kto_completions` for the KL term.

---

## 12. Roadmap beyond the TRL trainer port

The package gains shape over time. Each item below is its own follow-on workstream that lands as additional modules in `opaque-alignment`:

| Item | Where it lives | Why opaque-alignment is the right home |
|---|---|---|
| **Reward modeling primitives** | `losses/_reward.py` (`bt_pairwise_loss`) | Pure per-example loss; consumed by `RewardTrainer` in trl subnamespace and by `train_reward.py` example. |
| **ORPO loss** (`losses/_orpo.py`) | losses/ | Same pattern as DPO; small loss module. |
| **CPO loss** | losses/ | Same pattern. |
| **SimPO loss** | losses/ | Same pattern. |
| **GRPO primitives** (`losses/_grpo.py`, `logprob.py` extension for `compute_grpo_advantages`) | losses/ + logprob/ | Trajectory-level needs new logprob helpers but reuses the same chunked-kernel base. |
| **Decoupled DP-RLHF recipe** | `recipes/decoupled_dp_rlhf.py` | Scripted multi-stage workflow chaining `train_reward.py` (DP) → `train_ppo.py` (non-DP actor) per arXiv:2603.22563. Recipe = pure function over (dataset, model, config) → trained artifact. |
| **Alignment eval harnesses** | `eval/{reward_bench, alpaca_eval, kl_drift}.py` | Pure functions over `(policy_model, ref_model, dataset) → metrics`. Notebook-friendly. |
| **Recipe DSL** | `recipes/_registry.py` | `@register_recipe("sft+dpo")` for paper recipes (SquareχPO defaults, DP-AdamW + DPO). |
| **Per-architecture chunked-loss specializations** (Phase η.2+) | losses/_fused.py | Loss-specific dispatchers added as KTO, ORPO, CPO, SimPO land. |

None of these require structural changes to the core package. The Greek-letter phases above ship a *foundation*; the Roman-numeral roadmap items are *additions*.

---

## 13. Risk register

| ID | Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|---|
| α1 | `(B, ...)` collator layout drifts from TRL's `(2B, L)` for DPO and causes user confusion when porting code | M | L | Document the layout choice in `collators/_preference.py` docstring with a clear "TRL uses `(2B, L)`; we use `(B, ...)` because per-example DP-SGD clips per pair." |
| α2 | DP-purity test misses a subtle batch-level dependency | M | H | NaN-injection test + a tracer that walks the function body and flags any tensor reduction over the batch dim. |
| α3 | Liger chunked-preference port (Phase η) breaks under vmap | M | M | Phase η has explicit vmap composition test before declaring done. Eager loss path remains the default; fused is opt-in. |
| α4 | Reference-logp precompute cache fingerprint collisions | L | H | Use `Hasher.hash((dataset._fingerprint, hash_module(ref), explicit_cache_key))`. The explicit caller-provided `cache_key` is the escape hatch for users who notice collisions. |
| α5 | PEFT `null_ref_context` doesn't compose cleanly with `make_functional`-captured params | L | H | Documented "outside vmap only" + fixture test in Phase ζ. If broken, fall back to "explicit ref_model required for PEFT." |
| α6 | Phase ζ blocks on `trl-trainers-plan.md` Phase 0.5 (`opaque.distributed` extensions) | M | M | Coordinate via explicit dependency note. Phase 0.5 is S (1-2 days); can land in parallel with Phase β. |
| α7 | DFT loss DP-correctness deviates from TRL behavior in ways users notice | L | M | Document the DP-correct formula explicitly. The difference (per-example divisor vs batch-level) is mathematically identical under per-example mean interpretation. |
| α8 | New loss variants from papers require contract violations (e.g., they sort across batch) | M | L | Reject at function level (don't expose). Document rejection in module docstring. Same pattern as `aot`/`aot_pair`. |
| α9 | FlexAttention + vmap composition (Phase θ) doesn't work | M | M | Phase θ fixture test before relying on it. Fallback: SDPA + 4D block mask. |
| α10 | `clone_chat_template`'s `resize_token_embeddings` interaction with `make_functional` | L | M | Document: chat template clone MUST happen before `make_functional` snapshot. Runtime check in trainer init if possible. |

---

## 14. References

### Within this plan family
- `docs/development/trl-trainers-plan.md` — sibling plan for the TRL-style class trainers consuming these primitives. See especially §4 (cross-cutting decisions: ref-model paths, KTO rotation, DPO collator, DP-purity rule, PEFT integration, loss-type coverage, kernel optimization), §6 (trainer responsibilities) + §8.1 (the unified `compute_per_example_loss` hook), §8 (DPTrainer foundational hooks), §10 (`opaque.distributed` extensions).

### Opaque packages on `feat/dptrainer-main-integration`
- `packages/opaque-engine/src/opaque/api/engine/clipping/_clipped_grad.py` — `clipped_grad`.
- `packages/opaque-engine/src/opaque/api/engine/functional/` — `make_functional`.
- `packages/opaque-engine/src/opaque/api/engine/distributed/` — current public surface (`is_distributed`, `get_rank`, `get_world_size`, `all_reduce`, `sum_gradients`, `sum_gradients_`, `sync`, `reduce_pytree`, `reduce_pytree_`, `local_shard`). Missing `gather_for_metrics`, `is_main_process`, `wait_for_everyone`, `num_processes`, `process_index` — added in `trl-trainers-plan.md` Phase 0.5.
- `packages/opaque-optimizers/src/opaque/api/optimizers/_adam.py:117,191-203,205-245` — DP-AdamW (`noise_bias_correction=True`).
- `packages/opaque-patches/src/opaque/api/patches/kernels/` — fused kernels (CE, LCE, SwiGLU, GeGLU, RoPE, RMSNorm, LoRA-MLP/QKV/W).
- `packages/opaque-base/src/opaque/api/base/serialization/` — `state_dict`, `from_state_dict` (for precompute cache).

### TRL (analyzed at `/tmp/trl_src`, v1.5.0.dev0)
- `trl/trainer/dpo_trainer.py:152-211` — `DataCollatorForPreference` (reference for our `(B, ...)` layout).
- `trl/trainer/dpo_trainer.py:1000-1084,1042-1084` — `_precompute_ref_logps`, `compute_ref_log_probs`.
- `trl/trainer/dpo_trainer.py:1224-1402` — f-divergence remap + loss-type dispatch loop.
- `trl/trainer/dpo_trainer.py:1389-1400` — WPO weighting.
- `trl/trainer/dpo_trainer.py:1075-1081,1182-1188` — LD-DPO `ld_alpha` decomposition.
- `trl/trainer/sft_trainer.py:788-802` — DFT loss (TRL formula; we reimplement DP-corrected).
- `trl/experimental/kto/kto_trainer.py:83-90,609-623,875-887` — KTO rotation + KL math.
- `trl/trainer/utils.py:1056-1093` — `use_adapter` (reference for `null_ref_context`).
- `trl/data_utils.py:686-789` — packing.
- `trl/chat_template_utils.py:28-119` — `clone_chat_template`.

### DP-alignment papers
- **arXiv:2505.21395** — SquareχPO. Loss `dpo_squarechipo` in Phase γ.
- **arXiv:2505.08849** — DP-AdamW. Already implemented (no opaque-alignment work).
- **arXiv:2406.11827** — WPO weighting (`losses/_wpo.py`).
- **arXiv:2409.10524** — LD-DPO (`losses/_ld_dpo.py`).
- **arXiv:2603.22563** — Decoupled DP-RLHF. Roadmap recipe.
- **arXiv:2501.19080** — DP-PolicyGradient. Out of scope.
