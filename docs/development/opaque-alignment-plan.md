# opaque-alignment — Package Plan

**Status:** Planning. Sibling doc to `trl-trainers-plan.md`; the two evolve together but address different audiences. Each phase is structured as a Claude Code dynamic-workflow run (parallel work units + adversarial validation — see §10.0).

**Scope:** A new distribution `opaque-alignment` that ships **functional, mechanism-agnostic primitives for DP-safe preference learning**: per-example loss functions (DPO / KTO / SFT families), logprob helpers, preference collators, dataset transforms, reference-model helpers, alignment metrics, and a small alignment-specific kernel catalog. Built on `opaque-engine` (clipping, functional, distributed) and `opaque-base` (serialization); consumed by both functional training scripts (`examples/train_dpo.py`-style) and the TRL-style class trainers in `opaque.transformers.trl`.

**Branches:**
- Planning (this doc): `claude/add-trl-trainers-plan-nB07O`, rebased onto `feat/dptrainer-main-integration`.
- Implementation phases: per-phase sub-branches.

**Read alongside:** `docs/development/trl-trainers-plan.md` — covers the class trainers built on top of these primitives. **Read first:** `AGENTS.md` "Namespace contract" section — `opaque-alignment` strictly follows the nine rules there.

---

## Table of contents

1. [Goals and non-goals](#1-goals-and-non-goals)
2. [Relationship to `opaque.transformers.trl`](#2-relationship-to-opaquetransformerstrl)
3. [Architectural philosophy](#3-architectural-philosophy)
4. [Module layout (api/façade pattern)](#4-module-layout-apifaçade-pattern)
5. [Dependency pin](#5-dependency-pin)
6. [Public API surface](#6-public-api-surface)
7. [Per-module spec](#7-per-module-spec)
8. [Loss API tiers and `DPSpec`](#8-loss-api-tiers-and-dpspec)
9. [DDP-aware loss design](#9-ddp-aware-loss-design)
10. [Phase plan](#10-phase-plan)
11. [Test strategy](#11-test-strategy)
12. [Cross-package interactions](#12-cross-package-interactions)
13. [Functional examples](#13-functional-examples)
14. [Roadmap beyond the TRL trainer port](#14-roadmap-beyond-the-trl-trainer-port)
15. [Risk register](#15-risk-register)
16. [References](#16-references)

---

## 1. Goals and non-goals

### 1.1 Goals

- **Standalone usability.** A researcher can build a DP-DPO training script using only `opaque-alignment` + the mechanism of their choice (DP-SGD or DP-FTRL) + an optimizer from `opaque.optimizers`. **No requirement to subclass `DPTrainer`.**
- **Mechanism-agnostic.** Depends only on `opaque-engine` (clipping, functional, distributed) and `opaque-base` (serialization). It does NOT depend on `opaque-dpsgd`, `opaque-dpftrl`, or `opaque-optimizers`. The mechanism + optimizer are chosen at the call site.
- **Two-tier DP-purity invariant.** Every public per-example loss is labelled either Tier 1 (strict per-example, vmap-safe, NaN-injection passes) or Tier 2 (per-example + detached batch aggregate with bounded O(1/n) leverage; aggregate-detach audit passes). Tier 3 (rank/sort/quantile across batch) is rejected. See §3.3 + §8.
- **TRL-faithful loss math.** Every DPO / KTO / SFT loss variant that fits Tier 1 or Tier 2 lands at full numeric parity with TRL eager (within `1e-3` at `σ=0, C=∞`).
- **First-class functional examples.** `examples/train_sft.py`, `train_dpo.py`, `train_kto.py` ship as siblings of the existing `train_causal_lm.py`, demonstrating the "primitives → training loop" pattern.
- **Citable home for DP-alignment paper recipes.** SquareχPO (arXiv:2505.21395), DP-DPO theory (arXiv:2502.18014), DP non-decomposable objectives (Kumar et al. arXiv:2310.03104) — all referenced where their results inform the design.
- **DDP-aware loss interface.** Losses that need a cross-rank aggregate declare it explicitly via `LossAggregateSpec(cross_rank=True)`; the trainer routes the collective. No DDP-aware-loss recipes existed in the library before this package — see §9.
- **Reusable across trainer families.** Future RewardTrainer, ORPOTrainer, CPOTrainer, SimPOTrainer, GRPOTrainer (under `opaque.transformers.trl` or a recipe DSL) all consume the same primitives — no copy-paste of loss math between trainer subclasses.

### 1.2 Non-goals

- **No trainer classes.** Trainers live in `opaque.transformers.trl` (see `trl-trainers-plan.md`). This package ships primitives only.
- **No `accelerate` dependency.** Reductions go through `opaque.distributed` functional primitives.
- **No DeepSpeed / FSDP / Accelerate integration.** Out of architectural reach for per-example DP-SGD.
- **No VLM** (vision-language models). Out of scope.
- **No `aot` / `aot_pair` loss variants.** Sort-across-batch with O(1) leverage breaks per-example DP sensitivity. No published DP-safe variant exists. Rejected at function level (not even exposed in the loss registry). See §3.3 Tier 3.
- **No bundled mechanism choice.** Users explicitly import `opaque.dpsgd.*` or `opaque.dpftrl.*` at call site.
- **No `nn.Module` subclasses.** Everything functional or factory-built. PyTorch's `nn.Module` lifecycle is incompatible with `vmap(grad(...))` and conflicts with our "no hidden state" principle.
- **No DP-PPO actor** (arXiv:2501.19080). Trajectory-level DP is a separate design.

---

## 2. Relationship to `opaque.transformers.trl`

```
                ┌────────────────────────────────────────────────────────────┐
                │  opaque.transformers.trl                                   │
                │  ─────────────────────────                                 │
                │  SFTTrainer, DPOTrainer, KTOTrainer                        │
                │  configs, signature columns, log() drain, prediction_step  │
                │  (~30-line override of compute_per_example_loss — one hook)│
                └─────────────────────┬──────────────────────────────────────┘
                                      │ consumes
                                      ▼
                ┌────────────────────────────────────────────────────────────┐
                │  opaque-alignment                                          │
                │  ────────────────                                          │
                │  loss/{dpo,kto,sft}/   pure per-example loss fns + types  │
                │  logprob/              selective_log_softmax, sequence_logp│
                │  collator/             factory fns returning callables    │
                │  data/                 prompt, packing, chat template, KTO rotation
                │  reference/            precompute, null_ref_context, sync │
                │  metric/               rewards, KL, token accuracy        │
                │  kernel/               fused linear preference (paired + unpaired)
                └─────┬──────────────────────────┬──────────────────────────┘
                      │                          │
       ┌──────────────┘                          └─────────┐
       │ depends on                                        │ depends on
       ▼                                                   ▼
┌─────────────────────┐                          ┌───────────────────────┐
│  opaque-engine      │                          │  opaque-base          │
│  (clipping,         │                          │  (serialization for   │
│   functional,       │                          │   ref-logp cache)     │
│   distributed)      │                          │                       │
└─────────────────────┘                          └───────────────────────┘

Optional runtime dep (used by SFT recipe path, not by the kernel itself):
       opaque-patches  ← consumed by examples + trainers for opaque_linear_cross_entropy_loss
```

Per `trl-trainers-plan.md` §6/§8.1, the TRL trainers override a **single** unified hook in `DPTrainer`: `compute_per_example_loss(fmodel, params, inputs, *, return_logits)`. The trainer wraps it with `vmap` for both training (→ grad → clip → noise) and per-example eval. The hook orchestrates `opaque-alignment` primitives; that single override point is what makes the primitive layer load-bearing.

---

## 3. Architectural philosophy

### 3.1 Functional, no hidden state

Every public symbol in `opaque-alignment` is a pure function, a factory function returning a callable, or an inert dataclass. **No `nn.Module` subclasses; no user-instantiated classes** (per AGENTS.md namespace rule 9). Collators, for example, are factory functions returning callables — `language_modeling_collator(pad_token_id, max_length, ...) → Callable[[list[dict]], dict[str, Tensor]]` — not `DataCollatorForLanguageModeling(...)` classes.

This mirrors `opaque-engine`'s design (`AGENTS.md`: "every component uses explicit state — no hooks, no subclassing, no hidden mutation"). The reasoning is the same: `vmap(grad(...))` composes cleanly only over pure functions; hidden state breaks per-example independence.

### 3.2 Mechanism-agnostic

Dependencies are pinned to **substrate** packages only (`opaque-engine`, `opaque-base`), not to **mechanism** packages (`opaque-dpsgd`, `opaque-dpftrl`). A DPO loss does not care whether you're going to add Gaussian noise via DP-SGD or matrix-factorized noise via DP-FTRL — it's the caller's choice at clip + noise time.

Practical consequence: a researcher running DP-FTRL DPO writes the same loss closure as DP-SGD DPO. Only the mechanism imports change at the bottom of the training loop. See §13 for the worked example.

### 3.3 DP-purity invariant — Tier 1 / Tier 2 / rejected

The DP-SGD privacy proof (Abadi et al. 2016) bounds **sensitivity** of the sum-of-clipped-gradients with respect to record swap. A blanket "no cross-batch aggregates" rule would be sufficient but is unnecessarily restrictive: the actual requirement is only that swapping one record changes the released gradient by at most `O(C)`. Three regimes follow:

**Tier 1 — strict per-example.** Loss output for example `i` depends only on example `i`'s data. Sensitivity to record swap is trivially `O(C)` after clipping. Verified by **NaN-injection contract test** (§11.3): replace one example's input with NaN; only that example's gradient is affected. *Examples:* most DPO variants (sigmoid, hinge, ipo, robust, apo_*, exo_pair, nca_pair, bco_pair, sppo_hard, discopop, squarechipo, sigmoid_norm, sft), KTO `apo_zero_unpaired`, SFT `nll`, SFT `dft`.

**Tier 2 — per-example + detached batch aggregate with bounded leverage.** Loss output for example `i` depends on example `i`'s data and on a `detached_aggregate: Tensor` argument that the trainer computes once over the batch outside the vmap and broadcasts in. The aggregate must be `.detach()`-ed before passing in (verified by autograd-graph audit). Per-example sensitivity bound: aggregate leverage is `O(1/n)` for arithmetic means; the `DPSpec` records which. The aggregate is **unreleased private information**: it is consumed inside the gradient computation but is never an output of the mechanism, so it does not add to the privacy ledger. Verified by **aggregate-detach audit + leverage test** (§11.4): swapping one example's contribution to the aggregate changes the per-example loss by `O(1/n)`, not `O(1)`. *Examples:* KTO `kto` (detached batch-mean KL, leverage `O(1/n)` — matches TRL `kto_trainer.py:882-884` and KTO paper Eq. 8: "we do not backpropagate through z_0").

**Tier 3 — rejected.** Losses whose aggregate dependence on a single example is `O(1)` (sort, rank, quantile, max/min without smoothing). Per-example clipped gradient norm can flip entirely when one example is swapped, blowing sensitivity from `C` to `O(n·C)`. No known efficient DP-safe variant. *Examples:* DPO `aot`, `aot_pair`, `aot_unpaired`. Rejected at function level — not exposed in the loss registry. The rejection rationale is captured in `loss/dpo/types.py` per-variant `DPSpec`.

**Background theory.** Kumar et al., "Differentially Private Optimization for Non-Decomposable Objective Functions" (NeurIPS 2023, arXiv:2310.03104) provides the rigorous treatment of Tier 2: their Lemma 4.1 + Theorem 4.2 show that for InfoNCE-style softmax contrastive losses, per-example partial derivatives with respect to swapped examples are `O(1/n)`, giving sensitivity `O(B)` instead of `O(n·B)`. The same argument applies to KTO's batch-mean detached KL.

**The pre-clip / post-noise distinction.** Division **before** gradient computation is DP-correct: per-example divisors (mask token counts, per-example sequence lengths) are part of the per-example loss. Division **after** clipping or **after** noising must use only public quantities (`expected_batch_size`, `args.max_length`, fixed constants) — the realized batch token count is private and cannot be used as post-processing. See §8.1 for the divisor rule applied to each loss family.

**Telemetry rule.** Metrics computed during training (entropy, mean_token_accuracy, etc.) are private if **not released**. They can be displayed in logs, written to W&B with epsilon accounting, or used for early stopping — but only if the release path is itself DP-safe (e.g., gated by `_metrics` accumulator that only emits aggregates with appropriate noise). For v1, we treat all training-time metrics as private internal state; explicit DP-aware metric release is a roadmap item.

### 3.4 Vmap-safety contract

Every public loss + logprob + collator-output-consuming primitive must be safe to call under `torch.func.vmap(torch.func.grad(...))` over the trainable params. Specifically:

- Pure tensor operations only.
- No `nn.Module` state mutation inside the function body.
- No Python control flow on tensor values (`torch.where` instead of `if`).
- No `torch.no_grad()` on module instance attributes (PEFT adapter flags etc.).
- No `.item()` on dynamic-shape tensors.

Primitives that genuinely need to break vmap-safety (PEFT adapter toggles, ref-model precompute over a dataset) are documented as "**outside vmap only**" and labelled in the module docstring. These are typically the `reference/` helpers.

### 3.5 Loss organization — registry + factory

Each loss family (DPO, KTO, SFT) ships **both**:

- A **registry dict** `LOSSES: dict[str, LossFn]` for trainer/config string dispatch (`args.loss_type="sigmoid"` → `DPO_LOSSES["sigmoid"]`). The registry is the single source of truth for what variants exist.
- **Individual function exports** so users can import `from opaque.alignment.loss.dpo import dpo_sigmoid, dpo_ipo, ...` and call them directly in functional examples.

**Open design question** (acknowledged from PR review L142): whether a "central factory" per family that returns a configured loss callable would help reduce parameter-explosion as variants accumulate (e.g., `dpo_loss(variant="sigmoid", beta=0.1, label_smoothing=0.0, ld_alpha=None, f_divergence="reverse_kl")`). For Phase γ we ship the registry + direct exports; the factory is added in Phase δ.2 if the parameter surface bloats. The reviewer's instinct that "it would have a mess of parameters" is correct — the registry is preferable until variant-grouping crystallizes.

---

## 4. Module layout (api/façade pattern)

Strictly follows AGENTS.md namespace rules 1–9. Concern dirs are singular; each loss family is its own sub-concern under `loss/` (not a sibling of `loss/`); `types.py` lives with its concern; every `__init__.py` has explicit `__all__`.

```
packages/opaque-alignment/
├── pyproject.toml
├── README.md
└── src/opaque/
    ├── api/                                          ← (PEP 420 — no __init__.py)
    │   └── alignment/                                ← IMPLEMENTATION namespace
    │       ├── __init__.py                           # __all__ = []
    │       ├── loss/
    │       │   ├── __init__.py                       # __all__ = ["dpo", "kto", "sft"]
    │       │   ├── types.py                          # LossFn, DPSpec, LossAggregateSpec
    │       │   ├── dpo/
    │       │   │   ├── __init__.py                   # __all__ = ["LOSSES", "dpo_sigmoid", ...]
    │       │   │   ├── types.py                      # DpoVariant Literal, DpoSpec instances
    │       │   │   ├── _sigmoid.py
    │       │   │   ├── _ipo.py
    │       │   │   ├── _hinge.py
    │       │   │   ├── _robust.py
    │       │   │   ├── _apo.py                       # apo_zero, apo_down
    │       │   │   ├── _exo.py                       # exo_pair
    │       │   │   ├── _nca.py                       # nca_pair
    │       │   │   ├── _bco.py                       # bco_pair
    │       │   │   ├── _sppo.py                      # sppo_hard
    │       │   │   ├── _discopop.py
    │       │   │   ├── _sft.py                       # sft term for MPO blend
    │       │   │   ├── _sigmoid_norm.py
    │       │   │   ├── _squarechipo.py               # arXiv:2505.21395
    │       │   │   ├── _f_divergence.py              # DPO-specific remap (reverse_kl, …)
    │       │   │   ├── _mpo.py                       # DPO-specific combinator
    │       │   │   ├── _wpo.py                       # DPO-specific per-pair weight fn
    │       │   │   └── _ld_dpo.py                    # DPO-specific logp split
    │       │   ├── kto/
    │       │   │   ├── __init__.py                   # __all__ = ["LOSSES", "kto_loss", "apo_zero_unpaired"]
    │       │   │   ├── types.py                      # KtoVariant Literal, KtoSpec instances
    │       │   │   ├── _kto.py                       # Tier-2 (detached KL aggregate)
    │       │   │   └── _apo_zero_unpaired.py         # Tier-1
    │       │   └── sft/
    │       │       ├── __init__.py                   # __all__ = ["LOSSES", "nll_loss", "dft_loss"]
    │       │       ├── types.py                      # SftVariant Literal, SftSpec instances
    │       │       ├── _nll.py                       # per-example divisor (DP-corrected)
    │       │       └── _dft.py                       # per-example divisor (DP-corrected)
    │       ├── logprob/
    │       │   ├── __init__.py                       # __all__ = ["sequence_logp", "selective_log_softmax", "get_batch_logps"]
    │       │   ├── _gather.py                        # selective_log_softmax
    │       │   ├── _sequence.py                      # sequence_logp + ld split helper
    │       │   └── _batch.py                         # get_batch_logps
    │       ├── collator/
    │       │   ├── __init__.py                       # __all__ = ["language_modeling_collator", "preference_collator", "unpaired_preference_collator"]
    │       │   ├── types.py                          # CollatorOutput TypedDicts
    │       │   ├── _language_modeling.py             # factory fn returning callable
    │       │   ├── _preference.py                    # (B, ...) layout per trl-trainers-plan §4.3
    │       │   └── _unpaired_preference.py
    │       ├── data/
    │       │   ├── __init__.py
    │       │   ├── _prompt.py                        # extract_prompt
    │       │   ├── _packing.py                       # pack_bfd, pack_wrapped, pack_bfd_split
    │       │   ├── _chat_template.py                 # clone_chat_template, get_training_chat_template
    │       │   └── _kto_rotation.py                  # rotate_kto_completions
    │       ├── reference/
    │       │   ├── __init__.py
    │       │   ├── types.py                          # RefSpec (LoRA / separate_model / disable_adapter)
    │       │   ├── _precompute.py                    # compute_ref_logprobs_for_dataset (cached)
    │       │   ├── _adapter.py                       # null_ref_context, with_disabled_adapter
    │       │   └── _sync.py                          # ema_update_reference (TR-DPO core)
    │       ├── metric/
    │       │   ├── __init__.py
    │       │   ├── _reward.py                        # reward_metrics
    │       │   ├── _kl.py                            # kl_estimator
    │       │   └── _token.py                         # entropy_from_logits, mean_token_accuracy
    │       └── kernel/                               ← alignment-specific fused kernels
    │           ├── __init__.py                       # __all__ = ["opaque_fused_linear_dpo_loss", "opaque_fused_linear_kto_loss"]
    │           ├── types.py
    │           ├── _fused_linear_preference.py       # Opaque_FusedLinearPreference (paired base)
    │           ├── _fused_linear_unpaired.py         # Opaque_FusedLinearUnpairedPreference (KTO base)
    │           ├── _dpo_dispatch.py                  # opaque_fused_linear_dpo_loss + per-variant fn
    │           ├── _kto_dispatch.py                  # opaque_fused_linear_kto_loss
    │           └── _utils.py                         # shared kernel helpers (duplicated from opaque-patches)
    └── alignment/                                    ← PUBLIC FAÇADE namespace
        ├── __init__.py                               # headline re-exports with __all__
        ├── loss/
        │   ├── __init__.py
        │   ├── types.py
        │   ├── dpo/__init__.py                       # re-exports from opaque.api.alignment.loss.dpo
        │   ├── kto/__init__.py
        │   └── sft/__init__.py
        ├── logprob/__init__.py
        ├── collator/__init__.py
        ├── data/__init__.py
        ├── reference/__init__.py
        ├── metric/__init__.py
        └── kernel/__init__.py
```

**Conformance to AGENTS.md namespace rules:**
- Rule 1: No `__init__.py` at `src/opaque/`, `src/opaque/api/`, or the alignment-namespace roots — they're PEP 420 namespaces.
- Rule 2: Façade `__init__.py` files contain only re-exports + `__all__`.
- Rule 6: Every `__init__.py` declares explicit `__all__`.
- Rule 7: Concern dirs are singular (`loss/`, `collator/`, `reference/`, `metric/`, `kernel/`, `data/`, `logprob/`).
- Rule 8: `types.py` lives next to each concern.
- Rule 9: Collators, kernels, and reference helpers are factory functions returning callables, not user-instantiated classes.

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
    # NO opaque-patches: alignment-specific kernels live inside this package.
    #   See §7.7 — Opaque_FusedLinearPreference is self-contained pure-PyTorch
    #   (Liger's design uses torch.func.grad_and_value, no Triton).
    # NO opaque-dpsgd, NO opaque-dpftrl, NO opaque-optimizers
    # — mechanism + optimizer are chosen by the caller (trainer or example).
]

[project.optional-dependencies]
# Recipe path for SFT users who want opaque_linear_cross_entropy_loss
# memory optimization. Not needed for the alignment primitives themselves.
patches = ["opaque-patches"]

[tool.setuptools.packages.find]
where = ["src"]
include = ["opaque.alignment*", "opaque.api.alignment*"]
namespaces = true
```

**Why no `opaque-patches` core dep?** Per the kernel research (§7.7), `Opaque_FusedLinearPreference` does not depend on `Opaque_LinearCrossEntropyLoss` — the chunked-preference kernel is self-contained pure-PyTorch + `torch.func.grad_and_value`. Users who want LCE for the SFT recipe path install `opaque-alignment[patches]`; the alignment library proper has no patches dep.

CI matrix entry: `pytest packages/opaque-alignment/tests/`.

---

## 6. Public API surface

```python
# opaque/alignment/__init__.py — headline re-exports (with __all__)
from opaque.alignment.logprob import (
    sequence_logp, selective_log_softmax, get_batch_logps,
)
from opaque.alignment.loss.dpo import DPO_LOSSES         # dict[str, LossFn]
from opaque.alignment.loss.kto import KTO_LOSSES
from opaque.alignment.loss.sft import SFT_LOSSES
from opaque.alignment.loss.types import DPSpec, LossAggregateSpec
from opaque.alignment.collator import (
    language_modeling_collator,
    preference_collator,
    unpaired_preference_collator,
)
from opaque.alignment.data import (
    extract_prompt, pack_bfd, pack_wrapped, pack_bfd_split,
    clone_chat_template, get_training_chat_template,
    rotate_kto_completions,
)
from opaque.alignment.reference import (
    compute_ref_logprobs_for_dataset, null_ref_context, ema_update_reference,
)
from opaque.alignment.metric import (
    reward_metrics, kl_estimator,
    entropy_from_logits, mean_token_accuracy,
)

__all__ = [
    "sequence_logp", "selective_log_softmax", "get_batch_logps",
    "DPO_LOSSES", "KTO_LOSSES", "SFT_LOSSES",
    "DPSpec", "LossAggregateSpec",
    "language_modeling_collator", "preference_collator", "unpaired_preference_collator",
    "extract_prompt", "pack_bfd", "pack_wrapped", "pack_bfd_split",
    "clone_chat_template", "get_training_chat_template",
    "rotate_kto_completions",
    "compute_ref_logprobs_for_dataset", "null_ref_context", "ema_update_reference",
    "reward_metrics", "kl_estimator",
    "entropy_from_logits", "mean_token_accuracy",
]
```

Stable from Phase β onward; populated incrementally as phases land.

---

## 7. Per-module spec

### 7.1 `loss/dpo/` — DPO family

DPO variants are pure scalar functions on per-pair `(chosen_logratio, rejected_logratio)`. The `beta` parameter is **DPO-specific** (the reference-deviation temperature; not common to all loss families — KTO uses its own `beta`, SimPO uses `gamma`, etc.).

Generic signature:

```python
LossFn = Callable[..., Tensor]    # signatures vary per variant

def dpo_<variant>(
    chosen_logratio: Tensor,      # scalar per example (post-ref-subtraction)
    rejected_logratio: Tensor,    # scalar per example
    *,
    beta: float,
    **variant_kwargs,             # label_smoothing, discopop_tau, etc.
) -> Tensor:                      # scalar per example
    ...
```

**Variants** (all Tier 1, formula sketches given Δ = chosen_logratio − rejected_logratio):

| Variant | Formula | Source | DPSpec.tier |
|---|---|---|---|
| `sigmoid` | `-logσ(βΔ)` | Rafailov 2023 | 1 |
| `hinge` | `relu(1 − βΔ)` | Liu 2023 | 1 |
| `ipo` | `((chosen_avg − rejected_avg) − 1/(2β))²` where avg = logp / completion_len | Azar 2024 | 1 |
| `robust` | `(−(1−ε)logσ(βΔ) + ε·logσ(−βΔ)) / (1−2ε)` | label-smoothed Rafailov | 1 |
| `apo_zero` | `(1 − σ(β·chosen_lr)) + σ(β·rejected_lr)` | APO | 1 |
| `apo_down` | `σ(β·chosen_lr) + (1 − σ(βΔ))` | APO | 1 |
| `exo_pair` | `qw·(log qw − log(1−ε)) + ql·(log ql − log ε)`, `q = σ(±β·Δ)` | EXO | 1 |
| `nca_pair` | `−logσ(β·chosen_lr) − 0.5(logσ(−β·chosen_lr) + logσ(−β·rejected_lr))` | NCA | 1 |
| `bco_pair` | `−logσ(β·chosen_lr) − logσ(−β·rejected_lr)` | BCO | 1 |
| `sppo_hard` | `(chosen_lr − 0.5/β)² + (rejected_lr + 0.5/β)²` | SPPO | 1 |
| `discopop` | logistic/exp blend at temperature `τ` | DiscoPOP | 1 |
| `sft` | CE on chosen completion (broadcast scalar) | DPO + SFT regularizer for MPO | 1 |
| `sigmoid_norm` | sigmoid loss using length-normalized log-ratios | length-normalized DPO | 1 |
| **`squarechipo`** | `0.5·(σ(βΔ) − 1)²` | arXiv:2505.21395, first optimal-rate DP-DPO | 1 |

**Rejected (Tier 3):** `aot`, `aot_pair`, `aot_unpaired` — sort across batch (Melnyk et al. arXiv:2406.05882). No published DP-safe variant. Rejection rationale captured in `loss/dpo/types.py`.

**DPO-specific helpers** (per PR review L372/L395/L407/L421 — these live **inside** `loss/dpo/`, not as siblings):

- `_f_divergence.py` — `reverse_kl` (identity), `forward_kl` (`-exp(-logratio)`), `js_divergence` (`logsigmoid(logratio)`), `alpha_divergence` (`exp((α-1)·logratio)/(α-1)` with bf16/fp16 clamp). Output remapped log-ratios feed any DPO variant.
- `_mpo.py` — `mpo_combine(losses: dict[str, Tensor], weights: dict[str, float]) -> Tensor`. Trivial weighted sum for `loss_type=list` in TRL DPO.
- `_wpo.py` — `wpo_weights(per_token_logps, logits_detached, completion_mask) -> Tensor`. Per-pair marginal-logp weighting from arXiv:2406.11827. Caller passes detached logits upstream (vmap-incompatible if autograd attached).
- `_ld_dpo.py` — `ld_dpo_split(per_token_logps, completion_mask, shared_prefix_len, alpha) -> Tensor`. LD-DPO from arXiv:2409.10524.

`loss/dpo/types.py` exports the `DpoVariant: Literal[...]` enum, `DpoSpec` instances (one per variant, embedding the `DPSpec` from §8.1), and the `DPO_LOSSES` registry mapping name → callable.

### 7.2 `loss/kto/` — KTO family

KTO is unpaired (per-example label) and the `kto` variant requires a **detached batch-mean KL** (Tier 2 — see §3.3 and §8.1).

```python
def kto_loss(
    chosen_logratio: Tensor | None,    # scalar per example (None when label=False)
    rejected_logratio: Tensor | None,  # scalar per example (None when label=True)
    label: Tensor,                     # bool scalar per example
    *,
    beta: float,
    kl: Tensor,                        # SCALAR DETACHED BATCH-MEAN (broadcast to all examples)
    desirable_weight: float = 1.0,
    undesirable_weight: float = 1.0,
) -> Tensor:                           # scalar per example
    """KTO loss. Tier 2 — kl must be .detach()-ed before passing in.

    Formula matches TRL `kto_trainer.py:892-912` and KTO paper Eq. (8):

      L_i = desirable_weight  · (1 − σ(β·(chosen_lr_i − kl))) if label_i else
            undesirable_weight · (1 − σ(β·(kl − rejected_lr_i)))

    The kl term is computed by the caller as:
        kl = (policy_KL_logps - ref_KL_logps).mean().detach().clamp(min=0)
    over the active microbatch, OUTSIDE the per-example vmap. Under DP-SGD
    the detach + O(1/n) leverage keeps per-example sensitivity at O(C). See
    §3.3 and §8.1.

    Under Poisson-batch-size ≤ 1: pass kl=0 (degenerates to apo_zero_unpaired
    for that step).
    """

def apo_zero_unpaired(
    chosen_logratio: Tensor | None,
    rejected_logratio: Tensor | None,
    label: Tensor,
    *,
    beta: float,
    desirable_weight: float = 1.0,
    undesirable_weight: float = 1.0,
) -> Tensor:
    """APO unpaired variant. Tier 1 — strict per-example, no KL term."""
```

**Variants confirmed complete at 2** (per research; the KTO paper defines exactly one loss, and `apo_zero_unpaired` comes from arXiv:2408.06266).

`loss/kto/types.py` registers `KTO_LOSSES = {"kto": kto_loss, "apo_zero_unpaired": apo_zero_unpaired}` plus `DPSpec` instances marking `kto` as Tier 2 with `cross_batch_aggregate="kl_mean"`, `aggregate_must_detach=True`, `aggregate_leverage="O(1/n)"`.

### 7.3 `loss/sft/` — SFT family

Three variants:

| Variant | Formula | DPSpec.tier | Note |
|---|---|---|---|
| `nll` | Standard CausalLM CE with `ignore_index=-100`. Per-example mean over non-ignored tokens. | 1 | OK as-is — per-example mean is per-example data. |
| `dft` | `(per_token_loss * mask).sum() / mask.sum()` **per example** (DP-corrected — TRL divides by batch `num_items_in_batch`; we divide by per-example mask sum). Arises from `-detach(softmax_prob) · logprob` per-token, as in Yang et al. arXiv:2508.05629. | 1 | DP-corrected divisor; pre-clip division (per §3.3 pre-clip rule). |
| `chunked_nll` | **Math-equivalent to `nll`.** The chunked variant restricts the `lm_head` matmul to non-ignored tokens and processes in chunks to reduce peak memory. | 1 | Aliased to `nll` in `SFT_LOSSES`; the kernel selection (chunked vs eager LCE) lives in `kernel/` — see §7.7. |

`loss/sft/types.py` registers `SFT_LOSSES = {"nll": nll_loss, "dft": dft_loss, "chunked_nll": nll_loss}` with the third as a documented alias.

### 7.4 `loss/types.py` — shared loss types

```python
@dataclass(frozen=True)
class DPSpec:
    """Per-loss DP compatibility declaration. Read by trainers + audit harness."""
    tier: Literal[1, 2, 3]
    cross_batch_aggregate: str | None = None    # "kl_mean", "softmax_partition", etc.
    aggregate_must_detach: bool = True
    aggregate_leverage: Literal["O(1)", "O(1/n)", "sort"] | None = None
    dp_safe: bool = True
    rejection_reason: str | None = None

@dataclass(frozen=True)
class LossAggregateSpec:
    """Declares a Tier-2 loss's required cross-batch aggregate.
    Trainer reads this to know whether to compute the aggregate
    pre-vmap (and whether to all-reduce across ranks)."""
    name: str                                    # "kl_mean"
    reduction: Literal["mean", "sum"] = "mean"
    detach: bool = True
    cross_rank: bool = False                     # True → trainer routes through opaque.distributed.all_reduce
```

These types are imported by every loss-family `types.py` to populate `DPSpec` instances.

### 7.5 `logprob/` — logprob helpers

```python
# selective_log_softmax: vmap-safe gather of log p_i = log_softmax(logits)[indices_i]
def selective_log_softmax(logits: Tensor, indices: Tensor) -> Tensor: ...

# sequence_logp: per-token log_softmax → gather → mask → sum (or LD-decomposed sum)
def sequence_logp(
    logits: Tensor,
    input_ids: Tensor,
    completion_mask: Tensor,
    *,
    ld_alpha: float | None = None,
    shared_prefix_len: Tensor | None = None,
) -> Tensor: ...

# get_batch_logps: KTO-style helper with shift + ignore_index handling
def get_batch_logps(
    logits: Tensor,
    labels: Tensor,
    *,
    average_log_prob: bool = False,
) -> Tensor: ...
```

Tier 1; vmap-safe; pure functions.

### 7.6 `collator/` — factory functions

Per PR review L490, collators are **factory functions returning callables**, not user-instantiated classes (AGENTS.md rule 9):

```python
def language_modeling_collator(
    pad_token_id: int,
    max_length: int,
    *,
    completion_only_loss: bool = False,
    padding_free: bool = False,
    pad_to_multiple_of: int | None = None,
) -> Callable[[list[dict]], dict[str, Tensor]]:
    """Returns a collator callable. Output schema:
        input_ids: (B, L), labels: (B, L) with -100 masking, attention_mask: (B, L),
        optional completion_mask, assistant_masks, seq_lengths.
    """

def preference_collator(
    pad_token_id: int,
    max_length: int,
    *,
    pad_to_multiple_of: int | None = None,
) -> Callable[[list[dict]], dict[str, Tensor]]:
    """DPO collator. Output schema:
        chosen_input_ids: (B, L), chosen_attention_mask, chosen_completion_mask,
        rejected_input_ids: (B, L), rejected_attention_mask, rejected_completion_mask,
        optional ref_chosen_logps: (B,), ref_rejected_logps: (B,).
    """

def unpaired_preference_collator(
    pad_token_id: int,
    max_length: int,
    *,
    calculate_KL: bool = True,
) -> Callable[[list[dict]], dict[str, Tensor]]:
    """KTO collator. Output schema:
        completion_input_ids, completion_attention_mask, completion_labels,
        optional KL_completion_input_ids/_attention_mask/_labels (when calculate_KL),
        label: list[bool], optional reference_logps, reference_KL_logps.
    """
```

`collator/types.py` exports TypedDicts for the output schemas so trainers and tests have static-typing hooks. Internal class-based implementations are allowed inside `_language_modeling.py` etc. as private helpers — but the user-visible API is the factory function returning a callable, not the class.

### 7.7 `data/` — dataset transforms

| Module | Function |
|---|---|
| `_prompt.py` | `extract_prompt(example) -> dict` — same heuristic TRL uses (longest common prefix of chosen+rejected for preference; just `prompt+completion` for unpaired). |
| `_packing.py` | `pack_bfd(dataset, max_length)`, `pack_bfd_split(dataset, max_length)`, `pack_wrapped(dataset, max_length)` — port of `trl/data_utils.py:686-789`. Generates `seq_lengths` column. Requires FlexAttention or SDPA-with-block-mask downstream (Phase θ documents). |
| `_chat_template.py` | `clone_chat_template(model, tokenizer, source_template_path)` — port of `trl/chat_template_utils.py:28-119`. Mutates model embedding via `resize_token_embeddings`; documented that this MUST happen before any `make_functional` snapshot. `get_training_chat_template(tokenizer)` — inserts `{% generation %}` markers for `assistant_only_loss`. |
| `_kto_rotation.py` | `rotate_kto_completions(dataset, batch_size, seed)` — `dataset.map(_get_kl_dataset, batched=True, batch_size=...)` + column rename + `concatenate_datasets(..., axis=1)`. Runtime assertion that rotation is non-identity. |

### 7.8 `reference/` — reference-model handling

Per PR review L280, the diversity of ref-model configurations needs an explicit design table:

| Config | Detected by | `null_ref_context` behavior | Precompute path |
|---|---|---|---|
| **Separate model** (`ref_model` arg, not PEFT-derived) | `ref_model is not None and not is_peft_model(ref_model)` | No-op (no adapter to disable). | `compute_ref_logprobs_for_dataset(dataset, ref_model, collator, ...)` runs `ref_model` directly. |
| **LoRA with `ref` adapter clone** | `is_peft_model(model) and "ref" in model.peft_config` | `model.set_adapter("ref")` on enter; restore active adapter on exit. | Precompute runs `model` with the `ref` adapter activated. |
| **LoRA without separate ref** (TR-DPO seed, ad-hoc) | `is_peft_model(model) and ref_model is None` | `model.disable_adapter()` — bypasses the LoRA delta, base model serves as reference. | Precompute runs `model` with adapter disabled. |
| **Explicit reference callable** (advanced; ref is a pre-computed function over tokens) | user passes `ref_fn: Callable` | No-op. | Caller-supplied function evaluated per-batch. |

The four configs are dispatched by `null_ref_context` and `compute_ref_logprobs_for_dataset` based on the inputs. `reference/types.py` exposes the `RefSpec` discriminated union so trainers can declare which config they expect.

```python
def compute_ref_logprobs_for_dataset(
    dataset: Dataset,
    ref: PreTrainedModel | Callable,
    collator: Callable,
    output_columns: Sequence[str],
    *,
    batch_size: int | None = None,
    cache_key: tuple[Any, ...] = (),
) -> Dataset:
    """One-shot pass over dataset under torch.no_grad() + autocast.
    Gather across ranks via opaque.distributed.gather_for_metrics
    (added in trl-trainers-plan.md Phase 0.5). Cache to .npz via
    opaque.serialization keyed on (Hasher.hash(dataset._fingerprint),
    hash_module(ref), cache_key). Returns dataset with output_columns added.
    """

@contextmanager
def null_ref_context(model):
    """Outside vmap only. Dispatches per RefSpec table above."""

def ema_update_reference(
    ref_params: PyTree, policy_params: PyTree, alpha: float,
) -> PyTree:
    """Functional EMA: ref ← (1-α)·ref + α·policy. TR-DPO core."""
```

### 7.9 `metric/` — alignment metrics

```python
def reward_metrics(
    chosen_logratio: Tensor, rejected_logratio: Tensor, *, beta: float,
) -> dict[str, Tensor]:
    """Returns rewards/chosen, rewards/rejected, rewards/accuracies, rewards/margins."""

def kl_estimator(
    policy_logp: Tensor, ref_logp: Tensor, *,
    detach: bool = True, clamp_min: float = 0.0,
) -> Tensor: ...

def entropy_from_logits(logits: Tensor, mask: Tensor) -> Tensor: ...

def mean_token_accuracy(logits: Tensor, labels: Tensor, mask: Tensor) -> Tensor: ...
```

Per §3.3 telemetry rule: these are computed during training and accumulated in `_metrics["train"|"eval"]` (private internal state) before being logged. Explicit DP-aware release machinery is a roadmap item.

### 7.10 `kernel/` — alignment-specific fused kernels

**Decision (per PR review L89 + kernel research):** alignment-specific kernels live **inside** `opaque-alignment`. `opaque-patches` stays for model-general per-layer kernels (CE, LCE, SwiGLU, RoPE, RMSNorm, LoRA) + HF compat patches. The chunked-preference kernel is self-contained pure-PyTorch (Liger uses `torch.func.grad_and_value` + `torch.compile`, no Triton), so no patches dependency is needed for the kernel itself.

**Tier 1 — ship in this plan:**

```python
def opaque_fused_linear_dpo_loss(
    hidden_states: Tensor,         # (B, T, H) chosen, (B, T, H) rejected concatenated along batch dim
    lm_head_weight: Tensor,        # (V, H)
    target_ids: Tensor,            # (B, T) chosen + rejected (concat)
    completion_mask: Tensor,       # (B, T)
    ref_chosen_logp: Tensor,       # (B,)
    ref_rejected_logp: Tensor,     # (B,)
    *,
    beta: float,
    loss_type: str = "sigmoid",
    chunk_size: int = 1,
) -> Tensor:                       # (B,) per-pair loss
    """Chunked-preference kernel. Wraps Opaque_FusedLinearPreference.
    Memory: peak (chunk_size · 2T · V) instead of (B · 2T · V).
    Covers DPO + ORPO + CPO + SimPO + 10+ DPO variants via loss_type plugin.
    """

def opaque_fused_linear_kto_loss(
    hidden_states: Tensor,         # (B, T, H) completion
    lm_head_weight: Tensor,
    target_ids: Tensor,
    completion_labels: Tensor,
    label: Tensor,                 # (B,) bool
    ref_logp: Tensor,              # (B,)
    *,
    beta: float,
    kl: Tensor,                    # SCALAR (broadcast) — caller-computed detached batch-mean
    chunk_size: int = 1,
) -> Tensor:                       # (B,) per-example loss
    """Chunked-preference kernel for KTO. Wraps Opaque_FusedLinearUnpairedPreference.
    Note: chunks only the completion forward. The KL_completion forward (which
    materializes its own (B, T, V) logits) is NOT chunked by this kernel — see
    Tier 2 deferred item below.
    """
```

Two underlying autograd.Functions:
- `Opaque_FusedLinearPreference` (paired base, used by DPO/ORPO/CPO/SimPO).
- `Opaque_FusedLinearUnpairedPreference` (unpaired base, used by KTO).

Both follow the two-level `Opaque_Foo / _FooBackward` pattern with explicit `vmap` rules (per AGENTS.md kernel pattern). Three known interactions documented in the kernel docstrings:
1. `torch.compile` ↔ `vmap`: compile flag is auto-disabled under vmap; functional path still works.
2. Per-sample `dW` under vmap: handled via `compute_dc=needs_input_grad[1]` skip (LoRA frozen case) or per-sample scatter (full-FT case).
3. Nested `torch.func.grad_and_value` inside the custom vmap rule body: composes naturally because the inner call runs after the rule has flattened.

**Tier 2 — defer, document the win:**

```python
def opaque_selective_log_softmax(
    hidden_states, lm_head_weight, target_ids, *, seq_chunk, vocab_chunk,
) -> Tensor:
    """Dual-chunked selective log-softmax. Eliminates the KL_completion (B, T, V)
    materialization in KTO when online ref is used. Deferred until KTO precompute
    path is exhausted — under precompute (the default), this kernel is unused."""
```

**Tier 3 — not in this package:**
- DFT-as-kernel: `use_token_scaling=True` flag belongs on `Opaque_LinearCrossEntropyLoss` in `opaque-patches`. The DP-correct per-example DFT computes the scaling at the loss level on already-materialized per-token CE — fast enough in pure torch.
- Per-loss-variant kernels (sigmoid/hinge/exo/...): scalar math, kernel overhead would exceed compute time.

`kernel/_utils.py` duplicates ~50 lines of helpers from `opaque-patches/kernels/_utils.py` (autocast follow-along, dtype promotion). The duplication is intentional — making `opaque-patches` a hard dep just for two helpers is worse than the duplication.

---

## 8. Loss API tiers and `DPSpec`

### 8.1 Tier 1 vs Tier 2 — signatures and caller responsibilities

**Tier 1 (most losses).** Per-example pure function:

```python
LossFnTier1 = Callable[..., Tensor]  # (example_inputs..., *kwargs) -> per-example scalar
```

Caller responsibility: vmap over the batch axis. Verification: NaN-injection contract test (§11.3).

**Tier 2 (KTO `kto`, future ORPO-with-batch-norm-style aggregates).** Per-example pure function + detached aggregate:

```python
LossFnTier2 = Callable[..., Tensor]  # (example_inputs..., detached_aggregate, *kwargs) -> per-example scalar
```

Caller responsibility:
1. Compute the aggregate **outside** the vmap region (in `_prepare_inputs` or just before the vmap'd `compute_per_example_loss` call).
2. Call `.detach()` on the aggregate.
3. If `LossAggregateSpec.cross_rank=True`, all-reduce via `opaque.distributed.all_reduce` (see §9).
4. Broadcast the scalar into each per-example closure as a vmap-`None` argument.

Verification: aggregate-detach audit (§11.4) — walk autograd graph, assert no path from `detached_aggregate` back to model parameters; aggregate-swap test asserts `O(1/n)` leverage.

### 8.2 Per-loss-family divisor rules (DP-purity pre-clip)

Division **before** gradient computation is DP-safe; division **after** clipping must use only public hyperparameters. Concrete per-family rules:

| Loss family | Per-example divisor inside loss | Aggregate divisor after clipping |
|---|---|---|
| SFT `nll` | per-example `mask.sum()` (token count of THIS example) | `expected_batch_size` via `clipped_grad(normalize_by=...)` |
| SFT `dft` | per-example `mask.sum()` (DP-corrected from TRL's batch-level `num_items_in_batch`) | `expected_batch_size` |
| SFT `chunked_nll` | per-example mask (math-equivalent to `nll`) | `expected_batch_size` |
| DPO most variants | per-example `completion_len` for length-normalized variants (`ipo`, `sigmoid_norm`); none for the rest | `expected_batch_size` |
| KTO `apo_zero_unpaired` | none (sigmoid is already bounded) | `expected_batch_size` |
| KTO `kto` | none in the per-example body; the `kl` aggregate is detached so its computation flows through `mean()` over the microbatch (using the realized count is OK because the result is detached and bounded-leverage) | `expected_batch_size` |

**Forbidden:** dividing the per-example loss by a value that the trainer materializes from cross-example token counts (TRL's `num_items_in_batch` style). Caught by the DP-purity test in §11.

### 8.3 `DPSpec` declarations per variant

Every `LOSSES` registry entry has a matching `DPSpec`. Read by the trainer to decide:
- Which Tier-2 aggregates to compute pre-vmap.
- Which losses are flagged Tier 3 and must raise at init.
- Which losses require cross-rank all-reduce (DDP).

Example declarations:

```python
DPO_SPEC = {
    "sigmoid":     DPSpec(tier=1),
    "ipo":         DPSpec(tier=1),
    "squarechipo": DPSpec(tier=1),
    # ... rest are Tier 1 with no special metadata
    "aot":         DPSpec(tier=3, dp_safe=False, rejection_reason="sort-across-batch O(1) leverage"),
    "aot_pair":    DPSpec(tier=3, dp_safe=False, rejection_reason="sort-across-batch O(1) leverage"),
}

KTO_SPEC = {
    "kto": DPSpec(
        tier=2,
        cross_batch_aggregate="kl_mean",
        aggregate_must_detach=True,
        aggregate_leverage="O(1/n)",
    ),
    "apo_zero_unpaired": DPSpec(tier=1),
}
```

---

## 9. DDP-aware loss design

**New section — open design space, not in any prior Opaque package.** The PR review noted: "DDP is dependent on the batch size, and sometimes it causes problems, having losses that can manage DDP would unblock larger models with KTO. Requires special design as there are no cases of DDP aware losses in the library yet and interpolation of sync()."

### 9.1 The problem

KTO's `kl` aggregate is currently scoped to the per-rank microbatch. Under DDP with small per-rank batches, this estimator has high variance (the KTO paper assumes batch sizes ≥ 16; per-rank batches of 2–4 give noisy KL). The fix in TRL (`kto_trainer.py:884`) is `kl = self.accelerator.gather_for_metrics(kl).mean()` — an all-reduce of the local KL means. Under DP, this all-reduce is privacy-safe (the gathered means are detached and never released outside the gradient mechanism) but must be **explicit in the trainer**, not buried inside the loss closure.

### 9.2 Design

Extend `LossAggregateSpec` (§7.4):

```python
@dataclass(frozen=True)
class LossAggregateSpec:
    name: str                                    # "kl_mean"
    reduction: Literal["mean", "sum"] = "mean"
    detach: bool = True
    cross_rank: bool = False                     # NEW: True → all-reduce
```

The trainer (or functional example) reads `LossAggregateSpec.cross_rank` from the loss's `DPSpec` and, when True, wraps the aggregate computation:

```python
local_aggregate = compute_local_aggregate(...)  # already .detach()
if spec.cross_rank:
    aggregate = opaque.distributed.all_reduce(local_aggregate, op="mean")
else:
    aggregate = local_aggregate
# Then broadcast aggregate into the vmap'd loss closure
```

This belongs in the trainer (not the loss), because:
1. Cross-rank communication is a trainer concern, not a math concern.
2. Functional examples can opt in by calling `opaque.distributed.all_reduce` themselves.
3. The collective happens **before** the vmap'd gradient computation, so it does not interact with the DP mechanism's noise + sensitivity calibration.

### 9.3 DP correctness of cross-rank detached aggregates

The aggregate is `.detach()`-ed and bounded-leverage. The all-reduce is among ranks of the same training run, sharing the same model parameters and noise mechanism. The aggregate's contribution to per-rank per-example gradients still satisfies the `O(1/n_global)` leverage condition (in fact `n_global = n_rank · per_device_batch`, so leverage is *smaller* than single-rank). Per-rank sensitivity is unchanged.

### 9.4 Roadmap

- v1 (this plan): KTO `kto` uses **per-rank batch-mean KL** (no cross-rank). Documented as a degradation vs TRL for very small per-rank batches; mitigated by the `kl=0` fallback for batches ≤ 1.
- v2 (post-merge): expose `LossAggregateSpec.cross_rank` and wire the trainer + functional example to perform the all-reduce. Unblocks larger-model KTO under DDP.
- v3+ (research): generalize to other potential Tier-2 losses (e.g., contrastive-style across-rank InfoNCE).

---

## 10. Phase plan

Phases use Greek letters to keep them separable from the trainer plan's Roman numerals. They sequence loosely: α before everything; β foundation primitives unlock γ/δ/ε in parallel; ζ before any TRL trainer Phase 2/3; η optional optimization; θ + ι finish.

### 10.0 Execution model (Claude Code dynamic workflows)

Each phase below carries a **work-unit DAG** so the phase can be run as a Claude Code **dynamic workflow** (Opus 4.8+, research preview). A dynamic workflow is a JavaScript orchestration script Claude writes from the phase's DAG; a runtime executes it in the background, fanning the units out to parallel subagents. The constraints that shape the DAG ([docs](https://code.claude.com/docs/en/workflows)):

- **≤ 16 concurrent agents, ≤ 1000 total per run** — size each phase's ready-set so no more than 16 units are unblocked at once.
- **The script holds the DAG, loop, and intermediate results in variables** — deps, the iterate-until-green loop, and the fan-in live in JS, not in any agent's context.
- **The script has no filesystem/shell; agents do.** Worktree creation, the gate run, and the git merge are *agent actions*, instructed via each agent's prompt. The script only coordinates (spawns, awaits results, branches).
- **No mid-run user input.** Human sign-off therefore happens **between phases**: each phase is its own workflow run (`/workflow …` or a saved `/<name>` command), approved before it starts. The plan's phase boundaries are the natural approval gates.
- **Resumable within a session** — completed units return cached results on resume.

**One phase = one workflow run.** Invoke per phase; the cross-phase deps in the §10 totals table are the human-approval boundaries.

**Work-unit schema.** Every unit `<phase>.<n>` declares:

| Field | Meaning |
|---|---|
| **Produces** | The `_*.py` files (+ their tests) the unit owns. The agent edits **only** these, inside its own worktree. |
| **Deps** | Units whose merge must complete before this unit's agent spawns. Encoded as `await` ordering in the script. |
| **Gate** | One of the gate profiles below; the agent runs it and reports `{pass, diff, gate_output}`. |

**Shared-file ownership (conflict avoidance under 16-way parallelism).** Parallel agents never touch shared files (`__init__.py`, `__all__`, `LOSSES`/`*_SPEC` registries, `pyproject.toml`). Each phase ends with a terminal **wire-up unit** `<phase>.W` (deps = all siblings) whose agent performs the only edits to shared files: re-export public names, populate `__all__`, register variants + `DPSpec`. Every non-wire-up unit is then a pure file-add, so the per-unit worktrees merge without conflict.

**Gate profiles.** A unit's agent reports done only when its gate is fully green in its worktree:

- **`pure`** (loss functions, logprob, metric, f_div/mpo/wpo/ld helpers): unit tests vs hand-computed reference · vmap-safety contract (§11.2) · DP-purity audit — Tier-1 NaN-injection (§11.3) or Tier-2 aggregate-detach + leverage (§11.4) per the unit's `DPSpec` · TRL numeric parity at σ=0,C=∞ within 1e-3 (loss units) · ruff + format + smoke import. **CPU-only.**
- **`data`** (collator, packing, rotation, prompt, chat-template): unit tests · determinism (same input→same output; collator key-set stability) · rotation non-identity assertion (KTO) · ruff + format. **CPU-only.**
- **`reference`** (precompute, adapter, sync): unit tests · cache round-trip · `null_ref_context` dispatch per RefSpec · **GPU smoke** (1-step forward on a 2-layer model, via Cadence preset) · ruff + format.
- **`kernel`** (fused-linear-preference, unpaired): eager-vs-fused parity 1e-4 · vmap-safety composition · **GPU memory benchmark** (peak < `(B,T,V)`) · **GPU smoke train** · `torch.compile`↔vmap interaction · ruff + format.
- **`example`** (functional `train_*.py`): runs end-to-end · **GPU smoke** (2 DP-SGD steps on a 2-layer model) · **ε-budget regression** (50-step run at ε=10; snapshot final loss for drift tracking) · ruff + format.
- **`infra`** (skeleton, packaging, distributed extensions): unit/contract tests · namespace-compose smoke import · existing suite green (no regression) · ruff + format.

The GPU gates realize the **full-empirical** validation: no kernel/example/reference unit merges without a real (tiny-model) GPU run, and every example unit posts an ε-budget snapshot.

**Comprehensive validation = adversarial reviewer per unit.** The workflow's signature quality pattern is to pair each implementing agent with an **adversarial reviewer agent** that tries to *refute* the unit before it merges: hunt a DP-purity hole (construct a record-swap that moves the gradient by more than `O(C)`), a vmap break (find an op that silently drops the batch axis), or a parity drift (a config where the σ=0 loss diverges from TRL beyond 1e-3). The implementing agent must answer every refutation or fix the code; the unit merges only when the reviewer cannot break it. The script runs the implement→refute→fix loop until convergence, mirroring `/deep-research`'s cross-check-then-vote pattern.

**Worktree-per-unit isolation (agent-instructed).** Each phase runs on a phase branch `phase/alignment-<greek>` cut from the package branch. The script's per-unit agent prompt instructs the agent to:
1. `git worktree add worktrees/alignment-<greek>.<n> phase/alignment-<greek>`.
2. Implement only the unit's *Produces* files + tests in that worktree.
3. Run the unit's gate; loop with the adversarial reviewer until green.
4. Emit a patch; the script records `{pass, diff}` in a variable.

The terminal `<phase>.W` agent applies the recorded green diffs to the phase branch, makes the shared-file edits, and runs the closing gate (full namespace smoke + `test_all_exports_match`). Phase done = `<phase>.W` green.

**Orchestrator script shape (what Claude writes).** Topologically sort units by `Deps`; maintain a ready-set; `await` up to 16 unit-agents concurrently; on each returned green diff, recompute the ready-set; finish by spawning `<phase>.W`. The DAG tables below are the direct input to that script — each row is one `spawnAgent({ prompt, deps, gate })` call.

### Phase α — Package skeleton (S, ≤ 1 day)

- Create `packages/opaque-alignment/` with `pyproject.toml` (§5), `README.md`, the api/façade dir tree (§4) all containing empty `__init__.py` files.
- Each `__init__.py` declares `__all__: list[str] = []` (will populate as primitives land).
- Smoke test (`tests/test_import.py`): both `import opaque.alignment` and `import opaque.api.alignment` succeed; `dir(opaque.alignment)` equals its `__all__`.
- Contract test (`tests/contracts/test_all_exports_match.py`): for every submodule under `opaque.alignment.*`, asserts `dir(module) - {dunder}` equals `__all__`. This is the AGENTS.md rule 6 enforcement.
- Register in workspace tooling: `uv.lock`, all-packages CI matrix entry, ruff/pyright config inheritance.
- README documenting the api/façade pattern, functional-primitives philosophy, mechanism-agnostic posture, link to forthcoming `examples/train_dpo.py`.

**Work units (DAG).** Single workflow run; one terminal unit (skeleton is irreducible — the whole package tree + tooling must land atomically).

| Unit | Produces | Deps | Gate |
|---|---|---|---|
| α.1 | full dir tree + empty `__init__.py` (`__all__=[]`) + `pyproject.toml` + README + `tests/test_import.py` + `tests/contracts/test_all_exports_match.py` + workspace registration | — | `infra` |

### Phase β — Foundation primitives (M, 3–4 days)

Independent of any specific loss family. Unlocks everything downstream.

- `logprob/`: `selective_log_softmax`, `sequence_logp` (with optional `ld_alpha` decomposition stub), `get_batch_logps`.
- `collator/_language_modeling.py`: `language_modeling_collator` factory (text-only; VLM out of scope; packing deferred to Phase θ).
- `collator/_preference.py`: `preference_collator` factory with `(B, ...)` layout.
- `collator/_unpaired_preference.py`: `unpaired_preference_collator` factory.
- `data/_prompt.py`: `extract_prompt`.
- `metric/`: `reward_metrics`, `kl_estimator`, `entropy_from_logits`, `mean_token_accuracy`.
- `loss/types.py`: `DPSpec`, `LossAggregateSpec` dataclasses.

Tests:
- Unit tests for `selective_log_softmax`, `sequence_logp` vs hand-computed reference.
- Collator determinism: same input → same output (used by `_discover_batch_keys` upstream).
- Vmap-safety contract test (§11.2) for `sequence_logp`.

**Work units (DAG).** 6 parallel units + wire-up. Max concurrency 6 (well under 16). Deps = α.W.

| Unit | Produces | Deps | Gate |
|---|---|---|---|
| β.1 | `logprob/_gather.py`, `_sequence.py`, `_batch.py` + tests | α.W | `pure` |
| β.2 | `collator/_language_modeling.py` + tests | α.W | `data` |
| β.3 | `collator/_preference.py` + tests | α.W | `data` |
| β.4 | `collator/_unpaired_preference.py` + tests | α.W | `data` |
| β.5 | `data/_prompt.py` + tests | α.W | `data` |
| β.6 | `metric/_reward.py`, `_kl.py`, `_token.py` + tests; `loss/types.py` (`DPSpec`, `LossAggregateSpec`) | α.W | `pure` |
| β.W | `logprob/__init__.py`, `collator/__init__.py`, `data/__init__.py`, `metric/__init__.py`, top-level `__all__` updates | β.1–β.6 | `infra` |

### Phase γ — DPO loss family + functional example (L, 5–6 days)

- `loss/dpo/`: all 14 variants + helpers (`_f_divergence`, `_mpo`, `_wpo`, `_ld_dpo`) + `LOSSES` registry + `DPO_SPEC` declarations.
- `examples/train_dpo.py`: functional, hand-rolled DPO training loop (DP-SGD mechanism). Uses precomputed ref logps for simplicity.

Tests:
- Per-variant unit test vs TRL eager on synthetic batches at `σ=0, C=∞` → within `1e-3`.
- Vmap closure test on a 4-pair synthetic batch → finite per-pair gradients.
- DP-purity Tier-1 NaN-injection test on every variant.
- Tier-3 rejection test: instantiating with `loss_type="aot"` raises `NotImplementedError` with the documented rationale.
- MPO combinator: `loss_type=["sigmoid", "sft"]` with equal weights → matches hand combination.

**Work units (DAG).** Variants clustered to amortize the per-unit parity harness; helpers parallel to variants; example last. Max concurrency 5. Deps = β.W.

| Unit | Produces | Deps | Gate |
|---|---|---|---|
| γ.1 | `loss/dpo/_sigmoid.py`, `_hinge.py`, `_robust.py` + tests | β.W | `pure` |
| γ.2 | `loss/dpo/_apo.py`, `_exo.py`, `_nca.py`, `_bco.py`, `_sppo.py` + tests | β.W | `pure` |
| γ.3 | `loss/dpo/_ipo.py`, `_sigmoid_norm.py`, `_discopop.py`, `_sft.py`, `_squarechipo.py` + tests | β.W | `pure` |
| γ.4 | `loss/dpo/_f_divergence.py` + tests | β.W | `pure` |
| γ.5 | `loss/dpo/_mpo.py`, `_wpo.py`, `_ld_dpo.py` + tests | β.W | `pure` |
| γ.W | `loss/dpo/__init__.py` + `loss/dpo/types.py` (`DpoVariant`, `DPO_SPEC`, `DPO_LOSSES`) + Tier-3 rejection wiring; top-level `__all__` | γ.1–γ.5 | `pure` |
| γ.X | `examples/train_dpo.py` | γ.W | `example` |

### Phase δ — KTO loss + rotation + functional example (M, 3 days)

- `loss/kto/`: `kto_loss` (Tier 2 with detached `kl` parameter), `apo_zero_unpaired` (Tier 1), `LOSSES` registry, `KTO_SPEC` declarations.
- `data/_kto_rotation.py`: `rotate_kto_completions`.
- Extension to `collator/_unpaired_preference.py` to emit `KL_*` keys when present.
- `examples/train_kto.py`: functional. Demonstrates the Tier-2 caller pattern: compute `kl = (policy_KL_logps - ref_KL_logps).mean().detach()` OUTSIDE vmap, broadcast into the closure.

Tests:
- Per-variant unit test vs TRL eager.
- Rotation determinism: same seed → same rotation; non-identity assertion.
- Batch-size-0/1 Poisson edge case: `kl=0` produces sensible behavior.
- DP-purity Tier-2 aggregate-detach audit (§11.4): autograd graph from `kl` does not reach model params; swapping one example's KL contribution changes per-example loss by `O(1/n)`.

**Work units (DAG).** Max concurrency 3. Deps = β.W (and γ.W only for the `_unpaired_preference.py` KL-key extension, to avoid a collator edit-conflict with γ; sequence after β suffices since γ does not touch collators — keep dep on β.W).

| Unit | Produces | Deps | Gate |
|---|---|---|---|
| δ.1 | `loss/kto/_kto.py` (Tier 2), `_apo_zero_unpaired.py` (Tier 1) + tests | β.W | `pure` |
| δ.2 | `data/_kto_rotation.py` + tests | β.W | `data` |
| δ.3 | `collator/_unpaired_preference.py` KL-key extension + tests | β.W | `data` |
| δ.W | `loss/kto/__init__.py` + `loss/kto/types.py` (`KtoVariant`, `KTO_SPEC`, `KTO_LOSSES`); top-level `__all__` | δ.1–δ.3 | `pure` |
| δ.X | `examples/train_kto.py` (demonstrates Tier-2 caller pattern) | δ.W, ζ.W | `example` |

### Phase ε — SFT loss family + functional example (S, 2 days)

- `loss/sft/`: `nll`, `dft` (with DP-corrected per-example divisor), `chunked_nll` (aliased to `nll`, kernel selection deferred), `LOSSES` registry.
- `examples/train_sft.py`: functional, code dataset.

Tests:
- `dft` vs hand-computed reference matching TRL's formula but with per-example divisor.
- DP-purity Tier-1 NaN-injection.
- Document `chunked_nll` as math-alias with TODO for kernel selection.

**Work units (DAG).** Max concurrency 1 (small phase). Deps = β.W.

| Unit | Produces | Deps | Gate |
|---|---|---|---|
| ε.1 | `loss/sft/_nll.py`, `_dft.py` + tests | β.W | `pure` |
| ε.W | `loss/sft/__init__.py` + `loss/sft/types.py` (`SftVariant`, `SFT_LOSSES` with `chunked_nll`→`nll` alias); top-level `__all__` | ε.1 | `pure` |
| ε.X | `examples/train_sft.py` | ε.W | `example` |

### Phase ζ — Reference handling (M, 3 days)

Blocks DPO/KTO TRL trainer phases (per `trl-trainers-plan.md` §11/§12/§13) that need precomputed ref logps.

- `reference/_precompute.py`: `compute_ref_logprobs_for_dataset` with `.npz` caching via `opaque.serialization`.
- `reference/_adapter.py`: `null_ref_context` with dispatch per RefSpec table (§7.8).
- `reference/_sync.py`: `ema_update_reference` (functional EMA over a params pytree).
- `reference/types.py`: `RefSpec` discriminated union.

Tests:
- Precompute round-trip: cache miss → compute + save → cache hit on re-call.
- `null_ref_context` per dispatch row (separate model, LoRA-with-ref, LoRA-disable, callable).
- `ema_update_reference` on a small pytree; values match expected formula.

**Work units (DAG).** Max concurrency 3. Deps = β.W; also blocks on `trl-trainers-plan.md` Phase 0.5 (`opaque.distributed` extensions) for the precompute gather.

| Unit | Produces | Deps | Gate |
|---|---|---|---|
| ζ.1 | `reference/_precompute.py`, `reference/types.py` (`RefSpec`) + tests | β.W, trl-0.5 | `reference` |
| ζ.2 | `reference/_adapter.py` (`null_ref_context`) + tests | β.W | `reference` |
| ζ.3 | `reference/_sync.py` (`ema_update_reference`) + tests | β.W | `pure` |
| ζ.W | `reference/__init__.py`; top-level `__all__` | ζ.1–ζ.3 | `infra` |

### Phase η — Chunked fused-linear preference kernel (L, 4–6 days)

Self-contained inside `opaque-alignment.kernel`. No `opaque-patches` dep.

- `kernel/_fused_linear_preference.py`: `Opaque_FusedLinearPreference` (paired base for DPO/ORPO/CPO/SimPO). Two-level pattern with explicit `vmap` rule. `compute_dc=needs_input_grad[1]` skip for LoRA-frozen case.
- `kernel/_fused_linear_unpaired.py`: `Opaque_FusedLinearUnpairedPreference` (KTO base).
- `kernel/_dpo_dispatch.py`: `opaque_fused_linear_dpo_loss(loss_type="sigmoid")` plugin.
- `kernel/_kto_dispatch.py`: `opaque_fused_linear_kto_loss(kl=...)`.

Tests:
- Eager-vs-fused numeric parity within `1e-4` on synthetic batches.
- Memory benchmark: peak memory < `(B, T, V)` materialization.
- Vmap-safety: kernel composes with `vmap(grad(...))` on a 4-pair batch.
- `torch.compile` ↔ `vmap` interaction documented and tested (compile auto-disabled under vmap).

Optional optimization; not blocking for Phase ι. Tier-2 `opaque_selective_log_softmax` deferred.

**Work units (DAG).** The two base kernels parallel; dispatchers each depend on their base. Max concurrency 2. Deps = γ.W (DPO dispatch) / δ.W (KTO dispatch) for the per-variant plugins.

| Unit | Produces | Deps | Gate |
|---|---|---|---|
| η.1 | `kernel/_fused_linear_preference.py` (`Opaque_FusedLinearPreference` + `_Backward`, vmap rules) + `kernel/_utils.py` + tests | β.W | `kernel` |
| η.2 | `kernel/_fused_linear_unpaired.py` (`Opaque_FusedLinearUnpairedPreference` + `_Backward`) + tests | β.W | `kernel` |
| η.3 | `kernel/_dpo_dispatch.py` (`opaque_fused_linear_dpo_loss`) + tests | η.1, γ.W | `kernel` |
| η.4 | `kernel/_kto_dispatch.py` (`opaque_fused_linear_kto_loss`) + tests | η.2, δ.W | `kernel` |
| η.W | `kernel/__init__.py`, `kernel/types.py`; top-level `__all__` | η.1–η.4 | `infra` |

### Phase θ — Advanced data pipeline (M, 4–5 days)

- `data/_packing.py`: `pack_bfd`, `pack_wrapped`, `pack_bfd_split`.
- `data/_chat_template.py`: `clone_chat_template`, `get_training_chat_template`.

Open question (Phase θ design subsection): FlexAttention + vmap composition. If it works, FlexAttention is the documented attention backend for packed data. Otherwise fall back to SDPA with explicit 4D block-diagonal mask.

**Work units (DAG).** Max concurrency 3. Deps = β.W. θ.1 carries a research sub-step (FlexAttention↔vmap fixture) before the packing collator path is finalized.

| Unit | Produces | Deps | Gate |
|---|---|---|---|
| θ.0 | FlexAttention↔vmap composition fixture (decides backend) | β.W | `data` (+ GPU fixture) |
| θ.1 | `data/_packing.py` (`pack_bfd`, `pack_bfd_split`, `pack_wrapped`) + `seq_lengths` collator support + tests | θ.0 | `data` |
| θ.2 | `data/_chat_template.py` (`clone_chat_template`, `get_training_chat_template`) + tests | β.W | `data` |
| θ.W | `data/__init__.py` updates; top-level `__all__` | θ.1, θ.2 | `infra` |

### Phase ι — Docs and recipe scaffolding (S, 2 days)

- `docs/alignment/index.md` — package overview.
- `docs/alignment/loss.md` — per-loss reference (formula, paper, `DPSpec`).
- `docs/alignment/collator.md` — per-collator output schema reference.
- `docs/alignment/reference.md` — the four ref-model configs.
- `docs/alignment/recipes.md` — placeholder for future recipe documentation (decoupled DP-RLHF, etc.).

**Work units (DAG).** Docs are independent files → fully parallel. Max concurrency 5. Deps = all prior wire-ups (docs describe shipped API).

| Unit | Produces | Deps | Gate |
|---|---|---|---|
| ι.1 | `docs/alignment/index.md` | γ.W, δ.W, ε.W, ζ.W | `infra` (docs-build) |
| ι.2 | `docs/alignment/loss.md` | γ.W, δ.W, ε.W | `infra` |
| ι.3 | `docs/alignment/collator.md` | β.W | `infra` |
| ι.4 | `docs/alignment/reference.md` | ζ.W | `infra` |
| ι.5 | `docs/alignment/recipes.md` | — | `infra` |
| ι.W | mkdocs nav wiring; docs-build green | ι.1–ι.5 | `infra` |

### Phase totals

| Phase | Effort | Cumulative |
|---|---|---|
| α — skeleton | S | 0.5d |
| β — foundation | M | 4d |
| γ — DPO | L | 10d |
| δ — KTO | M | 13d |
| ε — SFT | S | 15d |
| ζ — reference | M | 18d |
| η — fused kernel | L | 24d (optional, can defer) |
| θ — advanced data | M | 28d |
| ι — docs | S | 30d |

**Total: ~6 working weeks** for the full package surface. The TRL trainers can start consuming primitives from end of Phase γ onward.

---

## 11. Test strategy

### 11.1 Unit tests (per loss function)

≥3 hand-computed reference cases per variant. Pure, no model.

### 11.2 Vmap-safety contract test

Meta-test walks every public symbol in `opaque.alignment.loss.*.LOSSES` and `opaque.alignment.logprob.*`, verifies the function survives `torch.func.vmap(torch.func.grad(...))` on a 4-example synthetic batch with finite gradients.

### 11.3 DP-purity NaN-injection test (Tier 1)

For every Tier-1 loss: replace one example's input with NaN; verify only that example's gradient is affected. Catches accidental cross-example divisors.

### 11.4 DP-purity aggregate-detach audit (Tier 2)

For every Tier-2 loss:
- **Autograd-graph audit:** trace the autograd graph from `loss_value` backward; assert no path from `detached_aggregate` to a leaf that flows back into model parameters (the `.detach()` is honored).
- **Aggregate-swap leverage test:** modify one example's contribution to the aggregate (without modifying the example's own data); assert per-example loss changes by `O(1/n)`, not `O(1)`. The expected leverage scaling is read from `DPSpec.aggregate_leverage`.

NaN-injection is NOT used for Tier 2 because the aggregate by construction couples all examples; the audit replaces NaN-injection.

### 11.5 TRL eager parity

Parameterized test per loss variant per supported config. Disable DP on Opaque side (`σ=0, C=∞`); compare per-batch loss to TRL within `1e-3`. For Tier-2 losses, also assert detached-aggregate value matches TRL's gathered aggregate.

### 11.6 Mechanism-agnostic integration test

Same loss closure run end-to-end under DP-SGD (`gaussian_noise` + `OpaqueEpochPoissonBatchSampler`) and DP-FTRL (`band_mf` + `b_min_sep_sampler`). Smoke test that the package contract holds under mechanism substitution.

### 11.7 Tier-3 rejection test

Instantiating a Tier-3 loss raises `NotImplementedError` with the rejection rationale string from `DPSpec.rejection_reason`. No exposure path exists in the public API (registry does not contain Tier-3 entries).

### 11.8 Cross-rank aggregate test (Phase η+ or v2)

For Tier-2 losses with `LossAggregateSpec.cross_rank=True`: distributed test (2-rank smoke) verifying the all-reduced aggregate equals the global mean and the per-example loss is bit-identical across ranks given the same seed.

### 11.9 Cache round-trip test (Phase ζ)

`compute_ref_logprobs_for_dataset` called twice on same dataset → second call is a cache hit (verified by mocking the forward pass).

---

## 12. Cross-package interactions

### 12.1 With `opaque-engine`

`opaque-alignment` consumes:
- `opaque.clipping.clipped_grad` — at the call site (trainer or example), not inside primitives.
- `opaque.functional.make_functional` — at the call site, not inside primitives.
- `opaque.distributed.gather_for_metrics`, `is_main_process`, `wait_for_everyone` — used by `compute_ref_logprobs_for_dataset` (Phase ζ) and by the Tier-2 cross-rank aggregate path (§9). These are added in `trl-trainers-plan.md` Phase 0.5; `opaque-alignment` Phase ζ blocks on that.
- `opaque.distributed.all_reduce` — used by the optional cross-rank aggregate path.
- `opaque.distributed.barrier` — used by precompute to synchronize across ranks before writing cache.

### 12.2 With `opaque-base`

- `opaque.serialization.{state_dict, from_state_dict}` — for precompute cache state.

### 12.3 With `opaque-patches` (optional, not required)

- The chunked-preference kernel does NOT depend on `opaque-patches` (per §5 + §7.10).
- The SFT recipe path can optionally use `opaque_linear_cross_entropy_loss` from `opaque-patches` for memory-efficient SFT. Installed via `opaque-alignment[patches]`.
- All other kernel usage is implicit through HF model forwards (`use_performance_kernels=True` on the trainer side routes through opaque-patches; transparent to the alignment loss layer).

### 12.4 With `opaque.transformers.trl`

Consumed per `trl-trainers-plan.md` §6.2. The trainers' single `compute_per_example_loss` override orchestrates `opaque-alignment` primitives. The dependency edge runs `opaque-transformers → opaque-alignment`; never the reverse.

### 12.5 With `opaque-dpsgd` / `opaque-dpftrl` / `opaque-optimizers`

**No dependency from `opaque-alignment`.** Mechanism + optimizer chosen at the call site.

---

## 13. Functional examples

Each example is a sibling of an existing functional script. Three deliverables.

### `examples/train_sft.py` (Phase ε)

```python
from opaque.alignment import language_modeling_collator
from opaque.alignment.loss.sft import SFT_LOSSES
from opaque.clipping import clipped_grad
from opaque.functional import make_functional
from opaque.dpsgd.noise import gaussian_noise
from opaque.dpsgd.sampling import OpaqueEpochPoissonBatchSampler
from opaque.optimizers import adamw

fmodel, trainable, frozen = make_functional(model, partition_trainable=True)
collator = language_modeling_collator(tokenizer.pad_token_id, max_length=1024)

def per_example_loss(trainable_params, input_ids, attention_mask, labels):
    merged = {**frozen, **trainable_params}
    out = fmodel(merged, input_ids=input_ids, attention_mask=attention_mask)
    return SFT_LOSSES["nll"](out.logits, labels)

grad_fn, clip_state = clipped_grad(per_example_loss, normalize_by=expected_batch_size, ...)
# ... DP-SGD glue identical to train_causal_lm.py ...
```

### `examples/train_dpo.py` (Phase γ)

```python
from opaque.alignment import preference_collator, compute_ref_logprobs_for_dataset, sequence_logp
from opaque.alignment.loss.dpo import DPO_LOSSES

dataset = preprocess_preference(raw, tokenizer, max_length=1024)
dataset = compute_ref_logprobs_for_dataset(
    dataset, ref_model, collator=preference_collator(...),
    cache_key=("dpo", model_name),
    output_columns=("ref_chosen_logps", "ref_rejected_logps"),
)

def per_example_loss(trainable_params, *batch):
    chosen_ids, chosen_mask, chosen_completion_mask = batch[:3]
    rejected_ids, rejected_mask, rejected_completion_mask = batch[3:6]
    ref_chosen_logps, ref_rejected_logps = batch[6:]
    merged = {**frozen, **trainable_params}
    chosen_out = fmodel(merged, input_ids=chosen_ids, attention_mask=chosen_mask)
    rejected_out = fmodel(merged, input_ids=rejected_ids, attention_mask=rejected_mask)
    chosen_logp = sequence_logp(chosen_out.logits, chosen_ids, chosen_completion_mask)
    rejected_logp = sequence_logp(rejected_out.logits, rejected_ids, rejected_completion_mask)
    return DPO_LOSSES["sigmoid"](
        chosen_logp - ref_chosen_logps,
        rejected_logp - ref_rejected_logps,
        beta=0.1,
    )

# Mechanism swap demo (comment): from opaque.dpftrl.noise import band_mf …
```

### `examples/train_kto.py` (Phase δ)

The Tier-2 caller pattern is illustrated explicitly:

```python
from opaque.alignment import unpaired_preference_collator, rotate_kto_completions
from opaque.alignment.loss.kto import KTO_LOSSES, KTO_SPEC

dataset = rotate_kto_completions(dataset, batch_size=8, seed=42)
dataset = compute_ref_logprobs_for_dataset(
    dataset, ref_model,
    output_columns=("reference_logps", "reference_KL_logps"),
)

def step(trainable, batch):
    # TIER 2: compute kl OUTSIDE vmap, detach, then broadcast
    with torch.no_grad():
        kl = (policy_KL_logp_batch - batch["reference_KL_logps"]).mean().detach().clamp(min=0)
    # (optional cross-rank, when LossAggregateSpec.cross_rank=True is wired in v2)
    # kl = opaque.distributed.all_reduce(kl, op="mean")

    def per_example_loss(params, completion_ids, completion_mask, label, ref_logp):
        merged = {**frozen, **params}
        out = fmodel(merged, input_ids=completion_ids, attention_mask=completion_mask)
        chosen_logp = sequence_logp(out.logits, completion_ids, completion_mask) * label.float()
        rejected_logp = sequence_logp(out.logits, completion_ids, completion_mask) * (~label).float()
        return KTO_LOSSES["kto"](
            chosen_logp - ref_logp, rejected_logp - ref_logp, label,
            beta=0.1, kl=kl,
        )

    return clipped_grad(per_example_loss, normalize_by=expected_batch_size)(trainable, ...)
```

---

## 14. Roadmap beyond the TRL trainer port

Sibling workstreams that become natural under the `opaque-alignment` + `opaque.transformers.trl` split:

| Item | Where it lives | Why opaque-alignment is the right home |
|---|---|---|
| **Reward modeling primitives** (`loss/reward/`) | new sub-concern | Pure per-example BT loss; consumed by `RewardTrainer` and by `train_reward.py` example. |
| **ORPO loss family** (`loss/orpo/`) | new sub-concern | Same pattern as DPO; odds-ratio + NLL terms. |
| **CPO loss family** (`loss/cpo/`) | new sub-concern | Same pattern. |
| **SimPO loss family** (`loss/simpo/`) | new sub-concern | Length-normalized, no ref. Distinct enough from DPO to warrant its own sub-concern. |
| **GRPO primitives** (`loss/grpo/`, `logprob/_grpo.py`) | trajectory-level | Needs `compute_grpo_advantages`, reuses fused-preference base. |
| **Decoupled DP-RLHF recipe** | new `recipes/` subnamespace | Scripted multi-stage workflow chaining `train_reward.py` (DP) → `train_ppo.py` (non-DP actor). |
| **Cross-rank Tier-2 wiring** (DDP-aware loss v2) | `loss/types.py` + trainer-side hook | Unblocks larger-model KTO under DDP. |
| **Tier-2 `opaque_selective_log_softmax` kernel** (Phase η.2) | `kernel/` | Eliminates KL-completion `(B, T, V)` materialization when online ref is used. |
| **DP-aware metric release** | `metric/_release.py` (new) | Replaces the v1 "all training metrics are private internal state" rule with an explicit ε-accounted release path. |
| **Alignment-specific eval harnesses** | new `eval/` subnamespace | `reward_bench`, `alpaca_eval`, `kl_drift` as pure functions. |
| **Recipe DSL** | `recipes/_registry.py` | `@register_recipe("sft+dpo")` for paper recipes (SquareχPO defaults, DP-AdamW + DPO). |

None of these require structural changes to the core package; they are *additions* to the Greek-letter foundation.

---

## 15. Risk register

| ID | Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|---|
| α1 | `(B, ...)` collator layout drifts from TRL's `(2B, L)` for DPO and causes user confusion when porting code | M | L | Document the layout choice in `collator/_preference.py` docstring with a clear "TRL uses `(2B, L)`; we use `(B, ...)` because per-example DP-SGD clips per pair." |
| α2 | Tier-2 audit misses a subtle non-detached path | M | H | Aggregate-detach audit walks autograd graph + leverage test; combined they're stronger than NaN-injection for Tier 2. CI runs both on every loss flagged Tier 2. |
| α3 | Chunked-preference kernel (Phase η) breaks under vmap | M | M | Phase η has explicit vmap composition test before declaring done. Eager loss path remains the default; fused is opt-in. `torch.compile` auto-disabled under vmap. |
| α4 | Reference-logp precompute cache fingerprint collisions | L | H | `Hasher.hash((dataset._fingerprint, hash_module(ref), explicit_cache_key))` with caller-provided escape hatch. |
| α5 | PEFT `null_ref_context` doesn't compose cleanly with `make_functional`-captured params | L | H | Documented "outside vmap only" + fixture test in Phase ζ. Dispatch per RefSpec (§7.8). |
| α6 | Phase ζ blocks on `trl-trainers-plan.md` Phase 0.5 (`opaque.distributed` extensions) | M | M | Coordinate via explicit dependency note. Phase 0.5 is S (1-2 days); can land in parallel with Phase β. |
| α7 | KTO under per-rank batches gives noisy `kl` estimate vs TRL's accelerator-gathered one | M | M | v1 ships per-rank; document the degradation; mitigation via `kl=0` for batches ≤ 1. v2 wires `cross_rank=True` aggregate (§9). |
| α8 | New loss variants from papers require Tier 3 (cross-batch with `O(1)` leverage) | M | L | Reject at registry level; document in `loss/<family>/types.py` `DPSpec`. Same pattern as `aot`. |
| α9 | FlexAttention + vmap composition (Phase θ) doesn't work | M | M | Phase θ fixture test before relying on it. Fallback: SDPA + 4D block mask (slower). |
| α10 | `clone_chat_template`'s `resize_token_embeddings` interaction with `make_functional` | L | M | Document: chat template clone MUST happen before `make_functional` snapshot. |
| α11 | `loss_type=list` (MPO) with mixed-Tier variants (e.g. `["sigmoid", "kto"]`) is ambiguous | L | M | Reject mixed-tier MPO at config validation; document only same-tier combinations are allowed. |
| α12 | `DPSpec.aggregate_leverage` claims drift from actual implementation as variants are added | M | M | Aggregate-swap test (§11.4) empirically verifies the claimed leverage; fails if implementation violates it. |

---

## 16. References

### Within this plan family
- `docs/development/trl-trainers-plan.md` — sibling plan for the TRL-style class trainers. See especially §4 (cross-cutting decisions), §6 (trainer responsibilities) + §8.1 (the unified `compute_per_example_loss` hook), §10 (`opaque.distributed` extensions).
- `AGENTS.md` "Namespace contract" — nine package-design rules; all are load-bearing for this plan.

### Opaque packages on `feat/dptrainer-main-integration`
- `packages/opaque-engine/src/opaque/api/engine/clipping/_clipped_grad.py` — `clipped_grad`.
- `packages/opaque-engine/src/opaque/api/engine/functional/` — `make_functional`.
- `packages/opaque-engine/src/opaque/api/engine/distributed/` — current public surface; missing functions added in `trl-trainers-plan.md` Phase 0.5.
- `packages/opaque-optimizers/src/opaque/api/optimizers/_adam.py:117,191-203,205-245` — DP-AdamW.
- `packages/opaque-patches/src/opaque/api/patches/kernels/` — per-layer fused kernels (optional dep for SFT recipe).
- `packages/opaque-base/src/opaque/api/base/serialization/` — `state_dict`, `from_state_dict` (precompute cache).

### TRL (analyzed at `/tmp/trl_src` and `/tmp/trl_v2`)
- `trl/trainer/dpo_trainer.py:152-211` — `DataCollatorForPreference` reference.
- `trl/trainer/dpo_trainer.py:1000-1084,1042-1084` — `_precompute_ref_logps`, `compute_ref_log_probs`.
- `trl/trainer/dpo_trainer.py:1224-1402` — f-divergence remap + loss-type dispatch loop.
- `trl/trainer/dpo_trainer.py:1389-1400` — WPO weighting.
- `trl/trainer/dpo_trainer.py:1075-1081,1182-1188` — LD-DPO `ld_alpha` decomposition.
- `trl/trainer/sft_trainer.py:795-809` — DFT loss (TRL formula; we reimplement DP-corrected).
- `trl/trainer/sft_trainer.py:104-339` — `chunked_nll` monkey-patch (we replace with kernel-level chunking).
- `trl/experimental/kto/kto_trainer.py:81-88,609-623,882-887` — KTO rotation + KL math + detach.
- `trl/trainer/utils.py:1056-1093` — `use_adapter` (reference for `null_ref_context`).
- `trl/data_utils.py:686-789` — packing.
- `trl/chat_template_utils.py:28-119` — `clone_chat_template`.

### Liger (analyzed at `/tmp/liger`)
- `liger_kernel/chunked_loss/fused_linear_preference.py:9-433` — paired chunked-preference base.
- `liger_kernel/chunked_loss/fused_linear_unpaired_preference.py:9-341` — unpaired (KTO) chunked-preference base.
- `liger_kernel/chunked_loss/{dpo,kto,cpo,orpo,simpo}_loss.py` — per-algorithm dispatchers (≤ 5 lines per variant).
- `liger_kernel/chunked_loss/fused_linear_ppo.py:120-158` — dual-chunked selective_log_softmax (Phase η.2 target).
- `liger_kernel/ops/fused_linear_cross_entropy.py:111-138,187-206` — DFT via `use_token_scaling` flag.

### DP-alignment papers
- **arXiv:2310.03104** — Kumar et al., NeurIPS 2023, "Differentially Private Optimization for Non-Decomposable Objective Functions". **Theoretical foundation for Tier 2** — the `O(1/n)` leverage argument for InfoNCE / contrastive losses applies to KTO's batch-mean detached KL.
- **arXiv:2505.21395** — SquareχPO (first optimal-rate DP-DPO). Loss `dpo_squarechipo`.
- **arXiv:2505.08849** — DP-AdamW (+15% over prior baselines at ε∈[2,5]). Already implemented in `opaque.optimizers.adamw`.
- **arXiv:2406.11827** — WPO weighting.
- **arXiv:2409.10524** — LD-DPO.
- **arXiv:2603.22563** — Decoupled DP-RLHF. Roadmap recipe.
- **arXiv:2402.01306** — KTO original paper. Eq. (8) prescribes stop-gradient on the KL term (`z_0`).
- **arXiv:2408.06266** — APO. Source of `apo_zero_unpaired` for KTO.
- **arXiv:2508.05629** — DFT (Dynamic Fine-Tuning) original.
- **arXiv:2406.05882** — AOT. Sorts across batch; no DP-safe variant (Tier 3 rejection).
- **arXiv:2501.19080** — DP-PolicyGradient. Out of scope.
