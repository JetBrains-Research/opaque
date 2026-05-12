# TRL Trainers on Opaque — Implementation Plan

**Status:** Planning. Slow phased implementation.
**Scope:** Add `opaque.transformers.trl.{SFTTrainer, DPOTrainer, KTOTrainer}` plus matching configs, faithful to TRL's interface, correct under DP-SGD. Includes a pre-phase that audits and ports any missing Opaque-patches kernels.
**Branch:** `claude/add-trl-trainers-plan-nB07O` (this plan); implementation phases land on `claude/add-trl-trainers-nB07O` or per-phase sub-branches.

---

## Table of contents

1. [Goals and non-goals](#1-goals-and-non-goals)
2. [Background — what we found and what it changes](#2-background--what-we-found-and-what-it-changes)
3. [Cross-cutting design decisions](#3-cross-cutting-design-decisions)
4. [Phase −1 — Kernel parity pass (`opaque-patches`)](#4-phase-1--kernel-parity-pass-opaque-patches)
5. [Phase 0 — DPTrainer foundational changes](#5-phase-0--dptrainer-foundational-changes)
6. [Phase 0.5 — `opaque.distributed` extensions](#6-phase-05--opaquedistributed-extensions)
7. [Phase 1 — SFTTrainer](#7-phase-1--sfttrainer)
8. [Phase 2 — DPOTrainer](#8-phase-2--dpotrainer)
9. [Phase 3 — KTOTrainer](#9-phase-3--ktotrainer)
10. [Phase 4 — Advanced SFT data pipeline](#10-phase-4--advanced-sft-data-pipeline)
11. [Phase 5 — Polish, examples, parity tests, docs](#11-phase-5--polish-examples-parity-tests-docs)
12. [Roadmap beyond this plan](#12-roadmap-beyond-this-plan)
13. [Risk register](#13-risk-register)
14. [DP correctness checklist (used at every loss port)](#14-dp-correctness-checklist-used-at-every-loss-port)
15. [Test strategy](#15-test-strategy)
16. [References](#16-references)
17. [Glossary](#17-glossary)

---

## 1. Goals and non-goals

### 1.1 Goals

- Three new trainers under `opaque.transformers.trl`: `SFTTrainer`, `DPOTrainer`, `KTOTrainer`, each TRL-faithful at the API level.
- Matching configs `SFTConfig`, `DPOConfig`, `KTOConfig`, each extending `DPTrainingArguments` with TRL-parity fields.
- Every TRL loss variant that is mathematically compatible with per-example DP-SGD lands in the appropriate phase. Variants that structurally violate per-example DP (e.g. `aot`, `aot_pair` which sort across the batch) are deliberately rejected with a documented reason.
- Every TRL advanced feature that is mathematically compatible lands in some phase by the end of the plan. Features deliberately skipped (vision-language models) are documented.
- Strong DP correctness story: per-example loss closures audited against the [DP correctness checklist](#14-dp-correctness-checklist-used-at-every-loss-port). No cross-example data-dependent quantities inside the per-example loss.
- TRL numeric parity at `σ=0, C=∞` to within `1e-3` on representative batches.

### 1.2 Non-goals (this plan)

- Vision / multimodal (VLM) trainers and collators — out of scope.
- DeepSpeed, FSDP, Accelerate-driven multi-device — out of architectural reach for per-example DP-SGD.
- Liger as a runtime dependency — Opaque reimplements equivalent kernels in `opaque-patches` (see [Phase −1](#4-phase-1--kernel-parity-pass-opaque-patches)).
- TRL DPO/KTO loss variants whose math sorts or aggregates across batch (`aot`, `aot_pair`): rejected at init.
- Mid-training mutation of the model's PEFT adapter set, embedding size, or tokenizer vocabulary (must happen before trainer construction).
- DP-PPO / DP policy-gradient (paper arXiv:2501.19080) — trajectory-level DP is a fundamentally different design.

### 1.3 Final-phase parity bar

By the end of Phase 5:

- Every TRL `SFTConfig`, `DPOConfig`, `KTOConfig` field has either (a) a matching Opaque field with documented semantics or (b) a deliberate-not-supported note with rationale (DP-incompatibility, VLM-only, etc.).
- Every TRL loss type either ships or is rejected with rationale.
- Every TRL behavioral feature either ships or is rejected with rationale.
- A maintained parity table in this document lists current status of every TRL surface element.

---

## 2. Background — what we found and what it changes

### 2.1 Architectural facts about Opaque DPTrainer

These are properties of the existing code that constrain the plan:

- **Functional training path.** `_dp_trainer.py:1274` does `make_functional(model, partition_trainable=True)` → `(fmodel, trainable_params, frozen_params)`. The training-time forward is `fmodel(merged_params, **inputs)`. The bound `nn.Module` is used only at eval (when `self._ctx is None`).
- **Per-example loss via `clipped_grad`.** `_dp_trainer.py:3037` defines `_build_per_example_loss(fmodel, frozen_params, batch_keys) → (per_example_loss_fn, batch_argnums)`. The closure runs inside `vmap(grad(...))` via `clipped_grad` (`opaque/clipping/_clipped_grad.py:74`).
- **Normalization by `expected_batch_size`, not realized batch size.** `_dp_trainer.py:3689` calls `clipped_grad(..., normalize_by=expected_batch_size, ...)`. The realized Poisson batch size is private; only the public expectation enters the gradient scale.
- **Compute_loss runs only at eval today.** Training uses `_build_per_example_loss`; `compute_loss` (`_dp_trainer.py:2054`) is invoked only from `prediction_step` / eval loop. This is the largest single gap from HF/TRL parity. Phase 0 fixes this.
- **Poisson batch sampler.** `OpaqueEpochPoissonBatchSampler` / `OpaqueEpochTruncatedPoissonBatchSampler` (`_dp_trainer.py:3273-3293`). Each example sampled independently with probability `ctx.sample_rate`. Batches have variable realized size including 0 and 1.
- **No `self.accelerator`.** DPTrainer talks to `opaque.distributed` directly. Phase 0.5 extends `opaque.distributed` with the small set of primitives TRL needs; we do not introduce an Accelerator shim.
- **CPU activation offloading already implemented.** `_config.py:305` `cpu_offload_activations: bool` wraps the per-step gradient in `torch.autograd.graph.save_on_cpu(pin_memory=True)` (`_dp_trainer.py:1268-1270`). Needs a rename to `activation_offloading` for TRL parity.
- **`opaque.optimizers` includes DP-AdamW.** `opaque.optimizers.adamw(noise_bias_correction=True)` at `_adam.py:235-251` implements the bias-corrected v-moment subtraction `v̂ = max(v/bc2 − φ̂, floor)`. This is mathematically the DP-AdamW recipe from arXiv:2505.08849. No new optimizer needed.

### 2.2 Architectural facts about `opaque-patches`

- **Comprehensive fused-kernel coverage** with two-level `Opaque_Foo / _FooBackward` vmap-safe autograd.Functions: `opaque_cross_entropy_loss`, `opaque_linear_cross_entropy_loss`, `opaque_swiglu`, `opaque_geglu_exact`, `opaque_geglu_approx`, `opaque_rope`, `opaque_rope_qk`, `opaque_slow_rope`, `opaque_rms_norm`, `opaque_fused_add_rms_norm`, `opaque_lora_w`, `opaque_lora_qkv`, `opaque_lora_mlp`. Pure-PyTorch fallbacks for non-Triton environments.
- **Wide architectural coverage** in `patches/transformers/models/`: Llama, Gemma/Gemma2/Gemma3, Qwen2/Qwen3, Mistral/Ministral, Phi3, Olmo2/Olmo3, GLM4, Granite, Cohere/Cohere2, SmolLM3, Exaone4.
- **PEFT-aware kernel components** in `patches/peft/components/`: fused LoRA MLP, fused LoRA Linear, fused LoRA QKV.
- **The DP-SGD-aware optimization** in `linear_cross_entropy.py:920-921`: skip `dC` (weight gradient) when `weight.requires_grad=False`. Saves ~⅓ of backward time for LoRA-frozen-base training.

What this means for the plan: **the SFT/DPO/KTO trainers do not need to write any kernel code.** Setting `use_liger_kernel=True` in their configs routes through `_liger.apply_liger_kernel_via_opaque_patches` and the existing kernel coverage is inherited transparently — including inside vmap.

### 2.3 Architectural facts about TRL

These constrain how we mirror TRL behavior:

- **`compute_loss` is the canonical override surface.** TRL subclasses override `compute_loss` for both training and eval. Opaque must support this contract.
- **Reference model has multiple legitimate paths.** Explicit `ref_model=`, PEFT `disable_adapter` / `use_adapter`, precomputed logps cached as dataset columns, or auto-load a second copy from the same model path. All four are supported by TRL and we mirror.
- **PEFT adapter toggling is module-flag based, not weight-swap based.** `model.disable_adapter()` flips `_disable_adapters` on each LoRA module; base weights are untouched. `torch.func.functional_call` doesn't capture or override these flags, so toggling them around a non-vmap'd forward is safe.
- **KTO rotation is dataset-prep-time, not sampler-time.** `_get_kl_dataset` (`kto_trainer.py:83-90`) rotates `completion_ids` once, then `concatenate_datasets(axis=1)` produces a dataset where every row has both `completion_ids` and `KL_completion_ids`. Any sampler that preserves row identity (including Poisson) works; the only constraint is realized batch size ≥ 2.
- **TRL collator emits `(2B, L)` for single-forward efficiency.** No semantic dependency — `(B, 2, L)` with two forwards is mathematically identical. The DP unit is the (chosen, rejected) pair, so `(B, 2, L)` is the natural fit.
- **`num_items_in_batch` is a data-dependent batch-level quantity.** TRL uses it as a divisor in DFT loss and grad-accum scaling. Under DP, using it would leak information about the batch composition into the per-example gradient. We replace it with public quantities (per-example mask sums, `expected_batch_size`, or `args.max_length`) in every loss formula we port.

### 2.4 DP-alignment papers — relevance summary

| Paper | arXiv | Action |
|---|---|---|
| SquareχPO — first optimal-rate DP-DPO | 2505.21395 | Ships as `loss_type="squarechipo"` in Phase 2. |
| DP-AdamW for alignment (+15% over prior baselines) | 2505.08849 | Already implemented as `adamw(noise_bias_correction=True)`. Default optimizer for DPO/KTO configs. |
| Decoupled DP-RLHF (DP reward + non-DP PPO actor) | 2603.22563 | Roadmap example/recipe; no new trainer needed beyond a future DP reward trainer. |
| DP-PolicyGradient (trajectory-level DP for RL) | 2501.19080 | Out of scope. |
| Sample complexity of DP-PO (theory) | 2510.21060 | Informational only; cite in docs. |

### 2.5 Liger / Unsloth — what's relevant

- **No FastDPOTrainer or FastKTOTrainer exists in Unsloth.** `unsloth/models/dpo.py` is an empty stub. Unsloth's "fast DPO" is just TRL+Liger underneath. Skip Unsloth integration entirely.
- **Liger's chunked preference loss** (`LigerFusedLinearPreferenceBase` in `liger_kernel/chunked_loss/`) is the one pattern not yet in `opaque-patches`. Pure PyTorch, uses nested `torch.func.grad_and_value` inside `autograd.Function`. The pattern (chunk hidden states along seq dim, accumulate gradients in fp32 buffers, never materialize full `(B, T, V)` logits) is the right shape to port natively into `opaque-patches` as `Opaque_FusedLinearPreference / _FusedLinearPreferenceBackward`. Scheduled in Phase −1.
- **Liger's RoPE / RMSNorm / SwiGLU / GeGLU / fused linear CE / LoRA kernels** are all already in `opaque-patches`. Phase −1 audit confirms no gaps in the per-layer surface for currently-supported architectures.

---

## 3. Cross-cutting design decisions

These hold across all phases. Each is binding once recorded here.

### 3.1 Package and module layout

- Public façade: `opaque.transformers.trl.{SFTTrainer, SFTConfig, DPOTrainer, DPOConfig, KTOTrainer, KTOConfig}`.
- Implementation: `opaque.api.transformers.trl/` mirroring the existing `opaque.api.transformers.trainer/` split.
- Internal modules grouped by concern (data collators, losses, logprob helpers, base mixin). Exact file layout is not load-bearing — the broader trainer rearrangement work will align it.
- Tests under `packages/opaque-transformers/tests/opaque_transformers/trl/`.
- Examples under `examples/`.

### 3.2 No `self.accelerator`

DPTrainer and the TRL subclasses never expose an Accelerator-shaped object. Where TRL code reads `self.accelerator.X`, the port replaces it with the equivalent `opaque.distributed.X` module function. New primitives needed by TRL are added to `opaque.distributed` in [Phase 0.5](#6-phase-05--opaquedistributed-extensions).

### 3.3 Reference-model strategy — all four TRL paths supported

For DPO and KTO. The path is selected from constructor arguments + config, in this priority:

| Path | Trigger | Implementation |
|---|---|---|
| **Precompute** | `args.precompute_ref_log_probs=True` | One-shot pass via `compute_ref_logprobs_for_dataset(...)` in subclass `__init__`, *before* `super().__init__()`. Adds `ref_chosen_logps` / `ref_rejected_logps` (DPO) or `reference_logps` / `reference_KL_logps` (KTO) as dataset columns. Collator picks them up. Vmap closure consumes them as constant tensor args. Cached on disk via `Hasher.hash((dataset._fingerprint, hash_module(ref_or_model), extra))`. |
| **Explicit `ref_model=`** | User passes `ref_model` and `precompute=False` | Stored on `self.ref_model`. `disable_dropout_in_model(self.ref_model)` if `args.disable_dropout`. Per-batch ref forward under `torch.no_grad()` happens **outside vmap**, in `_prepare_inputs`. The resulting per-example logp tensors are injected back into the `inputs` dict and consumed in the vmap'd `compute_loss`. |
| **PEFT `disable_adapter`** | `ref_model is None`, `is_peft_model(model)`, `precompute=False` | Per-batch ref forward under `torch.no_grad()`, wrapped in `with self.null_ref_context(): ...` which calls `model.disable_adapter()` (or `set_adapter("ref")` if a "ref" adapter has been added). Module flag toggle is invisible to `functional_call`'s param-dict capture; this is safe because the toggle happens entirely outside vmap. |
| **Auto-load** | All three of the above false; not PEFT | Load a second copy of the model via `AutoModelForCausalLM.from_pretrained(get_config_model_id(self.model.config))`, treat as explicit ref_model. |

DP semantics: in every path, ref logps are deterministic functions of `(example_data, public_ref_model_params)`. They contribute to the per-example loss as constants (no gradient flows through them). Per-example clipping covers the policy logp computation. The cache produced by the precompute path is itself private (it embeds training data) and must not be released outside the training process; standard care applies.

### 3.4 KTO under Poisson — TRL-faithful

TRL's KTO rotation is implemented in dataset-prep, not sampler. We reproduce that exactly:

1. `_prepare_dataset` runs `dataset.map(_get_kl_dataset, batched=True, batch_size=args.per_device_train_batch_size, ...)` to produce a sister dataset with rotated `completion_ids`.
2. `concatenate_datasets([base, kl], axis=1)` produces a single dataset where every row has both `completion_ids` and `KL_completion_ids`.
3. The Poisson sampler picks rows. The rotation is already baked in; the sampler doesn't need to know.
4. Constraint: realized Poisson batch size ≥ 2. TRL enforces this at config level (`per_device_train_batch_size > 1`). Under Poisson, individual realized batches may be smaller. **Behavior:** if realized batch size ≤ 1, the KL term contribution that step is set to 0 (mathematically equivalent to running `apo_zero_unpaired` on that step). Privacy unchanged because no cross-example dependency is introduced.

We do not invent a self-pairing fallback (it is mathematically degenerate — the KL term collapses into the implied reward; see `kto_trainer.py:357`).

### 3.5 DPO collator layout — `(B, 2, L)`

The collator emits separate `chosen_input_ids` and `rejected_input_ids` tensors of shape `(B, L)` each. The vmap'd per-example loss closure does two `fmodel(...)` calls per example, one for chosen and one for rejected. Vmap auto-batches over `i`. No semantic loss vs TRL's `(2B, L)` layout — DP-SGD per-pair clipping captures both forwards.

### 3.6 `num_items_in_batch` is private under DP

Any TRL loss formula that divides by `num_items_in_batch` (or any cross-example sum of `(labels != -100)`) is rewritten before porting:

| TRL formula | DP-correct rewrite |
|---|---|
| `(per_token_loss * mask).sum() / num_items_in_batch` (DFT) | `(per_token_loss * mask).sum() / mask.sum()` evaluated per-example inside the vmap closure |
| HF Trainer's grad-accum loss scaling | Not applicable — Opaque reinterprets `gradient_accumulation_steps` as a sample-rate scaler |
| Aggregations over `outputs.aux_loss` for MoE | Per-example reduction (each example contributes its own aux loss; clipped gradient handles the batch sum) |

The general rule: **the per-example loss closure must depend only on the example's own data, not on any aggregate of the batch.** The [DP correctness checklist](#14-dp-correctness-checklist-used-at-every-loss-port) enforces this at every loss port.

### 3.7 Loss variants we reject

Loss variants whose math depends on cross-example computation are rejected at config validation with a `ValueError`:

| Variant | Reason |
|---|---|
| `aot` | Sorts logratios across batch examples — breaks per-example DP. |
| `aot_pair` | Same. |
| Any future TRL loss that uses `torch.sort`, `topk`, or batch-aggregate divisors over the per-example values | Same. |

Each rejection emits a docstring pointing at the closest DP-safe alternative (typically `apo_zero` or `apo_down` for AOT-family losses).

### 3.8 Kernel layer — no new code in trainers

The TRL trainers set `use_liger_kernel=True` in their default configs (subject to user opt-out). The kernel acceleration routes through `_liger.apply_liger_kernel_via_opaque_patches` and inherits the full `opaque-patches` surface transparently. No kernel work in any trainer phase. Kernel gaps (vs current Liger) are handled in [Phase −1](#4-phase-1--kernel-parity-pass-opaque-patches).

### 3.9 Activation offloading

Rename `cpu_offload_activations` → `activation_offloading` to match TRL convention. Old name kept as a deprecated alias for one release. No new behavior.

### 3.10 PEFT-as-reference handling

Supported (see [§3.3 path 3](#33-reference-model-strategy--all-four-trl-paths-supported)). Constraint: the ref forward must happen outside vmap. Adapter toggle inside vmap is undefined under `torch.func.functional_call` because module instance flags are not part of the param dict.

### 3.11 DP-AdamW

Already implemented in `opaque.optimizers.adamw(noise_bias_correction=True)` per `_adam.py:235-251`. The TRL trainer configs default `optim="adamw"` and `optim_args="noise_bias_correction=True"`. No new optimizer.

### 3.12 Vocab / tokenizer mismatch between policy and reference

HF and TRL silently trust the user. Opaque is stricter:

- At `__init__`, if `ref_model is not None`, assert `policy.config.vocab_size == ref_model.config.vocab_size`. Raise `ValueError` on mismatch.
- For precomputed logps loaded from cache, assert the cache fingerprint includes `policy.config.vocab_size`. Mismatched cache → re-precompute.
- Tokenizer divergence is not directly verified (no canonical tokenizer hash), but the vocab-size check catches the common failure mode.

### 3.13 Dataset preprocessing — pre-init, not post-init

TRL writes `self.train_dataset = preprocessed` after `super().__init__()`. We instead preprocess before super:

```python
def __init__(self, model, ref_model, args, train_dataset, ...):
    self.args = args
    self.processing_class = processing_class
    self.ref_model = ref_model
    train_dataset = self._prepare_dataset(train_dataset, ...)
    if args.precompute_ref_log_probs:
        train_dataset = self._precompute_ref_logps(train_dataset)
    super().__init__(model=model, args=args, train_dataset=train_dataset, ...)
```

This avoids needing mutable `train_dataset` / `eval_dataset` setters on DPTrainer.

### 3.14 `compute_loss` is the override surface

Both training and eval go through `compute_loss(model, inputs, return_outputs, num_items_in_batch)`. At training, `model` is `fmodel` (a `Callable` accepting merged params dict) and the closure runs inside `vmap(grad(...))`; at eval, `model` is the bound `nn.Module` and the closure runs in full Python latitude. Subclasses detect via `self._ctx is not None` (training) vs `is None` (eval).

This is the load-bearing Phase 0 change ([§5.2](#52-route-training-time-loss-through-self-compute_loss)).

### 3.15 Examples use code datasets

Per project context (JetBrains/Opaque): SFT/DPO/KTO examples use code-domain datasets. Candidates:

- SFT: `JetBrains/KExercises`, `bigcode/the-stack-smol`, `HuggingFaceH4/CodeAlpaca_20K`.
- DPO: a code-leaning preference dataset (TBD at example-PR time).
- KTO: a code-labeled binary preference dataset (TBD at example-PR time).

---

## 4. Phase −1 — Kernel parity pass (`opaque-patches`)

**Workstream:** `opaque-patches`. **Independent of the trainer phases; can run in parallel.**

**Goal:** Audit current `opaque-patches` against current Liger and Unsloth heads, identify and port any missing fused kernels, kwargs, or fused-loss variants that benefit alignment training.

### 4.1 Audit deliverables

A maintained comparison table:

| Component | Opaque-patches | Liger (HEAD) | Unsloth (HEAD) | Status / action |
|---|---|---|---|---|
| RoPE (Q-only, QK-fused, slow) | ✅ | ✅ | ✅ | confirm parity on new arches |
| RMSNorm, fused-add-RMSNorm | ✅ | ✅ | ✅ | confirm parity |
| SwiGLU, GeGLU (exact + approx) | ✅ | ✅ (SwiGLU; GeGLU?) | ✅ | confirm |
| LayerNorm (vs RMSNorm) | ? | ✅ | ✅ | port if missing |
| Cross-entropy (fused) | ✅ | ✅ | ✅ | confirm kwargs parity: `softcap`, `logit_scaling`, `label_smoothing`, `reduction`, `z-loss` |
| Linear cross-entropy (fused, chunked) | ✅ | ✅ | ✅ | confirm kwargs; verify `use_token_scaling`, `return_token_accuracy` paths |
| LoRA-W, LoRA-QKV, LoRA-MLP (fused) | ✅ | ❌ | ✅ | confirm parity |
| **Chunked fused preference loss** (`LigerFusedLinearPreferenceBase`) | ❌ | ✅ | ❌ | **port natively in Phase −1.b** |
| Chunked unpaired preference loss (KTO base) | ❌ | ✅ | ❌ | port in Phase −1.b |
| Chunked PPO/GRPO base | ❌ | ✅ | (partial, `UnslothEfficientGRPO`) | optional; port if useful for future GRPO |
| Per-arch model patches (Llama, Gemma1/2/3, Qwen2/3, Mistral, Phi3, Olmo2/3, GLM4, Granite, Cohere1/2, SmolLM3, Exaone4) | ✅ | ✅ + Gemma4, Qwen3.5, GPT-OSS, Llama4 | varies | port missing arches |

### 4.2 Phase −1.a — Audit (1 day)

- Produce the table above with concrete file:line and version-pin citations.
- Identify each gap. For each: (Opaque has? / Liger has? / Unsloth has? / blocker?).
- Output: `docs/development/opaque-patches-parity.md` maintained alongside this plan.

### 4.3 Phase −1.b — Port chunked preference loss (2–3 days)

The headline gap. `LigerFusedLinearPreferenceBase` chunks `(2B, T, H)` hidden states along the seq dim into pieces; per chunk, computes `logits = chunk @ lm_head.T`, log-softmax, gathers per-token logps for chosen/rejected; calls `torch.func.grad_and_value(compute_loss, ...)` to get `grad_input + grad_weight + grad_bias`; accumulates in fp32 buffers; only the accumulated grads survive past the chunk loop.

Port natively as `opaque.patches.kernels.fused_linear_preference.Opaque_FusedLinearPreference` + `_FusedLinearPreferenceBackward`, both with explicit `vmap` rules matching the existing two-level pattern (e.g. `linear_cross_entropy.py:868,944`).

Then expose per-algorithm wrappers in the same module: `opaque_fused_linear_dpo_loss(...)`, `opaque_fused_linear_cpo_loss(...)`, `opaque_fused_linear_orpo_loss(...)`, `opaque_fused_linear_simpo_loss(...)`, `opaque_fused_linear_kto_loss(...)`.

The Opaque port additionally bakes in the [DP correctness checklist](#14-dp-correctness-checklist-used-at-every-loss-port) (no cross-example divisors). At call time from `DPOTrainer.compute_loss`, the kernel sees per-example inputs (via vmap) and computes a per-example scalar.

### 4.4 Phase −1.c — Port other gaps from the audit (variable)

For each gap surfaced by Phase −1.a, port natively to `opaque-patches/kernels/` with the same `Opaque_Foo / _FooBackward` + vmap-rule pattern. Per-arch model patches go to `opaque-patches/transformers/models/`.

### 4.5 Phase −1 acceptance

- All P-1 audit gaps either ported or documented as "not relevant / not portable / wait-and-see".
- Existing `opaque-patches` tests pass; new kernel tests added per port.
- DPOTrainer (Phase 2) can opt into `opaque_fused_linear_dpo_loss` once landed (default off until measurement; on by default once benched).

---

## 5. Phase 0 — DPTrainer foundational changes

**Workstream:** `opaque-transformers`. **Goal:** add the minimum scaffolding in DPTrainer that the TRL subclasses need, *without* polluting DPTrainer with TRL-specific concepts.

The guiding principle: **DPTrainer mirrors `transformers.Trainer`; the TRL subclasses mirror `trl.SFTTrainer/DPOTrainer/KTOTrainer`.** Anything TRL-specific (ref-model precompute, `null_ref_context`, RLHF metrics dict) lives in the TRL subnamespace.

### 5.1 Audit current DPTrainer parity vs HF Trainer

Maintain a parity table in `docs/development/dptrainer-hf-parity.md` (separate doc) with the methods, attributes, and config fields HF Trainer exposes. Items marked **present-different** or **missing** are the items Phase 0 either reconciles or documents as intentional deviation.

This is reference material, not a Phase 0 deliverable itself. The parity table is maintained as the trainer evolves.

### 5.2 Route training-time loss through `self.compute_loss`

**Most consequential change in Phase 0.** Today the per-example loss closure built by `_build_per_example_loss` (`_dp_trainer.py:3037-3175`) hard-codes `output = fmodel(merged, **kwargs); loss = output.loss`. Subclasses overriding `compute_loss` are ignored at training time.

Change: the closure delegates to `self.compute_loss(fmodel, kwargs, return_outputs=False, num_items_in_batch=None)`. Default `compute_loss` keeps the current behavior (read `output.loss` from `model(**inputs)`), preserving observable semantics.

Constraints documented on `compute_loss`:

- At training (`self._ctx is not None`), `model` is `fmodel` (a `Callable(params, **inputs)`) and the call site is inside `vmap(grad(...))`. Constraints: no `nn.Module` state mutation, no `torch.no_grad()` toggling on captured module instance attributes, no `.item()` calls on shape-dependent tensors. Inputs arrive vmap-stripped (batch dim removed); `_VMAP_BATCH_UNSQUEEZE_KEYS` re-adds them for standard kwargs.
- At eval (`self._ctx is None`), `model` is the bound `nn.Module`; full latitude.
- Subclasses MUST NOT divide by any cross-example aggregate (e.g. `num_items_in_batch`); see [DP correctness checklist](#14-dp-correctness-checklist-used-at-every-loss-port).
- Return value is a scalar loss tensor (no division by `expected_batch_size` — `clipped_grad` does that via `normalize_by`).

### 5.3 Smaller DPTrainer hooks

Land alongside §5.2:

| Hook | Purpose | Implementation |
|---|---|---|
| `compute_loss_context_manager()` | HF parity; used by `prediction_step` in subclasses | Return `torch.autocast(device_type, dtype=self._amp_dtype)` when set, else `contextlib.nullcontext()`. |
| `_default_signature_columns()` | Subclass-override hook for fixed column lists | Default returns `None` → use HF's introspection. Subclass returns hard-coded list. `_set_signature_columns_if_needed` consults this first. |
| `_extra_forward_kwargs` allowlist | Allow subclass to inject kwargs not in `forward()` signature (e.g. Liger `skip_logits`, `use_token_scaling`) without `remove_unused_columns` stripping them | Subclass-overridable set; `_set_signature_columns_if_needed` adds to the allowlist. |
| `activation_offloading` arg (rename) | TRL parity | Add to `DPTrainingArguments`, deprecate `cpu_offload_activations` for one release. |

### 5.4 What Phase 0 does NOT touch

These belong in the TRL subnamespace, not DPTrainer:

- `compute_ref_logprobs_for_dataset` — TRL mixin (`opaque.api.transformers.trl._base`).
- `null_ref_context` — TRL mixin.
- Per-mode `_metrics: dict[Literal["train","eval"], defaultdict[str, list]]` accumulator + log-drain logic — TRL mixin.
- Any reference-model state attribute (`self.ref_model`) — DPO/KTO subclass attribute.

This matches HF's separation: HF Trainer is RLHF-agnostic; TRL adds the RLHF concepts in its own package.

### 5.5 What Phase 0 does NOT need (rejected from prior drafts)

| Item | Reason |
|---|---|
| `self.accelerator` shim | Opaque deliberately avoids Accelerator parallels. TRL idioms are rewritten to use `opaque.distributed` at port time. |
| Mutable `train_dataset` / `eval_dataset` setters | Subclasses preprocess before `super().__init__()`. |
| `model_accepts_loss_kwargs` flag | Opaque doesn't run HF's grad-accum loss scaling path. Flag is irrelevant. |
| Generic ref-logprob precompute helper | Lives in TRL mixin per §5.4. |
| Generic `null_ref_context` | Lives in TRL mixin per §5.4. |

### 5.6 Phase 0 acceptance

- Existing DPTrainer tests still pass.
- New test: a `DPTrainer` subclass overriding `compute_loss` is shown to drive training (per-example loss observably equal to the override's output).
- New test: `compute_loss_context_manager()` returns autocast when `bf16=True`, nullcontext otherwise.
- New test: `_default_signature_columns()` subclass override is consulted by `_set_signature_columns_if_needed`.
- `activation_offloading` arg works; `cpu_offload_activations` emits deprecation warning but still works.

---

## 6. Phase 0.5 — `opaque.distributed` extensions

**Workstream:** `opaque-core`. **Goal:** expose the small set of distributed primitives TRL needs, as module functions in `opaque.distributed`, not under an Accelerator shim.

### 6.1 New / confirmed primitives

| Function | Purpose | Notes |
|---|---|---|
| `opaque.distributed.gather_for_metrics(tensor)` | Gather across ranks, deduplicating distributed-sampler padding | In DP-Poisson setting there's no padding, but parity matters |
| `opaque.distributed.is_main_process()` | True on rank 0 | exists? confirm |
| `opaque.distributed.is_local_main_process()` | True on local rank 0 | confirm |
| `opaque.distributed.wait_for_everyone()` | Barrier | exists as `barrier()`; add alias for TRL parity |
| `opaque.distributed.num_processes()` | World size | confirm |
| `opaque.distributed.process_index()` | Global rank | confirm |
| `opaque.distributed.local_process_index()` | Local rank | confirm |

If `opaque.distributed` already has these under different names, add aliases. If they're class methods on `DDPState`, surface as module functions.

### 6.2 No Accelerator-shaped API

These are module-level functions, used as `from opaque.distributed import gather_for_metrics; x = gather_for_metrics(x)`. No `self.accelerator` aggregator.

### 6.3 Phase 0.5 acceptance

- All seven primitives importable from `opaque.distributed`.
- Behavior verified on single-process and DDP fixtures (existing test patterns).

---

## 7. Phase 1 — SFTTrainer

**Workstream:** `opaque-transformers/trl`. **Estimated effort:** ~5 days.

### 7.1 Module layout

```
opaque/api/transformers/trl/
├── __init__.py
├── _base.py                  # RLHFMixin: ref helpers, metrics dict, null_ref_context
├── _data/
│   ├── language_modeling.py  # DataCollatorForLanguageModeling
│   └── ...                   # preference, unpaired-preference in later phases
├── _losses/
│   ├── sft.py                # nll, dft, MoE aux
│   └── ...                   # dpo, kto in later phases
├── _logprob.py               # selective_log_softmax, get_batch_logps
├── _sft_trainer.py
└── _sft_config.py

opaque/transformers/trl/__init__.py  # public re-exports
```

Exact file layout will be adjusted during the broader trainer rearrangement; module names are not load-bearing.

### 7.2 `SFTConfig`

Extends `DPTrainingArguments`:

| Field | Default | Phase | Notes |
|---|---|---|---|
| `dataset_text_field` | `"text"` | 1 | |
| `dataset_kwargs` | `None` | 1 | supports `{"skip_prepare_dataset": bool}` |
| `dataset_num_proc` | `None` | 1 | |
| `max_length` | `1024` | 1 | |
| `truncation_mode` | `"keep_start"` | 1 | `"keep_end"` deprecated per TRL |
| `shuffle_dataset` | `False` | 1 | DP path ignores; Poisson sampler. Documented. |
| `eos_token` | `None` | 1 | optional override |
| `loss_type` | `"nll"` | 1 (nll, dft); 4 (chunked_nll) | `"nll"` \| `"dft"` \| `"chunked_nll"` |
| `completion_only_loss` | `None` | 1 | auto-detect from sample schema |
| `assistant_only_loss` | `False` | 1 (consume) / 4 (template auto-insert) | requires `{% generation %}` template at consume time |
| `chat_template_path` | `None` | 4 | clone_chat_template + embedding resize |
| `formatting_func` | `None` | 1 | optional callable transforming rows |
| `model_init_kwargs` | `None` | 1 | from_pretrained kwargs |
| `activation_offloading` | `False` | 1 | uses Phase 0 rename |
| `packing` | `False` | 4 | requires FlexAttention / FA2 / SDPA-masked |
| `packing_strategy` | `"bfd"` | 4 | `bfd` \| `bfd_split` \| `wrapped` |
| `padding_free` | `False` | 4 | requires FA2 / FlexAttention |
| `pad_to_multiple_of` | `None` | 1 | |
| `eval_packing` | `None` | 4 | defaults to `packing` |
| `pad_token` | `None` | — | deprecated per TRL |

Defaults differ from `DPTrainingArguments`:

- `learning_rate = 2e-5` (TRL SFT default).
- `optim = "adamw"`, `optim_args = "noise_bias_correction=True"` (DP-AdamW default).
- `use_liger_kernel = True` (transparently routes to `opaque-patches`).
- `bf16 = True` if available.

### 7.3 `SFTTrainer`

```python
class SFTTrainer(DPTrainer, RLHFMixin):
    def __init__(
        self,
        model: str | PreTrainedModel | PeftModel,
        args: SFTConfig | TrainingArguments | None = None,
        data_collator: DataCollator | None = None,
        train_dataset: Dataset | IterableDataset | None = None,
        eval_dataset: Dataset | IterableDataset | dict | None = None,
        processing_class: PreTrainedTokenizerBase | ProcessorMixin | None = None,
        compute_loss_func: Callable | None = None,
        compute_metrics: Callable | None = None,
        callbacks: list[TrainerCallback] | None = None,
        optimizers: tuple[Optimizer | None, LambdaLR | None] = (None, None),
        optimizer_cls_and_kwargs: tuple[type[Optimizer], dict[str, Any]] | None = None,
        preprocess_logits_for_metrics: Callable | None = None,
        peft_config: PeftConfig | None = None,
        formatting_func: Callable[[dict], str] | None = None,
    ):
        args = self._resolve_args(args)
        # IterableDataset guard documented but no dispatch_batches toggle needed (no Accelerate)
        model = self._load_model(model, args.model_init_kwargs)
        processing_class = self._load_processing_class(processing_class, model)
        if peft_config is not None:
            model = self._wrap_peft(model, peft_config, args)
        train_dataset = self._prepare_dataset(train_dataset, processing_class, args, "training", formatting_func)
        eval_dataset = self._prepare_eval_dataset(eval_dataset, processing_class, args, formatting_func)
        data_collator = data_collator or self._default_collator(processing_class, args)
        super().__init__(
            model=model, args=args, data_collator=data_collator,
            train_dataset=train_dataset, eval_dataset=eval_dataset,
            processing_class=processing_class,
            compute_loss_func=compute_loss_func, compute_metrics=compute_metrics,
            callbacks=callbacks, optimizers=optimizers,
            optimizer_cls_and_kwargs=optimizer_cls_and_kwargs,
            preprocess_logits_for_metrics=preprocess_logits_for_metrics,
        )
        self._init_metrics()
        if hasattr(self.model, "add_model_tags"):
            self.model.add_model_tags(["opaque-sft"])

    def _default_signature_columns(self):
        return ["input_ids", "labels", "attention_mask",
                "completion_mask", "assistant_masks", "position_ids", "seq_lengths"]

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        # DP-safe: never divide by num_items_in_batch.
        if self.args.loss_type == "nll":
            return self._nll_loss(model, inputs, return_outputs)
        elif self.args.loss_type == "dft":
            return self._dft_loss(model, inputs, return_outputs)
        elif self.args.loss_type == "chunked_nll":
            return self._chunked_nll_loss(model, inputs, return_outputs)   # Phase 4
        else:
            raise ValueError(f"Unknown loss_type: {self.args.loss_type}")

    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        inputs["_prediction_loss_only"] = prediction_loss_only
        return super().prediction_step(model, inputs, prediction_loss_only, ignore_keys=ignore_keys)
```

### 7.4 SFT loss functions

Each is a pure function, no module state, vmap-safe. Located in `_losses/sft.py`.

- `nll_loss`: `model(**inputs).loss` from HF's `LOSS_MAPPING` (cross-entropy with `ignore_index=-100`). HF returns per-example mean over non-ignored tokens — DP-safe.
- `dft_loss`: per-example detached entropy-weighted CE. **DP rewrite of TRL's formula:**
    ```
    logprobs = selective_log_softmax(shift_logits, shift_labels)
    per_token_loss = -logprobs.exp().detach() * logprobs * mask  # mask = (shift_labels != -100)
    return per_token_loss.sum() / mask.sum()  # per-example self-normalization
    ```
    Replaces TRL's `/ num_items_in_batch` divisor.
- MoE aux loss: when `model.config.output_router_logits` is set, add `coef * outputs.aux_loss` per-example. Outputs the per-example aux loss; per-example clipping handles the batch sum.

### 7.5 Data collator — `DataCollatorForLanguageModeling`

Ported from TRL. Output keys: `input_ids`, `labels`, `attention_mask`, optional `completion_mask` (when `completion_only_loss`), optional `assistant_masks` (when `assistant_only_loss`).

Labels padded with `-100`. Padding positions and (if applicable) non-completion / non-assistant positions all forced to `-100`.

Deterministic key set across batches — verified by Phase 0 `_discover_batch_keys` check.

### 7.6 Phase 1 features

| TRL feature | Phase | Implementation note |
|---|---|---|
| `loss_type="nll"` | 1 | HF default CE via model.loss |
| `loss_type="dft"` | 1 | DP-rewritten formula (per-example self-normalization) |
| `loss_type="chunked_nll"` | 4 | Routes through `opaque_linear_cross_entropy_loss` (already in patches); no monkey-patch needed |
| `completion_only_loss` | 1 | Label masking with `-100` |
| `assistant_only_loss` (consume) | 1 | Reads `assistant_masks` from collator; masks labels |
| `assistant_only_loss` (auto-template-insert) | 4 | `get_training_chat_template` injects `{% generation %}` markers |
| MoE aux loss | 1 | Per-example aux loss |
| `formatting_func` | 1 | Trivial dataset.map wrapper |
| `chat_template_path` (clone + resize embeddings) | 4 | `clone_chat_template`; resize embeddings before make_functional |
| `packing` (BFD, BFD-split, wrapped) | 4 | Requires FlexAttention or SDPA-masked |
| `padding_free` | 4 | Requires FA2 / FlexAttention |
| VLM | — | Out of scope |
| `activation_offloading` | 1 | Reuses Phase 0 rename |

### 7.7 Phase 1 tests

- Construct with str model, with `PreTrainedModel`, with PEFT.
- `loss_type="nll"`: 2-step training run on tiny Qwen2 + dummy text.
- `loss_type="dft"`: same.
- `completion_only_loss=True`: dataset with `prompt` / `completion` columns; assert per-example loss masks prompt tokens.
- TRL numeric parity: load TRL `SFTTrainer` with `learning_rate=0, weight_decay=0`; load ours with `noise_multiplier=0, clipping_norm=∞`; assert per-batch loss matches within `1e-3`.
- DP-purity test: replace example `i`'s data with NaN; assert only example `i`'s gradient is NaN.

---

## 8. Phase 2 — DPOTrainer

**Workstream:** `opaque-transformers/trl`. **Estimated effort:** ~7 days. **Heaviest phase.**

### 8.1 `DPOConfig`

Extends `DPTrainingArguments`. Full TRL parity:

| Field | Default | Phase | Notes |
|---|---|---|---|
| `beta` | `0.1` | 2 | |
| `loss_type` | `["sigmoid"]` | 2 | list to support MPO |
| `loss_weights` | `None` | 2 | equal weights if None |
| `label_smoothing` | `0.0` | 2 | Robust-DPO / EXO knob |
| `ld_alpha` | `None` | 2 | LD-DPO tail discount |
| `f_divergence_type` | `"reverse_kl"` | 2 | `forward_kl`/`js_divergence`/`alpha_divergence` |
| `f_alpha_divergence_coef` | `0.5` | 2 | |
| `use_weighting` | `False` | 2 | WPO |
| `discopop_tau` | `0.05` | 2 | DiscoPOP |
| `precompute_ref_log_probs` | `False` | 2 | |
| `precompute_ref_batch_size` | `None` | 2 | defaults to per_device batch size |
| `disable_dropout` | `True` | 2 | applied to policy + ref |
| `sync_ref_model` | `False` | 2 | TR-DPO |
| `ref_model_mixup_alpha` | `0.6` | 2 | |
| `ref_model_sync_steps` | `512` | 2 | |
| `max_length` | `1024` | 2 | |
| `max_prompt_length` | `512` | 2 | |
| `truncation_mode` | `"keep_start"` | 2 | |
| `padding_free` | `False` | — | currently disabled in TRL; defer |
| `pad_to_multiple_of` | `None` | 2 | |
| `dataset_num_proc` | `None` | 2 | |

Defaults differ from `DPTrainingArguments`:

- `learning_rate = 1e-6` (TRL DPO default).
- `optim = "adamw"`, `optim_args = "noise_bias_correction=True"`.
- `use_liger_kernel = True`.
- `bf16 = True` if available.

### 8.2 `DPOTrainer.__init__`

```python
class DPOTrainer(DPTrainer, RLHFMixin):
    def __init__(
        self,
        model: str | PreTrainedModel | PeftModel,
        ref_model: PreTrainedModel | None = None,
        args: DPOConfig | None = None,
        data_collator: DataCollator | None = None,
        train_dataset: Dataset | IterableDataset | None = None,
        eval_dataset: ... = None,
        processing_class: PreTrainedTokenizerBase | None = None,
        compute_metrics: Callable | None = None,
        callbacks: list[TrainerCallback] | None = None,
        optimizers: tuple[Optimizer | None, LambdaLR | None] = (None, None),
        peft_config: PeftConfig | None = None,
    ):
        args = self._resolve_args(args)
        model = self._load_model(model, args.model_init_kwargs)
        processing_class = self._load_processing_class(processing_class, model)
        if peft_config is not None:
            model = self._wrap_peft(model, peft_config, args)
            # If PEFT already and ref_model is None and no precompute, clone "ref" adapter
            if ref_model is None and not args.precompute_ref_log_probs:
                self._setup_ref_adapter(model, peft_config)
        # Vocab-size guard
        if ref_model is not None:
            self._assert_vocab_compatible(model, ref_model)
        # Disable dropout on both
        if args.disable_dropout:
            disable_dropout_in_model(model)
            if ref_model is not None:
                disable_dropout_in_model(ref_model)
        # Reference model resolution
        if ref_model is None and not is_peft_model(model) and not args.precompute_ref_log_probs:
            ref_model = self._auto_load_ref_model(model)
        self.ref_model = ref_model
        # Tokenize prompt+chosen+rejected into prompt_ids, chosen_ids, rejected_ids
        train_dataset = self._prepare_dataset(train_dataset, processing_class, args, "training")
        eval_dataset = self._prepare_eval_dataset(eval_dataset, processing_class, args)
        # Precompute reference logps if requested
        if args.precompute_ref_log_probs:
            train_dataset = self._precompute_ref_logps(train_dataset, "training")
            if eval_dataset is not None:
                eval_dataset = self._precompute_ref_logps(eval_dataset, "eval")
        data_collator = data_collator or DataCollatorForPreference(...)
        super().__init__(
            model=model, args=args, data_collator=data_collator,
            train_dataset=train_dataset, eval_dataset=eval_dataset,
            processing_class=processing_class,
            compute_metrics=compute_metrics, callbacks=callbacks, optimizers=optimizers,
        )
        # Place ref_model on device (no Accelerator.prepare; just .to(device).eval())
        if self.ref_model is not None:
            self.ref_model = self.ref_model.to(self._device).eval()
            for p in self.ref_model.parameters():
                p.requires_grad_(False)
        # TR-DPO callback wiring
        if args.sync_ref_model:
            self._assert_tr_dpo_compatible(args, peft_config)
            self.add_callback(SyncRefModelCallback(ref_model=self.ref_model, args=args))
        self._init_metrics()
        if hasattr(self.model, "add_model_tags"):
            self.model.add_model_tags(["opaque-dpo"])
```

### 8.3 Reference logp paths in `_prepare_inputs` and `compute_loss`

If logps were precomputed, the collator emits `ref_chosen_logps` / `ref_rejected_logps` as tensor columns and they arrive in `inputs`. Done.

If live (explicit ref_model or PEFT-disable), `_prepare_inputs` runs the ref forward outside vmap:

```python
def _prepare_inputs(self, inputs):
    inputs = super()._prepare_inputs(inputs)
    if self._ctx is not None and self._needs_live_ref(inputs):
        with torch.no_grad(), self.null_ref_context():
            ref_chosen, ref_rejected = self._eager_ref_logps(inputs)
        inputs["ref_chosen_logps"] = ref_chosen
        inputs["ref_rejected_logps"] = ref_rejected
    return inputs
```

`null_ref_context` toggles to ref adapter (or disables adapter) when PEFT, no-op otherwise. The ref forward uses `self.ref_model` (explicit case) or `self.model` (PEFT-disable case).

### 8.4 `DPOTrainer.compute_loss`

Vmap-safe, no cross-example divisors:

```python
def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
    # Policy forwards — two per pair, inside vmap closure at train time
    chosen_out = model(input_ids=inputs["chosen_input_ids"],
                       attention_mask=inputs["chosen_attention_mask"])
    rejected_out = model(input_ids=inputs["rejected_input_ids"],
                         attention_mask=inputs["rejected_attention_mask"])
    chosen_logps = self._seq_logps(chosen_out.logits,
                                    inputs["chosen_input_ids"],
                                    inputs["chosen_completion_mask"],
                                    ld_alpha=self.args.ld_alpha)
    rejected_logps = self._seq_logps(rejected_out.logits,
                                      inputs["rejected_input_ids"],
                                      inputs["rejected_completion_mask"],
                                      ld_alpha=self.args.ld_alpha)
    # Reference logps from inputs (precomputed or set by _prepare_inputs)
    ref_chosen = inputs["ref_chosen_logps"]
    ref_rejected = inputs["ref_rejected_logps"]

    chosen_logratios = chosen_logps - ref_chosen
    rejected_logratios = rejected_logps - ref_rejected

    # f-divergence remap
    chosen_scores, rejected_scores = self._f_divergence_remap(
        chosen_logratios, rejected_logratios)
    delta = chosen_scores - rejected_scores

    # Loss-type dispatch (MPO: list + weights, per-example losses)
    loss = sum(
        w * _DPO_LOSSES[lt](
            delta, chosen_logratios, rejected_logratios,
            beta=self.args.beta,
            label_smoothing=self.args.label_smoothing,
            discopop_tau=self.args.discopop_tau,
        )
        for lt, w in zip(self.args.loss_type, self.args.loss_weights)
    )

    # WPO weighting (per-example)
    if self.args.use_weighting:
        loss = loss * self._wpo_weights_per_example(chosen_out, rejected_out, inputs)

    # Per-example metrics into self._metrics
    self._log_dpo_metrics(chosen_logratios, rejected_logratios, chosen_out, rejected_out)

    if return_outputs:
        return loss, {"chosen_out": chosen_out, "rejected_out": rejected_out}
    return loss
```

Each `_DPO_LOSSES[lt]` returns a per-example scalar (no batch aggregation inside).

### 8.5 DPO loss variants (Phase 2)

| `loss_type` | Math (per example) | Phase | Notes |
|---|---|---|---|
| `sigmoid` | `−logσ(βΔ)` | 2 | standard DPO |
| `hinge` | `relu(1 − βΔ)` | 2 | |
| `ipo` | `((chosen_avg − rejected_avg) − 1/(2β))²`, avg = logp / completion_len | 2 | per-example length normalization |
| `robust` | `(−(1−ε)logσ(βΔ) + ε·logσ(−βΔ))/(1−2ε)`, ε = `label_smoothing` | 2 | |
| `exo_pair` | `qw·(log qw − log(1−ε)) + ql·(log ql − log ε)` | 2 | EXO |
| `nca_pair` | `−logσ(β·chosen) − 0.5·(logσ(−β·chosen) + logσ(−β·rejected))` | 2 | NCA |
| `bco_pair` | `−logσ(β·chosen) − logσ(−β·rejected)` | 2 | BCO |
| `sppo_hard` | `(chosen − 0.5/β)² + (rejected + 0.5/β)²` | 2 | SPPO |
| `apo_zero` | `(1 − σ(β·chosen)) + σ(β·rejected)` | 2 | APO |
| `apo_down` | `σ(β·chosen) + (1 − σ(βΔ))` | 2 | APO |
| `discopop` | logistic/exp blend at temperature `τ` | 2 | DiscoPOP |
| `sft` | CE on chosen completion tokens | 2 | for MPO blend |
| **`squarechipo`** | `0.5 · (σ(βΔ) − 1)²` | 2 | arXiv:2505.21395; first optimal-rate DP-DPO |
| `sigmoid_norm` | sigmoid using length-normalized scores | 2 | |
| ~~`aot`~~ | rejected | — | Sorts across batch; breaks DP |
| ~~`aot_pair`~~ | rejected | — | Same |

### 8.6 f-divergence variants (Phase 2)

5 LOC each:

- `reverse_kl` (default): identity.
- `forward_kl`: `score = −exp(−logratio)`.
- `js_divergence`: `score = logsigmoid(logratio)`.
- `alpha_divergence`: `score = exp((α−1)·logratio) / (α−1)` with bf16/fp16 clamp.

### 8.7 MPO, WPO, LD-DPO

- **MPO** (`loss_type=list`, `loss_weights=list`): summation loop in `compute_loss`. Validate `len(loss_weights) == len(loss_type)` in `DPOConfig.__post_init__`.
- **WPO** (`use_weighting=True`): per-example weight computed under `torch.no_grad()` from chosen/rejected logits and completion mask. Multiplies per-example loss. Refuses combination with `aot`/`aot_pair` (moot since those are rejected entirely).
- **LD-DPO** (`ld_alpha`): logp decomposition into shared prefix vs tail. Applied to both policy and reference logps. ~15 LOC.

### 8.8 TR-DPO (SyncRefModelCallback)

`TrainerCallback` firing `on_step_end` at `state.global_step % args.ref_model_sync_steps == 0`. EMA update: `ref.param ← (1 − α) · ref.param + α · policy.param`. Iterates `zip(ref_model.parameters(), self.model.parameters())`.

Compatibility constraints (TRL precedent):

- Incompatible with PEFT (raise at `__init__`).
- Incompatible with `precompute_ref_log_probs=True` (raise at `__init__`).

### 8.9 `DPOTrainer.prediction_step`

```python
def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
    inputs = self._prepare_inputs(inputs)
    with torch.no_grad(), self.compute_loss_context_manager():
        if prediction_loss_only:
            loss = self.compute_loss(model, inputs, return_outputs=False)
            logits, labels = None, None
        else:
            loss, outputs = self.compute_loss(model, inputs, return_outputs=True)
            logits = outputs["chosen_out"].logits
            labels = inputs["chosen_input_ids"]
    return loss, logits, labels
```

Forces `compute_loss` at eval to get reward metrics (TRL precedent).

### 8.10 Phase 2 tests

- Construct with all four ref-model paths (precompute, explicit, PEFT-disable, auto-load).
- Each ref-model path: 2-step training run.
- `loss_type=["sigmoid"]`: TRL numeric parity.
- Each loss variant: smoke test.
- Each f-divergence variant: smoke test.
- MPO `loss_type=["sigmoid", "sft"]`, `loss_weights=[0.5, 0.5]`: smoke.
- WPO: smoke.
- LD-DPO: smoke with `ld_alpha=0.5`.
- TR-DPO: smoke; assert ref params change after `ref_model_sync_steps`.
- Precompute path: assert cache hit on re-run.
- Reward metrics (`rewards/chosen`, `rewards/rejected`, `rewards/accuracies`, `rewards/margins`, `logps/chosen`, `logps/rejected`) appear in log history.
- DP-purity test on each loss variant.
- Vocab-mismatch test: assert ValueError.

---

## 9. Phase 3 — KTOTrainer

**Workstream:** `opaque-transformers/trl`. **Estimated effort:** ~4 days.

### 9.1 `KTOConfig`

| Field | Default | Phase | Notes |
|---|---|---|---|
| `beta` | `0.1` | 3 | |
| `loss_type` | `"kto"` | 3 | `"kto"` or `"apo_zero_unpaired"` |
| `desirable_weight` | `1.0` | 3 | |
| `undesirable_weight` | `1.0` | 3 | |
| `precompute_ref_log_probs` | `False` | 3 | |
| `precompute_ref_batch_size` | `None` | 3 | |
| `disable_dropout` | `True` | 3 | |
| `max_length` | `1024` | 3 | |
| `model_init_kwargs` | `None` | 3 | |
| `dataset_num_proc` | `None` | 3 | |

Defaults differ from `DPTrainingArguments`:

- `learning_rate = 1e-6`.
- `optim = "adamw"`, `optim_args = "noise_bias_correction=True"`.
- `use_liger_kernel = True`.
- `bf16 = True` if available.

### 9.2 `KTOTrainer.__init__`

Parallel to `DPOTrainer.__init__` with deltas:

- `_prepare_dataset`:
    1. Extract prompt.
    2. If dataset has `chosen`/`rejected` columns: `unpair_preference_dataset` → split into two rows per pair, label `True`/`False`.
    3. Add EOS.
    4. Tokenize.
    5. If `calculate_KL = (loss_type != "apo_zero_unpaired")`: build KL dataset via `dataset.map(_get_kl_dataset, batched=True, batch_size=per_device_train_batch_size)`. Then `concatenate_datasets([base, kl], axis=1)` to attach `KL_completion_ids`.
- Desirability balance check: compute `(num_desirable, num_undesirable)`; warn if `(desirable_weight, undesirable_weight)` is far from the recommended ratio per the KTO paper (Eq. 8).
- Precompute `reference_logps` and (if `calculate_KL`) `reference_KL_logps` if `precompute_ref_log_probs=True`.
- Default collator = `DataCollatorForUnpairedPreference`.

### 9.3 `KTOTrainer.compute_loss`

Vmap-safe, per-example math:

```python
def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
    # Forward completion under model (gradient flows)
    completion_logits = model(input_ids=inputs["completion_input_ids"],
                              attention_mask=inputs["completion_attention_mask"]).logits
    completion_logps = get_batch_logps(completion_logits, inputs["completion_labels"])

    # KL completion under model (no_grad — KL is detached statistic for policy)
    if self.args.loss_type == "kto":
        with torch.no_grad():
            kl_logits = model(input_ids=inputs["KL_completion_input_ids"],
                              attention_mask=inputs["KL_completion_attention_mask"]).logits
        kl_logps = get_batch_logps(kl_logits, inputs["KL_completion_labels"])
        kl_per_example = (kl_logps - inputs["reference_KL_logps"]).clamp(min=0).detach()
    else:  # apo_zero_unpaired
        kl_per_example = 0.0

    logratios = completion_logps - inputs["reference_logps"]

    # Branch on label (use torch.where, not Python if — vmap-safe)
    label = inputs["label"]  # bool per example
    if self.args.loss_type == "kto":
        chosen_loss   = 1 - torch.sigmoid(self.args.beta * (logratios - kl_per_example))
        rejected_loss = 1 - torch.sigmoid(self.args.beta * (kl_per_example - logratios))
    elif self.args.loss_type == "apo_zero_unpaired":
        chosen_loss   = 1 - torch.sigmoid(self.args.beta * logratios)
        rejected_loss = torch.sigmoid(self.args.beta * logratios)

    per_example_loss = torch.where(
        label,
        self.args.desirable_weight * chosen_loss,
        self.args.undesirable_weight * rejected_loss,
    )

    self._log_kto_metrics(logratios, kl_per_example, label, completion_logits)

    return per_example_loss
```

The cross-batch mean in TRL's KL math is realized as a vmap reduction across the gradient sum, not inside the per-example closure.

### 9.4 KTO under variable Poisson batch size

- Normal path: realized batch size ≥ 2. Rotation is well-defined; loss runs.
- Edge case: realized batch size ≤ 1. The KL term contribution for that step is set to 0 (mathematically equivalent to running `apo_zero_unpaired` for that step). Privacy unchanged. Logged as a warning when frequency exceeds a threshold.

### 9.5 Data collator — `DataCollatorForUnpairedPreference`

Output keys: `completion_input_ids`, `completion_attention_mask`, `completion_labels`, optionally `KL_completion_input_ids` / `KL_completion_attention_mask` / `KL_completion_labels` (when `calculate_KL`), optional `reference_logps` and `reference_KL_logps`, plus `label` (bool).

`completion_labels` masks prompt positions with `-100`.

### 9.6 Phase 3 tests

- Construct with paired and unpaired datasets.
- `loss_type="kto"`: smoke, TRL parity at `σ=0, C=∞`.
- `loss_type="apo_zero_unpaired"`: smoke.
- Rotation correctness: `KL_completion_ids != completion_ids` after `_prepare_dataset`.
- Batch-0 and batch-1 Poisson steps: KL term falls back to 0, no crash, training continues.
- Reward + KL metrics in log history.
- DP-purity test.

---

## 10. Phase 4 — Advanced SFT data pipeline

**Workstream:** `opaque-transformers/trl`. **Estimated effort:** ~5 days.

Data-pipeline-heavy features for SFT, separated for review tractability.

### 10.1 Packing

- Port `_pack_bfd` (segment-tree best-fit, ~90 LOC) and `_pack_wrapped` (~15 LOC) from `trl/data_utils.py`.
- `packing_strategy ∈ {"bfd", "bfd_split", "wrapped"}`.
- Generates `seq_lengths` column.
- Collator generates `position_ids` with per-doc restarts.
- **Attention backend dependency:**
    | Backend | Available | Note |
    |---|---|---|
    | FlashAttention2 (`flash-attn` package) | optional dep | uses `cu_seq_lens` from `position_ids` |
    | FlexAttention (PyTorch ≥ 2.5 native) | preferred | score-mod for per-doc block-diagonal mask |
    | SDPA with explicit 4D block-diagonal mask | always | works but defeats flash kernel speed |

  Detection at runtime; auto-select the best available. Document. **Phase 4 subquestion:** verify FlexAttention composes with `torch.func.vmap` before defaulting to it.

### 10.2 Padding-free

- Flatten all sequences in a batch into one row with per-doc `position_ids` restarts.
- Same attention backend dependency as packing.
- ~30 LOC in collator.

### 10.3 Chunked NLL — via `opaque_linear_cross_entropy_loss`

The Opaque path is cleaner than TRL's monkey-patch:

- `SFTConfig.loss_type = "chunked_nll"` routes `compute_loss` to call `opaque_linear_cross_entropy_loss(hidden_states, lm_head.weight, labels, ignore_index=-100, label_smoothing=args.label_smoothing_factor)`.
- The Opaque kernel already chunks the matmul and never materializes full `(B, T, V)` logits — same memory benefit as TRL's `_chunked_cross_entropy_loss`.
- No `model.forward` monkey-patch.
- DP-SGD-aware: `dC` is skipped when `lm_head.weight.requires_grad=False` (LoRA-frozen-base case).

### 10.4 Chat template cloning with embedding resize

- Port `clone_chat_template` from `trl/chat_template_utils.py`.
- Calls `model.resize_token_embeddings(new_num_tokens=..., pad_to_multiple_of=64)` if new tokens added.
- **Order constraint:** must run before `super().__init__()` (which captures the model via `make_functional`). Document.
- If PEFT and `added_tokens` non-empty: mutate `peft_config` to include `trainable_token_indices["embed_tokens"]` and `modules_to_save=["lm_head"]` (TRL precedent).

### 10.5 Assistant-only loss with template auto-insertion

If chat template lacks `{% generation %}` markers and `assistant_only_loss=True`, call `get_training_chat_template` to inject markers. Affects the tokenizer / processing_class only; no model mutation.

### 10.6 Activation offloading hook wiring

The DPTrainingArguments field already exists ([§5.3](#53-smaller-dptrainer-hooks) rename). Phase 4 confirms the offload context composes with `functional_call` under vmap. Add a regression test.

### 10.7 Phase 4 tests

- Packing: BFD on synthetic variable-length dataset; verify pack density and per-doc position_ids.
- `padding_free`: end-to-end with FlexAttention (skip on CPU/MPS).
- `chunked_nll`: assert peak memory < `(B, T, V)` materialization on a vocab-100k fixture.
- `clone_chat_template`: assert embeddings grew and new tokens are usable.

---

## 11. Phase 5 — Polish, examples, parity tests, docs

**Workstream:** `opaque-transformers/trl`. **Estimated effort:** ~3 days.

### 11.1 Examples (code datasets)

- `examples/train_sft_trainer.py` — SFT on `JetBrains/KExercises` or `bigcode/the-stack-smol`.
- `examples/train_dpo_trainer.py` — DPO on a code preference dataset (TBD).
- `examples/train_kto_trainer.py` — KTO on a binary-labeled code dataset (TBD).

Each example mirrors the structure of `examples/train_causal_lm_trainer.py` (preset modes, logging, W&B integration optional).

### 11.2 TRL parity numerics test

For each trainer:

1. Load TRL's trainer (`SFTTrainer`/`DPOTrainer`/`KTOTrainer`) with identical model, tokenizer, dataset, optimizer state.
2. Load Opaque trainer with `privacy_noise_multiplier=0.0`, `clipping_norm=float("inf")`.
3. Run one forward+backward on the same batch.
4. Assert per-batch loss matches within `1e-3`.
5. Assert reward metrics (DPO/KTO) match within `1e-4`.

This is the strongest correctness gate available.

### 11.3 DP regression test

Each trainer: train ~50 steps on tiny model + dataset, `ε=10`, fixed seeds. Snapshot final loss values. Track for drift across PRs.

### 11.4 DP-purity test framework

Per loss type:

1. Construct a 4-example batch.
2. Replace example `i`'s data with NaN.
3. Compute per-example gradients via the vmap closure.
4. Assert only example `i`'s gradient is NaN; others are finite.

This catches any leaked cross-example dependency.

### 11.5 Docs

- `docs/trainers/sft.md` — usage, supported features, deferred features, DP knobs.
- `docs/trainers/dpo.md` — same, plus the four ref-model paths table.
- `docs/trainers/kto.md` — same, plus the KTO-under-Poisson note.
- Maintain `docs/development/trl-parity.md` — table of every TRL surface element with current Opaque status.

---

## 12. Roadmap beyond this plan

These are sibling workstreams, not subdivisions of the above:

| Item | Effort | Rationale |
|---|---|---|
| `RewardTrainer` (DP RM) | M | `AutoModelForSequenceClassification` + pairwise BT loss. Direct DPTrainer subclass. Required for the decoupled DP-RLHF recipe. |
| Decoupled DP-RLHF recipe (arXiv:2603.22563) | M | Notebook chaining `RewardTrainer` → vanilla TRL `PPOTrainer` (non-DP actor). No new trainer beyond a future DP reward trainer. |
| `ORPOTrainer`, `CPOTrainer`, `SimPOTrainer` | M each | Same pattern as DPO; different loss heads. Each can call `opaque_fused_linear_*_loss` from Phase −1.b. |
| `GRPOTrainer` | L | Trajectory-level; needs `old_logps`/`ref_logps` plumbing similar to DPO. Worth designing after DPO/KTO infrastructure lands. |
| DP-PPO (arXiv:2501.19080) | L | Trust-region-coupled noise budget; fundamentally different. Skip until customer demand. |
| Vision-language trainers | — | Out of scope. |

---

## 13. Risk register

| ID | Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|---|
| R1 | TRL/HF protocol drift | M | L | Maintain `docs/development/trl-parity.md`; implement compatible additions over time. |
| R2 | `_discover_batch_keys` ordering instability across collator runs | L | H | Phase 0 explicit determinism check: collator on 2 rows, assert key-tuple equality. |
| R3 | PEFT mid-training mutations (e.g., `merge_and_unload`) | L | M | User-action only; documented constraint. |
| R4 | `vmap(grad(...))` over `selective_log_softmax` produces wrong shapes for short sequences | M | H | Unit-test loss closure standalone vs TRL eager on 4–8 synthetic examples. |
| R5 | Reference model + policy use different tokenizers/vocab sizes | L | H | Vocab-size assertion at `__init__`; raise on mismatch. |
| R6 | KTO under Poisson — variance vs TRL | L | M | Match TRL approach (rotation at dataset-prep); fall back to KL=0 on batch ≤ 1 steps; document. |
| R7 | Liger kernels break per-example loss path | — | — | Removed — `opaque-patches` kernels are designed for this; we never depend on Liger directly. |
| R8 | DP-AdamW absent | — | — | Resolved — `adamw(noise_bias_correction=True)` IS DP-AdamW. |
| R9 | DPTrainer eval-time vs train-time loss split | — | — | Phase 0 unifies them: subclass `compute_loss` drives both. |
| R10 | Inventing shortcuts vs matching HF/TRL | M | M | Plan explicitly states HF/TRL semantics + DP adaptation per decision. |
| R11 | Phase 0 `compute_loss` redirect breaks existing DPTrainer subclasses | L | H | Keep `_build_per_example_loss` as a power-user override surface; default delegates to `compute_loss`. Existing tests guard. |
| R12 | Activation offloading composes badly with `functional_call` | L | M | Phase 4 fixture test before wiring. |
| R13 | FlexAttention doesn't compose with `torch.func.vmap` | M | M | Verify in Phase 4; fall back to SDPA-with-4D-mask if so. |
| R14 | Chunked preference loss port (Phase −1.b) is harder than estimated | M | M | Vmap-rule for the chunked autograd.Function is the main risk; budget 3 days, escalate if blocked. |
| R15 | TRL evolves rapidly; pinned version drifts | M | L | Pin TRL version in parity tests; bump deliberately. |
| R16 | `num_items_in_batch` slipped into a ported loss | M | H | DP-purity test framework (Phase 5) catches this automatically. |
| R17 | Precomputed ref-logp cache leaks via on-disk file | L | M | Cache files documented as private; never commit; honor user's `output_dir` permissions. |

---

## 14. DP correctness checklist (used at every loss port)

Every ported loss formula MUST pass this checklist before being accepted:

1. **No cross-example divisors.** The per-example loss closure may not divide by any quantity computed from data of examples other than the one being processed. Disallowed quantities include `num_items_in_batch`, `(labels != -100).sum()` over the batch, any reduction of `batch_size_dim` inside the closure.
2. **No `.item()` on data-dependent tensors.** Forces host sync and reveals batch composition.
3. **No `torch.sort` / `torch.topk` / `torch.argsort` over a batch axis.** Breaks per-example independence.
4. **No in-place mutation of tensors shared across the batch.** Inside vmap, the batch axis is hidden; in-place writes corrupt other examples.
5. **No conditional control flow keyed on data-dependent batch shapes.** Use `torch.where` instead of Python `if` on tensor comparisons.
6. **Reference model output enters as a constant.** Never as a graph leaf with `requires_grad=True`.
7. **Normalization happens via `clipped_grad`'s `normalize_by=expected_batch_size`**, not by hand inside the closure.
8. **DP-purity test passes.** Replace example `i`'s data with NaN; assert only example `i`'s per-example loss is NaN; others are finite.

Each loss type in `_losses/*.py` carries a comment block citing which items it satisfies and why.

---

## 15. Test strategy

Four tiers:

### 15.1 Unit tests (per loss function)

`tests/opaque_transformers/trl/test_dpo_losses.py`, etc. Each loss function called with hand-crafted inputs (`chosen=1.0, rejected=0.0, β=0.5`), expected scalar verified.

### 15.2 Loss-closure tests (per trainer)

`tests/opaque_transformers/trl/test_dpo_closure.py`, etc. Build `compute_loss`, call on a 4-example synthetic batch, verify per-example gradient w.r.t. trainable params is finite and non-zero.

### 15.3 Trainer-contract tests (per trainer)

`tests/opaque_transformers/trl/test_dpo_trainer_contract.py`, etc. Mirror the existing `test_trainer_contract.py` pattern:

- Construct with tiny Qwen2 (2 hidden layers) + LoRA + tiny dataset, CPU, `privacy_target_epsilon=10.0`, `privacy_noise_multiplier=1.0`.
- Run 2 training steps.
- Run `evaluate()`.
- `save_model(tmp_path)`, reload, verify state dict.

### 15.4 TRL parity tests (per trainer, GPU-required, marked slow)

Phase 5 deliverable. Side-by-side numerics with TRL at `σ=0, C=∞`.

### 15.5 DP-purity tests (per loss type)

Phase 5 deliverable. NaN-injection test framework asserts no cross-example dependency.

### 15.6 Test fixtures

- `tests/opaque_transformers/trl/conftest.py`: tiny Qwen2 config (2 layers), tiny tokenizer, tiny preference dataset (10 examples), tiny unpaired-preference dataset.
- Mark GPU-required tests with `@pytest.mark.cuda`.

---

## 16. References

### Opaque (verified at this branch)

- `packages/opaque-transformers/src/opaque/api/transformers/trainer/_dp_trainer.py:3037-3175` — `_build_per_example_loss` (Phase 0.2 redirect target)
- `packages/opaque-transformers/src/opaque/api/transformers/trainer/_dp_trainer.py:2054` — current `compute_loss` (eval-only today)
- `packages/opaque-transformers/src/opaque/api/transformers/trainer/_dp_trainer.py:3689` — `clipped_grad(..., normalize_by=expected_batch_size)`
- `packages/opaque-transformers/src/opaque/api/transformers/trainer/_config.py:305` — `cpu_offload_activations` (Phase 0 rename)
- `packages/opaque-transformers/src/opaque/api/transformers/trainer/_config.py:784` — `expected_batch_size` property
- `packages/opaque-core/src/opaque/optimizers/_adam.py:235-251` — DP-AdamW (`noise_bias_correction=True`)
- `packages/opaque-core/src/opaque/clipping/_clipped_grad.py:74` — `clipped_grad` API
- `packages/opaque-core/src/opaque/functional/__init__.py:58` — `make_functional`
- `packages/opaque-patches/src/opaque/patches/kernels/linear_cross_entropy.py:868,944` — vmap-safe fused LCE
- `packages/opaque-patches/src/opaque/patches/kernels/{swiglu,geglu,rope_embedding,rms_norm,lora}.py` — all two-level vmap-safe
- `AGENTS.md` — namespace contract, kernel pattern, patching model

### TRL (clone at /tmp/trl_src, v1.5.0.dev0 at audit time)

- `trl/trainer/sft_trainer.py` — SFT
- `trl/trainer/dpo_trainer.py` — DPO
- `trl/experimental/kto/kto_trainer.py` — KTO (moved to experimental in 1.5)
- `trl/trainer/utils.py:1056-1093` — `use_adapter`
- `trl/data_utils.py:686-789` — packing (BFD + wrapped)
- `trl/chat_template_utils.py:28-119` — `clone_chat_template`

### HF Transformers (clone at /tmp/transformers_src, 5.8.0.dev0 at audit time)

- `src/transformers/trainer.py:362,1870,1941,2883,3039` — `Trainer.__init__`, `training_step`, `compute_loss`, `prediction_step`, `_save_checkpoint`

### Liger / Unsloth (clones at /tmp/liger, /tmp/unsloth, /tmp/unsloth_zoo at audit time)

- `liger_kernel/chunked_loss/fused_linear_preference.py` — chunked preference base (Phase −1.b port target)
- `liger_kernel/chunked_loss/{dpo,kto,cpo,orpo,simpo,grpo}_loss.py` — per-algorithm losses
- `liger_kernel/transformers/monkey_patch.py:3411` — `_apply_liger_kernel`
- `unsloth/models/dpo.py` — empty stub (no FastDPOTrainer exists)

### DP-alignment papers

- arXiv:2505.21395 — SquareχPO (Phase 2 `loss_type="squarechipo"`)
- arXiv:2505.08849 — DP-AdamW (already implemented in `opaque.optimizers.adamw`)
- arXiv:2603.22563 — Decoupled DP-RLHF (roadmap recipe)
- arXiv:2501.19080 — DP-PolicyGradient (out of scope)
- arXiv:2510.21060 — Sample-complexity theory (informational)

---

## 17. Glossary

- **DP-SGD** — Differentially Private Stochastic Gradient Descent: per-example gradient clipping + Gaussian noise on the aggregated gradient.
- **Per-example loss closure** — the function passed to `clipped_grad`, run inside `vmap(grad(...))`. Must depend only on a single example's data.
- **Poisson sampling** — each example included in each batch with independent probability `q`. Realized batch size is `Binomial(N, q)`-distributed.
- **Reference model** — frozen policy used to compute log-prob baseline in DPO/KTO/etc.
- **PEFT adapter** — a LoRA / prompt-tuning / etc. attachment to a frozen base model. Toggleable via `disable_adapter` / `set_adapter`.
- **`functional_call`** — `torch.func.functional_call`; substitutes named params at forward time without mutating the module.
- **Vmap-safe** — code that produces correct results under `torch.func.vmap`: no in-place mutation, no `.item()`, no module state changes, no Python control flow on data-dependent tensor shapes.
- **MPO / WPO / LD-DPO / TR-DPO / EXO / NCA / BCO / SPPO / APO / DiscoPOP / SquareχPO** — DPO loss variants; see [Phase 2 table](#85-dpo-loss-variants-phase-2).
- **`expected_batch_size`** — public hyperparameter; the expectation of the realized Poisson batch size. Used for gradient normalization to keep the divisor data-independent.
