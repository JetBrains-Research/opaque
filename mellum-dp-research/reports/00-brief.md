# Mellum2 under DP — research brief (shared context for all agents)

## The ask (verbatim from the user)
"In the last PR we started improving Mellum support. And it kinda works but kinda. Its
original forward pass behaves differently and it is harder to represent in DP.
The goal is not to tell me that due to batch statistics we cannot do it right or with
reasonable clipping norm, but to find a solution. It doesn't matter whether it lives in
the area of mathematics of the model, mathematics of the backward pass, modified close
but easier loss, forward, tight DP accounting — anything that produces the right result."

"The right result" = DP fine-tuning of Mellum2 (MoE) inside Opaque's functional
per-example pipeline (vmap(grad) -> clip -> noise -> optimizer) that is (a) provably DP
with accounting that matches what runs, (b) faithful to the model's real training
objective/forward (including whatever batch-level parts exist), (c) usable with a
reasonable clipping norm / noise level, (d) numerically stable vs the non-DP HF path.

## Repo facts
- Monorepo at /home/user/opaque (branch claude/mellum-dp-representation-r6slaz). Python via
  `uv run python ...` (venv is synced: torch 2.14 CPU-only here? check `torch.cuda.is_available()`;
  transformers 5.16.1). No GPU in this container. Tests: `uv run pytest <path> -q`.
- Agent docs: AGENTS.md, .junie/architecture-contracts.md, .junie/differential-privacy-review.md
  (the DP review protocol — read it for any privacy claim).
- Packages: opaque-engine (clipping, noise_allocation/PerGroup, pytree, random, distributed),
  opaque-dpsgd (gaussian noise, adaptive clipping, Poisson sampling, accounting factories),
  opaque-dpftrl (MF mechanisms band-MF/BLT/BSR/BiSR/lambda-CGD, b-min-sep sampling),
  opaque-accounting (Rust PLD accountant; composition/calibration), opaque-patches
  (HF model patches: vmap-safe attention/masking, MoE kernel `opaque_moe`, chunked CE),
  opaque-transformers (DPTrainer: `compute_per_example_loss`, `augment_batch` hook run
  OUTSIDE vmap once per step, per-example `aux` telemetry, DDP sync), opaque-alignment
  (SFT/DPO per-example losses with per-example divisors), opaque-auditing.
- Mellum patch: packages/opaque-patches/src/opaque/api/patches/transformers/models/mellum.py
  (moe_kind="swiglu" -> `_make_moe_experts_forward` -> `opaque_moe` dense path;
  rms_norm_kind=None keeps upstream RMSNorm because Triton BF16 reduction order flipped
  routes; fused_add_rms_kind=None; chunked_linear_cross_entropy=2048).
- MoE kernel: packages/opaque-patches/src/opaque/api/patches/kernels/moe.py (dense
  every-token-through-every-expert `Opaque_MoE` with route weight 0 for unrouted experts,
  fp32 accumulation, custom vmap rules; grouped-GEMM fast path on CUDA/MPS).
- Upstream Mellum modeling: .venv/lib/python3.*/site-packages/transformers/models/mellum/modeling_mellum.py
  and configuration_mellum.py. HF MoE integration (grouped_mm/batched_mm experts):
  .venv/lib/python3.*/site-packages/transformers/integrations/moe.py
- Recent PRs: #978 "stabilize Mellum DP-SFT memory paths" (chunked CE), #980 "stabilize
  Mellum vmap precision" — body: "Mellum's route-sensitive MoE amplified small BF16
  differences in patched RMSNorm and masked SDPA execution into large per-example gradient
  drift. Preserve upstream Transformers RMSNorm and router-loss semantics, retain stock
  SDPA's causal fast path for fully valid vmapped batches, and align portable chunked
  cross-entropy with eager BF16 precision staging. ... reducing oracle gradient relative L2
  from 29.26% to approximately 1.3%." (The "oracle" = HF non-vmap reference gradient.)
- Existing DP presets for Mellum2 (examples/train_dpftrl.py `mellum2-kstack`,
  examples/train_dpo.py `mellum2-codesec`): LoRA r=16 on q/k/v/o ONLY (experts + router
  frozen), bf16, microbatch 8-16, batch 128-256, seq 1024, eps 3-8, band-MF / DP-DPO.
- TRL config conversion drops `router_aux_loss_coef` with a warning: "the MoE router
  load-balancing loss is a batch-level statistic with no per-example gradient to clip, so
  it cannot enter opaque's DP objective; training proceeds as if the coefficient were 0.0"
  (packages/opaque-transformers/src/opaque/api/transformers/trl/_convert.py). The fused-CE
  causal-LM forward patch falls back to the original HF forward when
  output_router_logits=True (components/cross_entropy.py ~L212).

## Mellum2-12B-A2.5B(-Base) architecture (public config.json, transformers 5.8.1)
- hidden 2304, 28 layers, ALL 28 MLPs sparse MoE: 64 experts, top-8 per token,
  moe_intermediate 896, SwiGLU (silu), norm_topk_prob=True (renormalize top-8 softmax
  probs), router = plain linear (no bias, no jitter, no capacity factor, no token dropping,
  no expert bias / loss-free balancing in the checkpoint), softmax computed in fp32.
- router_aux_loss_coef=0.001, output_router_logits=False (default), so plain HF
  `model(labels=...)` returns CE only; HF Trainer users set output_router_logits=True to
  add 0.001 * Switch-style load-balancing loss computed over the WHOLE BATCH:
  aux = E * sum_e f_e * P_e, f_e = fraction of (masked) top-k assignments to expert e over
  all tokens in the batch (all layers pooled), P_e = mean softmax prob of expert e over all
  tokens in the batch. Only P_e is differentiable (f_e is argmax-derived).
- Attention: 32 heads, 4 KV heads, head_dim 128; layer_types = 3 sliding (window 1024)
  then 1 full, repeated (7 full-attention layers at idx 3,7,...,27); YaRN rope on full
  layers (factor 16, orig 8192), default rope on sliding; max positions 131072.
- vocab 98304, untied embeddings, bf16 checkpoint. Intended use: code completion / FIM.
- Model card: https://huggingface.co/JetBrains/Mellum2-12B-A2.5B-Base (README fetchable
  via `curl -sL https://huggingface.co/JetBrains/Mellum2-12B-A2.5B-Base/raw/main/README.md`;
  the non-Base repo is gated/401).

## Upstream forward (what the DP path must be faithful to)
- MellumTopKRouter: logits = x @ W^T (bf16 linear), probs = softmax(fp32), top-k, renorm,
  cast back. Returns (router_logits, scores, indices).
- MellumExperts.forward: loop over hit experts; gather tokens; gate_up linear; silu(gate)*up;
  down linear; scale by top-k weight; index_add. Under HF `use_experts_implementation` the
  real execution is grouped_mm (bf16) or batched_mm. Opaque replaces this forward with
  `opaque_moe` (dense compute, fp32 accumulate, custom vmap rules).
- Loss: HF `loss_function` (ForCausalLM CE, token-mean over batch with num_items_in_batch)
  + optional 0.001 * aux (batch-level). Opaque's per-example loss: model `.loss` per example
  (token-mean over that example) — see DPTrainer.compute_per_example_loss and
  opaque-alignment nll_loss (per-example divisor).

## Preliminary hypotheses to TEST (not assume)
H1. The only genuinely batch-coupled quantity in Mellum's training objective is the
    load-balancing aux loss (f_e and P_e pooled over the batch). Everything else in the
    forward is per-token/per-example, so vmap is exact up to floating point.
H2. Because f_e has zero gradient, grad(aux_batch) = (E/T_total) * sum_x sum_t sum_e
    f_e(B) * grad p_e(x,t): i.e. a per-example-separable surrogate exists once the batch
    load vector f(B) is treated as a constant. A DP-released (noised) or lagged estimate
    f~ makes the surrogate a legitimate per-example loss with a small, accountable extra
    privacy cost; a per-example aux (f_e(x) over the example's own tokens) is a
    materially different regularizer (pushes within-document balance) — quantify.
H3. Route flips (top-k discontinuity) under bf16/vmap explain the residual DP-vs-oracle
    gradient drift; pinning routing decisions (computed once per example, e.g. in fp32 or
    from the frozen base model) removes the discontinuity from the per-example gradient
    and may tighten the per-example gradient-norm distribution (better clipping).
H4. With attention-only LoRA (router + experts frozen), the aux loss gradient still flows
    into LoRA params through router probs, but with coefficient 0.001 it is negligible;
    when experts/router are trained, balance matters and H2 is needed.
H5. Per-example gradients of expert weights are sparse across experts only for short
    sequences; at T=1024 with 64 experts/top-8 nearly every expert is hit by every
    example, so noise-vs-sparsity is a utility (not correctness) concern; per-group
    clipping / LoRA on experts / shared-A MoE-LoRA are the levers.

## Output discipline
Return facts with file:line citations for repo claims and URLs/theorem numbers for
literature claims. Mark each claim VERIFIED (you ran/read it) or PLAUSIBLE. Never invent
citations. If something could not be verified, say so explicitly.
