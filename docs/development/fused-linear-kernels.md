# Fused-linear kernels for opaque-alignment — design note

**Status:** design locked; `token_weight` on LCE shipped (GPU CI pending); the
alignment fused glue (A/B) is the remaining build. Supersedes
`opaque-alignment-plan.md` §7.10 (pure-torch, KTO, string registry — all
changed).

## Decisions (locked, post-review)

1. **No bespoke kernel.** `sequence_logp = −Σ_completion CE`, so the existing
   `opaque-patches` `Opaque_LinearCrossEntropyLoss` *is* the preference-logp
   kernel: mask non-completion tokens to `ignore_index`, call it (unreduced),
   negate. No new Triton.
2. **`token_weight` on LCE (shipped).** Per-token weight multiplied into the
   per-token CE; backward scales the per-token `do`. Enables completion masking
   (0/1) and soft/turn masks. **It does NOT enable DFT** — DFT's weight is
   `detach(exp(−CE))`, *derived from the logits*, so it needs a kernel-internal
   `use_token_scaling` mode (deferred). DFT stays eager until then.
3. **Auto-select inside one entry.** The fused `*_loss` entries pick the LCE
   fast path (when `[patches]` + CUDA + half) or the pure-torch fallback —
   mirroring the patches `use_fused_ce` gate. No user branching.
4. **Signature.** Fused entries take `(hidden_states, lm_head_weight, …)`; eager
   take `(logits, …)`. Fused requires the model to emit hidden (skip the
   in-forward `lm_head`).
5. **Packaging.** `opaque-alignment` stays Triton-free; reuses patches via the
   `[patches]` extra; the pure-torch chunked loop is the CPU/no-Triton fallback.
6. **Sequencing.** A-NLL + B (DPO) use `ignore_index` masking, *not*
   `token_weight`, so they don't depend on the unverified `token_weight` and are
   buildable now. A-DFT waits on `use_token_scaling`. C (ref precompute) folds
   into the fused-logp helper (the precompute already batches).

> The original "build a Triton `Opaque_FusedLinearSequenceLogp`" plan below is
> **superseded** by decision 1 (reuse LCE); kept for the analysis/rationale.

## 0. Why these and not others (constraint envelope)

- The **only** memory-bound tensor in the alignment forward is the vocab
  projection `hidden @ Wᵀ → (B, T, V)` (~1 GB/sample at V≈128K). Everything
  downstream (logp reduction, scalar DPO/CE math) is cheap. A kernel is worth it
  **iff it avoids materializing those logits**.
- DP-SGD runs `vmap(grad(per_example_loss))`. A kernel that actually saves
  **training** memory must **recompute logits in backward** — which means a
  custom `torch.autograd.Function` with an explicit `vmap` rule (the two-level
  `Opaque_Foo / _FooBackward` pattern). `torch.utils.checkpoint` does **not**
  compose with `torch.func`.
- The current `fused_linear_preference` / `fused_linear_sft_loss` are plain
  chunked Python loops: they lower the **forward-transient** peak only; under
  `grad`, every chunk's `log_softmax` is saved → peak stays `O(B·T·V)`. They are
  correct and stay as the **no-Triton fallback**, but they are not the win.
- **Template:** `opaque-patches`' `Opaque_LinearCrossEntropyLoss` already does
  the right thing for plain CE — Triton (Apple cut-cross-entropy lineage),
  recompute-LSE-in-backward, explicit `vmap` staticmethods, shift handled in
  Python so the vmap merge is a trivial reshape. B copies this template.

## 1. Workstream B — `Opaque_FusedLinearSequenceLogp` (patches)

### Computes
Per sequence `n`, the completion log-probability
```
logp[n] = Σ_{t ∈ completion(n)} ( logit[n,t,target] − LSE(e[n,t] @ Cᵀ) )
        = − Σ_{t ∈ completion(n)} CE(e[n,t], target[n,t])
```
- **Inputs:** `hidden (N, T, H)`, `weight (V, H)`, `target_ids (N, T)`,
  `completion_mask (N, T)`.  **Output:** `logp (N,)`.
- `N = 2B` (chosen rows `[0:B]`, rejected `[B:2B]`) — the layout the current
  `opaque_fused_linear_dpo_loss` dispatcher already uses.

### Relationship to LCE (what's reused vs new)
Sequence-logp is **the LCE per-token core with a different reduction**:
| Aspect | LCE (`Opaque_LinearCrossEntropyLoss`) | This kernel |
|---|---|---|
| per-token term | `neg_dot + LSE` (CE) | same |
| reduction | global `nll.sum()` → scalar | **per-sequence sum** → `(N,)` |
| masking | `ignore_index` | `completion_mask` (+ ignore_index) |
| output | CE sum | **logp = −CE sum** (no division) |
| shift | pre-shift in Python | pre-shift in Python |
| backward | recompute LSE | recompute LSE (same) |
Reuse the cut-CE `_utils` (autotuned matmul+LSE Triton tiles, `b_bin_fn`,
autocast follow-along). The new code is the per-sequence grouped reduction +
`(N,)` output + completion masking.

### Forward / backward
- **Forward (Triton):** stream vocab blocks, accumulate `LSE[n,t]` online and
  `logit[n,t,target]`; reduce over completion tokens per sequence → `logp (N,)`.
  Never materialize `(N,T,V)`.
- **Backward (recompute):** `∂logp[n]/∂e[n,t] = −(softmax(e[n,t]@Cᵀ) −
  onehot(target))` on completion tokens, scaled by upstream `g[n]`; `dW`
  accumulates `Σ_{n,t}`. Recompute softmax over vocab blocks — no logits saved.
  Peak `O(N + V_block)` vs `O(N·T·V)`.
- **vmap rule:** explicit `vmap` staticmethod (copy LCE's): labels/targets
  batched at dim 0, pre-shifted, so vmap merge is a reshape; frozen-`W` (LoRA)
  case returns `dc = 0`.

### DP-correctness
`logp[n]` depends only on sequence `n`'s `hidden`/`targets` (W is shared
parameter, not per-example data) → strict per-example (Tier 1). The scalar DPO
variant runs **outside** the kernel on the `(B,)` log-ratios (elementwise) →
Tier 1 preserved. Enforced by the NaN-injection per-example isolation test.

### alignment wrapper (no API change)
`opaque_fused_linear_dpo_loss(..., per_pair_loss_fn=dpo_sigmoid, ...)` keeps its
signature. New internal path:
```
if patches available:
    cl = Opaque_FusedLinearSequenceLogp(chosen_hidden, W, chosen_ids, chosen_cmask)
    rl = Opaque_FusedLinearSequenceLogp(rejected_hidden, W, rejected_ids, rejected_cmask)
    return per_pair_loss_fn(cl - ref_chosen_logp, rl - ref_rejected_logp)
else:
    <current pure-torch chunked loop>     # CPU / no-Triton fallback
```
The kernel replaces only the logp computation, never the scalar loss.

## 2. Workstream A — SFT NLL via LCE

Make the opt-in `fused_linear_sft_loss` delegate to
`Opaque_LinearCrossEntropyLoss` when `[patches]` is present; pure-torch loop
otherwise. **DP-correctness gotcha:** use the **unreduced `nll_sum`** and divide
by the **per-example** completion-token count (plan §3.3/§8.2 divisor) — *not*
`opaque_linear_cross_entropy_loss`'s global-mean wrapper. This matches the
current `nll_loss` divisor exactly.

- **NLL:** ships now (LCE is a drop-in for the per-token CE core).
- **DFT:** needs per-token confidence scaling *inside* the CE sum →
  `use_token_scaling=True` on `Opaque_LinearCrossEntropyLoss` (a patches
  extension, plan Tier 3). **Recommendation:** NLL via LCE now; DFT stays on the
  pure-torch fused path until the patches kernel grows token-scaling.

## 3. Workstream C — chunk the reference precompute

`compute_ref_logprobs_for_dataset` already runs under `no_grad`, so chunking the
lm-head projection fully realizes the memory win (nothing saved for backward) —
even a plain chunk loop suffices, no custom autograd needed. Lowest-risk; ship
first. Once B lands it can call `Opaque_FusedLinearSequenceLogp` in no-grad mode
for one code path, but that is optional.

## 4. Validation

- **CPU** (extend `tests/dpo/kernel/test_fused_linear_dpo.py`): fused-vs-eager
  parity `1e-4`; chunk/vmap invariance; `grad` parity; `vmap(grad)` finite.
- **GPU** (Cadence preset): peak memory `< (B,T,V)`; throughput; tiny smoke
  train; per-example ε snapshot.
- **DP**: NaN-injection per-example isolation (Tier 1) on the kernel-backed path.

## 5. Risks / open items

- The three `vmap` interactions to copy from LCE: `compile`↔`vmap`
  auto-disable; per-sample `dW`; nested `grad_and_value` inside the vmap body.
- LSE-recompute numerical stability (float32 accumulation) — cut-CE handles it.
- Autocast entry: alignment's `_utils.follow_autocast` already exists; the
  patches kernels carry their own.
- DFT token-scaling is a deliberate patches-kernel extension — decide whether to
  do it in this effort or defer.
- The fused path is `[patches]`-gated; the pure-torch fallback **must** stay
  green on CPU CI.

## 6. Sequencing

1. **C** — no autograd subtlety, no patches dep; validates chunked no-grad win.
2. **A** — establishes the `[patches]`-gated wrapper pattern + per-example
   divisor on the simpler (CE) kernel.
3. **B** — the real preference kernel, reusing A's wrapper plumbing and the LCE
   Triton/vmap template.
