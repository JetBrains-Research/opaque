# Phase 1 — `divergence`: where Opaque's DP per-example path differs from the upstream HF Mellum2 training path

Agent: `divergence`. Repo: `/home/user/opaque` @ `claude/mellum-dp-representation-r6slaz` (HEAD `ef1abc5`, PR #980).
Environment: torch 2.14.0+cu130 (CPU only, `torch.cuda.is_available()==False`), transformers 5.16.1.
Scripts + raw logs: `/tmp/claude-0/-home-user-opaque/a97cca5e-900f-5617-b2df-d73b55081cac/scratchpad/research/divergence/`
(`exp1_vmap_vs_loop.py` → `exp1_log.txt`/`exp1_results.json`; `exp2_grouped_padding_ce.py` → `exp2_log.txt`/`exp2_results.json`;
`exp3_padding_debug.py`; `exp4_bisect.py`; `exp5_leftpad.py`; `exp6_composition.py`; `exp7_maskedrow_autocast.py`).

Legend for every item below: **BATCH-COUPLED** = semantics differ between batched HF and per-example vmap;
**NUMERICAL** = same math, floating-point differences (magnitude estimated); **STRUCTURAL** = different code path, same math.
Status tags: VERIFIED (I read the code lines cited and/or ran it here), PLAUSIBLE (reasoned from code but not executable here, e.g. CUDA/Triton), REFUTED.

All file paths below are relative to `/home/user/opaque/` unless they start with `.venv/`
(`.venv/lib/python3.11/site-packages/transformers/...`).

---

## 0. Executive answer / decision on H1

**H1 is CONFIRMED with one amendment.** Within the model forward and the causal-LM loss, the *only* term that is genuinely
inseparable across examples is the load-balancing auxiliary loss (both `f_e(B)` and `P_e(B)` are batch-pooled; VERIFIED
from `.venv/.../models/mellum/modeling_mellum.py:540-606`). Every other computation (embedding, RMSNorm, sliding/full
attention with RoPE, router, top-k, renormalisation, experts, LM head, per-token CE) is per-token or per-example, and Opaque's
`vmap(grad)` reproduces the upstream per-example gradient to **3.5e-7 relative L2 in fp32** (VERIFIED, `exp1_log.txt`,
`exp2_log.txt`) across: dense `Opaque_MoE`, grouped `Opaque_GroupedMoE` (`torch._grouped_mm`), sliding-window + full layers,
right and left padding, `attention_mask=None` (Boolean SDPA band path), HF gradient checkpointing, and the chunked linear-CE path.

The amendment: HF Trainer's CE reduction is *also* batch-coupled but *separably*: with `num_items_in_batch`, the batch loss is
`Σ_i n_i·mean_i / N` (token-weighted), and its gradient equals the token-weighted sum of per-example token-mean gradients to
7.5e-7 (VERIFIED). Opaque's per-example token-mean loss weights every example equally instead (rel-L2 difference of the
*aggregate* gradient 0.40 in a toy batch with valid-token counts 15/13/10/7). This is a re-weighting, not a new coupling: it
is representable per example via a public batch-level constant `N` (see §4), unlike the aux loss where `f_e(B)` is a data-
dependent argmax statistic.

Two further batch couplings exist *outside the objective*: (i) the collator pads every example of a microbatch to the
longest sequence (shape coupling; loss/gradients exact under right padding — VERIFIED), and (ii) `vmap_create_causal_mask`
decides the SDPA `is_causal` fast path from the **whole physical microbatch mask** (PR #980), so one padded example switches
every example in that microbatch to the materialised-mask SDPA kernel (NUMERICAL only; VERIFIED from
`packages/opaque-patches/src/opaque/api/patches/transformers/runtime/masking.py:195-216`).

Secondary results: H2 VERIFIED exactly (surrogate with detached `f(B)` has bit-identical gradient); H4 supported in the toy
(aux gradient is 2.4e-4 of the CE gradient at coef 0.001); H3 REFUTED at toy scale (bf16 drift of 4.6e-3 to 5e-3 rel-L2 exists
with **zero** route flips, so it is accumulation-order/precision-staging, not routing discontinuity); HF's aux loss **cannot run
under vmap at all** when an `attention_mask` is passed (in-place `scatter_add_` into an unbatched buffer) — VERIFIED.

---

## 1. Experts forward / backward

### 1.1 What upstream HF executes (batched, non-DP)

* Class `MellumExperts` is decorated with `@use_experts_implementation` (`modeling_mellum.py:283-284`). The decorator
  (`.venv/.../integrations/moe.py:523-582`) replaces `forward` by a dispatcher on `config._experts_implementation`
  (`moe.py:567-570`). The default is **`grouped_mm`** (`.venv/.../modeling_utils.py:1965`:
  `applicable_experts = "grouped_mm" if requested_experts is None`), falling back to `eager` only if
  `_grouped_mm_can_dispatch()` fails. In this venv a freshly constructed Mellum reports `grouped_mm` (VERIFIED, exp1 log line 1).
* `grouped_mm_experts_forward` (`moe.py:377-478`): sort the `T·K` (token, slot) rows by expert (`393-395`), `histc` +
  `cumsum` offsets (`401-403`), `_grouped_linear` for gate/up (`440-442`) → `_apply_gate` = `silu(gate)*up` in the activation
  dtype (`509-520`) → `_grouped_linear` for down (`457-459`) → multiply by `top_k_weights` (`462`) → un-permute (`468-470`) →
  `view(T, K, H).sum(dim=1)` (`476`; comment 472-475: reshape+sum accumulates in fp32 for bf16 inputs) → cast back (`478`).
* `_grouped_mm` (`moe.py:308-336`) calls `torch.nn.functional.grouped_mm(input.to(weight.dtype), weight, offs)` — **the input
  is cast to the weight dtype** (`332`, comment 327-330), which matters under autocast (§1.5).
* `batched_mm_experts_forward` (`moe.py:112-173`): gather per-row weights `gate_up_proj[expert_ids]` and `bmm`; same
  reduction (`171`).
* Eager `MellumExperts.forward` (`modeling_mellum.py:296-320`): per hit expert, `index_add_` in the output dtype.
* `_can_use_grouped_mm` (`moe.py:260-305`) reads `weight.data_ptr() % 16` on CPU for torch ≤ 2.10 — under functorch this
  raises (no storage); Opaque shims it (`packages/opaque-patches/src/opaque/api/patches/transformers/runtime/moe.py:321-354`,
  VERIFIED). Irrelevant on torch 2.14 (guard is version-gated) but the shim is still installed by `apply_runtime_patches`.
* functorch coverage (VERIFIED by `torch._C._dispatch_has_kernel_for_dispatch_key`): `aten::_grouped_mm`, `aten::bincount`,
  `aten::histc` have **no** `FuncTorchBatched` kernel; `aten::scatter_add_`, `aten::index_add_`, `aten::topk`, `aten::sort`
  do. So HF's `grouped_mm` experts forward is not vmappable; this is *why* the Opaque kernel exists (STRUCTURAL).

### 1.2 What Opaque executes (`opaque_moe`)

* Patch: `mellum.py` (`packages/opaque-patches/src/opaque/api/patches/transformers/models/mellum.py:28-45`) sets
  `moe_kind="swiglu"`; `_factory.py:316-324` installs `_make_moe_experts_forward(grouped=kwargs.get("grouped_moe", kernels))`
  on `MellumExperts.forward` via `_patch_forward` (`_router.py:59-92`). Only `forward(hidden_states, top_k_index,
  top_k_weights)` is replaced (`components/moe.py:281-294`); the router, aux loss and parameters are untouched (VERIFIED).
* **Process-wide capture of `grouped`** (STRUCTURAL, VERIFIED experimentally): `_patch_forward` patches the *class* once
  (`__opaque_patched__` guard, `_router.py:74-80`) and skips instances whose class forward is already patched (`84-90`), so the
  `grouped` value of the **first** `apply_model_patches` call in a process wins for every later Mellum model in that process.
  In exp1 the 16-expert model dispatched 2 dense / 0 grouped backward calls because the first model had been patched with
  `kernels=False`. In `DPTrainer`, `kernels=bool(args.use_performance_kernels)` (`_dp_trainer.py:841-847`) whose default is
  `False` (`_training_arguments.py:436`) ⇒ **the DPTrainer default is the dense every-token-through-every-expert `Opaque_MoE`
  path on every host, including CUDA**, unless `use_performance_kernels=True` or `performance_kernels_config={"grouped_moe": True}`.
* Dispatch (`kernels/moe.py:578-633`): CUDA + Triton + bf16/fp16 → `Opaque_FusedMoE` (`606-612`); otherwise, if
  `torch._grouped_mm` exists and `use_grouped_route` (memory estimate + `experts >= _SPARSE_MOE_MIN_EXPERTS=16`, `575`,
  `624-632`) → `Opaque_GroupedMoE`; else dense `Opaque_MoE`. On this CPU: 8-expert toy → dense; 16-expert toy with
  `grouped_moe=True` → grouped (exp2: `{'dense': 0, 'grouped': 14}`).
* Dense forward `_moe_forward` (`kernels/moe.py:75-104`): for **every** expert `e` and every token, `F.linear(x, W1[e])`,
  `silu(gate.float()).to(dtype) * up` (`98-101`), `F.linear(h, W2[e]) * route` accumulated into an **fp32** output (`83`, `103`),
  where `route = Σ_k [idx_k==e]·w_k` (`_expert_route`, `67-72`) is 0 for unrouted experts. Mathematically identical to
  upstream (unrouted experts contribute exactly 0·y = 0 unless `y` is inf/NaN). STRUCTURAL + NUMERICAL.
* Grouped forward (`_grouped_moe.py:80-115`): sort routes by expert, `torch._grouped_mm` for gate/up and down, `index_add_`
  into an fp32 accumulator (`112-113`) — same as upstream except fp32 accumulation of the top-k sum.
* Backward (`kernels/moe.py:215-434`, `_grouped_moe.py:270-415`) is a hand-written `autograd.Function` pair
  (`_MoEBackward`/`Opaque_MoE`, `437-568`) with explicit `vmap` rules; per-sample expert-weight gradients are written into
  `(B, E, ...)` buffers (never summed across samples; `_grouped_moe.py:21-24`, `moe.py:255-264`); all reductions in fp32
  (`moe.py:245-254, 326-340, 366, 380`; `_grouped_moe.py:59-71`). Double backward unsupported (`moe.py:473`).
* **Frozen experts skip weight grads** (VERIFIED, exp1): `Opaque_MoE.backward` reads `ctx.needs_input_grad[1]/[2]`
  (`moe.py:536-539`; grouped `_grouped_moe.py:576-580`; Triton `fused_moe.py:510-513`) and passes
  `compute_gate_wgrad/compute_down_wgrad`. With only q/k/v/o trainable the spy recorded
  `[('dense', False, False), ('dense', False, False)]` and the per-example gradient pytree contained exactly the 8 attention
  projections. `compute_route_grad = needs_input_grad[4]` stays `True` whenever hidden states carry grad (LoRA upstream of the
  router), which is required for the chain into earlier layers.

### 1.3 Routing weights and the router gradient (identical math)

* Upstream: `MellumTopKRouter.forward` (`modeling_mellum.py:332-341`): bf16/fp32 `F.linear`, `softmax(dtype=float)`,
  `topk`, in-place renormalisation `router_top_value /= sum` when `norm_topk_prob` (`337-338`), cast back to logits dtype
  (`339`). `MellumSparseMoeBlock.forward` (`350-355`) passes `(routing_weights, selected_experts)` to the experts. The
  gradient reaches the router **only through `top_k_weights`** (indices are integer). In Opaque the router is untouched
  (no `MellumTopKRouter` entry in `classes`, `mellum.py:32-38`) and `opaque_moe` returns `dtw[t,k] = Σ_h go[t,h]·y_e[t,h]`
  in fp32 (`moe.py:353-361`; grouped `_grouped_moe.py:360-362`), cast to the weights' dtype, after which HF autograd continues
  through renorm → softmax → linear exactly as upstream. VERIFIED: fp32 `mlp.gate.weight` per-example grads match to 5.9e-7.
* Top-k ties: same `torch.topk` in both paths (`FuncTorchBatched` kernel exists, so under vmap it is the batched kernel over
  `(B·T, E)`); tie-breaking is implementation-defined in both. Under bf16 the toy showed 0/128 route flips between
  Opaque-vmap and HF-batched (VERIFIED), and 1/128 flips between bf16 and fp32 upstream itself.

### 1.4 Numerical differences in the experts path (magnitudes)

Kernel-level, identical inputs, 16 experts / top-4 (`exp2_log.txt`, section "MoE kernel"):

| dtype | fwd HF-vs-Opaque | fwd vs fp64 (HF / Opaque) | dx | dW1 | dW2 | dtw |
|---|---|---|---|---|---|---|
| fp32 | 4.5e-8 | 2.7e-7 / 2.7e-7 | — | — | — | — |
| bf16 | **0.0** (bit-identical) | 4.5e-3 / 4.5e-3 | see log | see log | see log | see log |

Model-level per-example gradients (bf16 model weights, no autocast, `exp1`/`exp2`):

| comparison | overall rel-L2 |
|---|---|
| HF bf16 loop vs HF fp32 loop (upstream's own bf16 floor) | 1.10e-2 (worst param 5.6e-1 on `mlp.gate.weight[2]`) |
| Opaque dense bf16 vmap vs HF grouped_mm bf16 loop | 4.98e-3 |
| Opaque grouped bf16 vmap vs HF grouped_mm bf16 loop | 4.63e-3 |
| Opaque bf16 vmap vs HF fp32 loop | 1.10e-2 (same as upstream floor) |
| HF bf16 batched (B=4) vs mean of HF bf16 loop (B=1) — upstream vs itself | 2.7e-3 |
| HF fp32 batched vs mean of HF fp32 loop | 6.2e-7 |

Reading: in bf16 the Opaque path is as far from the HF bf16 loop (≈5e-3) as the HF batched forward is from the HF loop
(≈2.7e-3) — both sit well inside upstream's own bf16-vs-fp32 floor (1.1e-2). The Opaque drift comes from fp32 accumulation of
the top-k sum / backward reductions (accumulation order), **not** from routing (0 flips). Classification: NUMERICAL, ~5e-3
rel-L2 on a 2-layer toy; expected to grow with depth (28 layers) and route sensitivity at 12B (PLAUSIBLE; PR #980 body reports
1.3% oracle drift at scale).

### 1.5 Autocast precision staging (NUMERICAL; partly PLAUSIBLE)

* HF `grouped_mm` under autocast with fp32 master weights computes the expert GEMMs in **fp32** (`moe.py:332` upcasts the
  input to the weight dtype; VERIFIED on CPU autocast: `input dtype float32 weight dtype float32`). HF eager experts under
  autocast run `F.linear` in bf16.
* Opaque dense on CPU ignores CPU autocast at the Function boundary (`_active_cuda_dtype` is CUDA-only, `moe.py:45-48`), but
  the inner `F.linear` calls follow CPU autocast. On CUDA, `Opaque_MoE.forward`/`vmap` cast `x, W1, W2, top_k_weights` to the
  autocast dtype (`moe.py:524-527, 562-564`; Triton path `fused_moe.py:559`), i.e. **experts run in bf16 while HF grouped_mm
  runs them in fp32** when weights are fp32 masters (PLAUSIBLE — not executable here).
* Measured on CPU autocast (exp7): Opaque vmap vs HF autocast grouped_mm loop 5.5e-3; HF autocast grouped_mm vs HF autocast
  eager-experts 5.4e-3; all three vs fp32 ≈ 8e-3. So under autocast the Opaque-vs-HF gap equals HF's own backend-to-backend gap.
* Relevance: `DPTrainer` `bf16=True` **enables autocast on the loss closure and does not cast the model**
  (`_dp_trainer.py:849-858`; autocast wraps `grad_fn` at `2123-2129`); the `examples/train_dpftrl.py` presets instead load the
  model in bf16 (`examples/train_dpftrl.py:1021-1039`, `dtype="bfloat16"` at `891`) with no `bf16=` training-arg hit — pure
  bf16 arithmetic, no master weights, so the autocast staging difference does not apply to those presets (VERIFIED by grep).

---

## 2. Router

* Unchanged by Opaque (VERIFIED §1.3). Numerics: linear in model dtype, softmax fp32, renorm fp32, cast back — identical
  ops in both paths; under vmap `F.linear` over `(T,H)` with physical `(B,T,H)` is the same batched GEMM HF runs on `(B·T,H)`.
* `norm_topk_prob` renormalisation: differentiable, in-place on the fp32 topk output; upstream and Opaque share the exact
  autograd graph for it.
* NUMERICAL only: bf16 logits → route flips near ties. Toy: 0/128 flips Opaque-bf16-vmap vs HF-bf16-batched; 1/128 flips
  bf16-vs-fp32 upstream. At 12B/28 layers flips will occur (PLAUSIBLE), but in both paths equally — they are a property of
  bf16 routing, not of vmap.

---

## 3. RMSNorm, attention, masking, RoPE

### 3.1 What the mellum family patch installs (VERIFIED from code)

* `apply_mellum_family_patches = make_apply_family_patches(family="mellum", module_path=..., rope_replacement=None)`
  (`mellum.py:21-25`). With defaults (`_family.py:120-129`): `mod.repeat_kv → vmap_repeat_kv` and
  `mod.eager_attention_forward → vmap_eager_attention_forward` (`_family.py:206-215`), module-scoped masking patch
  (`create_causal_mask`/`create_sliding_window_causal_mask` → `vmap_create_causal_mask`/`vmap_create_sliding_window_causal_mask`,
  `components/masking.py:1077-1098`), **no SDPA replacement** (`sdpa_attention_replacement=None`), **no RoPE replacement**
  (test `test_mellum_keeps_transformers_rotary_embedding`).
* Per-class: `rms_norm_kind=None`, `fused_add_rms_kind=None` (`mellum.py:41-42`) ⇒ upstream `MellumRMSNorm.forward`
  (`modeling_mellum.py:130-135`: fp32 variance, `weight * x.to(input_dtype)`) is kept (test
  `test_mellum_keeps_transformers_rms_norm`). `activation_kind="swiglu"` only affects the dense `MellumMLP` (none in Mellum2:
  `mlp_layer_types` all `"sparse"`, `configuration_mellum.py:114-115`) and only on CUDA+Triton (`_factory.py:293-298`,
  `components/swiglu.py:11-16`). STRUCTURAL: no-op for Mellum2.
* `apply_runtime_patches(compat=True)` (`patches/__init__.py:89-135`, called by DPTrainer at `_dp_trainer.py:586`) also
  patches the shared `transformers.masking_utils.create_causal_mask`/`create_sliding_window_causal_mask`,
  `_ignore_causal_mask_sdpa` (`runtime/masking.py:450-465`) and `sdpa_attention.repeat_kv` (`468-518`), the grouped_mm shim,
  the collator empty-batch shim (`runtime/collator.py:22-50`), and the torch checkpoint patch.
* Batchify (`components/batchify.py:388-416`): `MellumForCausalLM.forward` is wrapped by `with_batch_dim` so 1-D
  `input_ids/attention_mask/labels/position_ids` (2-D `inputs_embeds`) get a leading batch dim under vmap and the output is
  squeezed. KV-cache disabler (`_factory.py:404-405`).

### 3.2 Attention numerics under vmap

* SDPA: HF's `sdpa_attention_forward` (`.venv/.../integrations/sdpa_attention.py:79-170`) is used unchanged. Under vmap on
  **CUDA** the flash / mem-efficient / cuDNN SDPA ops have `FuncTorchBatched` kernels (VERIFIED via dispatch-key query), so the
  batched-over-B call folds into the same fused kernel HF's batched forward uses — with `enable_gqa` (`:97-102`) when no mask
  is passed. On **CPU** `_scaled_dot_product_flash_attention_for_cpu` has no batching rule, so functorch falls back to a
  per-example loop of the same kernel (warning in exp logs) — bit-identical result (VERIFIED: vmap-vs-batched SDPA 0.0 in fp32
  and bf16). STRUCTURAL; NUMERICAL differences on CUDA only through kernel selection (see 3.4).
* Eager (`attn_implementation="eager"`): `vmap_eager_attention_forward` (`components/attention.py:679-723`) is the upstream
  formula with negative-dim indexing; VERIFIED exact in fp32 (exp3: 0.0 rel-L2 for eager, right and left padding).
* `q_norm`/`k_norm` (`modeling_mellum.py:237-238, 252-253`) and RoPE (`141-171`, per-layer-type `rotary_emb`,
  `518-520`; YaRN on full layers in the real config) are upstream code in both paths. Exact.

### 3.3 Sliding-window layers under vmap (VERIFIED correct)

* Upstream window semantics: `sliding_window_overlay` keeps `kv_idx > q_idx - sliding_window` (`masking_utils.py:92-99`), i.e.
  keys in `(q-W, q]`, W keys including self.
* Opaque: `vmap_create_sliding_window_causal_mask` (`runtime/masking.py:318-447`) keeps `k_abs >= q_abs - W + 1`
  (`437-439`) — identical set. Paths: (a) `attention_mask is None` + sdpa + no cache → returns `None` if `T <= W`
  (SDPA `is_causal`), else a Boolean `(1,1,T,T)` band `tril().triu(1-W)` expanded to batch (`60-75`, `361-372`);
  (b) otherwise additive mask from `vmap_create_causal_mask` (`242-247`: `finfo(dtype).min`, padding at `300-313`) with the
  window applied by `masked_fill` (`442-445`). Mellum does **not** get the compact chunked SDPA path
  (`vmap_sdpa_attention_forward_sliding_window`, `components/attention.py:638-676`) because its family passes no
  `sdpa_attention_replacement`; that path exists for families that opt in.
* Experiments (T=16, W=8, layers `[sliding, full]`): fp32 vmap vs HF loop 3.5e-7 with all-ones mask (additive path),
  3.5e-7 with `attention_mask=None` (Boolean band path), and exact at valid positions with right/left padding (exp3, exp5 for
  every pad count 0..7 and W ∈ {4, 8, 16}). Tests: `packages/opaque-patches/tests/transformers/runtime/test_masking.py:146-260,
  330-434` cover look-back limits, cache offsets, SDPA band, and `None` passthrough when `W >= T`.
* Real config: `sliding_window=1024` (`configuration_mellum.py:96`) with `max_seq_len=1024` in the presets ⇒ `T <= W`, so the
  window never activates; sliding layers behave as full causal layers and `vmap_create_sliding_window_causal_mask` returns
  `None` (or the padded additive mask). Any DP run with `T > 1024` exercises the Boolean band path, which is exact (VERIFIED).

### 3.4 The PR #980 `all_valid_attention` change (NUMERICAL, microbatch-coupled)

`runtime/masking.py:195-216`: when an `attention_mask` is passed, the code unwraps the functorch batched tensor
(`torch._C._functorch.get_unwrapped`, `201-206`) and tests `physical_mask.all()` over the **entire physical microbatch**. If all
ones, it returns `None` so SDPA uses `is_causal=True` (the same fast path HF's `_ignore_causal_mask_sdpa` takes for an all-ones
mask, `masking_utils.py:235-280`). Consequences:
* All-valid microbatches: same SDPA kernel as HF batched ⇒ per-example gradients match the batched HF kernel numerics
  (this is what reduced the reported oracle drift from 29% to ≈1.3% — PR body; the mechanism is VERIFIED from the diff
  `git show ef1abc5 -- .../runtime/masking.py`).
* Any padded example in a microbatch flips **every** example of that microbatch to the additive-mask SDPA kernel
  (math/mem-efficient on CUDA). Semantics unchanged (VERIFIED exact in fp32), but the bf16 kernel numerics of unpadded
  examples now depend on which other examples were sampled into the same vmap chunk. This is a microbatch-composition
  numerical coupling; it does not affect the DP guarantee (each example's gradient is still a deterministic function of the
  microbatch, and clipping bounds it) but it is a source of run-to-run drift vs the oracle.
* `_safe_seq_length(past_key_values) <= 0` guard: DPTrainer's kv-cache patch removes `DynamicCache` allocation
  (`_factory.py:400-405`).

### 3.5 Fully-masked query rows (left padding) — NUMERICAL/convention, upstream-inconsistent

* HF SDPA builds a **Boolean** mask (`masking_utils.py:372-500`); PyTorch SDPA returns an all-zero row for a fully masked
  query (VERIFIED: `boolean-mask row0 = [0,0,0,0]`). HF eager and Opaque use an additive `finfo.min` mask, for which softmax is
  uniform and the row equals `mean(V)` (VERIFIED: `additive-min row0 == mean(v)`).
* Only left-padded pad *query* rows are fully masked. Their hidden states differ between the two conventions (rel-L2 ≈ 0.83,
  exp5) but are never attended by valid tokens. They **do** enter the loss when a label sits at the first valid position: HF's
  label shift (`loss_utils.py:61-64`) makes pad position `p-1` predict token `p`. Upstream itself disagrees across backends on
  this token (ex2 left-pad-4: HF sdpa 4.82941 vs HF eager 4.81524; identical 4.80801 once that boundary target is masked —
  VERIFIED exp7). Opaque-vmap reproduces HF-eager (4.93683 vs HF-sdpa 4.90940 on the same example, exp6) and is exact at all
  valid positions in every batch composition (exp6: B=1, [ex2,ex2], [ex0,ex2], [ex1,ex2], B=4 all → 6.5e-8..1.9e-7).
  The large exp2 "padding" gradient error (rel-L2 10) was entirely this boundary token.
* Position ids are `arange` regardless of padding in both paths (`modeling_mellum.py:496-499`; no shift for left padding).
* Practical note: right padding (the usual causal-LM collator setting) never creates a fully masked query row. Left-padded
  prompts (DPO-style collators) would put the first response... only if a label sits at the first *non-pad* position, which
  SFT/DPO prompt masking normally prevents. Flag for the alignment agent: verify the opaque-alignment DPO collator's padding
  side and whether the first completion token is ever predicted from a pad row.

---

## 4. Loss (CE)

* HF: `MellumForCausalLM.forward` → `self.loss_function(logits, labels, vocab_size, **kwargs)` (`modeling_mellum.py:688-689`)
  = `ForCausalLMLoss` (`.venv/.../loss/loss_utils.py:49-71`): `logits.float()`, pad labels with `-100` and shift, flatten,
  `fixed_cross_entropy` (`32-46`): `reduction="sum"` divided by `num_items_in_batch` when the Trainer passes it, else `"mean"`
  over non-ignored tokens of the batch. HF `Trainer` counts `num_items_in_batch` over `labels[..., 1:]` for shift-label losses
  (`.venv/.../trainer.py:518-528`) and across gradient-accumulation micro-steps (`1735`).
* Opaque `DPTrainer.compute_per_example_loss` (`_dp_trainer.py:2314-2434`) calls `fmodel(params, **inputs)` per example and
  reads `output["loss"]` (`2400-2402`) — i.e. HF's own `ForCausalLMLoss` with `reduction="mean"` over that example's tokens
  (no `num_items_in_batch` is ever passed inside vmap). SFT: `nll_loss` (`packages/opaque-alignment/src/opaque/api/alignment/
  sft/loss/_nll.py:43-88`) with a per-example divisor `clamp(min=1)`; `_sft_trainer.py:554-615`. Opaque's CE patches also
  hard-code per-example mean (`components/cross_entropy.py:11-47`, `338-342`).
* **BATCH-COUPLED (separable) — VERIFIED**: `grad(L_batch) = Σ_i (n_i/N)·grad(mean_i)` to 7.5e-7; vs the plain mean of
  per-example means: 0.40 rel-L2 (token counts 15/13/10/7). Effect: HF weights tokens equally, Opaque weights examples equally.
  For fixed-length packed/FIM code data (`max_seq_len=1024`, all tokens labelled) `n_i ≡ N/B` and the two coincide exactly.
  If exact HF weighting is wanted, the per-example loss `(n_i/N̄)·mean_i` with a *public* constant `N̄` (e.g. `B·T_max`, or a
  DP-released mean token count) is a legitimate per-example objective; the sensitivity bound is unaffected by clipping but the
  per-example gradient scale changes (clip threshold should be reconsidered).
* Label shift: identical (`loss_utils.py:61-64` ≡ `components/cross_entropy.py:35-37`; chunked path `_linear_ce_chunked.py:
  255-256`). `label_smoothing`: HF drops it silently; Opaque honours it (`cross_entropy.py:27-31`, `_dp_trainer.py:2380-2432`).
  `logits.float()` upcast in both.
* Chunked linear CE (`chunked_linear_cross_entropy=2048`, `mellum.py:44`; opt-in by `fused_linear_cross_entropy=True` at
  patch time and `opaque_fused_loss_only=True` per call, `components/cross_entropy.py:273-283`; DPTrainer SFT `chunked_nll` /
  eligible `nll`, `_sft_trainer.py:575-590`): streams LSE over 2048-vocab tiles in ≥fp32 (`_linear_ce_chunked.py:47-51, 62-110`)
  with the projection in the input precision (`54-59`), backward re-crosses the bf16 cast boundary like eager
  (`202-236`). VERIFIED: fp32 per-example grads 3.7e-7 vs HF loop; bf16 identical to the non-chunked Opaque path (4.63e-3 vs HF
  bf16 loop). It returns `logits=None` (STRUCTURAL) and falls back to the original forward when `output_router_logits` is set
  (`212-234`), when `logits_to_keep != 0`, `shift_labels`, or class `weight` are used (`142-163`).
* Triton `Opaque_LinearCrossEntropyLoss` / `Opaque_CrossEntropyLoss` (CUDA only; `kernels/linear_cross_entropy.py`, fp32 LSE
  accumulators `623-625`, weight grad gated by `needs_input_grad[1]` `1027-1031`) — PLAUSIBLE same contract, not executed here.

---

## 5. Auxiliary (load-balancing) loss

### 5.1 Exactly what HF computes (VERIFIED from `modeling_mellum.py:540-606, 666-700`)

With `output_router_logits=True` (kwarg or config; default `False`, `configuration_mellum.py:102`), the `capture_outputs`
hooks (`.venv/.../utils/output_capturing.py:100-119, 215-290`; `_can_record_outputs["router_logits"] =
OutputRecorder(MellumTopKRouter, index=0)`, `modeling_mellum.py:432`) collect every layer's `router_logits` of shape
`(B·T, E)` (all 28 layers). Then `load_balancing_loss_func(gate_logits, E, K, attention_mask)`:
* per layer: `routing_weights = softmax(logits)` (fp32 if logits fp32; note: HF passes the raw logits, softmax here is in the
  logits dtype — `584`), `selected = topk(K)`;
* with a mask: `tokens_per_expert_sum += scatter_add(one_hot(selected)·mask)`, `router_prob_sum += Σ_rows routing_weights·mask`,
  `total_rows += mask.sum()` (`594-600`); without a mask: `bincount`/plain sums (`586-593`);
* pooled **over all layers and the whole batch**: `f = tokens_per_expert_sum/total_rows`, `P = router_prob_sum/total_rows`,
  `aux = E · Σ_e f_e·P_e` (`602-606`). Note `Σ_e f_e = K·L` (K assignments per token per layer, L layers), so the layer
  pooling is a sum of per-layer terms with a common denominator, not a mean.
* `loss += router_aux_loss_coef · aux` (`699-700`), coef `0.001` (`configuration_mellum.py:103`). Only `P_e` is
  differentiable; `f_e` is argmax-derived (VERIFIED: H2 identity below). Reference: HF cites Switch Transformer eqs (4)-(6)
  (`549-551`, https://huggingface.co/papers/2101.03961); I did not fetch the paper — equation numbering PLAUSIBLE.

### 5.2 Per-example vs batch (VERIFIED, exp1)

`aux_batch = 2.010357`; per-example aux (each example alone): `[2.1284, 2.0426, 2.0880, 2.0275]`, mean `2.0716 ≠ 2.0104`.
The per-example version penalises within-document imbalance; the batch version penalises imbalance of the pooled batch
(a document can be arbitrarily imbalanced and still contribute a low batch aux if others compensate). BATCH-COUPLED and
**inseparable** as written.

**H2 identity (VERIFIED exactly)**: with `f(B)` detached, `surrogate = E·Σ_e f_e·(Σ_x,t p_e(x,t))/T_total` has gradient equal
to `grad(coef·aux_batch)` with rel-L2 **0.0** over all 21 parameters. So `grad(aux_batch) = (E/T_total)·Σ_x Σ_t Σ_e f_e(B)·∇p_e(x,t)`,
i.e. per-example separable once `f(B)` is treated as a constant. Any lagged / DP-released / public-prior `f~` turns
`E·Σ_e f~_e·(Σ_t p_e(x,t))/T_x` into a legitimate per-example loss (privacy cost only for releasing `f~`, which is a
histogram of K·L·T_x per-example counts — bounded sensitivity, cheap to release).

### 5.3 Magnitude (VERIFIED toy; H4)

`‖grad CE‖ = 3.16`, `‖grad coef·aux‖ = 7.7e-4` (ratio 2.4e-4 overall, 2.4e-4 restricted to q/k/v/o). The aux gradient reaches
`embed_tokens`, `experts.*`, `gate.weight`, both layernorms, `q/k/v/o_proj`, `q_norm/k_norm` (all params upstream of routers).
With attention-only LoRA and coef 0.001 it is ≈1e-4 of the CE gradient in the toy — negligible against DP noise (PLAUSIBLE
at scale; the ratio depends on how imbalanced the fine-tuning data routes).

### 5.4 What happens under vmap (VERIFIED)

* `output_router_logits=True` **with an `attention_mask` fails** inside vmap:
  `RuntimeError: vmap: scatter_add_(self, *extra_args) is not possible because ... other being vmapped over but self not`
  (`modeling_mellum.py:596-598` writes into an unbatched `torch.zeros(E)`). Without a mask the `bincount` branch runs (functorch
  falls back to a loop for `bincount`, which has no batching rule) and yields a **per-example** aux (each example's own tokens).
* `DPTrainer` never passes `output_router_logits`; the config default is `False`, so the DP objective is CE only. The fused-CE
  forward defers to the original HF forward when `output_router_logits` is set (`components/cross_entropy.py:212-234`,
  test `test_fused_ce_preserves_router_auxiliary_loss_contract`). The TRL converters drop `router_aux_loss_coef` with a warning
  (`packages/opaque-transformers/src/opaque/api/transformers/trl/_convert.py:69-76`; wired at `_sft_convert.py:90`,
  `_dpo_convert.py:140`).
* Route information is still obtainable under vmap by calling the backbone `model.model(..., output_router_logits=True)`
  (no aux computation there) — used in exp1/exp4 to extract per-example routes.

---

## 6. Parameter partition with the `mellum2-kstack` preset

* Preset (`examples/train_dpftrl.py:873-895`): LoRA r=16, α=32 on `["q_proj","k_proj","v_proj","o_proj"]`, bf16 model,
  microbatch 8, batch 256, seq 1024, ε=3, band-MF (64 bands), b-min-sep sampling. Experts + router + norms + embeddings frozen.
* Per-example gradients that exist: exactly the LoRA A/B of the 4 attention projections (toy with q/k/v/o trainable:
  8 keys, VERIFIED). No expert/router per-example gradients are materialised (`compute_gate_wgrad=compute_down_wgrad=False`,
  VERIFIED); the `(B,E,2I,H)` buffers that would cost `B·64·1792·2304·2 bytes ≈ 8·0.53 GB` per layer are never allocated.
* PEFT fused-LoRA kernels (`packages/opaque-patches/src/opaque/api/patches/peft/_router.py:151-190`) are performance-gated
  (`lora = kwargs.get("lora", performance)`, `_lora_patching_allowed()`); STRUCTURAL, not exercised here.
* Where the aux loss would enter: only via `top_k_weights → router → hidden states → LoRA` (§5.3) — with the trainer default it
  does not enter at all.

---

## 7. Other batch-coupled or path-dependent items

| # | item | class | status | evidence |
|---|---|---|---|---|
| 7.1 | Collator pads to longest in microbatch (`DataCollatorWithPadding`, `_dp_trainer.py:484-497`); pad positions masked; per-example loss ignores `-100` | BATCH-COUPLED shape, exact semantics (right padding) | VERIFIED | exp3: right/left padding grads 0.0 rel-L2 (eager) / ≤3.4e-7 (sdpa) at valid positions |
| 7.2 | `all_valid_attention` reads the whole physical microbatch mask → SDPA kernel choice depends on microbatch composition | NUMERICAL | VERIFIED (code) | `runtime/masking.py:195-216` |
| 7.3 | Left-padding boundary token predicted from a fully masked query row: Boolean-SDPA (zeros) vs additive (mean V) | NUMERICAL/convention; upstream-inconsistent | VERIFIED | exp5/6/7 |
| 7.4 | Position ids `arange` in both paths (no left-pad shift) | same | VERIFIED | `modeling_mellum.py:496-499` |
| 7.5 | Gradient checkpointing: HF `gradient_checkpointing_enable(use_reentrant=False)` (`_dp_trainer.py:1336-1341`) + Opaque `torch.utils.checkpoint` patch (`patches/torch/checkpoint`) | STRUCTURAL | VERIFIED exact (3.5e-7) | exp2 "gc_fp32" |
| 7.6 | Dropout: `attention_dropout=0.0` by config (`configuration_mellum.py:97`); Opaque ships `disable_dropout` (`components/dropout.py`) because fused SDPA dropout is vmap-incompatible | STRUCTURAL (no-op for Mellum2) | VERIFIED (code) | — |
| 7.7 | KV cache: DPTrainer/kv-cache patch skips `DynamicCache`; HF training also passes no cache when `use_cache=False`; masks built with `past_seen_tokens=0` | STRUCTURAL | VERIFIED (code) | `_factory.py:400-405`, `runtime/masking.py:214, 229` |
| 7.8 | `grouped` flag captured at the first class-level patch per process; DPTrainer default `use_performance_kernels=False` ⇒ dense MoE path everywhere | STRUCTURAL (perf) | VERIFIED | §1.2 |
| 7.9 | HF `grouped_mm` experts not vmappable (no batching rule for `_grouped_mm`, `histc`, `bincount`) | STRUCTURAL | VERIFIED | dispatch-key query |
| 7.10 | bf16 autocast staging: HF grouped_mm upcasts activations to fp32 weights; Opaque CUDA casts to bf16 | NUMERICAL (~5e-3 toy on CPU; CUDA PLAUSIBLE larger) | VERIFIED CPU / PLAUSIBLE CUDA | exp7 |
| 7.11 | Distributed: HF DDP averages the batch loss gradient; Opaque sums clipped per-example grads across ranks (`_dp_trainer.py:2140+`) — equivalent to the single-process mechanism by construction | STRUCTURAL | not exercised (single process) | — |
| 7.12 | `num_items_in_batch` is also summed across gradient-accumulation steps in HF (`trainer.py:1735`); DP has no accumulation, microbatches are vmap chunks of one logical batch | BATCH-COUPLED (separable, see §4) | VERIFIED (code) | — |

---

## 8. Hypothesis verdicts

* **H1** — CONFIRMED with amendment (aux loss inseparable; CE normalisation separably batch-coupled; forward exact to fp).
* **H2** — VERIFIED exactly (surrogate gradient identical, rel-L2 0.0). Per-example aux differs materially from batch aux
  (2.07 mean vs 2.01 batch in the toy; it is a different regulariser).
* **H3** — REFUTED at toy scale: bf16 Opaque-vs-HF drift (4.6e-3–5.0e-3) exists with 0/128 route flips, so residual drift is
  accumulation/staging, not routing discontinuity; and the drift is within upstream's own batched-vs-loop bf16 spread (2.7e-3)
  and far under its bf16-vs-fp32 floor (1.1e-2). Whether flips dominate at 12B/28 layers is PLAUSIBLE but unverified here;
  pinning routes would remove a source of *variance* but cannot make the two paths agree better than the accumulation floor.
* **H4** — supported in the toy (aux/CE gradient ratio 2.4e-4 at coef 0.001; same ratio on q/k/v/o).
* **H5** — not tested here (short sequences only); note that with the preset partition no expert gradients exist at all.

---

## 9. Open questions / things I could not verify

1. CUDA-only paths (`Opaque_FusedMoE` Triton, `Opaque_LinearCrossEntropyLoss`, fused LoRA, cuDNN/flash SDPA batching under
   vmap, autocast bf16 staging on CUDA): read only, PLAUSIBLE.
2. The 12B oracle drift figure (29% → 1.3%) is from the PR body; not reproducible here.
3. Switch Transformer equation numbering (4)-(6) for the aux loss: quoted from the HF docstring, paper not fetched.
4. Padding side used by the opaque-alignment DPO/SFT collators for Mellum (left-pad boundary token issue §3.5).
5. Whether `bincount`'s functorch fallback loop (no-mask aux path) is acceptable at scale, if a per-example aux is wanted —
   a vmap-native per-example aux should be written with out-of-place `scatter_add`/one-hot sums instead.
