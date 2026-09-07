# Phase 1 — empirical characterization of the Mellum2 DP-representation problem (agent: `empirical`)

All experiments: CPU-only, `torch 2.14.0+cu130` (no CUDA), `transformers 5.16.1`, repo venv via
`uv run python`, `torch.set_num_threads(2)`. Scripts, logs and raw JSON/pt outputs live in
`/tmp/claude-0/-home-user-opaque/a97cca5e-900f-5617-b2df-d73b55081cac/scratchpad/research/empirical/`
(`common.py`, `e1_ref.py`, `e1_vmap.py`, `e1b_ref.py`, `e1b_vmap.py`, `e2_routes.py`, `e3_norms.py`,
`e3b_norms.py`, `e4_aux.py`, `e4b_imbalanced.py`, `e5_load.py`, `e6_sens.py`; `*.log`, `*_results.json`).
Total compute was about 2 minutes (every script ran in <= 10 s), far under the 25-minute budget.

**Global caveat (applies to every number below).** All models are *randomly initialised* tiny Mellum
configs (`hidden 64, 2 layers (1 for E5/E6), 4 heads / 2 KV heads, vocab 128, E=8 experts, top-2,
moe_intermediate 32`, plus `E=64/top-8` in E5/E6), built with the repo's own
`build_moe_model('mellum', 'cpu', ...)` helper (`packages/opaque-patches/tests/transformers/models/_test_utils.py:109-170`),
which applies `apply_model_patches(model, eager_attention=True)` exactly as the Mellum tests do.
Data are synthetic random token sequences, or "structured" sequences in which each example samples
its tokens from its own 6-token vocabulary subset (mimicking within-document topical routing).
Nothing here measures a trained 12B checkpoint; the *mechanisms* transfer, the *magnitudes* do not.

The "unpatched HF" reference is a fresh `MellumForCausalLM` in a **separate process** (the Opaque
patch rebinds `MellumExperts.forward` at class level, so an in-process "unpatched" copy would be
patched too). On this CPU the unpatched HF model executes experts with
`config._experts_implementation == "grouped_mm"` (printed by `e1_ref.py`; dispatch at
`.venv/lib/python3.11/site-packages/transformers/integrations/moe.py:569`).

---

## 0. Facts established while setting up (VERIFIED by running)

| # | Finding | Evidence |
|---|---|---|
| 0.1 | HF `load_balancing_loss_func` **cannot run under `torch.func.vmap` when an `attention_mask` is passed**: it uses in-place `scatter_add_` on an unbatched tensor. | `modeling_mellum.py:598`; `smoke.py` traceback: `RuntimeError: vmap: scatter_add_(self, *extra_args) is not possible because there exists a Tensor 'other' in extra_args that has more elements than 'self'` |
| 0.2 | Without a mask the `torch.bincount` path (`modeling_mellum.py:589`) *does* run under vmap (via a slow batching fallback with a warning) and yields a **per-example** aux loss (each example's own f and P). | `smoke2.py`: `no-mask vmap OK; per-example aux: tensor([2.0265, 2.0663, 2.1494, 2.0951])` |
| 0.3 | An out-of-place re-implementation (`common.py::aux_loss_vmap_safe`, one-hot via `==` comparison, no `F.one_hot` which also breaks under vmap) reproduces HF's batched value to < 1e-5 and is vmap-safe with masks. Used for E3/E4. | `e4_aux.py` assertion `abs(aux_hf - aux_mine) < 1e-5` passed for all 3 batches |
| 0.4 | On CPU, SDPA under vmap falls back to `BatchedFallback` for `_scaled_dot_product_flash_attention_for_cpu` (perf warning only; results exact, see E1). | warning in `e1_vmap.log` |
| 0.5 | HF's *batched* CE loss (token-mean with `num_items_in_batch` absent, equal lengths) has gradient exactly equal to (1/B) x sum of per-example gradients: max relative L2 error over all 26 parameter tensors **4.5e-7 (fp32)**, 2.5e-3 (bf16). I.e. on the HF side the *only* batch coupling is the aux loss (H1, HF side). | `e1_ref.py` output |

---

## E1 — Exactness of the Opaque per-example path vs the unpatched HF loop

Setup: B=8, T=32 random tokens, **all 313,152 parameters trainable**, same weights (state_dict copied
from the reference process). Opaque side: `opaque.dpsgd.clipping.clipped_grad(..., clipping_norm=1e9)`
(the real DP path, returns the sum) and `torch.func.vmap(torch.func.grad(...))` (what `clipped_grad`
runs internally) for per-example vectors. Reference: Python loop of `loss.backward()` on the unpatched
model, `output_router_logits=False`. Router top-k sets captured per token in both paths (Opaque side:
`output_router_logits=True` with `router_aux_loss_coef=0` so the loss is exactly CE; the router
logits come back through HF's `OutputRecorder`, which works under vmap).

| dtype | pair | route flips (tokens) | per-example rel. L2 (whole grad vector) | max per-tensor rel. L2 (worst tensor) | sum-vs-sum max rel. L2 | loss max abs err |
|---|---|---|---|---|---|---|
| fp32 | Opaque vmap vs HF loop | 0 / 256 | 2.4e-7 ... 3.4e-7 | 6.1e-7 (`layers.0.input_layernorm.weight`) | 4.4e-7 | 4.8e-7 |
| bf16 | Opaque vmap vs HF loop (both bf16) | 0 / 256 | 4.0e-3 ... 4.9e-3 | 8.8e-3 (`layers.0.input_layernorm.weight`) | 6.6e-3 | 4.8e-7 |
| bf16 | Opaque vmap (bf16) vs HF loop (fp32) | 3 / 256 (layer 1) | 0.92% ... 1.18% | — | — | — |
| bf16 | HF loop (bf16) vs HF loop (fp32) | 3 / 256 | 0.92% ... 1.17% | — | — | — |

Raw: `e1_results.json`, `e1_vmap.log`.

**Reading.** (VERIFIED) In fp32 the Opaque path (dense `Opaque_MoE`, `moe.py:518`, fp32 accumulation
`moe.py:83`; SDPA fallback; upstream RMSNorm retained per `mellum.py:41`) is exact to fp32 round-off
(~3e-7) with identical routing. In bf16 the two bf16 implementations differ by ~0.45% with **no**
route flips — pure accumulation-order noise between HF `grouped_mm` (bf16 accumulate) and
`Opaque_MoE` (fp32 accumulate). Both bf16 paths are ~1.0% from fp32 and are equally far, i.e.
essentially all the "bf16 vmap vs oracle" discrepancy is bf16 itself, not vmap or the patch.

### E1b — larger bf16 batch, route-flip attribution, and a pinned-routing control (H3)

B=32, T=64 (2048 tokens), all params trainable. Reference process computes fp32 loop grads, bf16 loop
grads, and **bf16 loop grads with routing pinned to the fp32 top-k choice** (a process-local
`MellumTopKRouter.forward` that gathers the probabilities at the pinned indices and renormalises
exactly as `modeling_mellum.py:335-339`). Errors are relative L2 against the fp32 loop.

| quantity | examples with >= 1 flipped token (22/32) | examples with no flip (10/32) |
|---|---|---|
| tokens whose top-2 set differs bf16 vs fp32 | 41 / 2048 (2.0%) | — |
| whole-vector rel. L2, bf16 | mean 1.34%, max 1.58% | mean 1.13%, max 1.22% |
| whole-vector rel. L2, bf16 **pinned** to fp32 routes | mean 1.17%, max 1.29% | mean 1.13% |
| **router** (`mlp.gate`) grads, bf16 | **mean 12.9%, max 23.5%** | 1.7% |
| **router** grads, bf16 pinned | **1.66%** | — |
| **experts** grads, bf16 | **mean 11.1%, max 15.3%** | 1.4% |
| **experts** grads, bf16 pinned | **1.47%** | — |
| attention q/k/v/o grads, bf16 | 1.31%, max 1.55% | 1.12% |
| attention grads, bf16 pinned | 1.16% | — |

Opaque bf16 vmap vs HF bf16 loop on the same 2048 tokens: **0 route flips**, error 0.47% (max 0.51%).
Opaque bf16 vmap vs fp32 loop: the *same* 41 flipped tokens / 22 examples as HF bf16 (i.e. the patch
reproduces HF's bf16 routing decisions exactly here), flip-examples 1.34% vs no-flip 1.13%.
Raw: `e1b_results.json`, `e1b_vmap_results.json`, logs.

**Reading (H3).** (VERIFIED, tiny model) Route flips are the dominant source of *excess* bf16 gradient
error for **router and expert** parameters (about 8x the no-flip floor for affected examples), and
pinning the routes removes that excess entirely (back to the ~1.5% bf16 floor). For **attention-only**
parameters (the current LoRA presets) flips add only ~0.2 percentage points. So "pin routing decisions
per example" is a real lever if router/experts are trained, and mostly irrelevant for attention-only LoRA.
Note pinning does not change the DP analysis of a per-example gradient (the pinned indices are a
deterministic function of that example alone), but it does change *which* function's gradient is
clipped — the fp32-routed forward — which is the more faithful one.

---

## E2 — Route sensitivity of the forward pass (bf16 vs fp32), unpatched model

B=32, T=64, eval mode, per layer; the router weight was additionally scaled by x1 / x5 / x20 to emulate
sharper routing than random init produces. `prob margin` = p(k-th) - p(k+1-th) of the fp32 softmax.

| router scale | data | L0 flip frac | L1 flip frac | median prob margin | frac margin < 1e-3 | frac margin < 1e-2 | median top-1 prob |
|---|---|---|---|---|---|---|---|
| x1 | random | 1.0% | 0.8% | 6.1e-3 | 10.2% | 68.6% | 0.153 |
| x1 | structured | 0.8% | 1.1% | 6.0e-3 | 9.8% | 72.3% | 0.152 |
| x5 | random | 0.8% | 1.1% | 3.7e-2 | 2.0% | 16.2% | 0.284 |
| x5 | structured | 0.9% | 0.9% | 3.6e-2 | 2.0% | 16.1% | 0.280 |
| x20 | random | 0.8% | 1.4% | 7.5e-2 | 2.9% | 14.2% | 0.694 |
| x20 | structured | 0.9% | 1.2% | 7.5e-2 | 3.7% | 15.8% | 0.696 |

Raw: `e2_results.json` (also logit-space margins and margins of the flipped rows).

**Reading.** (VERIFIED, tiny model) ~1% of tokens per layer change their top-2 set between bf16 and fp32
forward. The flip rate is *invariant to router sharpness*: scaling the router scales both the margin
and the logits, and bf16 has fixed *relative* resolution (~8 bits), so the fraction of tokens whose
margin is below the bf16 rounding of the logits stays ~1%. Sharper routers do not fix flips; only
higher-precision routing (fp32 router matmul / fp32 hidden states into the router) or pinning does.
With random init nearly all tokens are within 1e-2 of a tie; with x20 still ~15%. A trained checkpoint
sits somewhere in between — unknown from this experiment.

---

## E3 — Per-example gradient-norm distributions (fp32, T=64, B=32)

Via `clipped_grad(..., clipping_norm=1e9, return_aux=True)` -> `aux.grad_norms`. Partitions: `attn` =
full q/k/v/o weights (proxy for LoRA on q/k/v/o; no LoRA adapters used — caveat), `attn+router`,
`attn+experts`. Loss variants: `ce` (HF per-example CE), `ce+0.001aux` (CE + 0.001 x *per-example*
Switch aux computed from the example's own tokens, pooled over layers, via the vmap-safe function),
`aux_only` (the per-example aux alone, coefficient 1, to show its scale).

Default init (`initializer_range=0.02`) — `e3_results.json`:

| partition | loss | data | median | p95 | max | max/median |
|---|---|---|---|---|---|---|
| attn | ce | random | 1.941 | 2.262 | 2.506 | 1.29 |
| attn | ce | structured | 5.667 | 7.092 | 7.534 | 1.33 |
| attn | ce+0.001aux | random | 1.941 | 2.262 | 2.506 | 1.29 |
| attn | aux_only | random | 2.603 | 3.716 | 4.111 | 1.58 |
| attn | aux_only | structured | 4.085 | 6.460 | 7.565 | 1.85 |
| attn+router | ce | random | 1.941 | 2.262 | 2.506 | 1.29 |
| attn+router | aux_only | random | 2.869 | 4.264 | 4.525 | 1.58 |
| attn+router | aux_only | structured | 4.959 | 7.802 | 8.580 | 1.73 |
| attn+experts | ce | random | 1.942 | 2.263 | 2.507 | 1.29 |
| attn+experts | ce | structured | 5.668 | 7.093 | 7.535 | 1.33 |
| attn+experts | aux_only | random | 2.603 | 3.717 | 4.111 | 1.58 |

Init `1/sqrt(hidden) = 0.125` so all branches are O(1) (closer to a trained regime) — `e3b_init0.125_results.json`:

| partition | loss | data | median | p95 | max | max/median |
|---|---|---|---|---|---|---|
| attn | ce | random | 2.595 | 2.953 | 3.236 | 1.25 |
| attn | ce | structured | 6.094 | 7.383 | 7.934 | 1.30 |
| attn | ce+0.001aux | random | 2.596 | 2.953 | 3.235 | 1.25 |
| attn | aux_only | random | 3.535 | 6.395 | 7.097 | 2.01 |
| attn | aux_only | structured | 5.181 | 8.878 | 9.763 | 1.88 |
| attn+router | ce | random | 2.606 | 2.961 | 3.239 | 1.24 |
| attn+router | aux_only | random | 4.130 | 7.139 | 7.819 | 1.89 |
| attn+router | aux_only | structured | 5.865 | 9.993 | 11.887 | 2.03 |
| attn+experts | ce | random | 2.647 | 3.010 | 3.286 | 1.24 |
| attn+experts | ce | structured | 6.158 | 7.436 | 7.976 | 1.30 |
| attn+experts | aux_only | random | 3.556 | 6.433 | 7.139 | 2.01 |

**Reading.** (VERIFIED, tiny model)
1. The CE per-example norm distribution is tight (max/median 1.24-1.33) for all three partitions; a
   clipping norm near the median clips roughly half the examples with modest bias. Structured
   sequences give 2-3x larger attention gradients (repeated tokens are highly predictable by attention).
2. Adding router or expert weights barely moves the CE norm at random init (the MoE branch contributes
   little); with the larger init the increase is still small (2.595 -> 2.647). Not informative about a
   trained model where expert grads may dominate — caveat.
3. With the HF coefficient 0.001, the per-example aux changes the norm by < 0.1% (e.g. 1.9409 -> 1.9413):
   **H4 holds** in this regime (aux is negligible for clipping). The *unscaled* per-example aux gradient
   is comparable to the CE gradient (median 2.6-5.9 vs 1.9-6.1) with a heavier tail (max/median 1.6-2.0),
   and it *does* flow into attention-only parameters (attn|aux_only is nonzero and of the same order as
   attn+router|aux_only) — the "aux still reaches LoRA params through router probs" part of H4 is verified.

---

## E4 — Aux-loss gradient representations (one batch, unpatched model, autograd, fp32)

B=16, T=32, all params, coefficient 1 on the aux term (errors are relative, so the coefficient cancels).
Definitions (HF, `modeling_mellum.py:540-608`): N = total rows pooled over layers (L x sum_x T_x),
f_e = (#top-k assignments to e)/N, P_e = (sum of softmax probs of e)/N, aux = E * sum_e f_e P_e.
(i) HF batched aux gradient. (ii) mean over x of grad of aux_x = E * sum_e f_e(x) P_e(x) (each example's own
f and P — what `output_router_logits=True` under vmap produces). (iii) surrogate with f~ = f(B) constant:
sum_x grad of l_x, l_x = E * sum_e f~_e * Pbar_e(x) * T_x / T_tot. (iv) f~ = f(B) + N(0, (rel * k/E)^2 I),
5 seeds. Plus: f~ from a *different* batch ("lagged"), and f~ = k/E ("uniform").

Random init (`e4_results.json`), f(B)/(k/E) in [0.66, 1.26] i.e. nearly balanced:

| batch | ||(i)|| | (ii) rel err / cos | **(iii) rel err** | noise rel 0.05 | 0.1 | 0.3 | 1.0 | lagged f (its dev from f(B), in k/E) | uniform f |
|---|---|---|---|---|---|---|---|---|---|
| random | 0.330 | 2.54 / 0.39 | **2.4e-7** | 0.32 | 0.65 | 1.94 | 6.47 | 0.65 (0.083) | 1.00 (||grad|| 2.8e-8) |
| structured | 0.406 | 3.80 / 0.28 | **2.6e-7** | 0.24 | 0.48 | 1.44 | 4.81 | 1.66 (0.324) | 1.00 |
| random, ragged lengths (right padding, T_x = 32-2b) | 0.337 | 3.85 / 0.26 | **3.3e-7** | 0.32 | 0.63 | 1.90 | 6.34 | 0.83 (0.117) | 1.00 |

Induced imbalance (router rows 0-1 x3 in every layer; f(B)/(k/E) in [0.51, 1.71]) — `e4b_imbalanced_results.json`:

| batch | ||(i)|| | (ii) rel err / cos | (iii) rel err | noise rel 0.05 | 0.1 | 0.3 | 1.0 | lagged f |
|---|---|---|---|---|---|---|---|---|
| random | 1.711 | 0.98 / 0.79 | 2.5e-7 | 0.098 | 0.20 | 0.59 | 1.97 | 0.23 (0.082) |
| structured | 2.343 | 1.10 / 0.76 | 2.7e-7 | 0.069 | 0.14 | 0.42 | 1.39 | 0.55 (0.299) |

**Reading.**
1. (VERIFIED) **H2's separable surrogate is exact**: with f~ = f(B) held constant, the sum of per-example
   gradients of l_x equals the HF batch aux gradient to fp32 round-off (2-3e-7), including the ragged-length
   case with the T_x/T_tot weighting. This is the algebraic fact that f_e is argmax-derived (zero gradient,
   `modeling_mellum.py:589/598` are integer counts) so grad(aux) = E * sum_e f_e * grad(P_e) and P_e is a
   plain mean over rows.
2. (VERIFIED) The **per-example aux (ii) is a materially different regulariser**: relative error 2.5-3.9,
   cosine 0.26-0.39 at balanced init; even with induced imbalance cosine 0.76-0.79. It penalises
   *within-example* imbalance (mean per-example aux 2.06-2.18 > batch aux 2.005-2.05), which the
   batch loss never asked for. Using `output_router_logits=True` under vmap (the no-mask path that
   "works") silently trains this other objective.
3. (VERIFIED, exact identity) Since sum_e P_e(x) = 1 for every row, sum_e grad P_e = 0 and therefore
   grad(surrogate) = E * sum_e (f~_e - k/E) grad Pbar_e: **the aux gradient signal is proportional to the
   imbalance f~ - k/E**, and a uniform f~ gives an identically-zero gradient (measured ||grad|| ~3e-8).
   Consequently the *relative* error of a noised f~ is ~ ||noise|| / ||f(B) - k/E|| and grows linearly
   with the noise (0.32 / 0.65 / 1.94 for rel 0.05 / 0.1 / 0.3 at balanced init; 3.3x more tolerant with the
   induced imbalance). The absolute perturbation of the *total* gradient is coef (0.001) x E x ||noise|| x
   ||grad Pbar||, which is tiny — but whether the noised aux term still *balances* depends on the
   imbalance-to-noise ratio, not on k/E.
4. (VERIFIED) A lagged f~ from another batch is only useful when the imbalance is persistent: at random
   init the batch-to-batch deviation of f(B) (0.08 k/E random, 0.32 structured) is the same size as the
   imbalance itself, so the lagged gradient error is 65-166%; with a persistent imbalance it drops to
   23-55%. Structured (topically clustered) batches make f(B) itself much noisier (E5).

---

## E5 — Load-vector statistics: per-example f(x) vs batch f(B) (1-layer model, B=32)

Mean |f_e(x) - f_e(B)| over examples and experts, in units of k/E; "iid-null" = same quantity after
randomly permuting token routes across examples (pure sampling-noise floor for the same f(B)).
Router scaling (x1 vs x5) changed nothing (scaling logits does not change argmax) — rows deduplicated.

| E/k | T | data | mean abs dev (k/E) | iid-null | ratio | max abs dev | batch-to-batch dev of f(B) | frac experts unused by an example | f(B)/(k/E) range |
|---|---|---|---|---|---|---|---|---|---|
| 8/2 | 64 | random | 0.41 | 0.17 | 2.4 | 1.54 | 0.12 | 0.0% | [0.79, 1.21] |
| 8/2 | 64 | structured | 0.81 | 0.17 | 4.9 | 2.94 | 0.19 | 16.4% | [0.77, 1.19] |
| 8/2 | 512 | random | 0.17 | 0.06 | 2.8 | 0.65 | 0.03 | 0.0% | [0.78, 1.27] |
| 8/2 | 512 | structured | 0.76 | 0.06 | 13.4 | 3.04 | 0.15 | 15.6% | [0.75, 1.64] |
| 64/8 | 64 | random | 0.53 | 0.25 | 2.1 | 2.87 | 0.17 | 5.2% | [0.27, 2.14] |
| 64/8 | 64 | structured | 1.09 | 0.25 | 4.4 | 6.79 | 0.31 | 42.7% | [0.32, 2.32] |
| 64/8 | 512 | random | 0.24 | 0.09 | 2.8 | 1.58 | 0.05 | 0.0% | [0.29, 2.24] |
| 64/8 | 512 | structured | 1.15 | 0.09 | 12.9 | 6.85 | 0.23 | 40.1% | [0.28, 2.03] |

Raw: `e5_results.json`.

**Reading.** (VERIFIED, tiny model) Even random-token examples deviate from the batch load vector by
2-3x the iid floor (routing is context-dependent, so tokens within an example are correlated); topically
structured examples deviate 5-13x and the deviation does *not* shrink with T (0.81 -> 0.76 at E=8;
1.09 -> 1.15 at E=64) whereas random examples shrink as ~1/sqrt(T). With E=64/top-8 and structured
T=512 sequences, ~40% of experts receive **zero** tokens from a given example (H5's "every expert is hit
by every example at T=1024" is REFUTED for topically clustered documents in this toy — for random
tokens it holds: 0% unused at T=512). This matters for two things: (a) per-example expert gradients are
genuinely sparse across experts for real documents, so the noise-vs-sparsity utility concern of H5 is real;
(b) a per-example aux loss (E4-(ii)) is heavily penalising exactly this natural within-document
specialisation.

## E6 — Sensitivity of the load vector for a DP release of f(B)

Each per-example load vector f(x) has entries in [0,1] and sum_e f_e(x) = k, hence ||f(x)||_2 <= sqrt(k)
(equality iff the example uses exactly k experts). Measured maxima confirm the bound is tight-ish for
structured data (E=64: max ||f(x)||_2 = 2.26 vs bound 2.83; random: 1.25-1.43).

Under add/remove adjacency, releasing S = sum_x f(x) with the Gaussian mechanism (sensitivity sqrt(k)) and
dividing by the public expected batch size B gives per-coordinate noise std sigma * sqrt(k)/B. In k/E
units: E/(B sqrt(k)) * sigma -> **E=64, k=8: 0.177 sigma at B=128, 0.022 sigma at B=1024**; E=8, k=2:
0.044 sigma at B=128. (Arithmetic, VERIFIED; the Gaussian-mechanism calibration itself is standard, see
Dwork & Roth / Balle & Wang, PLAUSIBLE as applied here, not re-derived.) Comparing with E4: at
sigma ~ 1 and B=128, the surrogate's aux gradient would be ~35-65% off at the induced-imbalance level
and >100% off at balanced init; at B=1024 or with an EMA over ~64 steps (noise / 8) it is a few percent.
Measured ||f(B) - k/E||_2/(k/E) for the toy: 0.37-0.79 (E=8), 3.2-4.1 (E=64, i.e. random init with 64
experts is already quite unbalanced per batch).

---

## Summary table (what each hypothesis got)

| Hyp. | Verdict (this toy) | Key number |
|---|---|---|
| H1 (only aux is batch-coupled; vmap exact) | **VERIFIED** | fp32 Opaque vmap vs HF loop 3e-7; HF batched vs per-example 4.5e-7 |
| H2 (constant-f surrogate is exact; per-example aux differs) | **VERIFIED** (exactness 2e-7; per-example aux cos 0.26-0.39); DP-release cost/utility trade-off characterised: error ~ ||noise||/||f - k/E|| | E4, E4b, E6 |
| H3 (route flips explain residual drift; pinning removes it) | **VERIFIED for router/expert grads** (12.9%/11.1% -> 1.7%/1.5% when pinned); **minor for attention-only** (1.34% vs 1.13%) | E1b |
| H4 (aux negligible at 0.001 for attention-only LoRA) | **VERIFIED** (norm change < 0.1%); aux does reach attention params (attn-only aux grad ~ CE grad size at coef 1) | E3 |
| H5 (every expert hit by every example at long T) | **REFUTED for structured docs** (40% experts unused at T=512, E=64/k=8), holds for random tokens | E5 |

## Caveats, in order of importance

1. Random-init tiny models: routing is near-tie (median top-1 prob 0.15), aux is near its minimum,
   expert branch is small. Flip *rates* (~1%/layer) and the *linearity* of noise error are structural and
   should transfer; absolute norms, f(B) ranges, and how much the surrogate matters at coef 0.001 do not.
2. "Structured" = 6-token vocabularies per example — an extreme of topical clustering.
3. `attn` partition is full q/k/v/o weights, not LoRA adapters; per-example norms of LoRA grads will differ.
4. CPU only: the Triton fused MoE / grouped-GEMM CUDA paths of `opaque_moe` were not exercised; the dense
   `Opaque_MoE` fallback was. The HF side used `grouped_mm` on CPU.
5. E6's noise-scale arithmetic assumes add/remove adjacency and Poisson/expected batch normalisation;
   Opaque's `clipped_grad` documents the same convention (`_clipped_grad.py:137-140`).
