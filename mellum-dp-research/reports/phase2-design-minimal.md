# Phase 2 — design-minimal: the smallest correct change to Opaque for a DP-faithful Mellum2 objective

Agent: `design-minimal`. Lens: minimal-change engineering (patches + trainer; engine untouched).
Inputs: BRIEF.md, PHASE1-DIGEST.md (F1–F11, G1–G10), the five phase-1 reports, the critic report,
`.junie/differential-privacy-review.md`. New numerics for this note (CPU, ≤ 1 min each) live in
`scratchpad/research/design-minimal/cost_table.py` → `cost_table.out` (run with `uv run python`).
Tags: **VERIFIED** = read the cited lines / ran the cited script; **PLAUSIBLE** = derived or read,
not executed end to end. Repo paths relative to `/home/user/opaque`; HF paths under
`.venv/lib/python3.11/site-packages/transformers/`.

One-paragraph summary. Mellum2's HF objective has exactly one batch-coupled term, the Switch-style
load-balancing loss; its gradient is *exactly* a sum of per-example gradients once the batch load
vector `f(B)` is held constant (F3). The design turns that constant into a DP output: every step the
per-example load histogram `h_x ∈ [0,1]^64` rides in the clipped pytree as a second `PerGroup` group
via a zero "probe" parameter (F7), is noised by the *same* Gaussian / matrix mechanism as the
gradient (accountant literally unchanged — Mahalanobis budget, F6), and the noised aggregate is
post-processed (renormalised, smoothed) into the public constant `f̃_{t+1}` that the next step's
per-example loss uses. Routing runs in fp32 (pretraining-faithful; removes the bf16 tie flips). The
router statistics are obtained through HF's own output recorder from the backbone while keeping
Opaque's chunked cross-entropy, so no full-vocab logits are materialised. Cost at the preset regime
with the recommended budget split ρ = 0.1: gradient noise ×1.049, ε unchanged at 3.00, load
relative error 16.5 % per step → 2.6 % after EMA(β = 0.95) under DP-SGD, → ≈ 0.4 %·(nm_MF/0.5622)
after a 256-step window mean under band-MF.

---

## 1. Objective (G1)

### 1.1 Decision: target (a), the logical-batch HF pooling (Fact A), realised per example

Per step `t`, public constant `f̃_t ∈ ℝ^E` (`E = 64`, `Σ_e f̃_{t,e} = k = 8`), per-example loss

```
ℓ_x(θ; f̃_t) = CE_x(θ) + α · E · Σ_e ( f̃_{t,e} − k/E ) · P_e(x; θ)        [+ ⟨z, s·h_x(θ)⟩ , value 0]

P_e(x; θ) = (1 / (L·T_x)) Σ_{l=1..L} Σ_{t} m_{x,t} · softmax_fp32( z^l_{x,t}(θ) )_e
h_{x,e}(θ) = (1 / (L·T_x)) Σ_{l}      Σ_{t} m_{x,t} · 1[ e ∈ topk_k( z^l_{x,t}(θ) ) ]      (= f_e(x))
```

with `m` the **attention mask** (prompt tokens with label −100 still count, as in HF
`modeling_mellum.py:694-697`, VERIFIED), `T_x = Σ_t m_{x,t}`, `L = 28`, `z^l_{x,t}` the router logits
of layer `l`, `CE_x` the per-example token-mean CE over label-valid tokens (Opaque's existing
convention, `_dp_trainer.py:2400-2402`; SFT `_nll.py:15-24, 86-87` per phase-1). The bracketed probe
term is the mechanism carrier (Section 2); `z` is a zero parameter, so it contributes nothing to the
loss value or to `∇_θ`.

Why the centred form `f̃ − k/E`: `Σ_e P_e(x) = 1` for every example, so
`Σ_e (k/E) ∇P_e = 0` and the centred and uncentred surrogates have **identical gradients**
(F3, `phase1-math.md` §1.1, VERIFIED to 1e-17); the centred form only removes the constant `α·k`
from the loss value and makes the "signal is the imbalance" fact visible in code.

Faithfulness claim (VERIFIED, `phase1-math.md` §1.2, `phase1-empirical.md` E4 rel-err 2–3e-7):
for equal-length examples (the packed T = 1024 presets) and `f̃_t = f(B_t)`,
`(1/B) Σ_{x∈B_t} ∇ℓ_x = ∇[ CE_tokenmean(B_t) + α · L_aux^HF(B_t) ]` exactly, where `L_aux^HF` is
`load_balancing_loss_func` (`modeling_mellum.py:540-606`, product of layer-pooled means,
`Σ_e f_e = k`). With `f̃_t` a lagged DP estimate the aux gradient is the exact gradient of the same
functional form evaluated at a nearby load vector (Section 2.5 quantifies "nearby").

### 1.2 Why (a) and not (b)/(c)/(d)

| candidate | what it is | verdict |
|---|---|---|
| (a) logical-batch Fact A | HF formula on the whole batch | **chosen** — it is the documented loss of the checkpoint's HF class, and F3 makes it exactly per-example-separable. |
| (b) HF-Trainer-realised | per-*microbatch* `f`, coefficient effectively `G·α` (critic Exp A, VERIFIED rel-L2 0.0) | rejected — an accumulation artefact of `trainer.py:1961-1963`, not an objective; for the presets it would mean `f` on 8 sequences and `α_eff = 0.032`. The DP path is *more* faithful to Fact A than HF Trainer is. |
| (c) Megatron per-layer running average (pretraining) | mean over layers of per-layer `E Σ f_e^l P_e^l` with a lagged `f` | rejected as the *target* (different pooling: mean of products vs product of means; per-layer `f` costs √L = 5.3× more noise, `phase1-math.md` §3); **adopted in spirit**: our `f̃_t` is also a lagged running estimate, which the Mellum2 tech report documents as what pretraining actually used (`phase1-literature.md` §F.2, arXiv:2605.31268 §3.6, VERIFIED). |
| (d) per-sequence aux | `f(x)` from the example's own tokens | rejected as a stand-in — different regulariser (anti-specialisation; cos 0.26–0.39 to the batch gradient, E4), even though DeepSeek-V2/V3 define it as a first-class variant. Offered later as an explicit option, never as "faithful". |

### 1.3 Remaining G1 sub-decisions

- **Mask**: attention mask (HF). SFT prompt masking does not remove prompt tokens from `f`/`P`.
- **CE weighting**: equal example weights (Opaque's convention; coincides with HF token weighting
  iff `T_x ≡ T`, true for the presets). The public-`N̄` token-weighted variant
  (`phase1-divergence.md` §4) is representable but **not built** in v1.
- **Coefficient default**: `router_aux_loss_coef = None` → read `model.config.router_aux_loss_coef`
  (= 1e-3, the value an HF-Trainer user with `output_router_logits=True` gets;
  `configuration_mellum.py:103`, VERIFIED). The two Mellum2 presets set **1e-4**, the value
  JetBrains used for Mellum2's own SFT "since the router is already well-balanced after
  pre-training" (`phase1-literature.md` §F.2, VERIFIED from the tech report). The feature itself is
  **opt-in** (`router_load_balancing=False` by default) so existing runs are bit-identical.
- **`f` from executed routes**: yes, by construction — `h_x` is computed from the *same* fp32
  router logits tensor the router used (Section 3), with `topk(softmax_fp32(·))` = the router's own
  op on the same input. HF's aux instead recomputes top-k from a bf16 softmax (critic R9:
  0.03–1.7 % of tokens differ) — a documented departure from HF *in favour of* the executed routes.
- **Per-layer vs pooled**: pooled `E`-vector (HF). Per-layer is diagnostic only.
- **Initial value**: `f̃_0 = k/E · 1` → the aux gradient is identically zero at step 0 and
  switches on as the first releases arrive (Section 2.4).

---

## 2. Mechanism and accountant (G4)

### 2.1 Per-step mechanism under DP-SGD / Poisson (VERIFIED seams; numbers from `cost_table.out`)

Notation: `C` gradient bound (preset 0.9), `λ = ρ·C` load-group bound, `s = λ/√k` probe scale,
`B̄` expected batch (256 = `normalize_by`, `_dp_trainer.py:1394, 4289`), `nm` noise multiplier.

1. **Sample** `B_t` by Poisson with rate `q = B̄/N` (unchanged).
2. **Augment** (once per step, outside vmap, `_augment_inputs`, `_dp_trainer.py:2295-2312`):
   copy the public `f̃_t` into the loss closure's device tensor; assert the probe leaf
   `trainable_params["router_load_probe"]` is zero.
3. **Per-example gradient** of `ℓ_x` w.r.t. the trainable dict *including* the probe leaf:
   `∂ℓ_x/∂z = s·h_x` (detached inside the loss; F7 VERIFIED end to end at toy scale),
   `∂ℓ_x/∂θ` = CE + surrogate.
4. **Per-group clipping** (`per_group(trainable, router_load_probe=λ, fallback=C)`,
   `_per_group.py:44-75`; `_clip_pytree_per_group`, engine): gradient group scaled by
   `min(1, C/‖g_x‖)`; probe group bound `λ` is **structural** — `0 ≤ h ≤ 1`, `Σ_e h_e = k` ⇒
   `Σ_e h_e² ≤ k` ⇒ `‖s·h_x‖₂ ≤ s√k = λ` — so the probe group is never rescaled and the release is
   unbiased (math §4(b) remark (ii); critic C6). Sum over `B_t`, divide by `B̄`.
5. **Noise** (`gaussian_noise`, `_gaussian.py:320-325` → `per_group_noise_stddev`,
   `noise_allocation.py:44-110`):
   `σ_g = nm·√(C(C+λ))/B̄` on every gradient leaf, `σ_h = nm·√(λ(C+λ))/B̄` on the probe leaf.
   Mahalanobis check `(C/B̄)²/σ_g² + (λ/B̄)²/σ_h² = 1/nm²` holds with equality
   (`cost_table.out` column "1/nm² chk" = 1.0000 for every ρ; VERIFIED with the real function).
6. **Release** = the noised pytree `(ĝ_t, ẑ_t)`; this is the *only* DP output of the step.
7. **Post-processing** (public, Section 2.4): `f̂_t = ẑ_t / s`, then filter → renormalise →
   `f̃_{t+1}`.
8. **Accountant**: unchanged — `poisson(gaussian(nm), q) * T` exactly as `_build_mechanism`
   builds it today (`_dp_trainer.py:4310-4392`); `num_groups` is only consulted by the adaclip
   wrapper, which this design excludes.

Why step 8 is correct (VERIFIED chain): the per-example contribution to the concatenated statistic
`[g; s·h]` is bounded per group by `(C, λ)`; an anisotropic Gaussian with block stddevs `(σ_g, σ_h)`
is, after whitening, a sensitivity-`√(C²/σ_g² + λ²/σ_h²)` = `1/nm` Gaussian (Dong–Roth–Su 2019
Thm 2.7, https://arxiv.org/abs/1905.02383, critic R6 VERIFIED; the same form as Andrew et al. 2021
Thm 1, https://arxiv.org/abs/1905.03871, `z_Δ = (z⁻² − (2σ_b)⁻²)^{-1/2}`); Poisson subsampling of
that single Gaussian is the exact fast path in `poisson.rs:27-29` (Feldman–Shenfeld
arXiv:2602.17284 Lemma 3.2/Thm 3.3, critic R6 VERIFIED). The two releases share the sampling coin,
which is exactly why they must be one joint Gaussian and not two composed subsampled mechanisms
(`phase1-math.md` §4(c); `phase1-literature.md` §G.2).

### 2.2 Per-step mechanism under DP-FTRL / band-MF / b-min-sep

Steps 1–4 as above with `B_t` from `BMinSepSampler` (`_b_min_sep.py:30-`; per-iteration `p`
derived from `p₀` so `E|B_t| ≈ B̄` is constant, `_b_min_sep.py:6-11`, VERIFIED read) — the load
statistic is computed on the **same** batch (no second draw; F6). Then:

5'. **Noise** (`mf_gaussian_noise`, `_mf_gaussian_noise.py:163-167`): base per-group σ from the same
   `per_group_noise_stddev(max_norm, nm)`; the correlated noise `C⁻¹Z` is applied leaf-wise to the
   *whole* pytree, probe leaf included; realised per-step σ on the probe leaf =
   `base σ · ‖row_t(C⁻¹)‖` (`:186-188`). The latch `_validate_constant_max_norm` accepts the
   `PerGroup` because `(C, λ)` and the group map are constant for the run (primitives E2, VERIFIED
   for identity and band-MF at toy scale).
6'. **Release** = the noised stream row `(ĝ_t, ẑ_t)`.
7'. **Post-processing**: window mean over `W` steps (Section 6), renormalise, `f̃_{t+1}`.
8'. **Accountant**: unchanged — `b_min_sep(mf_gaussian(nm, band_mf_strategy(bands=64)), n_steps, p0)`
   (`_dpftrl.py:151-159`; `_mf_gaussian.py:118-128` folds `strategy.sensitivity(n_steps, min_sep,
   max_participations)` into the multiplier). Correctness: Denisov et al. 2022 Thm 2.1
   (https://arxiv.org/abs/2202.08312; rows chosen adaptively — `f̃_t` is post-processing of earlier
   rows, so it is exactly the adaptive-rows clause) applied to the per-group-whitened stream; the
   participation-pattern sensitivity is homogeneous of degree 1 in the row bound, so per-group
   whitening gives the same PLD as the scalar case (`phase1-math.md` §5(i)). The probe group shares
   the gradient's participation pattern (same example, same step), so the same `min_sep` /
   `max_participations` apply.

`normalize_by` consistency (critic M11): the trainer sets `normalize_by = expected_batch_size =
a.train_batch_size` (`_dp_trainer.py:1394`), and the b-min-sep sampler is constructed to keep the
expected batch at that value; the realised `Σ_e f̂_{t,e} = k·|B_t|/B̄` is corrected by the public
renormalisation in 2.4, never by the private `|B_t|`.

### 2.3 Adjacency and sensitivities

- Protected unit: one training example (one packed sequence); **add/remove** adjacency (repo
  default, `differential-privacy-review.md` "Adjacency"; `_clipped_grad.py:135-145`). Under
  replace-one both group bounds double: `(2C, 2λ)`; the structural bound becomes `2λ` (two
  histograms can differ by at most `‖h_x − h_x'‖ ≤ ‖h_x‖ + ‖h_x'‖`). Nothing in the code changes —
  the doubled bound is the documented `clipped_fun` contract (`_clipped_fun.py:520-530`).
- Per-example L2 sensitivities: gradient group `C` (clipping); probe group `λ = s√k` (structural,
  attained only by an example whose every token in every layer routes to the same `k` experts —
  `phase1-math.md` §3; measured max 2.26/2.83 on structured toy data, E6). Centred variant
  `√(k(1−k/E))·s` is tighter by 6.5 % (2.646 vs 2.828); v1 uses the plain bound (simpler; the
  probe carries `s·h`, not `s·(h − k/E)`).
- No clipping of the probe ⇒ no bias; `ClippedGradAux.group_norms["router_load_probe"] ≤ λ`
  always (test T8).

### 2.4 Post-processing (all public, all linear except the renormalisation)

Given the noised probe leaf `ẑ_t` (rank-identical under DDP because the noise key is shared,
`_dp_trainer.py:1478`, critic R4 VERIFIED):

```
f̂_t   = ẑ_t / s                                     # (1/B̄) Σ_{x∈B_t} h_x + N(0, (σ_h/s)² I)
u_t   = Filter(f̂_t, state)                          # DP-SGD: EMA  u_t = β u_{t-1} + (1-β) f̂_t   (β = 0.95)
                                                     # MF:     window mean over the last W steps (W = 256)
f̃_{t+1} = k · u_t / Σ_e u_{t,e}    if Σ_e u_{t,e} > k/2  else  f̃_t   # renormalise: fixes k·|B_t|/B̄ and lag
```

- The renormalisation uses the **noisy** sum (public); it must never use `aux.batch_size`
  (private, un-noised).
- No clamp on the loss path: `f̃` enters the loss linearly, so a slightly negative entry for a cold
  expert is mathematically fine (it pushes `P_e` up — the right direction) and clamping would bias
  the estimate; clamp to `[0, 1]` only for the *logged* summary.
- Lagged use is free (adaptive composition: Zhu–Dong–Wang Thm 10, https://arxiv.org/abs/2106.08567,
  critic R6 VERIFIED; the dependence of `ℓ_x` on `f̃_t` is the same kind as its dependence on
  `θ_t`; `phase1-math.md` §4(d)).
- Noise variance of the filtered estimate: EMA `(1−β)/(1+β)` × single-release
  (`√` factors 0.229 / 0.160 / 0.071 at β = 0.9 / 0.95 / 0.99, `cost_table.out`); MF window factors in
  Section 6.

### 2.5 THE cost table — preset regime `nm = 0.5622, B̄ = 256, k = 8, E = 64, C = 0.9, T = 15625, q = 5.12e-4, δ = 1e-6`

Baseline `poisson(gaussian(0.5622), q)·T` → **ε = 3.0004** (Opaque accountant, `cost_table.out`).
Single-release relative error of one entry of `f̂` at `σ_h = nm` on `f_e ≈ k/E`:
`r₀ = nm·E/(B̄√k) = 4.97 %`. The probe route gives `r = r₀·√(1 + 1/ρ)`.

| ρ = λ/C | λ | gradient-noise inflation √(1+ρ) | r per step | r after EMA β=.95 | r after EMA β=.99 | ε with the probe route (accountant unchanged) | equivalent load multiplier σ_h/(λ/B̄) = nm√(1+1/ρ) | ε **if nm were held** on the gradient and the load released same-batch at that multiplier |
|---|---|---|---|---|---|---|---|---|
| 0.05 | 0.045 | **1.025** | 22.8 % | 3.6 % | 1.6 % | **3.000** | 2.576 | 3.417 |
| **0.10** | **0.090** | **1.049** | **16.5 %** | **2.6 %** | **1.2 %** | **3.000** | 1.865 | 3.703 |
| 0.25 | 0.225 | 1.118 | 11.1 % | 1.8 % | 0.8 % | 3.000 | 1.257 | 4.453 |
| 0.50 | 0.450 | 1.225 | 8.6 % | 1.4 % | 0.6 % | 3.000 | 0.974 | 5.443 |

(All columns VERIFIED with `per_group_noise_stddev` and `poisson(gaussian(nm)|gaussian(σ_h), q)*T`
in `cost_table.py`; the two rightmost columns are the *same mechanism family* re-parametrised —
critic C7 — shown once so a reader can choose "pay in gradient σ" or "pay in ε".) **Default ρ = 0.1.**

Signal-to-noise reading (E4: surrogate gradient error ≈ ‖noise‖₂/‖f(B) − k/E‖₂): with
`r = 2.6 %` per entry, `‖noise‖₂ = 0.026·(k/E)·√E = 0.026`; the signal is
`‖f(B) − k/E‖₂ = δ_rel` (with `δ_rel` the rms relative imbalance), so the relative gradient error is
`≈ 0.026/δ_rel`: 26 % at `δ_rel = 10 %`, 9 % at 30 %. The absolute perturbation of the total
gradient is `α·E·‖noise‖·‖∇P̄‖` — negligible at `α = 1e-4`. `δ_rel` on KStack is the G3 number
that decides whether the term carries signal (Section 10).

Under band-MF the same table applies with `nm → nm_MF` (the trainer-calibrated multiplier of
`mf_gaussian`, which this box could not calibrate: one `b_min_sep` MC-PLD evaluation at
`n = 15625` exceeded 170 s) and the per-step factor `‖row_t(C⁻¹)‖ ≈ 3.80` replaced by the filter
factors of Section 6: at ρ = 0.1, `r_MF,window256 ≈ 16.5 % × 0.0216 × (nm_MF/0.5622) ≈ 0.36 %·(nm_MF/0.5622)`.

---

## 3. Router precision and routing (G5)

**Decision: fp32 router logits by default for the Mellum family; routes from the current model
inside vmap; no frozen-base pinning; `f` from the executed routes.**

- New patch `_make_fp32_router_forward(original)` for `MellumTopKRouter`
  (`modeling_mellum.py:323-341`, VERIFIED): `router_logits = F.linear(h.float(), W.float())` (fp32),
  `probs = softmax(router_logits)` (already fp32 upstream), `topk`, renormalise, cast **scores** back
  to `hidden_states.dtype` (so the expert kernel's inputs are unchanged), return
  `(router_logits_fp32, scores, indices)`. Everything downstream (`opaque_moe`, LoRA, experts) is
  untouched. Cost: `64 × 2304` MACs/token/layer in fp32 vs `≈ 4.95e7` MACs/token/layer for the
  routed experts — < 0.3 % (arithmetic, PLAUSIBLE).
- Why default-on: Mellum2 was pretrained with an FP32 router (tech report appendix, VERIFIED in
  `phase1-literature.md` §F.2), so fp32 is the pretraining-faithful routing; bf16 logits have fixed
  *relative* resolution so ~1 %/layer of tokens flip regardless of router sharpness (E2), and 4–7 %
  of tokens sit on an exact bf16 tie at the k/k+1 boundary (critic R9) — those flips are the source
  of the 8× excess gradient error on router/expert parameters (E1b) and of the vmap-vs-oracle
  drift when the two paths round differently (F8). The patch moves Opaque *away* from the HF-bf16
  path and *towards* the checkpoint's training router; Section 10 defines the oracle with the same
  patch installed so the comparison isolates vmap.
- Opt-out: `performance_kernels_config={"router_fp32": False}` (the gate lives in the `compat`
  bucket of `make_apply_model_patches`, default taken from a new per-family
  `router_precision="fp32"` declaration; other families keep the HF forward).
- Routing pinning: not needed for privacy (both variants are per-example functions,
  `phase1-math.md` §6) and frozen-base pinning would freeze training-time routes while inference
  routes drift (train/inference mismatch). With fp32 logits the remaining discontinuity is genuine
  near-ties at fp32 resolution, which the route-flip counter (Section 10) measures.
- `f` from executed routes: the statistics code calls `topk(softmax(router_logits))` on the *same*
  fp32 tensor the router used — a deterministic op on identical input gives identical indices —
  and the parity test (T2) additionally compares against the router's returned `indices`.
- Inference-time routing (vLLM etc.) may still run in bf16; that mismatch is a property of the
  checkpoint (Mellum2 TR §5.2, VERIFIED in `phase1-literature.md` §E.1) and is out of scope; the
  optional z-loss (Section 7) is the standard mitigation.

---

## 4. Clipping norm and per-example gradient norms

What the design does so that a reasonable `C` exists, and how to pick it:

1. **The load vector never competes with the gradient for the clipping budget.** That is the
   reason for the separate `PerGroup` group (2.1 step 4) rather than the joint-clip option 4(a) of
   the math note, which would shrink the gradient's admissible norm to `C√(1−ρ²)` and bias the
   histogram. With the probe group, `C` is chosen exactly as today.
2. **The aux term does not move the per-example norm.** At `α ∈ [1e-4, 1e-3]` the per-example
   gradient norm changes by < 0.1 % (E3, toy VERIFIED; the unscaled aux gradient is of CE size,
   so the bound is `α × O(1)`), and at balance its gradient is identically zero (F3). `C` is
   governed by CE.
3. **fp32 routing removes the route-flip variance** from router/expert gradients (E1b: 12.9 %/11.1 %
   → 1.7 %/1.5 %) — relevant when those parameters are trained (Section 7); for the attention-only
   presets it is +0.2 pp (E1b), i.e. `C` is unaffected either way.
4. **Fixed clipping for v1**; `clipping_mode="fixed"` is what both presets use (critic R11). AUTO-S
   on the gradient group is MF-compatible but `auto_scale_pytree` applies `R/(‖·‖+γ)` per group
   *without* `min(1, ·)` (`_auto.py`, primitives §1.4), which would rescale every example's load
   vector to norm ≈ λ and bias the release; the trainer therefore raises `ConfigurationError` when
   `router_load_balancing=True` and `clipping_mode != "fixed"`. Adding a per-group "fixed" override
   to `auto_scale_pytree` is the v2 engine change if AUTO-S is wanted with the feature.
5. **How to pick `C`** (G3 script, Section 10): one pass of `clipped_grad(..., clipping_norm=1e9,
   return_aux=True)` over 256 KStack examples under the preset partition (LoRA r = 16 q/k/v/o)
   gives `aux.grad_norms`; set `C` at the p50–p60 quantile (clip ≈ 40–50 %, the usual DP-SGD
   operating point; toy distributions are tight, max/median 1.24–1.33, E3). Report p10/p50/p90/p99.
   If experts are later trained, use `per_group(trainable, experts=C_e, gate=C_r, fallback=C_a,
   router_load_probe=λ)` — `per_group` already supports arbitrary substring groups
   (`_per_group.py:44-75`).
6. `normalize_by = B̄` keeps the released sensitivity `C/B̄` (`_clipped_grad.py:215-218`) so the
   optimizer sees a batch-mean gradient, as today.

---

## 5. Performance (G6)

### 5.1 Dense vs grouped MoE default

Facts (VERIFIED, critic R3): `DPTrainer` passes `kernels=bool(args.use_performance_kernels)`
(`_dp_trainer.py:841-847`), default `False` (`_training_arguments.py:436`); the factory sets
`grouped_moe = kwargs.get("grouped_moe", kernels)` (`_factory.py:316-324`); `grouped=False` selects
the dense every-token-through-every-expert `Opaque_MoE` on every host including CUDA; the flag is
captured by the first class-level patch per process (`_router.py:59-92`).

Cost arithmetic at Mellum2 shape (PLAUSIBLE — not measured; G6 measurement in Section 10):
per token per layer the experts cost `E·(2I·H + I·H) = 64 × 6.19e6 ≈ 3.96e8` MACs dense vs
`8 × 6.19e6 ≈ 4.95e7` grouped (8×). Per example (T = 1024, L = 28): 22.7 TFLOP dense vs 2.8 TFLOP
grouped forward; with frozen experts the backward skips weight grads (`moe.py:536-539`) so
≈ 2× forward. Per 256-example step ≈ 12–17 PFLOP dense vs 1.5–2 PFLOP grouped: on one H100 at
~200 TFLOP/s effective that is roughly 60–90 s/step vs 8–10 s/step, i.e. 11–16 days vs ~2 days
for 15 625 steps. The dense default is the single largest practical blocker for the presets.

**Decision: Mellum defaults to the grouped path whenever it is available**, independent of
`use_performance_kernels`: `grouped_moe = kwargs.get("grouped_moe", kernels or
_grouped_moe_available())` in `_factory.py`, where `_grouped_moe_available()` is true on CUDA +
Triton (fused kernel) or when `torch._grouped_mm` exists (CUDA/MPS/CPU with ≥ 16 experts, the
existing `use_grouped_route` gate in `kernels/moe.py:575, 624-632`). Justification: the component
docstring already states all paths are numerically equivalent (`components/moe.py:11-16`), and the
divergence report measured grouped-vs-HF bf16 4.63e-3 against dense 4.98e-3 (both inside HF's own
batched-vs-loop spread; fp32 both 3.5e-7, VERIFIED). Documentation: mention MoE in the
`use_performance_kernels` docstring (`_training_arguments.py:429-435`) and the first-patch capture
in the Mellum patch docstring. The dense path stays the compat fallback (`grouped_moe=False`).

### 5.2 Memory of the new buffers (per microbatch of 8, T = 1024, L = 28, E = 64, k = 8)

| buffer | shape / dtype | size | note |
|---|---|---|---|
| router logits tuple (recorder) | 28 × (T, E) fp32 refs | 0 extra | already live in the autograd graph (softmax → top-k weights); the recorder stores references, not copies (`output_capturing.py:104-115`) |
| probabilities for `P_e(x)` | (T, E) fp32 per layer, reduced immediately | 0.26 MB transient | computed layer by layer in a Python loop over the 28 tensors, summed into an (E,) accumulator |
| one-hot counts | (T, k, E) bool per layer, reduced immediately | 0.5 MB transient | broadcast compare `idx[..., None] == arange(E)` (vmap-safe; `F.one_hot` is not, F7) |
| probe leaf | (E,) fp32 | 256 B | + its per-example gradient (B, E) inside `clipped_grad` |
| filter state | EMA (E,) or window ring (W, E) fp32 | ≤ 64 KB at W = 256 | public |
| fp32 router logits | (T, E) fp32 instead of bf16 per layer | +0.26 MB/layer/example ≈ +7 MB/example | kept for backward through softmax (already fp32 upstream) |

Nothing here approaches the chunked-CE savings (full logits would be `1024 × 98304 × 4 B = 400 MB`
per example, which PR #978 removed and which this design keeps removed — Section 9.2).

---

## 6. Smoothing under MF (G7)

Computed for `band_mf_strategy(bands=64)` at the preset horizon `n_steps = 15625` (Opaque's
`optimize_toeplitz`, 55 s CPU; VERIFIED, `cost_table.out`): column-normalised coefficients
(`sensitivity = ‖c‖ = 1.0000`, `_band_mf.py:137-139`), `c₀ = 0.322`, `c₁ = 0.200`, `c₆₃ = 0.0855`;
`‖row_t(C⁻¹)‖` rises from 3.11 (t = 0) to its plateau **3.80** by t ≈ 512. So a per-step consumer
of the noised probe leaf sees 3.8× the base σ under band-MF — the same fact primitives E2 saw at
bands = 4 (0.1716 vs 0.1443).

Noise factor (in units of the base σ on the leaf) of linear filters applied to the noised stream
`x_t = leaf_t + σ_base·(C⁻¹z)_t`, computed as `‖(w * C⁻¹)_{row t}‖₂` with `w` the filter's Toeplitz
coefficients (steady state, t = 15624):

| filter | noise factor under band-MF(64) | iid (DP-SGD) equivalent | MF / iid |
|---|---|---|---|
| single step | 3.80 | 1.00 | 3.8× worse |
| EMA β = 0.95 | 0.185 | 0.160 | ≈ parity |
| EMA β = 0.99 | 0.038 | 0.071 | 1.9× better |
| window mean W = 64 (= bands) | 0.067 | 0.125 | 1.9× better |
| **window mean W = 256** | **0.0216** | 0.0625 | **2.9× better** |

Reading. The band-MF strategy is optimised so that *prefix sums* `A·C⁻¹Z` (workload
`momentum = 1.0` by default, `_band_mf.py:60, 142-147`) have small error; a window mean is a
difference of two prefix sums divided by `W`, so it inherits that guarantee — an EMA with small β
does not (it is closer to a per-step consumer and loses to the anti-correlation structure at
β = 0.95). **Decision:** under MF the filter is the boxcar window mean with `W = 4·bands = 256`
(mean lag 128 steps; noise factor 0.0216); under DP-SGD the filter is the EMA with β = 0.95 (lag
≈ 20 steps; factor 0.160). Both are linear post-processing of released outputs — nothing changes in
the accountant; only utility (lag bias vs noise) is at stake. The window also removes the
"first-64-steps" transient (row norm 3.11 → 3.80) from the estimate.

Caveat (PLAUSIBLE): the strategy's `lr_schedule` weighting (`_dpftrl.py:110-113` passes the
schedule) changes `C` slightly; the factors above use the unweighted recipe. Re-run
`cost_table.py` with the trainer's materialised schedule if the numbers matter to 10 %.

---

## 7. Scope (G8)

| item | v1 | what would be needed |
|---|---|---|
| Causal-LM SFT via `DPTrainer` default path and `DPSFTTrainer` (`nll` / `chunked_nll`) | **in** | Section 9 |
| DP-FTRL (band-MF, BLT, BSR, BiSR, λ-CGD) with b-min-sep / Poisson / balls-in-bins | **in** | nothing beyond the window filter; all strategies go through `mf_gaussian_noise`'s `PerGroup` path |
| DP-DPO (`mellum2-codesec` preset, `_dpo_trainer.py:1065-1195`) | **out** | aux over the policy's chosen + rejected forwards: `P_e(x)` and `h_x` pooled over both sequences with denominator `L·(T_c + T_r)` (same `√k` bound); reference-model forward excluded (no gradient; routes from the policy); the two forwards already exist in `compute_per_example_loss_and_metrics`, so it is the same helper called twice plus the converter mapping. TRL itself never added the aux to DPO (`phase1-literature.md` C13). |
| Router z-loss (`L_z(x) = (1/(L·T_x)) Σ_{l,t} m·(logsumexp z^l_{x,t})²`, ST-MoE §3.1 eq. 5, https://arxiv.org/abs/2202.08906) | **optional flag, default 0** | per-token, per-example separable, zero privacy cost (it is inside the clipped per-example gradient); Mellum2 pretraining used `1e-3`; not part of the HF objective so off by default; recommended with a trainable router. |
| Experts / router trainable (PEFT `target_parameters` on stacked `experts.gate_up_proj`, or full router) | **out** | (i) the M5 smoke test (PEFT `ParamWrapper` under `functional_call` + vmap — untested, primitives §5); (ii) per-group `C` by parameter class; (iii) fp32 router becomes important (E1b); (iv) memory of the `(B, E, 2I, H)` per-example expert-weight gradient buffers (`≈ 0.53 GB` per example per layer, `phase1-divergence.md` §6) — grouped path with weight grads and microbatch 1–2. The probe + surrogate mechanism itself is partition-agnostic. |
| Per-sequence aux (DeepSeek-style) | **out** (documented alternative) | `ℓ_x += α E Σ_e h_{x,e} P_e(x)` — trivially separable, zero extra cost, but a different regulariser (F3). |
| Token-weighted CE with public `N̄` | out | one-line loss change + clip-norm rethink (`phase1-divergence.md` §4). |

---

## 8. Privacy hygiene (G9)

| tensor / state | where it lives | class | logged | checkpointed | DDP |
|---|---|---|---|---|---|
| fp32 router logits, probabilities, `h_x`, `P_e(x)` (per example) | inside the vmapped loss closure | **private-internal** — never leave `clipped_grad` | no | no | no |
| per-example probe gradient `s·h_x` | inside `clipped_grad` before summation | private-internal | no | no | no |
| `ClippedGradAux.group_norms["router_load_probe"]` (per-example `‖s h_x‖`) and `aux.loss_aux` | existing telemetry channel | **private, un-noised** (pre-existing posture for grad norms, F11) — the trainer must **exclude the probe group** from the logged `group_norms` mean; nothing about `f(x)` or `f(B)` is ever added to `loss_aux` | excluded | no | gathered (existing) |
| clipped-summed probe leaf, pre-noise (`grads.pytree["router_load_probe"]`) | `ClippedPytree` in `training_step` | **private** — same status as the un-noised gradient sum; only `noise_fn` may consume it | no | no | all-reduced by `sum_gradients_` (private sum, same as gradients) |
| noised probe leaf `ẑ_t` (`noisy_grads.pytree["router_load_probe"]`) | `NoisedPytree` handed to `on_pre_optimizer_step` | **public** (DP output) | may be logged as `router_load/raw_*` | via filter state | rank-identical (shared noise key, `_dp_trainer.py:1478`) |
| `f̂_t`, filter state (EMA vector or `(W, E)` ring), `f̃_t` | `RouterLoadState` on the built-in callback | **public post-processing** | yes: `router_load/max`, `/min`, `/entropy`, `/sum_raw`, `/noise_std = σ_h/s` | yes — sidecar `router_load_state.pt` (registry `state_dict`) | rank-identical by construction; optional `register_sync_type(RouterLoadState, assert_equal)` |
| probe parameter `z` | `ctx.trainable_params["router_load_probe"]` | public constant **0** | no | yes (a zero vector inside the model state; harmless) | identical |
| `aux.batch_size` = realised `|B_t|` | existing | private (pre-existing) — **never** used to renormalise `f̂` | (pre-existing) | no | summed (existing) |
| `α, λ, ρ, s, β, W, E, k` | args | public hyperparameters | yes | yes (args) | — |

Privacy statement addition (for `docs/user-guide/accounting.md` / the mechanism page): "With
`router_load_balancing=True` each step releases one noised vector consisting of the clipped
gradient sum and the scaled per-example router-load histogram sum; the two are one Gaussian
(matrix) mechanism with per-group bounds `(C, λ)` and the accountant is unchanged. The
load-balancing target `f̃_t` used by the loss is post-processing of previous releases." Optimizer
note: the built-in callback zeros the probe entry of `noisy_grads` in place after reading it, so
AdamW's update for `z` is exactly 0 (zero moments, zero weight-decay on a zero parameter) and the
probe stays 0 without per-step re-zeroing; `_augment_inputs` asserts it as belt-and-braces.

---

## 9. Implementation plan (G10)

Engine (`opaque-engine`, `opaque-dpsgd`, `opaque-dpftrl`, `opaque-accounting`): **no change**.
Reused as-is: `per_group` (`_per_group.py:44`), `PerGroup`, `clipped_grad` (`_clipped_grad.py:85`),
`per_group_noise_stddev` (`noise_allocation.py:44`), `gaussian_noise` (`_gaussian.py:320-325`),
`mf_gaussian_noise` (`_mf_gaussian_noise.py:163-188`), the serialization registry
(`opaque-base` `_dispatch.py`, `opaque-engine` `_structural.py`), `sum_gradients_`.

### 9.1 `opaque-patches`

| file | change |
|---|---|
| `src/opaque/api/patches/transformers/components/router.py` (**new**) | `_make_fp32_router_forward(original)` — Section 3. Pure torch, vmap-safe (no in-place on batched tensors except the upstream renormalisation on the fresh top-k tensor, which is already vmap-safe). |
| `src/opaque/api/patches/transformers/components/moe_stats.py` (**new**) | `router_load_and_probs(router_logits: Sequence[Tensor], attention_mask: Tensor \| None, *, top_k: int) -> (h, P)`; `load_balancing_surrogate(P, f_tilde, *, num_experts, top_k)` = `E·⟨f̃ − k/E, P⟩`; `router_z_loss(router_logits, attention_mask)`. All out-of-place, layer-loop reductions (Section 5.2). |
| `src/opaque/api/patches/transformers/_factory.py` | (a) new `router_factory` / `classes["router"]` / `router_precision` recipe args; gate `kwargs.get("router_fp32", compat and router_precision == "fp32")` in the compat bucket; (b) `grouped_moe = kwargs.get("grouped_moe", kernels or _grouped_moe_available())` (Section 5.1). |
| `src/opaque/api/patches/transformers/models/mellum.py` | `classes["router"] = "MellumTopKRouter"`, `router_precision="fp32"`; docstring: fp32 router, grouped default, first-patch capture. |
| `src/opaque/api/patches/transformers/components/cross_entropy.py` | new explicit kwarg `opaque_router_logits: bool = False` on the fused/chunked causal-LM `forward` (`:166-190`): when true, call the backbone with `output_router_logits=True` (`:239-252`) **without** taking the `output_router_logits` fallback branch (`:212-234`, which exists to preserve HF's *aux-loss* contract and is kept for that case). The existing `hasattr(outputs, "router_logits")` return path (`:359-370`) already forwards `router_logits` in a `MoeCausalLMOutputWithPast` with `aux_loss=None` and `logits=None`. |
| `src/opaque/api/patches/transformers/runtime/…` | nothing. |
| docs `docs/mechanisms/dp-sgd/…` + `docs/user-guide/huggingface.md` | mechanism page "MoE load balancing under DP" (loss, mechanism, cost table, privacy statement, primary sources: Switch eqs. 4–6, Andrew Thm 1, Dong–Roth–Su Thm 2.7, Denisov Thm 2.1, Mellum2 TR §3.6). |

How the statistics reach the loss *without hooks* (G10): HF's `MellumModel.forward` is decorated
with `@capture_outputs` (`modeling_mellum.py:475`) and `_can_record_outputs["router_logits"] =
OutputRecorder(MellumTopKRouter, index=0)` (`:432`); the recorder hook appends only while a
`ContextVar` collector is active, and that collector is set immediately before the backbone forward
and reset in a `finally` right after it (`output_capturing.py:266-272`, VERIFIED read). Hence:

- **gradient checkpointing**: the recompute during backward runs with the collector reset → no
  double append; the captured logits are ordinary autograd tensors created inside a non-reentrant
  checkpoint region (`_force_non_reentrant`, `patches/torch/checkpoint/huggingface.py:28-33`), so
  gradients flow through them (PLAUSIBLE — test T5 makes it VERIFIED);
- **microbatch chunks**: each chunk is a separate vmapped call → separate collector;
- **torch.compile**: HF uses a `CompileableContextVar` (`output_capturing.py:97`); if dynamo still
  breaks, `_compile_with_fullgraph_fallback` (`_dp_trainer.py:4195-4228`) downgrades to
  `fullgraph=False` — the same fallback the adaptive/auto clip paths rely on today;
- **batchify**: `_squeeze_output` squeezes only top-level tensor values (`functional/__init__.py:189-214`);
  `router_logits` is a tuple of `(T, E)` tensors (the router reshapes to `(-1, H)`,
  `modeling_mellum.py:333`) so nothing needs squeezing — no shim.

### 9.2 `opaque-transformers`

| file | change |
|---|---|
| `trainer/_training_arguments.py` | fields: `router_load_balancing: bool = False`; `router_aux_loss_coef: float \| None = None` (None → model config); `router_load_group_scale: float = 0.1` (ρ); `router_load_smoothing: dict \| None = None` (default `{"kind": "ema", "beta": 0.95}` for Gaussian, `{"kind": "window", "steps": 4*bands}` for MF); `router_z_loss_coef: float = 0.0`. Validation: feature requires `clipping_mode == "fixed"` and a model whose backbone can record `router_logits` (checked at setup, else `ConfigurationError`). |
| `trainer/_router_load.py` (**new**) | frozen dataclass `RouterLoadState(f_tilde: Tensor, filter: Tensor, count: int, kind: str, beta: float, window: int, top_k: int)` (registry-serialisable, primitives §4.4 E2); pure functions `initial_state(num_experts, top_k, …)`, `update(state, f_hat_raw) -> state` (Section 2.4), `summary(state) -> dict[str, float]` (clamped, for logging). Built-in `RouterLoadCallback(TrainerCallback)` with `on_pre_optimizer_step(self, args, state, control, *, grads, trainable_params, **kw)`: reads `grads.pytree["router_load_probe"]`, divides by `s`, calls `update`, then zeros that leaf in place (Section 8 optimizer note). Registered automatically by the trainer when the feature is on — reuses the existing `call_event("on_pre_optimizer_step", …, grads=noisy_grads, …)` seam (`_dp_trainer.py:2210-2218`) with **no new seam in `training_step`**. |
| `trainer/_dp_trainer.py` `_setup_training` | if enabled: (1) `self._model.register_parameter("router_load_probe", nn.Parameter(torch.zeros(E, dtype=float32, device)))` **before** `make_functional(partition_trainable=True)` (`:1357-1361`) so the leaf lands in `trainable_params` under path `("router_load_probe",)`; (2) extend the clip-norm block (`:1455-1469`): a scalar `clipping_norm=C` becomes `per_group(trainable, router_load_probe=λ, fallback=C)`, a dict gets the `router_load_probe` key added; (3) instantiate `RouterLoadCallback` and add it to the callback handler; (4) store `s`, `α`, `E`, `k` on the trainer. |
| `_dp_trainer.py` `_augment_inputs` (`:2295-2312`) | copy `callback.state.f_tilde` into `self._router_load_target` (a device tensor read by the loss closure — unbatched under vmap, a plain graph input under compile); assert `ctx.trainable_params["router_load_probe"]` is zero. (Alternative kept in reserve: a seeded `(E,)` batch column overwritten here, TR-DPO pattern `_dpo_trainer.py:721-748, 807-841`.) |
| `_dp_trainer.py` `compute_per_example_loss` (`:2314-2434`) and `trl/_sft_trainer.py` `compute_per_example_loss` (`:554-615`) | shared helper `self._apply_router_load_terms(loss, outputs, params, inputs)`: `h, P = router_load_and_probs(outputs["router_logits"], inputs.get("attention_mask"), top_k=k)`; `loss = loss + α·load_balancing_surrogate(P, self._router_load_target, …) + (params["router_load_probe"] * (s*h).detach()).sum() [+ ζ·router_z_loss]`. The forward is called with `opaque_router_logits=True` (marker detected by signature exactly like `_fused_forward_uses_marker`, `_sft_trainer.py:287-291`); the SFT `chunked_nll` path (`:576-590`) keeps `opaque_fused_loss_only=True` so chunked CE stays in force. |
| `_dp_trainer.py` metrics (`:2258-2293`) | add `summary(state)` keys; exclude the probe group from the logged `group_norms` mean (Section 8). |
| `_dp_trainer.py` `_save_checkpoint` (`:4910`, bundle at `:5091-5120`) / `_apply_runtime_state` (`:5335-5356`) / resume (`:1176`, `:5317`) | write/read the sidecar `router_load_state.pt` = `opaque_state_dict(state)` next to `DP_STATE_NAME`; `save_dp_runtime_state`'s fixed signature (`_checkpoint.py:311-381`) is left alone (zero bundle-version churn). |
| `trl/_convert.py` `_drop_router_aux_loss` (`:69-76`) and the SFT/DPO converters | for SFT on a family with `router_precision`/recorder support: map `router_aux_loss_coef>0` → `router_load_balancing=True, router_aux_loss_coef=value` (info log instead of the warning); DPO keeps the warning (Section 7). |
| `examples/train_dpftrl.py` `mellum2-kstack` (`:873-896`), `examples/train_dpo.py` `mellum2-codesec` | kstack: `router_load_balancing=True`, `router_aux_loss_coef=1e-4`, ρ = 0.1, window 256; codesec: unchanged (out of scope). |

DDP: nothing to add — `sum_gradients_` all-reduces every leaf including the probe
(`distributed/gradients.py:150-200`, primitives §2.3), noise is added per rank with the shared key
after the reduction (`_dp_trainer.py:2172-2183`), so `ẑ_t`, the filter state and `f̃` are
bit-identical on all ranks.

### 9.3 Test plan (placement per ARC-006; markers per AGENTS.md)

`packages/opaque-patches/tests/transformers/models/test_mellum.py` (+ new
`test_moe_router_stats.py`, tiny random-init models via `build_moe_model`):

- T1 fp32 router: `router_logits.dtype == float32`, scores dtype = hidden dtype, indices equal to
  the unpatched router on an fp32 model; works under `vmap` and `vmap(grad)`.
- T2 `router_load_and_probs`: vmap-safe; `Σ_e h = k`, `0 ≤ h ≤ 1`, `‖h‖₂ ≤ √k`, `Σ_e P = 1`;
  equals an eager per-example loop; padded tokens excluded; `h` from the logits equals `h` from the
  router's returned `indices`.
- T3 surrogate identity (float64, equal lengths): `Σ_x ∇ℓ_x|_{f̃=f(B)} == ∇ load_balancing_loss_func`
  to 1e-12 relative; ragged lengths with the `T_x/T_tot` reweighting.
- T4 chunked-CE forward with `opaque_router_logits=True`: returns 28 `(T, E)` router logits under
  vmap, `logits is None`, loss equals the same forward without the kwarg; with
  `output_router_logits=True` the HF-aux fallback is still taken (existing
  `test_fused_ce_preserves_router_auxiliary_loss_contract` stays green).
- T5 gradient checkpointing: `gradient_checkpointing_enable()` → `(h, P, grads)` equal to the
  non-checkpointed run to fp32 tolerance; `len(router_logits) == L` (no double append).
- T6 grouped vs dense `opaque_moe` with the statistics enabled: identical `h`, grads within the
  parity-harness tolerance.

`packages/opaque-transformers/tests/opaque_transformers/test_router_load_balancing.py`:

- T7 setup: probe leaf in `trainable_params`; `clip_norm` is a two-group `PerGroup`; scalar
  `clipping_norm` converted; σ values equal `per_group_noise_stddev` closed form.
- T8 pre-noise probe leaf equals `(s/B̄) Σ_x h_x` exactly and `group_norms["router_load_probe"] ≤ λ`
  for adversarial single-expert routing (clipping never triggers).
- T9 post-processing: with `noise_fn` monkey-patched to inject known noise, `f̂`, EMA/window and
  renormalisation match closed form; `Σ_e f̃ = k`; `f̃_0 = k/E`; the guard for
  `Σ u ≤ k/2` keeps the previous `f̃`.
- T10 probe hygiene: after `n` steps `trainable_params["router_load_probe"] == 0` exactly; loss
  value equals `CE + α·surrogate` bit-for-bit; optimizer state for the probe is zero.
- T11 checkpoint round-trip: sidecar restores `RouterLoadState` exactly; resuming reproduces the
  next `f̃` bit-identically.
- T12 (`distributed` marker, 2 Gloo ranks): identical `f̃` and filter state on both ranks after 3
  steps; probe leaf all-reduced.
- T13 `microbatch_size=2` chunks == single chunk (grads, `h`, `f̂`).
- T14 DP-FTRL: `mf_gaussian_noise` + `band_mf_strategy(bands=4)` accepts the `PerGroup` latch;
  realised `noise_stddev.values["router_load_probe"] == base·row_l2(t)`; window filter state; the
  `ctx.mechanism` object is the same factory as a run with the feature off.
- T15 accounting invariance: `epsilon_at(δ)` identical with and without the feature for both
  stacks (this is the machine-checkable form of "accountant unchanged").
- T16 converter: `router_aux_loss_coef` maps to the flag for SFT on Mellum and still warns for DPO.
- T17 hygiene: no logged key is a function of un-noised `h`/`f(B)`; `router_load/*` values equal
  `summary(callback.state)`; probe group absent from the logged `group_norms` mean.
- T18 (`slow`) `torch_compile=True` one step == eager on CPU inductor.
- T19 `ConfigurationError` for `clipping_mode in {"auto", "adaptive"}` with the feature on.

Rust: no change. Docs build: the new mechanism page.

---

## 10. Validation plan on the real checkpoint (GPU) (G2 / G3)

**Oracle definition (G2).** `MellumForCausalLM` from transformers 5.16.1, `JetBrains/Mellum2-12B-A2.5B-Base`
in bf16, `_experts_implementation = "grouped_mm"`, PEFT LoRA r = 16 / α = 32 on q/k/v/o, eager
per-example loop of `loss.backward()` with `output_router_logits=False`, **with the same fp32-router
forward installed** (the patch is a plain `nn.Module.forward` replacement that works outside vmap).
Ladder: **O1** = the same loop in fp32 (ground truth), **O2** = bf16 loop (the "non-DP HF path"),
**O3** = Opaque bf16 `vmap(grad)` through `clipped_grad(clipping_norm=1e9, return_aux=True)` with
the Mellum patches (grouped and dense variants). The aux part is checked separately against the
float64 identity (T3) at `f̃ = f(B)` because HF's own aux (bf16-softmax top-k) is not the executed
routing (critic R9). Script: `examples/validation/mellum_oracle_drift.py` (one microbatch of 8,
T = 1024, KStack, fixed seed, one GPU, ~minutes).

**Drift metrics.** (m1) per-example relative L2 of the whole LoRA gradient, O3 vs O2 and vs O1;
(m2) **route-flip counter**: per token per layer, symmetric difference of the top-8 sets O3 vs O2
(and each vs O1), reported as flips/token/layer, plus the fraction of tokens whose fp32 margin
`p_(k) − p_(k+1)` is below `1e-6` (true near-ties); (m3) per-parameter-group relative L2 (q, k, v,
o adapters); (m4) all of the above with the fp32-router patch off (HF-bf16 routing on both sides);
(m5) loss absolute difference. **Acceptance:** with fp32 routing, flips O3 vs O2 = 0 (or every
flip's margin < 1e-6); (m1) O3-vs-O2 ≤ 1.5 × (O2-vs-O1) — inside HF's own bf16 floor; the PR #980
"≈ 1.3 %" figure reproduced to ±0.5 pp under this written-down definition (its own script is
untraceable, critic G2).

**Statistics to collect (G3)** — one GPU, 256 KStack examples under the preset, all from a
*validation-only* non-DP pass (these numbers are never logged by a DP run):

| statistic | how | decides |
|---|---|---|
| per-example gradient-norm quantiles p10/p50/p90/p99/max | `clipped_grad(…, clipping_norm=1e9, return_aux=True).grad_norms` | `C` (Section 4.5) |
| `δ_rel = ‖f(B) − k/E‖₂/(k/E)/√E` for 32 batches of 256, and its batch-to-batch drift over 100 consecutive LoRA steps | router recorder, eager | whether the term carries signal (need `r_smoothed ≲ 0.3·δ_rel·…`, Section 2.5) and the filter length (lag bias ≤ noise) |
| per-example `‖h_x‖₂` distribution and fraction of experts with `h_e = 0` | same pass | tightness of the structural bound (expect ≈ 1.0 balanced, ≤ 2.83), H5 |
| bf16-vs-fp32 router flip rate per layer on real code | m2 with the patch off vs on | confirms the fp32 default |
| dense vs grouped time / peak memory for one microbatch of 8 (frozen experts) | `torch.cuda.max_memory_allocated`, wall clock | G6 default |
| aux/CE gradient-norm ratio at α = 1e-4 and 1e-3 on the checkpoint | T3 machinery | H4 at scale |

**Mechanism acceptance on a 200-step DP run (ε = 3 calibrated, feature on, ρ = 0.1):**
(a) `f̃_t` tracks the validation-only `f(B_t)` with relative error ≤ 10 % per entry after warm-up
(200 steps > 3 EMA time constants / one window); (b) cosine between the DP surrogate aux gradient
and the exact batch aux gradient ≥ 0.8 whenever `δ_rel ≥ 10 %`; (c) eval loss within run-to-run
noise of the feature-off run (the term must not hurt at 1e-4); (d) reported ε identical with and
without the feature; (e) `group_norms["router_load_probe"]` max ≤ λ (never clipped); (f) throughput
with the grouped default within 20 % of the feature-off run.

---

## 11. Risks and what would falsify the design

1. **No usable signal on real data.** If G3 finds `δ_rel ≲ 3 %` on KStack (a well-balanced
   checkpoint on in-distribution code), the smoothed noise (2.6 % per entry at ρ = 0.1) is of the
   same size as the imbalance and the surrogate gradient is a random perturbation — harmless in
   absolute terms (`α·E·‖noise‖·‖∇P̄‖`, α = 1e-4) but useless. Falsifier: acceptance (b) fails at
   `δ_rel < 10 %` and never triggers because `δ_rel` never reaches 10 %. Mitigation: longer window /
   larger ρ (table), or leave the feature off (OLMoE and Tholoniat evidence that dropping the aux
   in fine-tuning is benign, `phase1-literature.md` C8/A1 — VERIFIED numbers, PLAUSIBLE transfer).
2. **Lag bias.** If `f(B)` drifts faster than the filter's time constant (20 steps EMA / 128-step
   window), `f̃` chases a stale target. Falsifier: G3 batch-to-batch drift over 100 steps larger
   than `r_smoothed`. Mitigation: shorter window at a higher ρ.
3. **Recorder under checkpointing/compile.** The claim that captured router logits inside a
   non-reentrant checkpoint region carry gradient and are not double-appended is PLAUSIBLE until T5
   / T18 pass. Fallback: the forward-hook variant of primitives E3 (clear list per call), which is
   VERIFIED at toy scale but graph-breaks under compile.
4. **Probe leaf and the optimizer.** If a future optimizer ignores a zero gradient differently
   (e.g. decoupled weight decay on a nonzero param), the probe could drift. T10 guards it; the
   `_augment_inputs` assertion converts silent drift into a hard error.
5. **AUTO-S / adaptive clipping with the probe group** is excluded (Section 4.4); a preset that
   switches to `"auto"` gets a `ConfigurationError` rather than a biased release. Falsifier: users
   need AUTO-S for the gradient group — then the engine change (per-group fixed override in
   `auto_scale_pytree`) is required, contradicting "engine untouched".
6. **fp32 router changes the model vs the HF-bf16 oracle.** By design; the oracle carries the
   same patch. Inference stacks that route in bf16 keep the documented train/inference route
   disagreement (Mellum2 TR §5.2). Falsifier: eval with bf16 routing degrades measurably vs fp32
   routing after fine-tuning — then ship the router in the served model or enable z-loss.
7. **Grouped-MoE default changes numerics of existing runs** (4.63e-3 vs 4.98e-3 vs HF bf16 loop,
   both inside the floor) and the process-wide first-patch capture can surprise multi-model
   processes. Falsifier: any parity test in `test_parity_harness.py` moving outside tolerance.
8. **Mahalanobis allocation is MSE-optimal only for equal group dimensions**
   (`noise_allocation.py:55-58`); privacy is unaffected (equality holds regardless, VERIFIED) but
   the split between a 64-dim group and a ~10⁶-dim group is chosen by ρ, not by the allocator.
   Falsifier: none for correctness; utility only.
9. **Faithfulness is to Fact A, not to HF Trainer.** A reviewer comparing against an
   HF-Trainer-with-accumulation run will see a different aux (per-microbatch `f`, `G·α`, critic Exp A).
   The design states this explicitly; if the product decision is "match HF Trainer's realised
   objective", set `α_eff = G·α` and accept microbatch-level `f` — which the same mechanism can
   emulate only approximately (the released `f̂` is batch-level).
10. **Concurrent releases under DP-FTRL.** The design deliberately avoids a separate per-step
    load mechanism under MF (that would need concurrent composition, Vadhan–Wang / Vadhan–Zhang,
    theorem numbers not verified in phase 1). Falsifier: someone later adds an independently
    sampled load release under b-min-sep — that is a different participation model and must be
    accounted separately (`phase1-primitives.md` §3.4, §7.3).
11. **Unverified at scale.** Every magnitude here except the accounting table and the MF row norms
    comes from random-init toys; requirement (c) of the brief ("usable clipping norm / noise") is
    settled only by Section 10's G3 pass.
