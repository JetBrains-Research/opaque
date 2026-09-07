# Phase 2 — design (lens: FAITHFULNESS-FIRST). Mellum2 MoE under Opaque DP: the "released-load surrogate" mechanism

Agent: `design-faithful`. Repo `/home/user/opaque` @ `ef1abc5` (branch `claude/mellum-dp-representation-r6slaz`).
Evidence tags: **VERIFIED** = I read the cited lines / ran the cited command in this session, or the
phase-1 report that established it executed it (named); **PLAUSIBLE** = derived or read, not executed.
Repo citations are `file:line` relative to `/home/user/opaque`; HF = `.venv/lib/python3.11/site-packages/transformers/…`
(transformers 5.16.1). Nothing in the tracked repo was modified. Two small checks were run with `uv run python`
(CPU): the accountant table in §2.6 and the band-MF row norms in §6.2.

**One-paragraph summary.** The only inseparable batch term in Mellum2's objective is the Switch-style load-balancing
loss `L_aux(B) = E·Σ_e f_e(B)·P_e(B)` (F1). Because `f` is argmax-derived, `∇L_aux(B) = Σ_x ∇S(x; f(B))` exactly, with
`S(x; f̃) = E·Σ_e f̃_e·P_e(x)` (F3, math §1.2 VERIFIED to 1e-17). The design therefore (i) keeps the model's real
objective term-for-term, (ii) replaces the one private batch constant `f(B)` by a **DP-released, workload-filtered
estimate `f̃_t`** carried as a second per-group leaf of the *same* clipped pytree, so that the accountant call is
**literally unchanged** (`poisson(gaussian(nm), q)*T` / `b_min_sep(mf_gaussian(nm, band_mf), …)`), and the price is a
√(1+ρ) inflation of gradient noise (default ρ=0.05 ⇒ ×1.025) — and (iii) removes the bf16 route-flip source of
DP-vs-oracle drift by routing in fp32, which is what Mellum2's pretraining did. `f` is defined from the **executed**
routes. Everything else (mask, weighting, coefficient, per-layer option, smoothing, hygiene, implementation) follows.

---

## 1. Objective (G1)

### 1.1 The per-example loss (exact)

For example `x` (right-padded, attention mask `m_{x,t}`, `T_x = Σ_t m_{x,t}`), layers `l = 1..L` (L = 28), experts
`e = 1..E` (E = 64), `k = 8`:

```
z^l_{x,t}  = W_l · h^l_{x,t}                       router logits, computed in fp32 (§3)
p^l_{x,t}  = softmax_fp32(z^l_{x,t})               full-E softmax (HF: modeling_mellum.py:335)
S^l_{x,t}  = topk_k(p^l_{x,t})                     the EXECUTED top-8 set (the experts that ran)
P_e(x)     = (1/(L·T_x)) Σ_l Σ_t m_{x,t} · p^l_{x,t,e}          per-example mean prob      (differentiable)
h_e(x)     = (1/(L·T_x)) Σ_l Σ_t m_{x,t} · 1{e ∈ S^l_{x,t}}     per-example load fraction  (piecewise constant)
d(x)       = h(x) − (k/E)·1                                    centred load, Σ_e d_e(x) = 0

ℓ_x(θ; f̃_t) = CE_x(θ) + α · E · Σ_e f̃_{t,e} · P_e(x; θ)          [+ ζ · Z_x(θ), z-loss, opt-in, §7]
```

`CE_x` = HF's own per-example token-mean causal-LM CE over label-valid tokens (label mask −100, shift by one), exactly
what `DPTrainer.compute_per_example_loss` reads from `fmodel(params, **inputs)["loss"]`
(`packages/opaque-transformers/src/opaque/api/transformers/trainer/_dp_trainer.py:2314-2434`, VERIFIED; chunked
LM-head path `packages/opaque-patches/src/opaque/api/patches/transformers/components/cross_entropy.py`).

The batch-level objective this implements is `(1/B̄)·Σ_{x∈B_t} ℓ_x(θ; f̃_t)`. Its aux part has **exactly** HF's
`∇[α·L_aux(B)]` when `f̃_t = f(B_t)` and all `T_x` are equal (math §1.2–1.3 VERIFIED: 1e-17 full mask; F4).

### 1.2 Target of faithfulness: the HF logical-batch formula (Fact A), executed-route f — and why

Four candidates existed (critic G1). Decision and reasons:

| candidate | what it is | verdict |
|---|---|---|
| (a) **HF logical-batch pooled formula** `E·Σ_e f_e·P_e` with f, P pooled over all layers and the whole batch, one denominator `L·T_tot` (`modeling_mellum.py:575-606`, VERIFIED) | the objective the released checkpoint *declares* (`router_aux_loss_coef=0.001`, `configuration_mellum.py:102-103`) and the only one reproducible from the HF artefact | **TARGET** |
| (b) HF-Trainer-realised: per-*microbatch* f with effective coefficient `G·α` (critic Exp A, rel-L2 0.0, VERIFIED) | an accumulation artefact of `trainer.py:1961-1963` (loss divided by G only when the model does not accept loss kwargs; Mellum accepts them) — not a stated objective; at the preset (B=256, mb=8) it would mean f over 8 sequences and α_eff = 0.032 | rejected; document that the DP path is *more* faithful to (a) than HF Trainer itself |
| (c) Megatron per-layer running-average f (Mellum2 pretraining/SFT, TR §3.6; literature F.2 VERIFIED) | `(1/L)Σ_l E Σ_e f^l_e P^l_e` (mean of per-layer products; per-layer averaging PLAUSIBLE — Megatron code not read) with a *lagged running-average* f | not reproducible from HF (no per-layer pooling in HF); needs an `L×E` release (×√L noise, §1.5). The **lagged/running-average** aspect of (c) *is* adopted: `f̃_t` is an EMA of released loads, i.e. the same kind of estimate Megatron used (literature 0.3) |
| (d) per-sequence aux (DeepSeek-V2/V3, Wang et al.) | a different regulariser: pushes within-document uniformity; cos 0.26–0.39 to the batch gradient (E4, VERIFIED); anti-specialisation closed form (math §2) | rejected as *faithful*; note it is a legitimate zero-cost variant some MoEs train with |

So: the objective is (a) evaluated on the *logical* DP batch, with `f(B_t)` replaced by the released estimate. The
DP path has no gradient accumulation (microbatches are vmap chunks of one logical batch, `_dp_trainer.py:2082-2131`),
so (a) is the natural target and (b)'s artefact never arises.

### 1.3 Mask, CE weighting, coefficient

- **Mask for f and P: the attention mask** (HF passes `attention_mask` at `modeling_mellum.py:697`, VERIFIED): prompt
  tokens with label −100 still count as routed tokens. Faithful and physically right — those tokens *were* routed.
  Under SFT prompt-masking the CE uses the label mask; the two masks differ by design, as in HF.
- **CE weighting: equal example weights** (Opaque convention, `packages/opaque-alignment/src/opaque/api/alignment/sft/loss/_nll.py:10-24`
  VERIFIED: pre-clip division by the example's own token count). HF weights tokens (`num_items_in_batch`). They
  coincide exactly for the packed T=1024 presets (F4). The consistent aux weighting is then also per-example
  (`w_x = 1/B`), so the released statistic is the *example-mean* load `(1/B̄)Σ_x h(x)`, not the token-weighted one.
  Ragged data: an optional public constant `N̄` (e.g. `B·T_max`) restores HF's token weighting per example
  (divergence §4) — same lever for CE and aux; not default.
- **Coefficient α.** Default = the checkpoint's `config.router_aux_loss_coef` (1e-3, the pretraining value). The
  Mellum2 presets (`examples/train_dpftrl.py:873-896`, `examples/train_dpo.py:1300-1318`) set **α = 1e-4**, Mellum2's
  own SFT coefficient ("router already well-balanced after pre-training", TR §5.1.2, literature F.2 VERIFIED).
  α = 0 recovers today's behaviour (aux dropped; Tholoniat/OLMoE precedent, literature A1/C8).
  Value at perfect balance: `L_aux = k = 8`, i.e. 0.008 nats at 1e-3 / 0.0008 at 1e-4.

### 1.4 f from executed routes (fp32), not HF's recomputation

HF's `load_balancing_loss_func` recomputes top-k from a softmax in the *logits dtype* (bf16 for a bf16 model,
`modeling_mellum.py:584`), while the forward routed with an fp32 softmax (`:335`); 0.03–1.7 % of tokens get a different
aux top-k than the one executed (critic R9, VERIFIED toy). **Definition adopted:** `S^l_{x,t}` is the index set the
experts actually ran with (the router's own `router_indices`), and `p` is the router's own fp32 softmax. With the fp32
router patch (§3) logits are fp32 too, so the executed routes are the fp32 routes — the pretraining router's routes.
Deviation from HF's *number* is bounded by 1e-3·k/E (critic R9) — below the noise floor of §2.6.

### 1.5 Pooled vs per-layer, and the √L price

HF pools layers (one product on layer-pooled means). Megatron's per-layer form differs by the across-layer covariance:

```
L_aux^{per-layer} − L_aux^{pooled} = E · Σ_e (1/L) Σ_l (f^l_e − f̄_e)(P^l_e − P̄_e)
```

Reproducing it needs `f̃^l_t` per layer: an `L×E = 1792`-dim release with structural bound `√(kL(1−k/E)) = 14.0`
vs `√(k(1−k/E)) = 2.646` pooled (math §3, VERIFIED), i.e. **×√L = ×5.29 relative noise** at equal privacy. Decision:
**pooled is the default** (it is the HF target); `router_load_per_layer=True` is an opt-in whose cost is one row of
the table in §2.6 (at ρ=0.2 under MF the smoothed per-layer error is ≈5 %, so it is usable, just not free).

---

## 2. Mechanism and accountant (G4)

### 2.1 Adjacency and protected unit

Example-level (one collated row = one document / one FIM sample; for DPO one preference pair, §7), **add/remove**
adjacency — Opaque's default (`.junie/differential-privacy-review.md` "Adjacency"; `clipped_grad` contract
`packages/opaque-engine/src/opaque/api/engine/clipping/_clipped_grad.py:135-145` per primitives §1.2). Replace-one
adjacency doubles every per-record bound below (both groups) — nothing else changes.

### 2.2 The per-step mechanism (DP-SGD, Poisson)

Public state entering step `t`: `θ_t`, `f̃_t ∈ [0,1]^E`. Constants: gradient bound `C_g` (preset 0.9), load scale
`λ`, structural load bound `Δ_h = √(k(1−k/E)) = 2.6458`, group bound `C_h = λ·Δ_h`, ratio `ρ = C_h/C_g`, public
expected batch `B̄ = q·N` (`normalize_by=expected_batch_size`, `_dp_trainer.py:4267/4277/4288` VERIFIED).

1. **Sample** `B_t` by Poisson(q) (existing sampler).
2. **Per example, inside vmap(grad)** (fp32 router §3): forward → `CE_x`, router logits `(L,T,E)` → `p`, executed `S`,
   `P(x)`, `h(x)`, `d(x)`; loss `ℓ_x = CE_x + α·E·⟨f̃_t, P(x)⟩ + ⟨z, λ·d(x).detach()⟩` where `z ∈ R^E` is the
   zero-valued probe parameter (value term is identically 0; `∂ℓ_x/∂z = λ·d(x)`; primitives E2 VERIFIED that other
   leaves are unaffected).
3. **Per-group clip** (`PerGroup`, `packages/opaque-engine/src/opaque/api/engine/clipping/_pytree.py:439-477`):
   gradient group(s) to `C_g`; probe group to `C_h`. Since `‖λ d(x)‖₂ ≤ λΔ_h = C_h` *structurally*, the probe group is
   never rescaled → the load release is **unbiased** (math 4(b) remark ii). (Set `C_h = λΔ_h(1+1e-6)` so fp round-off
   never triggers the ULP-guarded clip.)
4. **Sum + Gaussian noise**, MSE-optimal per-group allocation (`packages/opaque-engine/src/opaque/api/engine/noise_allocation.py:44-110`
   VERIFIED): `σ_g = nm·√(C_g·S)`, `σ_h = nm·√(C_h·S)`, `S = C_g + C_h` (all scaled by `1/B̄`). Output
   `(ĝ_t, ŷ_t)` with `ĝ_t = (1/B̄)(Σ_x clip g_x + N(0,σ_g²))`, `ŷ_t = (1/B̄)(λ Σ_x d(x) + N(0,σ_h²))`.
5. **Post-processing of `ŷ_t`** (public, free): `d̂_t = ŷ_t/λ`; project to the sum-zero subspace
   `d̂_t ← d̂_t − mean(d̂_t)`; filter `d̃_{t+1} = β_f·d̃_t + (1−β_f)·d̂_t`; `f̃_{t+1} = clamp(k/E + d̃_{t+1}, 0, 1)`
   (clamp is post-processing; the sum-k constraint is restored by the projection up to the clamp, which is inactive
   unless an expert is essentially dead). `d̃_0 = 0` (⇒ `f̃_0 = k/E`, zero aux gradient at step 0 — exactly the real
   objective's behaviour at balance, F3). Zero the probe parameter.
6. **Optimizer** on `ĝ_t` (probe leaf masked out / re-zeroed).
7. **Realised-batch scaling (critic M11).** `E[Σ_x d(x)] = B̄·E_x[d(x)]`, so `d̂_t` is unbiased for the population
   centred load; the realised `|B_t|/B̄` (rel. sd `1/√B̄ ≈ 6 %`) only multiplies the *imbalance signal* by a factor near
   1 — no renormalisation to Σ=k is needed for the centred release (its sum is 0 structurally; the projection removes
   the noise's sum component). This is the reason to release `d(x)` rather than `h(x)`.

**Sensitivities (add/remove).** Gradient group: `C_g` (as today). Load group: `sup_x ‖λ d(x)‖₂ = λΔ_h = C_h`
(math §3 VERIFIED: `‖h_x‖₂ ≤ √k`, centred `√(k(1−k/E))`, attained when every token in every layer routes to the same
8 experts). Both hold for *every* prefix of previous outputs (the bound does not depend on `f̃_t`), which is what
adaptive composition needs (Zhu–Dong–Wang Thm 10, https://arxiv.org/abs/2106.08567, VERIFIED by critic R6).

**Why the accountant is unchanged.** The step releases one Gaussian on the concatenation `[Σ clip g_x ; λ Σ d(x)]`
with diagonal covariance; whitening gives a sensitivity-1 Gaussian at multiplier `σ_eff` with
`σ_eff⁻² = C_g²/σ_g² + C_h²/σ_h²`; Opaque's allocation makes this exactly `1/nm²` (`noise_allocation.py:57-58`
docstring; primitives E1(f) numerically; math 4(b)). Precedent: Andrew et al. 2021 Thm 1
(https://arxiv.org/abs/1905.03871, VERIFIED by literature B1/critic R6) — the clipped-count release of adaptive
clipping is this same joint Gaussian. Subsampling then applies to the *one* joint mechanism (both releases share the
Poisson coin; Feldman–Shenfeld Lemma 3.2/Thm 3.3 as implemented in `src/amplification/poisson.rs:15-42`, critic R6
VERIFIED). Hence:

```
accountant (DP-SGD):  dpsgd.poisson(dpsgd.gaussian(nm), q) * T          ← unchanged (_dp_trainer.py:4301-)
```

The naive "σ·C_g on the gradient, σ·C_h on the load" is **not** `gaussian(nm)` (it is `gaussian(nm/√2)`; math 4(b)
REFUTED row) — the design never uses it.

### 2.3 The per-step mechanism (DP-FTRL, band-MF, b-min-sep)

Identical steps 1–7 with two substitutions:

- Sampling/participation: `BMinSepSampler` (`packages/opaque-dpftrl/src/opaque/api/dpftrl/sampling/_b_min_sep.py`);
  the load leaf shares the example's participation pattern (same example, same step), so `min_sep`,
  `max_participations` are the gradient's.
- Noise: `mf_gaussian_noise(trainable_params, band_mf_strategy(bands=64, momentum=β), n_steps, …, noise_multiplier=nm)`
  with the **same `PerGroup` max_norm**. VERIFIED (`packages/opaque-dpftrl/src/opaque/api/dpftrl/noise/_mf_gaussian_noise.py:163-179`):
  `base_stddev = per_group_noise_stddev(max_norm, nm)`, then the streaming `C⁻¹` multiplication; realised per-step
  σ on every leaf = `base·‖row_t(C⁻¹)‖` (`:186-192`). The constant-max_norm latch accepts a constant `PerGroup`
  (primitives E2 VERIFIED, `_engine.py:473-517`).
- Per-group MF correctness: per-group whitening gives `Σ_g ‖C(G_g−H_g)‖²_F/σ_g² ≤ sens(C)²·Σ_g C_g²/(nm²·C_g·S) = sens(C)²/nm²`
  because the participation-pattern sensitivity is degree-1 homogeneous in the row bound (math §5(i)); Denisov et al.
  Thm 2.1 (https://arxiv.org/abs/2202.08312, VERIFIED critic R6) covers adaptively chosen rows — `f̃_t` is a
  function of previous outputs, exactly the theorem's setting.

```
accountant (DP-FTRL): dpftrl.b_min_sep(dpftrl.mf_gaussian(nm, band_mf_strategy(64, β)), n_steps=T, p0=p0)   ← unchanged
```

Adaptive clipping remains excluded under MF (drifting bound); AUTO-S on the *gradient* group is allowed; the load
group must **not** be AUTO-S-scaled (it would rescale every `d(x)` to norm ≈C_h and bias the release; critic R11) —
the implementation forces `fixed` semantics on that group (§9).

### 2.4 Same-step vs lagged, and the alternative parametrisation

`f̃_t` is consumed **lagged** (from releases `< t`): it fits `_augment_inputs` (runs once per step outside vmap,
`_dp_trainer.py:2302-2312` VERIFIED), needs no second forward, and costs nothing extra (math 4(d); same dependence as
`θ_t`). A same-step variant (forward-only routing pass, then release, then gradient pass) is possible but is a second
mechanism per step and doubles forward cost; not adopted. An *independently subsampled* load release would be cheaper
in ε for DP-SGD (primitives E1(h)) but is unavailable under b-min-sep (no second draw) — rejected for stack uniformity.

Equivalent "pay in ε" form (for users who insist on gradient σ = nm·C_g): the joint allocation at ratio ρ is the
same mechanism family as `poisson(gaussian(nm) | gaussian(c·nm), q)` with `c = √(1+1/ρ)` on the load release;
holding nm raises ε as tabulated in §2.6. The design ships the joint form (ε fixed, ×√(1+ρ) gradient noise).

### 2.5 Defaults

`ρ = 0.05` ⇒ `λ = ρ·C_g/Δ_h = 0.05·0.9/2.6458 = 0.0170`, `C_h = 0.045`; `β_f = 0.95` under DP-SGD, and `β_f :=` the MF
workload momentum under DP-FTRL (§6). `f̃_0 = k/E`.

### 2.6 ONE cost table — preset regime `nm = 0.5622, B̄ = 256, k = 8, E = 64, C_g = 0.9, L = 28`, `q = 256/5e5, T = 15625, δ = 1e-6`

Single-step per-entry noise on `f̂` (pooled, centred): `nm·Δ_h·√(1+1/ρ)/B̄`; in units of `k/E = 0.125`:
`r₁ = 0.04648·√(1+1/ρ)`. Smoothing factors: DP-SGD EMA `√((1−β)/(1+β)) = 0.1601` (β=0.95, stationarity);
band-MF workload-matched EMA `(1−β)·‖row_t(A C⁻¹)‖ = 0.0824` (§6.2, VERIFIED numerically; MF single-step factor
`‖row_t(C⁻¹)‖ = 1.432`). ε column: accountant run this session (`poisson(gaussian(nm)|gaussian(c·nm), q)*T`,
`epsilon_at(1e-6)`; baseline 3.000).

| ρ = C_h/C_g | c = √(1+1/ρ) | **gradient-noise inflation √(1+ρ)** (ε held at 3.0) | **ε if nm held** (grad σ unchanged) | load r₁ single step, DP-SGD | load r after EMA β=.95, DP-SGD | load r single step, MF (×1.432) | **load r after workload EMA, band-MF b=64** | per-layer variant (×5.29) after EMA: SGD / MF |
|---|---|---|---|---|---|---|---|---|
| 0.5 | 1.732 | ×1.225 | 5.443 | 8.1 % | 1.3 % | 11.5 % | 0.66 % | 6.8 % / 3.5 % |
| 0.3 | 2.082 | ×1.140 | 4.673 | 9.7 % | 1.5 % | 13.9 % | 0.80 % | 8.2 % / 4.2 % |
| 0.2 | 2.449 | ×1.095 | 4.219 | 11.4 % | 1.8 % | 16.3 % | 0.94 % | 9.6 % / 5.0 % |
| 0.1 | 3.317 | ×1.049 | 3.703 | 15.4 % | 2.5 % | 22.1 % | 1.27 % | 13.1 % / 6.7 % |
| **0.05 (default)** | 4.583 | **×1.025** | 3.417 | 21.3 % | **3.4 %** | 30.5 % | **1.76 %** | 18.0 % / 9.3 % |
| 0.02 | 7.141 | ×1.010 | 3.234 | 33.2 % | 5.3 % | 47.5 % | 2.74 % | 28.1 % / 14.5 % |
| 0.01 | 10.05 | ×1.005 | 3.172 | 46.7 % | 7.5 % | 66.9 % | 3.85 % | 39.6 % / 20.4 % |

Reading the table against the objective: the surrogate gradient is `α·E·Σ_e (f̃_e − k/E)·∇P_e(x)` (F3), so the
*relevant* error is `‖noise‖/‖f(B)−k/E‖`, not `r` against `k/E`. If the real per-coordinate imbalance is ≥ 0.3·k/E
(unknown; G3 measurement §10), the default gives ≈ 11 % (SGD) / 6 % (MF) relative error on the aux gradient for a
2.5 % gradient-noise price and **no ε change**. If the imbalance is smaller, the term is small in the real objective
too (it vanishes at balance) — the design degrades to what the true objective does, not to a different regulariser.

---

## 3. Router precision and routing (G5)

**Decision: fp32 router logits, routes computed per example inside vmap from the current model, f from those routes.**

- Upstream computes `router_logits = F.linear(hidden_bf16, W_bf16)` in bf16 and only the softmax in fp32
  (`modeling_mellum.py:334-335` VERIFIED). bf16 has fixed *relative* resolution, so ≈1 % of tokens per layer sit
  within its rounding of a top-k tie regardless of router sharpness (E2 VERIFIED toy); those flips cause ≈8× excess
  per-example gradient error on router/expert parameters and +0.2 pp on attention-only parameters (E1b). Mellum2's
  pretraining ran the router in FP32 (TR appendix "Router precision FP32", literature F.2 VERIFIED) and the authors
  document train/inference route disagreement on this very checkpoint (TR §5.2).
- **Patch** (`opaque-patches`, new role `"router": "MellumTopKRouter"`, kind `"fp32"`, default ON for the mellum
  family under `compat`): `logits = F.linear(h.float(), W.float())`; `p = softmax(logits, -1)` (already fp32);
  `topk`; renormalise; `scores.to(h.dtype)`; return `(logits_fp32, scores, indices)`. Downstream consumers only use
  `scores`/`indices` (`MellumSparseMoeBlock.forward`, `modeling_mellum.py:350-355` VERIFIED), so the model output is
  unchanged except through the more precise routing decision. Cost: one `64×2304` fp32 GEMM per token per layer
  (≈0.3 MFLOP) against ≈99 MFLOP of routed expert compute — negligible; memory: an fp32 view of W (590 KB/layer) or an
  on-the-fly cast.
- **Pinning.** Routes are computed *inside* the per-example function from the current `θ_t` — they are already
  per-example constants of `(x, θ_t)`; no cross-example dependence, no DP consequence (math §6). The per-example loss
  is `C^∞` within a routing cell and jumps only at exact fp32 ties (measure zero). Frozen-base pinning is **rejected**:
  it freezes training-time routing while inference routing (whose inputs move through attention LoRA) drifts —
  train/inference mismatch that grows with the fine-tune (math §6). No STE / DenseMixer: that changes the backward
  of the real model (out of "faithful"; noted as a future utility lever).
- **Consistency:** `h(x)` and `P(x)` use the *same* fp32 `p` and the *same* executed `S` that the experts consumed
  (§1.4). The z-loss (§7) is the standard remedy for bf16 router round-off; with an fp32 router it is not needed for
  numerics and is offered only for faithfulness to the pretraining objective.
- **Oracle implication (§10):** the reference forward for drift measurement applies the same fp32-router patch, so the
  drift metric isolates vmap/per-example effects; the deviation from *unpatched* HF-bf16 routing is reported
  separately as a deliberate, pretraining-faithful choice.

---

## 4. Clipping norm and per-example gradient norms

What the design does so a reasonable clipping norm exists, and how to pick it:

1. **The load release never touches the gradient's clipping norm.** It is its own `PerGroup` group with a structural
   bound; the gradient group(s) keep exactly today's `C_g` (preset 0.9, `examples/train_dpftrl.py:611-613`,
   VERIFIED). With `α = 1e-3` the aux term changes per-example gradient norms by < 0.1 % (E3 VERIFIED toy); with the
   *surrogate* it is ∝ `‖f̃ − k/E‖` and is exactly 0 at balance, so it cannot inflate the norm distribution the way a
   per-example aux (`E·P_{S_x}` anti-specialisation term, math §2) would.
2. **fp32 routing tightens the distribution where it matters.** Per-example gradient error on router/expert params
   drops from 12.9 %/11.1 % to 1.7 %/1.5 % when routes are pinned to the fp32 choice (E1b VERIFIED toy); the
   heavy-tail contribution of near-tie flips is removed at the source. For attention-only LoRA the effect is +0.2 pp
   — the preset's norm distribution (max/median 1.24–1.33 at toy scale) is already tight.
3. **Partition-aware `PerGroup` when experts/router are trained** (§7): groups `{attention-LoRA, router, experts}` via
   `per_group(trainable, gate=…, experts=…, fallback=…)` (`packages/opaque-engine/src/opaque/api/engine/clipping/_per_group.py:44-`;
   trainer resolves a dict `clipping_norm` into `PerGroup` at `_dp_trainer.py:1456-1468` VERIFIED). Expert gradients
   are sparse across experts per example (E5 caveated by critic R8 — real fraction to be measured, §10) and
   high-dimensional; giving them their own bound stops a few hot experts from consuming the whole budget, and the
   MSE-optimal allocation keeps `gaussian(nm)` unchanged (§2.2).
4. **AUTO-S** (`auto_clipped_grad`, `packages/opaque-engine/src/opaque/api/engine/clipping/_auto.py:116-`) is allowed on
   the gradient groups under both stacks (constant per-record bound; MF-compatible); the load group is exempt (§2.3).
5. **How to pick `C_g`** (the G3 script, §10.3): run `clipped_grad(..., clipping_norm=1e9, return_aux=True)` on
   ~256 KStack examples with the preset partition; set `C_g` at the median of `aux.grad_norms` (clips ≈50 %, low bias)
   — Opaque's `adaptive` mode would do this online but is barred under MF, so the preset pins the measured value.
   Per group: the same statistic per `group_norms`. Expect the router group (if trained) to need a *smaller* bound
   than experts (E3: router aux-only norms are the heaviest-tailed).
6. **Numerical stability vs the non-DP path** (requirement (d)): fp32 accumulation in `Opaque_MoE`
   (`packages/opaque-patches/src/opaque/api/patches/kernels/moe.py:83,103`), upstream RMSNorm retained
   (`mellum.py:41`), SDPA causal fast path parity (`runtime/masking.py:195-216`), and now fp32 routing — the
   remaining bf16 drift is accumulation order (0.45–0.5 % rel-L2 with zero flips, E1/divergence §1.4 VERIFIED toy),
   which is inside HF's own batched-vs-loop spread.

---

## 5. Performance (G6)

- **Dense vs grouped MoE default.** `DPTrainer` defaults `use_performance_kernels=False`
  (`_training_arguments.py:436`) ⇒ `kernels=False` (`_dp_trainer.py:846`) ⇒ `grouped_moe=False` ⇒ dense
  every-token-through-every-expert `Opaque_MoE` on every host including CUDA (F9, critic R3 VERIFIED). Arithmetic
  (PLAUSIBLE, not measured): per token per layer the dense path runs 64 experts × 3 GEMMs (`2304×896` each, 2 flop/MAC)
  ≈ 793 MFLOP vs 99 MFLOP routed and ≈ 40 MFLOP attention at T=1024 — the dense default is ≈6× the FLOPs of a grouped
  run and dominates step time. **Decision:** the mellum family requests `grouped_moe=True` whenever
  `torch._grouped_mm` / Triton is available (`kernels/moe.py:578-633` dispatch already prefers `Opaque_FusedMoE` on
  CUDA bf16), independent of the CUDA-kernel group; the `use_performance_kernels` docstring
  (`_training_arguments.py:429-435`) gains the MoE entry; presets set `performance_kernels_config={"grouped_moe": True}`.
  The first-class-level-patch capture (`_router.py:69-80`) must be documented ("first `apply_model_patches` in a
  process wins").
- **Memory of new buffers** (per example, T=1024): router logits `(L,T,E)` fp32 = 28·1024·64·4 B = 7.3 MB (HF already
  keeps bf16 logits for autograd; fp32 doubles it) and the fp32 probs for `P(x)` (another 7.3 MB, freed after the
  reduction); ×microbatch 8 ⇒ ≈ 120 MB transient — negligible against expert activations. Per-example `h, P, d`:
  3×64 floats. Probe leaf: 64 floats (+ 128 floats optimizer state if not masked). EMA state: 64 floats. Nothing scales
  with parameters.
- **Compute of the aux term:** one `(L·T)×E` softmax (already computed by the router; reused), one masked mean, one
  top-k indicator sum — `O(L·T·E)` = 1.8 M elements per example.
- **Chunked CE is preserved** (the new `opaque_router_logits_only` kwarg, §9.2), so peak memory stays that of
  PR #978, not the full-vocab `98304×1024` logits of the current `output_router_logits` fallback.

---

## 6. Smoothing under MF (G7)

### 6.1 Which filter inherits the strategy's guarantee

Band-MF's Toeplitz `C` is optimised for the momentum workload `A` with entries `β^i`
(`packages/opaque-dpftrl/src/opaque/api/dpftrl/noise/_band_mf.py:35-60,86-91` VERIFIED: `_momentum_workload_coef`,
`optimize_toeplitz(n, bands, workload_coef=β^i, query_weights=lr)`). The released load stream is
`x̂_t = d_t + σ_h·(C⁻¹Z)_t`. An EMA with coefficient `β_f = β` is `y_t = (1−β)·Σ_{s≤t} β^{t−s} x̂_s = (1−β)·(A x̂)_t`,
i.e. **exactly the workload the strategy minimises error for, scaled by (1−β)**. Its noise is
`(1−β)·σ_h·‖row_t(A C⁻¹)‖`, and the strategy's max/mean per-query error guarantee transfers with the factor `(1−β)`.
An EMA with a different β, or a plain running mean, is a different linear map and does *not* inherit the guarantee.
**Decision:** `β_f := workload momentum` (`args.momentum` for SGD, `beta1` for Adam-family; preset 0.95,
`examples/train_dpftrl.py:495-498,1563-1581` VERIFIED). With an `lr_schedule` (query weights), the matched filter is the
lr-weighted momentum sum; the preset uses constant LR (warmup 0), so `A` is the plain momentum Toeplitz.

### 6.2 Numbers (VERIFIED this session: `band_mf_strategy(bands=64, momentum=0.95)`, `n_steps=4096`, `inverse_coef`)

`‖C‖ (sensitivity) = 1.0000`; `‖row_t(C⁻¹)‖`: 1.284 (t=0), 1.421 (t=1), **1.432 stationary from t≈7**;
`‖row_t(A C⁻¹)‖`: 1.284, 1.421, 1.597 (t=7), **1.649 stationary from t≈31**; `(1−β)·‖row_t(A C⁻¹)‖ = 0.0824`.
Under DP-SGD the same EMA has stationary factor `√((1−β)/(1+β)) = 0.1601`. So the load estimate is **≈2× less noisy
under band-MF than under DP-SGD at equal base σ**, because the correlated noise was shaped for precisely this
prefix-sum consumer; the stationary values at n=15625 are the same (Toeplitz; reached by t≈31).
Lag: `1/(1−β) = 20` steps ≈ 0.13 % of the horizon.

### 6.3 Per-step realised σ and the `NoisedPytree` metadata

The trainer reads realised σ from `NoisedPytree.noise_stddev` (`PerGroup` × `row_l2`, `_mf_gaussian_noise.py:186-197`);
the EMA consumer uses the *filtered* factor above for its own diagnostics, never the per-step one.

---

## 7. Scope (G8)

| item | in/out | what is needed |
|---|---|---|
| DP-SFT / causal-LM with attention-only LoRA (both presets) | **in** — the mechanism as specified | §9 |
| DP-DPO (`mellum2-codesec`) | **in** (mechanism unchanged): the protected unit is the preference pair; `h(x)`, `P(x)` pool *chosen + rejected* policy forwards (both are routed tokens of the same example); the reference forward (`_dpo_trainer.py:18-22,807-841` VERIFIED: precomputed or TR-DPO EMA via `_augment_inputs`) contributes **no** aux (HF adds aux only to the policy loss, `modeling_mellum.py:692-700`). The `load_target` column rides beside `ref_*_logps` (same seeding pattern, `_dpo_trainer.py:721-748`). TRL's converter must forward `router_aux_loss_coef` instead of dropping it (`trl/_convert.py:69-76`) | DPO trainer override of `compute_per_example_loss_and_metrics` to sum the two forwards' `(L·T)`-weighted loads; converter change |
| Router z-loss `Z_x = (1/(L·T_x)) Σ_{l,t} m_{x,t}·(logsumexp_e z^l_{x,t,e})²` (ST-MoE eq. 5, https://arxiv.org/abs/2202.08906 §3.1, literature C2 VERIFIED) | **in as opt-in**, default `ζ = 0`: it is per-token separable (zero DP cost, ordinary clipping), part of Mellum2's pretraining objective (1e-3) but absent from HF's, and its numerical motivation (bf16 round-off) is already removed by §3 | one extra reduction on the captured fp32 logits |
| Router-trainable / experts-trainable variants | **in for the mechanism** (nothing in §2 depends on which leaves exist; `opaque_moe` emits per-example expert-weight grads into `(B,E,…)` buffers when `needs_input_grad` is set, `kernels/moe.py:536-539`); **out for presets** until M5 is executed: PEFT `target_parameters` LoRA on stacked `experts.gate_up_proj/down_proj` under `functional_call`+vmap is untested (primitives §5). Full expert training costs `B·E·2I·H` bytes of per-example buffers per layer (≈0.53 GB/example/layer in bf16) — LoRA-on-experts or ESFT-style subsets are the practical forms; private expert selection (ESFT from private data) would itself need accounting (literature D.2) | M5 smoke test; per-group clipping preset (§4.3) |
| Per-layer `f̃^l` (Megatron pooling) | **opt-in**, ×√L cost (§1.5, §2.6) | `router_load_per_layer` flag; probe leaf `R^{L×E}` |
| Per-sequence aux, STE/DenseMixer, loss-free bias balancing | **out** (different objective / different backward / non-loss mechanism); loss-free balancing is noted as the authors' stated next step and would be a 64-sign release per step — a separate design | — |
| HF-Trainer-realised objective (per-microbatch f, `G·α`) | **out** (artefact, §1.2) | — |

---

## 8. Privacy hygiene (G9)

| tensor / state | where it lives | class | logged? | checkpointed? | DDP |
|---|---|---|---|---|---|
| probe parameter `z ∈ R^E` (value ≡ 0) | model `nn.Parameter` in `trainable_params` | public constant (re-zeroed in `_augment_inputs` every step; excluded from optimizer update) | no | as a zero tensor (harmless) | identical by construction |
| per-example router logits / fp32 probs / executed routes `S` | inside the vmapped per-example function only | **private-internal** — never leave the closure | never | never | n/a |
| per-example `h(x), P(x), d(x)`, per-example aux value `E⟨f̃,P(x)⟩` | inside the closure; `λ d(x)` becomes the probe leaf's per-example gradient | **private-internal** pre-clip/pre-noise | **not** added to `loss_aux` telemetry (the existing un-noised `loss`/`loss_aux` means at `_dp_trainer.py:2258-2293` are a pre-existing posture — F11 — this design adds nothing to it) | never | n/a |
| noised probe leaf `ŷ_t` (in `NoisedPytree.pytree["router_load_probe"]`) | output of `noise_fn` | **public** (post-noise) | may be logged | — | already all-reduced pre-noise by `sum_gradients_` (`_dp_trainer.py:2170-2173`); noise key shared across ranks ⇒ bit-identical (`:1478`, primitives §2.3 VERIFIED) |
| `RouterLoadState(ema d̃_t, f̃_t, step, beta, k, E, rho)` | trainer-side frozen dataclass | **public post-processing** | `f̃_t` logged (e.g. `router/load_min`, `router/load_max`, `router/imbalance_l2`) | **yes** — sidecar in the runtime checkpoint (§9.5); needed for reproducibility, not privacy | rank-identical; `register_sync_type` handler asserts equality (`packages/opaque-engine/src/opaque/api/engine/distributed/_state.py:537`) |
| `load_target` batch column `(B, E)` = broadcast `f̃_t` | injected by `_augment_inputs` | public | — | no | identical |
| fp32 router weight view | patch | public model parameter | — | no (derived) | — |
| realised batch size `aux.batch_size`, clip rates, group norms | existing `ClippedGradAux` | pre-existing telemetry (outside this design) | as today | as today | as today |

**Privacy statement addition** (for `docs/mechanisms/…` and the trainer docstring): "Each step releases one Gaussian
mechanism on the concatenation of the clipped per-example gradients and the per-example centred router-load vectors
(per-record bounds `C_g`, `C_h = λΔ_h`); the load estimate `f̃_t` consumed by the loss is post-processing of previous
releases. Accounting is `gaussian(nm)` per step under the stated sampler; no other quantity derived from private
routing is released." Anything that would log `f(B_t)` un-noised, `h(x)`, or per-example aux values is a violation
and is not implemented.

---

## 9. Implementation plan in Opaque (G10) and test plan

### 9.1 `opaque-patches` — model-side, DP-agnostic

1. **`components/router.py` (new):** `make_fp32_router_forward(original)` for `MellumTopKRouter` (§3). Register a
   `"router"` role in `make_apply_model_patches` (`_factory.py:190-260`: add `router_kind` next to `moe_kind`, gated by
   `compat`, user kwarg `router_fp32`), `_patch_forward` (`_router.py:59-92`) does the class+instance replacement.
   `mellum.py:32-45`: add `"router": "MellumTopKRouter"`, `router_kind="fp32"`.
2. **`components/cross_entropy.py:212-234`:** keep the existing fallback (it preserves the upstream aux contract,
   test `test_fused_ce_preserves_router_auxiliary_loss_contract`), and add a sibling kwarg
   `opaque_router_logits_only=True`: call the backbone with `output_router_logits=True` (HF's `capture_outputs`
   collects `router_logits` per layer via `OutputRecorder(MellumTopKRouter, index=0)`, `modeling_mellum.py:432,475`
   VERIFIED), compute the chunked CE as today, and return `MoeCausalLMOutputWithPast(loss, logits=None,
   router_logits=outputs.router_logits, aux_loss=None)` — **no** HF batch aux is added (that is the whole point).
   This is the memory-safe route the critic asked for (C10).
3. **`components/moe_aux.py` (new):** `per_example_router_load(router_logits: tuple[Tensor,…], attention_mask, *, k,
   per_layer=False) -> (h, P, Z)` — pure, out-of-place, vmap-safe (broadcast-compare one-hot, no `F.one_hot`, no
   `scatter_add_`, no `bincount`: `modeling_mellum.py:589,598` are the exact ops that break vmap, F7). `Z` = z-loss
   per example.
4. **`kernels/moe.py` dispatch / `_factory.py:316-324`:** family-level `grouped_moe` default (§5).

### 9.2 `opaque-transformers` — trainer-side (`trainer/_moe_balance.py`, new; small edits in `_dp_trainer.py`)

- **TrainingArguments** (`_training_arguments.py`): `router_aux_loss_coef: float|None = None` (None → model config),
  `router_load_release: bool = True` (auto-off when α=0), `router_load_rho: float = 0.05`,
  `router_load_ema: float|None = None` (None → workload momentum / 0.95), `router_load_per_layer: bool = False`,
  `router_z_loss_coef: float = 0.0`.
- **Setup** (after `make_functional(partition_trainable=True)`, `_dp_trainer.py:1357-1361`): register
  `router_load_probe = nn.Parameter(zeros(E or L·E), requires_grad=True)` on the model *before* functionalisation so it
  lands in `trainable_params`; extend the clipping dict with `{"router_load_probe": C_h}` before the `PerGroup` build
  at `:1456-1468` (a scalar user `clipping_norm` becomes `{"fallback": C_g, "router_load_probe": C_h}`); seed a
  `load_target` batch column at construction so `_discover_batch_keys` (`:1369, :3651`) includes it (TR-DPO pattern,
  `_dpo_trainer.py:721-748`).
- **`_augment_inputs`** (`:2302`): write `inputs["load_target"] = f̃_t.expand(B, E)`; zero the probe in
  `ctx.trainable_params`.
- **`compute_per_example_loss`** (Mellum override; also the SFT/DPO subclasses): `out = fmodel(params, **inputs,
  opaque_router_logits_only=True)`; `(h, P, Z) = per_example_router_load(out.router_logits, inputs["attention_mask"],
  k=k)`; `return out.loss + α·E·(inputs["load_target"]·P).sum() + ζ·Z + (params["router_load_probe"]·(λ·(h−k/E)).detach()).sum()`.
- **Post-noise hook:** wrap `ctx.noise_fn` at setup (cleaner than a callback): after `noise_fn`, read
  `noisy.pytree["router_load_probe"]`, run step 5 of §2.2, store `RouterLoadState`; then mask the probe leaf's update
  (torchopt `masked` chain on that key, or zero the update) before `ctx.opt.update` (`:2210-2238`).
- **Sampling-mode guards:** `clipping_mode="auto"` scales only the gradient groups (`R` as `PerGroup` with the probe
  group handled by fixed clipping — needs a small extension of `auto_scale_pytree` to accept a per-group "fixed" marker,
  or the probe leaf is pre-scaled so AUTO-S at `R=C_h`, `γ` is a no-op; simplest: reject `auto` + load release until
  the marker exists). `adaptive` + MF already rejected.
- **Checkpoint** (§8): add an `extra_state: dict[str, Any] | None` slot to `save_dp_runtime_state`
  (`trainer/_checkpoint.py:311-333`, fixed signature today) and restore in `_apply_runtime_state` (`_dp_trainer.py:5335`);
  the frozen dataclass round-trips through the serialization registry (primitives §4.4 VERIFIED: `LoadEmaState` example).
- **TRL converters** (`trl/_convert.py:69-76`): forward `router_aux_loss_coef` to the new argument when the family
  supports `router_logits_only`; keep the warning otherwise.
- **Presets:** `examples/train_dpftrl.py:873-896`, `examples/train_dpo.py:1300-1318`: `router_aux_loss_coef=1e-4`,
  `router_load_rho=0.05`, `performance_kernels_config={"grouped_moe": True}`.

### 9.3 Composition (G10)

| with | mechanism | status |
|---|---|---|
| chunked CE | `opaque_router_logits_only` keeps the chunked LM-head path; router logits are backbone outputs, independent of the head | design VERIFIED against `cross_entropy.py:212-300`; to test |
| gradient checkpointing | HF's `capture_outputs` registers its recorder hooks for the duration of one forward call and returns `router_logits` as tensors with grad_fn; the non-reentrant recompute (`opaque.patches.torch.checkpoint.huggingface._force_non_reentrant`, VERIFIED) happens in backward after the hooks are removed — no double capture. The alternative (`register_forward_hook` + Python list, primitives E3) *does* double-fire under recompute and is not used | PLAUSIBLE; test T3 |
| microbatching (`microbatch_size` chunks) | the probe leaf is an ordinary leaf: clipped per example, summed across chunks by `clipped_grad` exactly like every gradient leaf | VERIFIED by construction (`_clipped_fun.py:255-260` chunk path); test T2 |
| DDP | leaf all-reduced pre-noise; shared noise key ⇒ identical `ŷ_t`, identical EMA | VERIFIED (`_dp_trainer.py:2170-2183`, `:1478`) |
| `torch.compile` of the grad transform (`_dp_trainer.py:4191-4228`) | the per-example function is pure tensor code with tuple outputs (no Python-list side effects); HF's recorder hooks may graph-break → `fullgraph=False` fallback (`:198-232`) | PLAUSIBLE; test T5 |
| DP-FTRL latch | constant `PerGroup` incl. probe group | VERIFIED (primitives E2) |
| second-moment streams (`second_moment=True`) | the probe group would also get a squared stream and consume budget (`paired_noise_stddevs` sums all groups) — exclude the probe from the second-moment set or document the cost | needs a one-line exclusion; test T6 |

### 9.4 Test plan (behavioural, per ARC-006/ARC-012; no docstring-pinning)

- **T1 (opaque-patches, CPU, tiny Mellum):** surrogate identity on the patched model — `Σ_x ∇S(x; f(B))` from
  `vmap(grad)` equals HF `∇[α·L_aux(B)]` from the unpatched batched forward to ≤1e-5 rel-L2 in fp32 (E4 method),
  including ragged masks with the `N̄` option; `per_example_router_load` equals HF's `f(B)` for equal-length batches.
- **T2 (opaque-transformers):** end-to-end step with `router_load_release=True`: `ClippedPytree.max_norm` is a
  `PerGroup` with the probe group at `C_h`; group norm of the probe never exceeds `C_h`; the noised leaf's σ equals
  `per_group_noise_stddev`; result identical for `microbatch_size ∈ {None, 2}`; EMA update reproduces the closed form.
- **T3:** T2 with `gradient_checkpointing_enable()` — per-example `h, P` bit-identical to the unchecked run.
- **T4 (accounting):** the trainer's `_build_mechanism` output is unchanged (`poisson(gaussian(nm), q)` /
  `b_min_sep(mf_gaussian(...))`) with and without the load release (asserts "accounting literally unchanged").
- **T5:** `torch_compile=True` smoke (CPU inductor or `aot_eager`) — runs, values match eager.
- **T6:** `second_moment=True` excludes the probe from the squared stream.
- **T7 (privacy hygiene):** telemetry dict never contains per-example aux or `h`; checkpoint sidecar contains only
  `RouterLoadState`; DDP (2-rank Gloo, `distributed` marker) `RouterLoadState` equal on both ranks.
- **T8 (opaque-patches):** fp32 router patch — routes equal to an fp32 reference forward; unpatched HF contract
  untouched when `router_fp32=False`.

---

## 10. Validation plan on the real checkpoint (GPU) — G2 / G3

### 10.1 Oracle definition (G2)

Model `JetBrains/Mellum2-12B-A2.5B-Base`, bf16 weights, `attn_implementation="sdpa"`, LoRA r=16 on q/k/v/o (preset).
Three references on the **same** microbatch of 8 KStack sequences (T=1024, all-valid masks, and a second microbatch
with one padded example):

- **O1 "per-example HF loop"** (the correctness oracle): patched model (Opaque model patches incl. fp32 router), HF
  eager loop of `B=1` forwards with `output_router_logits=True` and `router_aux_loss_coef=0` (CE only) plus the
  surrogate term added by hand with the *same* `f̃`; `loss.backward()`; per-example LoRA gradients.
- **O2 "HF batched"** (what a non-DP HF Trainer computes): unpatched HF, batched forward with `num_items_in_batch`,
  `output_router_logits=True`, coefficient α, gradient of the *batch* loss; compared against the DP path's *sum* with
  `f̃ = f(B)` (identity check on the checkpoint).
- **O3 "fp32 floor"**: O1 with fp32 weights.

DP side: `clipped_grad(clipping_norm=1e9, return_aux=True)` + `vmap(grad)` per-example vectors (E1 method), bf16 and
fp32, dense and grouped MoE, with and without gradient checkpointing.

**Drift metric:** per-example rel-L2 `‖g_DP − g_O1‖/‖g_O1‖` (whole vector and per parameter group), **and separately
a route-flip counter**: fraction of `(token, layer)` whose executed top-8 set differs between DP and oracle (from the
captured logits on both sides), fraction of examples with ≥1 flip, and drift stratified by flip/no-flip (E1b method).
Also report DP-vs-O2 (sum) rel-L2 with `f̃ = f(B)` and DP-vs-O3 to bound the bf16 floor. This reproduces (and finally
makes traceable) PR #980's "29 % → 1.3 %" figure; the script lives in `examples/validate_mellum_oracle.py`.

### 10.2 Statistics to collect (G3)

On ≈20 batches of B=256 KStack samples (KStack is public; for a private dataset these would be *design-time*
measurements that must run on public proxy data):
(i) per-example gradient-norm quantiles (p50/p90/p99/max) for the preset partition, and for `{attn, router,
experts}` groups with the variants unfrozen; (ii) `‖f(B) − k/E‖₂/(k/E)`, per-coordinate RMS and max, per batch, plus
its batch-to-batch sd; (iii) per-example `‖d(x)‖₂` distribution vs the bound 2.646 and the fraction of experts unused
per example (pooled and per layer); (iv) route-flip rate bf16-router vs fp32-router on the same tokens; (v) per-layer
vs pooled imbalance (to decide whether the per-layer option is ever worth ×5.29); (vi) step time and peak memory,
dense vs grouped MoE, microbatch 8.

### 10.3 Acceptance criteria

1. fp32: DP-vs-O1 rel-L2 ≤ 1e-5, **0 flips**, on every configuration (dense/grouped, gc on/off, padded/unpadded).
2. bf16 + fp32 router: DP-vs-O1 rel-L2 ≤ HF's own batched-vs-loop bf16 spread measured on the same batch (O2-loop vs
   O2-batched), flips ≤ 0.1 % of (token, layer) pairs; report the number.
3. Surrogate identity on the checkpoint: `Σ_x ∇S(x; f(B))` vs O2's `∇[α L_aux]` rel-L2 ≤ 1e-4 (fp32).
4. Signal-to-noise: with the default ρ=0.05 and the measured imbalance from (ii), the smoothed
   `‖noise‖/‖f(B)−k/E‖ ≤ 0.3` (SGD) — otherwise raise ρ per §2.6 or accept that the term is silent.
5. `C_g` chosen at the p50 of (i); clip rate 40–60 % in the first 100 steps of a DP run.
6. Grouped MoE ≥ 3× faster than dense at microbatch 8 (informational; sets the default).

---

## 11. Risks and what would falsify the design

1. **The imbalance on real fine-tuning data is tiny** (`‖f(B)−k/E‖ ≪ 0.1·k/E`): the released estimate is then
   noise-dominated at any affordable ρ. This does not falsify faithfulness — the true aux gradient is ∝ the same
   imbalance and is equally silent — but it means the release buys nothing; the mitigation is `α=0` (drop), which the
   design supports. Decided by §10.2(ii).
2. **Residual route flips at 28 layers even with fp32 routing** (e.g. from SDPA kernel selection under vmap,
   `runtime/masking.py:195-216` microbatch-coupled fast path, or CUDA autocast staging in `Opaque_MoE`,
   divergence §1.5): would break acceptance criterion 2 and the "numerically stable vs HF" claim. Falsifier: flips
   > 0.1 % with the fp32 router. Mitigations: eager attention for the oracle, `attention_mask=None` fast path for
   fully valid batches, or routing-decision hysteresis (would deviate from the model — last resort).
3. **HF `capture_outputs` under vmap + non-reentrant checkpointing** double-captures or drops layers: T3 falsifies;
   fallback is the module-hook route with an explicit recompute guard (primitives E3), or returning the logits
   functionally from a patched `MellumDecoderLayer`.
4. **PEFT `target_parameters` on stacked experts under `functional_call`+vmap** fails (M5): blocks the
   experts-trainable variant only; the attention-LoRA presets are unaffected.
5. **Per-layer faithfulness demanded** (Megatron semantics): ×5.29 noise; usable under MF at ρ≥0.2 but no longer
   "free"; the pooled default is a deliberate deviation from the pretraining pooling, documented in §1.5.
6. **Non-constant LR schedules under MF**: the workload includes query weights; an unweighted EMA is then not the
   matched filter (§6.1). The implementation must derive `β_f`/weights from the same `lr_schedule` passed to
   `band_mf_strategy`, or fall back to the DP-SGD factor 0.160 (still valid, just 2× noisier).
7. **`torch.compile` graph breaks** on HF's recorder hooks: only performance; the `fullgraph=False` fallback exists.
8. **Telemetry posture**: the pre-existing un-noised `loss`/`loss_aux` logging (F11) is outside this design but sits
   next to it; a reviewer applying the review protocol's "all releases in the privacy statement" will (rightly) flag
   it — the design adds no new un-noised release and recommends the existing one be documented or noised.
9. **Second-moment streams**: forgetting the probe exclusion (T6) silently spends budget on a useless squared load
   stream — a utility bug, not a privacy bug (the allocation stays `gaussian(nm)`).
10. **The DPO pair pooling** (§7) has not been exercised; a wrong per-pair `T_x` divisor would change the aux weighting
    but not privacy (the structural bound holds for any masked mean).

**Sources relied on (all VERIFIED by phase-1/critic from primary text unless marked):** Switch Transformer eqs. (4)–(6)
https://arxiv.org/abs/2101.03961 §2.2; Andrew et al. 2021 Thm 1 https://arxiv.org/abs/1905.03871; Zhu–Dong–Wang
Def. 7 / Thm 10 https://arxiv.org/abs/2106.08567; Feldman–Shenfeld Lemma 3.2 / Thm 3.3 https://arxiv.org/abs/2602.17284
(as cited by `src/amplification/poisson.rs`); Denisov et al. Thm 2.1 https://arxiv.org/abs/2202.08312; Dong–Roth–Su
Thm 2.7 https://arxiv.org/abs/1905.02383 (whitening/GDP of the anisotropic Gaussian); ST-MoE §3.1 eq. (5)
https://arxiv.org/abs/2202.08906; Mellum 2 Technical Report https://arxiv.org/abs/2605.31268 §3.6 (running-average f),
§5.1.2 (SFT α=1e-4), appendix (FP32 router); Tholoniat et al. https://arxiv.org/abs/2402.07334 (prior DP-MoE, drops
aux). Megatron's per-layer averaging (§1.2 row (c)) is PLAUSIBLE — Megatron source not read.
