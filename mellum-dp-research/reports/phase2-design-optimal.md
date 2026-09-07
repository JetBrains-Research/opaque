# Phase 2 — design (lens: PRIVACY-UTILITY OPTIMAL). Mellum2 MoE under Opaque DP: the cheapest mechanism that keeps the real objective

Agent: `design-optimal`. Repo `/home/user/opaque` @ `ef1abc5` (branch `claude/mellum-dp-representation-r6slaz`), nothing tracked
modified. Evidence tags: **VERIFIED (this session)** = I read the cited lines or ran the cited script now;
**VERIFIED (phase-1 <agent>)** = established by a phase-1 report that executed/read it (F1–F11 of the digest);
**PLAUSIBLE** = derived, or read from secondary text, not executed. Repo paths relative to `/home/user/opaque`;
HF = `.venv/lib/python3.11/site-packages/transformers/…` (transformers 5.16.1). Scripts and raw output of this session:
`scratchpad/research/design-optimal/{cost_table.py,cost_table.json,mf_filter.py}` (CPU, 45 s total).

**The result in one paragraph.** Mellum2's only inseparable batch term is the Switch-style aux loss
`L_aux(B) = E·Σ_e f_e(B)·P_e(B)`; because `f` is argmax-derived, `∇L_aux(B) = Σ_x ∇S(x; f(B))` *exactly* with
`S(x; f̃) = E·Σ_e f̃_e·P_e(x)` (F3, VERIFIED phase-1 math to 1e-17), and since `Σ_e P_e(x) = 1` the signal is the
**imbalance** `f̃ − k/E` only. So the whole DP question is "what is the cheapest way to obtain a good estimate of a
64-dimensional imbalance vector that is a public constant of the step". This document ranks every lever by its price at the
preset regime (`nm = 0.5622, B̄ = 256, k = 8, E = 64, ε = 3, T = 15625`) and finds: (i) the price of a same-batch
release scales as `√(1+ρ)` in gradient noise where ρ is the fraction of the clipping budget given to the load leaf —
at **ρ = 0.02 that is ×1.010 gradient noise, ε literally unchanged, accountant call literally unchanged**; (ii) under band-MF
the correlated noise makes *any* low-pass filter of the load stream 2–3.5× more accurate than under DP-SGD (EMA β=0.99:
noise factor 0.0249 vs 0.0709, VERIFIED this session), so the smoothed load error at ρ = 0.02 is **0.83 % of k/E** under
the DP-FTRL preset and 2.4 % under DP-SGD; (iii) an *independently subsampled, forward-only* release is the true
privacy optimum under Poisson (`σ_h = 2·nm`, every 4 steps: ε 3.000 → 3.003, i.e. ×1.0002 gradient noise, 2.1 % smoothed
load error) at +8 % compute, and is impossible under b-min-sep; (iv) amortised, per-layer, replace-one, randomised-response
and "pay in ε" variants are all dominated and are tabulated to show it; (v) loss-free balancing under DP is a free
post-processing of the same release (sign of the noised deviation), but for a checkpoint with no bias tensor it is an
architecture change, admissible only as an opt-in shipped with the adapter. The recommended default for both stacks is
the in-stream ρ = 0.02 release with a noise-aware (James–Stein) shrinkage of the smoothed imbalance, fp32 routing, and
AUTO-S on the gradient group (the load group exempt).

---

## 1. Objective (G1)

### 1.1 The per-example loss, exactly

For example `x` (right-padded, attention mask `m_{x,t}`, `T_x = Σ_t m_{x,t}`), layers `l = 1..L` (L = 28), experts `e = 1..E`
(E = 64), `k = 8`, router weight `W_l ∈ R^{E×2304}`:

```
z^l_{x,t}   = W_l · h^l_{x,t}                                  router logits — fp32 (§3)
p^l_{x,t}   = softmax(z^l_{x,t})  over all E experts             fp32 (HF: modeling_mellum.py:335, VERIFIED this session)
S^l_{x,t}   = topk_k(z^l_{x,t})                                  the EXECUTED expert set (the one the experts ran with)
P_e(x)      = (1/(L·T_x)) Σ_l Σ_t m_{x,t} · p^l_{x,t,e}          per-example mean router prob   (differentiable)
h_e(x)      = (1/(L·T_x)) Σ_l Σ_t m_{x,t} · 1{e ∈ S^l_{x,t}}     per-example load fraction      (piecewise constant)
d(x)        = h(x) − (k/E)·1                                     centred load;  Σ_e d_e(x) = 0

ℓ_x(θ; f̃_t) = CE_x(θ) + α·E·Σ_e f̃_{t,e} · P_e(x; θ)   [ + ζ·Z_x(θ)  z-loss, opt-in, §7 ]
             = CE_x(θ) + α·E·Σ_e (f̃_{t,e} − k/E) · P_e(x; θ) + α·k          (identity, since Σ_e P_e = 1)
```

`CE_x` is HF's own per-example token-mean causal-LM CE over label-valid tokens, read by
`DPTrainer.compute_per_example_loss` from `fmodel(params, **inputs)["loss"]`
(`packages/opaque-transformers/src/opaque/api/transformers/trainer/_dp_trainer.py:2314`, VERIFIED this session) through
the chunked LM-head path (`packages/opaque-patches/src/opaque/api/patches/transformers/components/cross_entropy.py`,
`chunked_linear_cross_entropy=2048` in `packages/opaque-patches/src/opaque/api/patches/transformers/models/mellum.py:44`,
VERIFIED this session).

The batch objective implemented is `(1/B̄)·Σ_{x∈B_t} ℓ_x(θ; f̃_t)`; its aux gradient equals HF's `∇[α·L_aux(B_t)]`
**exactly** when `f̃_t = f(B_t)` and all `T_x` are equal (F3/F4; VERIFIED phase-1 math: 8e-18 full mask; 2.4e-7 fp32 on a
real forward, phase-1 empirical E4).

### 1.2 Target of faithfulness and why

| candidate | verdict under this lens |
|---|---|
| (a) HF logical-batch pooled formula (`modeling_mellum.py:540-606`, one denominator `L·T_tot`, `Σ_e f_e = k`; F1) | **TARGET.** It is the objective the released checkpoint declares (`router_aux_loss_coef=0.001`), the only one reproducible from the HF artefact, and — decisive for this lens — the cheapest: it needs a **64-dim** release with structural bound `√(k(1−k/E)) = 2.646`. |
| (b) HF-Trainer-realised (per-microbatch f, coefficient `G·α`; critic Exp A rel-L2 0.0, VERIFIED phase-1 critic) | rejected: an artefact of `trainer.py:1961-1963`; at the preset it would mean f over 8 sequences (≈ per-sequence regulariser) and `α_eff = 0.032`. The DP path has no accumulation (microbatches are vmap chunks of one logical batch), so it is *more* faithful to (a) than HF Trainer itself. |
| (c) Megatron per-layer running-average f (Mellum2 TR §3.6; literature F.2) | per-layer needs an `L×E = 1792`-dim release with bound `√(kL(1−k/E)) = 14.0` ⇒ **×√L = ×5.29 relative noise** at equal privacy (math §3 VERIFIED phase-1). Not reproducible from HF. The *running-average* aspect **is** adopted (`f̃_t` is a filtered stream of releases — literally the kind of estimate pretraining used, TR §3.6 VERIFIED phase-1 literature). Per-layer is an opt-in row of the cost table. |
| (d) per-sequence aux (DeepSeek-V2/V3, Wang et al.) | zero privacy cost but a different regulariser (cos 0.26–0.39 to the batch gradient, anti-specialisation closed form `E·P̄_{S_x}`; VERIFIED phase-1 math §2 / empirical E4). Not faithful; offered as the α-cost-free fallback only if the user asks for it. |

### 1.3 Mask, CE weighting, coefficient, pooled vs per-layer, executed routes

- **Mask:** attention mask for `h`, `P` (HF passes `attention_mask` at `modeling_mellum.py:697`; prompt tokens with label −100
  were routed and count). Label mask for CE (unchanged).
- **CE weighting:** equal example weights (Opaque convention, `packages/opaque-alignment/.../sft/loss/_nll.py:15-24`,
  VERIFIED phase-1 math); coincides with HF token weighting for the packed T = 1024 presets (F4). Consequently the
  released statistic is the *example-mean* load `(1/B̄)Σ_x d(x)`; a public `N̄` restores token weighting if ever wanted
  (same lever as CE; not default).
- **Coefficient α — default `1e-4`** (Mellum2's own SFT value, TR §5.1.2: "router already well-balanced after pre-training",
  VERIFIED phase-1 literature F.2). Rationale under this lens: the aux term's *utility* value during fine-tuning of a balanced
  checkpoint is small (OLMoE §4.3, Tholoniat; F10), so the coefficient JetBrains themselves chose for SFT is the
  utility-optimal prior; `α = config.router_aux_loss_coef = 1e-3` is the "faithful-to-config" alternative (one flag);
  `α = 0` recovers today's behaviour. Value at balance: `α·k = 0.0008` nats (1e-4) / `0.008` (1e-3).
- **Pooled (E-vector), not per-layer:** ×5.29 noise for the same budget (row in §2.6). Per-layer is opt-in.
- **f from executed fp32 routes, not HF's recomputation:** HF's `load_balancing_loss_func` recomputes top-k from a
  *bf16* softmax (`modeling_mellum.py:584`) while the forward routed on an fp32 softmax (`:335`); 0.03–1.7 % of tokens
  differ (critic R9, VERIFIED phase-1 toy). Definition adopted: `S` = the router's own `router_indices`, `p` = the router's
  own fp32 softmax. The deviation from HF's *number* is ≤ 1e-3·k/E — 10× below the smallest noise row of §2.6.

---

## 2. Mechanism and accountant (G4)

### 2.1 Adjacency, protected unit, sensitivities

Example-level (one collated row; for DPO one preference pair), **add/remove** adjacency — Opaque's default
(`.junie/differential-privacy-review.md` "Adjacency"; `clipped_grad` contract
`packages/opaque-engine/src/opaque/api/engine/clipping/_clipped_grad.py:135-145`, VERIFIED phase-1 primitives).
Per-record bounds (both hold for **every** prefix of previous outputs, because neither depends on `f̃_t`):

| group | per-record L2 bound | why |
|---|---|---|
| gradient leaves | `C_g` (fixed clip) or `R` (AUTO-S) | as today |
| load leaf `λ·d(x)` | `C_h = λ·Δ_h`, `Δ_h = √(k(1−k/E)) = 2.6458` | `‖h(x)‖₂ ≤ √k` (each token/layer contributes k distinct experts), centred `‖h−k/E‖² = ‖h‖² − k²/E` (math §3, VERIFIED phase-1; attained when every token in every layer routes to the same 8 experts). **Structural — never clipped — release unbiased.** |

Replace-one adjacency doubles both bounds; nothing else changes. Centring buys 6.5 % (`2.646` vs `2.828`), makes the sum
structurally zero (so the realised Poisson batch size `|B_t|/B̄`, rel. sd `1/√256 = 6 %`, only multiplies the *signal*
by ≈1 — no renormalisation to `Σ = k` is needed; critic M11 closed), and lets the sum-zero projection strip the noise's
mean component (variance ×63/64, VERIFIED this session, trivial but free).

### 2.2 Per-step mechanism — DP-SGD / Poisson (default: in-stream)

Public state entering step `t`: `θ_t`, `d̃_t ∈ R^E` (smoothed imbalance), `s_t` (its known noise std), `f̃_t = clamp(k/E + shrink(d̃_t), 0, 1)`.
Constants: `C_g`, ratio `ρ = C_h/C_g`, `λ = ρ·C_g/Δ_h`, `B̄ = q·N` (`normalize_by=expected_batch_size`,
`_dp_trainer.py:4267/4277/4288`, VERIFIED this session).

1. **Sample** `B_t ~ Poisson(q)` (existing sampler).
2. **Per example inside `vmap(grad)`** (fp32 router, §3): forward → `CE_x`; router stats → `p`, executed `S`, `P(x)`, `d(x)`;
   loss `ℓ_x = CE_x + α·E·⟨f̃_t, P(x)⟩ + ⟨z, λ·d(x).detach()⟩` with `z ∈ R^E` the zero-valued probe parameter
   (value term ≡ 0; `∂ℓ_x/∂z = λ·d(x)`; other leaves unaffected — VERIFIED phase-1 primitives E2).
3. **Per-group clip** (`_pytree.py:439`, VERIFIED this session): gradient group(s) to `C_g`, probe group to
   `C_h(1+1e-6)` (never active).
4. **Sum + Gaussian**, Opaque's per-group allocation (`packages/opaque-engine/src/opaque/api/engine/noise_allocation.py:103-110`):
   `σ_g = nm·√(C_g·S)/B̄`, `σ_h = nm·√(C_h·S)/B̄`, `S = C_g + C_h`. Output `(ĝ_t, ŷ_t)`.
5. **Post-processing (public, free):** `d̂_t = ŷ_t/λ`; `d̂_t ← d̂_t − mean(d̂_t)` (sum-zero projection);
   `d̃_{t+1} = β_f·d̃_t + (1−β_f)·d̂_t`; noise std of `d̃` is known exactly: `s = (σ_h/λ)·√((1−β_f)/(1+β_f))` at stationarity
   (`s_t` tracked by the same recursion on variances during burn-in);
   **shrinkage** `d̃⁺ = d̃ · max(0, 1 − (E−1)·s²/‖d̃‖²)` (positive-part James–Stein toward the balanced point — post-processing,
   removes the random regularisation direction the noise would otherwise inject when the true imbalance is below the noise
   floor; when `‖d̃‖ ≫ s√(E−1)` it is the identity); `f̃_{t+1} = clamp(k/E + d̃⁺, 0, 1)`. `d̃_0 = 0` ⇒ `f̃_0 = k/E` ⇒ zero aux
   gradient at step 0 (exactly the real objective's behaviour at balance, F3). Zero the probe parameter.
6. **Optimizer** on `ĝ_t` (probe leaf re-zeroed each step; its 64-float optimizer state is inert).

**Why the accountant is unchanged.** The step is one Gaussian on the concatenation `[Σ_x clip g_x ; λΣ_x d(x)]` with
diagonal covariance; whitening gives a sensitivity-1 Gaussian at multiplier `σ_eff` with
`σ_eff⁻² = C_g²/σ_g² + C_h²/σ_h²`, and Opaque's allocation makes this `1/nm²` exactly
(`noise_allocation.py:50-61` docstring; numerically `Σ(C_i/σ_i)² = 1/nm²` — VERIFIED phase-1 primitives E1(f); the
dominating pair is Zhu–Dong–Wang Def. 7 for the Gaussian, https://arxiv.org/abs/2106.08567, and Dong–Roth–Su Thm 2.7 for
the whitened statistic, https://arxiv.org/abs/1905.02383 — both VERIFIED phase-1 critic R6 from the PDFs). Precedent:
Andrew et al. 2021 Thm 1 (https://arxiv.org/abs/1905.03871, VERIFIED phase-1 literature B1): the clipped-count release of
adaptive clipping is this same joint Gaussian. Subsampling applies to the *one* joint mechanism (both releases share the
Poisson coin; Feldman–Shenfeld Lemma 3.2 / Thm 3.3 as implemented in `src/amplification/poisson.rs:15-42`, VERIFIED
phase-1 critic R6). `f̃_t` is a function of previous outputs — the same kind of dependence as `θ_t`; adaptive composition
(Zhu–Dong–Wang Thm 10) charges nothing for it (math §4(d)).

```
accountant (DP-SGD, in-stream):   dpsgd.poisson(dpsgd.gaussian(nm), q) * T         ← unchanged (_dp_trainer.py:4301 _build_mechanism)
```

The naive "σ·C_g on the gradient, σ·C_h on the load" is `gaussian(nm/√2)`, **not** `gaussian(nm)` (math 4(b) REFUTED row) —
never used here.

### 2.3 Per-step mechanism — DP-SGD / Poisson, the privacy optimum (opt-in: independent forward-only release)

Every `m` steps, draw a **second, independent** Poisson sample `B'_t ~ Poisson(q₂)` with its own key
(`fold_in(key, "opaque.moe.load_release")`), run a **forward-only** vmap of the routing statistic
(`clipped_fun`, `packages/opaque-engine/src/opaque/api/engine/clipping/_clipped_fun.py:492`, `clipping_norm = Δ_h`,
`normalize_by = q₂·N`), add `N(0, (c·nm·Δ_h)²)` with a separately rooted key, post-process as in §2.2 step 5 with the EMA
running over releases. The gradient mechanism is untouched (`C_g`, `nm`).

```
accountant:  dpsgd.poisson(dpsgd.gaussian(nm), q) * T  |  dpsgd.poisson(dpsgd.gaussian(c·nm), q₂) * (T/m)
```
Validity: two mechanisms with **fresh** sampling coins and fresh Gaussian noise, adaptively interleaved — exactly
Zhu–Dong–Wang Thm 10 with each factor dominated via Lemma 3.2 of Feldman–Shenfeld (both VERIFIED phase-1 critic R6);
Opaque's `Poisson` accepts any inner `DpProcess` and `|` composes at the process level
(`packages/opaque-dpsgd/src/opaque/api/accounting/dpsgd/amplification/_poisson.py:25-45`, `core/_base.py:483`, VERIFIED
this session). Why it is cheaper: two small-`q` draws at `(nm, 2nm)` compose almost additively in the GDP regime, whereas
one draw at `σ_eff = nm/√(1+1/c²)` sits on the strongly convex part of `ε(1/σ)` (primitives E1(g): ε 0.34 → 0.97 for
σ 1 → 0.707). **Unavailable under b-min-sep** (no second draw; §2.4).

### 2.4 Per-step mechanism — DP-FTRL / band-MF / b-min-sep (the actual Mellum2 preset)

Identical to §2.2 with two substitutions:
- Participation: `BMinSepSampler` (`packages/opaque-dpftrl/src/opaque/api/dpftrl/sampling/_b_min_sep.py:30`) — the load
  leaf shares the example's participation pattern (same example, same step), so `min_sep`, `max_participations` are the
  gradient's.
- Noise: `mf_gaussian_noise(trainable, band_mf_strategy(bands=64, momentum=β), n_steps, …, noise_multiplier=nm)` with the
  same constant `PerGroup` max_norm: `base_stddev = per_group_noise_stddev(max_norm, nm)`, then the streaming `C⁻¹`
  multiplication; realised per-step σ on every leaf `= base·‖row_t(C⁻¹)‖`
  (`packages/opaque-dpftrl/src/opaque/api/dpftrl/noise/_mf_gaussian_noise.py:163-186`, VERIFIED this session). The
  constant-max_norm latch (`_engine.py:473-493`, VERIFIED this session) accepts a constant `PerGroup` (VERIFIED phase-1
  primitives E2, identity and band-MF).
- Correctness of per-group MF: whitening per group gives
  `Σ_g ‖C(G_g−H_g)‖²_F/σ_g² ≤ sens(C)²·Σ_g C_g²/(nm²·C_g·S) = sens(C)²/nm²` because the participation-pattern sensitivity is
  degree-1 homogeneous in the row bound (math §5(i)); Denisov et al. Thm 2.1 (https://arxiv.org/abs/2202.08312, VERIFIED
  phase-1 critic R6) covers adaptively chosen rows — `f̃_t` is a function of previous outputs, the theorem's setting.

```
accountant (DP-FTRL):  dpftrl.b_min_sep(dpftrl.mf_gaussian(nm, band_mf_strategy(64, β)), n_steps=T, p0=p0)   ← unchanged
```
The trainer calibrates `nm` from the target ε (`_calibrate_noise`), so ε is *held* automatically and the entire price
shows up as `×√(1+ρ)` on the gradient noise. **Amortisation is impossible** (latch forbids toggling the group) **and
unnecessary** (§2.6 row "MF, EMA .99"). An independent Poisson side-release composed at the process level
(`Accountant(prefix=cached(b_min_sep(...))) | poisson(gaussian(c·nm), q₂)*(T/m)`, seam `_dp_trainer.py:287-299`) would need
a concurrent-composition argument for the interleaving of two mechanisms whose inputs depend on each other's outputs
(Vadhan–Zhang 2023 concurrent composition for f-DP — **PLAUSIBLE, theorem not re-fetched**) plus a second sampler over the
dataset; not adopted, not needed.

Adaptive clipping stays excluded under MF; AUTO-S is allowed on the gradient group; the load group must **not** be
AUTO-S-scaled (it would map every `d(x)` to norm ≈ `C_h`, i.e. a mean *unit direction* — a different statistic; critic R11) —
§4 specifies the mixed mode.

### 2.5 Levers compared (all at ε = 3, nm = 0.5622, B̄ = 256, q = 5.12e-4, T = 15625, δ = 1e-6; accountant runs VERIFIED this session unless marked)

| lever | what it buys | what it costs | verdict |
|---|---|---|---|
| same-batch joint `PerGroup` at ratio ρ | one mechanism, accountant unchanged, works under both stacks | `×√(1+ρ)` gradient noise (ε held) | **default**, ρ = 0.02 |
| same-batch separate Gaussian, `σ_h = c·nm` ("pay in ε") | gradient σ untouched | ε rises (3.0 → 3.234 at c = 7.14) — the *same family* as the row above (`c = √(1+1/ρ)`, math 4(c)); choosing it is a reparametrisation, not a saving | dominated by holding ε |
| independent Poisson draw, forward-only, `c = 2`, every `m = 4` steps | ε 3.000 → **3.003** (or ×1.0002 grad noise); 2.1 % smoothed load error | +8 % compute (one forward on ≈256 examples per 4 steps, ≈ 1/3 of a step's fwd+bwd per release, PLAUSIBLE ratio); DP-SGD only | **opt-in optimum** for DP-SGD |
| independent draw with 4× batch (`q₂ = 4q`, `c = 1`) | 4× less noise per release | ε 5.30 — amplification lost; noise per release scales `1/B̄₂` but ε scales far faster | rejected: use small `q₂`, not large |
| amortised same-batch every `m` steps (DP-SGD heterogeneous composition) | ρ = 0.2, m = 16: ε 3.194, 2.6 % smoothed | same budget spent in bursts: average gradient-variance inflation `1+ρ/m ≈ ×1.006` RMS, lag `m/(1−β)` | ≈ equal to small-ρ every step; not worth the complexity |
| centred `d(x)` instead of `h(x)` | 6.5 % less noise; no `Σ = k` renormalisation; sum-zero projection | none | **adopted** |
| pooled vs per-layer | per-layer reproduces Megatron's mean-of-products | ×5.29 relative noise; to match pooled accuracy needs ρ ×28 ⇒ ×1.25 gradient noise | pooled default; per-layer opt-in |
| EMA vs MF-workload-matched filter (§6) | under band-MF every low-pass filter is 2–3.5× better than under DP-SGD | lag `1/(1−β_f)` | β_f = 0.99 (lag 100 = 0.64 % of horizon) |
| public-data `f̃` (Davody et al. 2020, https://arxiv.org/abs/2006.10919; Ponomareva §5 option (a), VERIFIED phase-1 literature B2/B3) | zero privacy cost, zero release | estimates the *public* distribution's imbalance, not the private one; distribution-shift bias | opt-in when a public corpus exists; not faithful |
| loss-free balancing (§7.3) | no gradient term; balance controlled directly | architecture change for this checkpoint; needs the same release | opt-in |
| per-sequence aux | zero cost | different regulariser | fallback only |
| randomised response on `sign(d)` | 1 bit/expert | one example can flip all 64 signs ⇒ per-step ε = 64·ln((1−p)/p) = 12.8 (p = .45) … 70 (p = .25) — vs < 0.5 *total* for the Gaussian route | **dominated** (VERIFIED arithmetic this session) |

### 2.6 ONE cost table — preset regime `nm = 0.5622, B̄ = 256, k = 8, E = 64, C_g = 0.9, L = 28`

Single-release relative noise per entry of `d̂` (pooled, centred), in units of `k/E = 0.125`:
`r₁ = nm·Δ_h·c/(B̄·k/E) = 0.04648·c`, `c = √(1+1/ρ)` for the in-stream route (VERIFIED: `cost_table.py`). Smoothing factors
(noise std of the filtered stream / noise std of one release), VERIFIED this session (`mf_filter.py`, band-MF `bands=64,
momentum=0.95, n=2048`, exact row norms of `F·C⁻¹`): DP-SGD EMA `√((1−β)/(1+β))` = 0.1601 (β=.95) / 0.0709 (β=.99);
band-MF single step `‖row_t(C⁻¹)‖` = 1.4313 (stationary from t ≈ 7); band-MF EMA(.95) = 0.0824; band-MF EMA(.99) = **0.0249**.
ε column = `poisson(gaussian(nm) | gaussian(c·nm), q)*T`, `epsilon_at(1e-6)`, baseline 3.000.

| ρ = C_h/C_g | c | **gradient-noise inflation √(1+ρ)** (ε held) | ε if nm held instead | r₁ single release | DP-SGD after EMA .95 / **.99** | band-MF single step | band-MF after EMA .95 / **.99** | per-layer opt-in (×5.29), best filter SGD / MF |
|---|---|---|---|---|---|---|---|---|
| 0.5 | 1.732 | ×1.225 | 5.443 | 8.1 % | 1.3 % / 0.57 % | 11.6 % | 0.67 % / 0.20 % | 3.0 % / 1.1 % |
| 0.2 | 2.449 | ×1.095 | 4.219 | 11.4 % | 1.8 % / 0.81 % | 16.3 % | 0.94 % / 0.28 % | 4.3 % / 1.5 % |
| 0.1 | 3.317 | ×1.049 | 3.703 | 15.4 % | 2.5 % / 1.09 % | 22.1 % | 1.27 % / 0.38 % | 5.8 % / 2.0 % |
| 0.05 | 4.583 | ×1.025 | 3.417 | 21.3 % | 3.4 % / 1.51 % | 30.5 % | 1.76 % / 0.53 % | 8.0 % / 2.8 % |
| **0.02 (default)** | 7.141 | **×1.010** | 3.234 | 33.2 % | 5.3 % / **2.35 %** | 47.5 % | 2.73 % / **0.83 %** | 12.4 % / 4.4 % |
| 0.01 | 10.05 | ×1.005 | 3.172 | 46.7 % | 7.5 % / 3.31 % | 66.9 % | 3.85 % / 1.16 % | 17.5 % / 6.1 % |
| *indep. draw c=2, m=4 (DP-SGD)* | — | ×1.0002 | 3.003 | 9.3 % | 2.1 % (EMA .9 over releases, lag 40 steps) | n/a | n/a | 11 % |
| *indep. draw c=1, m=4 (DP-SGD)* | — | ×1.013 | 3.128 | 4.6 % | 1.1 % | n/a | n/a | 5.6 % |
| *indep. draw c=2, m=16 (DP-SGD)* | — | ×1.0001 | 3.001 | 9.3 % | 2.1 % (lag 160) | n/a | n/a | 11 % |

**How to read it against the objective.** The surrogate gradient is `α·E·Σ_e (f̃_e − k/E)·∇P_e(x)`, so the error that
matters is `‖noise‖/‖d_true‖`, not `r` against `k/E`. With the default row and an unknown real per-coordinate RMS imbalance
`δ·k/E`: relative aux-gradient error ≈ `0.0083/δ` (MF) or `0.0235/δ` (SGD) — at `δ = 0.1` (a well-balanced checkpoint)
that is 8 % / 24 %; the shrinkage of §2.2 step 5 zeroes the term automatically when `δ` is below `s·√63/(k/E)` ≈ 0.07 (MF)
/ 0.19 (SGD), which is exactly where the true objective's gradient is negligible too. The design therefore degrades to
*what the real objective does at balance*, never to a different regulariser.

**Recommended default with numbers.** Both stacks: in-stream `PerGroup`, `ρ = 0.02` ⇒ `λ = 0.02·0.9/2.6458 = 0.00680`,
`C_h = 0.0180` (with AUTO-S `R = 1`: `λ = 0.00756`, `C_h = 0.020`); `β_f = 0.99`; shrinkage on; `f̃_0 = k/E`;
`α = 1e-4`. Price: **ε = 3.0 unchanged, accountant call unchanged, gradient noise ×1.010** (equivalently 2 % of the
clipping budget), smoothed load error **0.83 % of k/E under band-MF b-min-sep, 2.35 % under DP-SGD**, lag 100 steps
(0.64 % of the 15625-step horizon), no extra forward. DP-SGD users who can afford +8 % compute switch to the independent
draw (`c = 2, m = 4`): ×1.0002 gradient noise, 2.1 % smoothed error. Experts/router-trainable variants (§7), where balance
matters more, raise ρ to 0.1 (×1.049; MF 0.38 %) or use the independent draw with `c = 1`.

---

## 3. Router precision and routing (G5)

**Decision: fp32 router logits by default for the mellum family; routes computed per example inside vmap from the current
model; `f` from those executed routes; no frozen-base pinning; no STE.**

- Upstream computes `router_logits = F.linear(hidden_bf16, W_bf16)` in bf16 and only the softmax in fp32
  (`modeling_mellum.py:334-335`, VERIFIED this session). bf16 has fixed *relative* resolution, so ≈1 % of tokens per layer
  sit inside its rounding of a top-k tie irrespective of router sharpness (E2, VERIFIED phase-1 empirical); flips cause ≈8×
  excess per-example gradient error on router/expert parameters and +0.2 pp on attention-only ones (E1b). Mellum2's
  pretraining ran the router in FP32 (TR appendix, VERIFIED phase-1 literature) and the authors document train/inference
  route disagreement on this checkpoint (TR §5.2). Under this lens fp32 routing is a pure win: cost one `64×2304` fp32 GEMM
  per token per layer (0.3 MFLOP vs ≈99 MFLOP routed expert compute — 0.3 %, PLAUSIBLE arithmetic), memory an fp32 view
  of `W` (590 KB/layer) or an on-the-fly cast, and it removes the bf16 tie set that makes the per-example loss
  discontinuous *between two numerically different evaluations of the same example* (math §6: pinned/continuous gradient
  vs an O(1) jump at a tie — VERIFIED phase-1 toy).
- **Patch:** `MellumTopKRouter.forward` replacement: `logits = F.linear(h.float(), W.float())`; `p = softmax(logits)`;
  `topk`; renormalise (`norm_topk_prob=True`); `scores.to(h.dtype)`; return `(logits_fp32, scores, indices)`. Downstream
  consumers use only `scores`/`indices` (`MellumSparseMoeBlock.forward`, `modeling_mellum.py:350-355`, VERIFIED this
  session), so the model output changes only through the more precise decision. Both HF's aux top-k and the executed
  top-k then coincide (critic R9 discrepancy disappears).
- **Pinning:** routes are per-example constants of `(x, θ_t)` already; no cross-example dependence, no DP consequence.
  Frozen-base pinning is rejected: it freezes training-time routing while inference routing drifts with the LoRA-modified
  hidden states (math §6 utility argument). Cross-step discontinuity is harmless for DP and expected for utility.
- **Consistency:** `h(x)`, `P(x)`, and (opt-in) `Z_x` use the same fp32 `p`/`z` and the same executed `S`.
- **Oracle implication (§10):** the drift oracle applies the same fp32-router patch so that the metric isolates
  vmap/per-example effects; the deviation from *unpatched* HF-bf16 routing is reported separately and is a deliberate,
  pretraining-faithful choice.

---

## 4. Clipping norm and per-example gradient norms

### 4.1 What the design does so that a reasonable clipping norm exists

1. **The load release never touches the gradient's clipping norm.** It is its own `PerGroup` group with a structural bound;
   the gradient group keeps `C_g` (preset 0.9, `examples/train_dpftrl.py:611`, VERIFIED this session). No clipping norm has
   to be chosen for the 64-dim statistic at all (this is the reason to release fractions, not counts).
2. **The aux term cannot inflate the norm distribution.** Its per-example gradient is `α·E·Σ_e(f̃_e−k/E)∇P_e(x)`: at
   `α = 1e-4` and a per-example aux gradient comparable to the CE gradient at coefficient 1 (E3: median 2.6–5.9 vs CE
   1.9–6.1, VERIFIED phase-1 toy) the norm change is < 0.01 %, and it vanishes at balance. The *per-example* aux variant
   (`E·P̄_{S_x}`) would add a heavier-tailed component (max/median 1.6–2.0) — one more reason not to use it.
3. **fp32 routing removes the tail source that matters for router/expert training** (§3): per-example gradient error on
   router/expert leaves 12.9 %/11.1 % → 1.7 %/1.5 % when routes are fp32-consistent (E1b).
4. **Partition-aware groups when experts/router are trained** (§7): `per_group(trainable, gate=C_r, experts=C_e, fallback=C_a,
   router_load_probe=C_h)` (`packages/opaque-engine/src/opaque/api/engine/clipping/_per_group.py:44`, VERIFIED this
   session) — the MSE-optimal allocation keeps `gaussian(nm)`; per-example expert gradients are sparse across experts for
   real documents (E5, caveated by critic R8) and high-dimensional, so their own bound stops hot experts from consuming
   the attention budget.
5. **AUTO-S on the gradient group, fixed on the load group (mixed mode).** `auto_clipped_grad`
   (`packages/opaque-engine/src/opaque/api/engine/clipping/_auto.py:117`, scaling `R·g/(‖g‖+γ)`, MF-compatible) gains
   `fixed_groups=("router_load_probe",)` so that named groups use `min(1, C/‖·‖)` instead (per-record bound unchanged:
   `R` for AUTO-S groups, `C_h` for the fixed one; the latch sees a constant `PerGroup`).

### 4.2 What the per-example gradient-norm distribution looks like under the preset, and how to pick C

Quantities that are known without the checkpoint (VERIFIED arithmetic this session): LoRA r = 16 on q/k/v/o
(`q: 2304→4096, k/v: 2304→512, o: 4096→2304`) is `294 912` parameters per layer, `d = 8 257 536` in total, `√d = 2874`.
The per-step noise vector has norm `nm·C·√d/B̄ = 0.5622·0.9·2874/256 = 5.68` at `C = 0.9`, i.e. **6.3·C** — 6× the largest
possible clipped per-example contribution and far above the norm of the *mean* clipped gradient. This is the regime in which
(i) the per-step signal-to-noise is set by the *coherence* of the clipped per-example directions across the batch, not by the
absolute scale of `C`, and (ii) `C` acts mostly as a learning-rate scale once most examples are clipped. Consequences:

- **Shape to expect** (PLAUSIBLE until G3 runs; toy E3 VERIFIED: max/median 1.24–1.33 for token-mean CE): a per-example loss
  that is a mean over 1024 tokens concentrates by the law of large numbers across tokens, so the norm distribution is tight
  within a data type; heterogeneity comes from language/file kind and FIM vs plain, so p90/p50 ≈ 1.5–2 and a thin tail from
  short or degenerate files (the token-mean divisor bounds growth with length).
- **Default: AUTO-S with `R = 1`, `γ = 0.01`** on the gradient group (Bu et al. 2023, https://arxiv.org/abs/2206.07136 — the
  claim that AUTO-S matches or beats tuned fixed clipping is from the paper's abstract/experiments, **PLAUSIBLE**, theorem
  numbers not re-fetched). It removes the hyperparameter, is MF-compatible (`_auto.py:15-29`), and — the point of this
  lens — equalises every example's contribution so no single hot example dominates a step. The load group is exempt (§4.1.5).
- **If fixed clipping is kept:** pick `C` by the measured curve, not by a rule of thumb. From one
  `clipped_grad(..., clipping_norm=1e9, return_aux=True)` pass over ≈256 KStack examples (`aux.grad_norms`,
  `aux.group_norms`), compute for candidate `C` the bias `‖mean(clip_C g_x) − mean(g_x)‖` and the noise `nm·C·√d/B̄`; choose
  the `C` minimising `bias² + noise²` — with `d = 8.3M` this lands at or *below* the median norm (clip rate ≥ 50 %), which is
  where AUTO-S already sits. Report p10/p50/p90/p99/max and the clip rate at the chosen `C` (§10.3). If experts/router are
  trained, do the same per group (expect the router group to need the smallest bound: E3 router aux-only norms are the
  heaviest-tailed).
- **Numerical stability vs the non-DP path** (requirement (d)): fp32 accumulation in `Opaque_MoE`
  (`packages/opaque-patches/src/opaque/api/patches/kernels/moe.py:83,103`, VERIFIED phase-1), upstream RMSNorm retained
  (`mellum.py:41`), SDPA causal fast-path parity (PR #980), fp32 routing (§3): the remaining bf16 drift is accumulation
  order — 0.45–0.5 % rel-L2 with zero route flips (E1/divergence, VERIFIED phase-1 toy), inside HF's own batched-vs-loop
  spread.

---

## 5. Performance (G6)

- **Dense vs grouped MoE default.** `DPTrainer` defaults `use_performance_kernels=False`
  (`packages/opaque-transformers/src/opaque/api/transformers/trainer/_training_arguments.py:436`) ⇒ `kernels=False`
  (`_dp_trainer.py:846`) ⇒ `grouped_moe = kwargs.get("grouped_moe", kernels) = False`
  (`packages/opaque-patches/src/opaque/api/patches/transformers/_factory.py:316-322`) ⇒ dense every-token-through-every-expert
  `Opaque_MoE` on every host including CUDA (F9; VERIFIED this session). Arithmetic (PLAUSIBLE): per token per layer the dense
  path runs 64 experts × 3 GEMMs of `2304×896` ≈ 793 MFLOP vs ≈ 99 MFLOP routed — ≈8× the expert FLOPs, dominating step time
  at 12B. **Decision:** the mellum family requests `grouped_moe=True` whenever `torch._grouped_mm`/Triton is available,
  independent of the CUDA-kernel group (the dispatcher already prefers the grouped path on CUDA bf16, `kernels/moe.py:578-633`,
  VERIFIED phase-1); the `use_performance_kernels` comment (`_training_arguments.py:429-435`, which lists rope/rms_norm/
  activation/cross_entropy only — VERIFIED this session) gains the `grouped_moe` entry; presets set
  `performance_kernels_config={"grouped_moe": True}`; the first-class-level-patch capture (`_router.py:59-92`) is documented.
- **Cost of the mechanism itself:** one `(L·T)×E` fp32 softmax per example (already computed by the router; reused), one masked
  mean, one top-k indicator reduction — `O(L·T·E) = 1.8 M` elements per example; negligible. The independent-draw variant adds
  one forward over `q₂N ≈ 256` examples every `m = 4` steps ≈ +8 % of step time with grouped MoE (≈ +25 % with the dense
  default — another reason to fix the default).
- **Memory of new buffers** (per example, T = 1024): fp32 router logits `(L, T, E)` = 28·1024·64·4 B = 7.3 MB (HF keeps bf16
  logits for autograd already; fp32 doubles it) plus the fp32 probs for `P(x)` (7.3 MB, freed after the reduction); ×microbatch
  8 ⇒ ≈ 120 MB transient. Per-example `P, h, d`: 3×64 floats. Probe leaf: 64 floats (+128 floats inert optimizer state).
  `RouterLoadState`: 2×64 floats + scalars. Nothing scales with parameters. Chunked CE is preserved (§9.2), so peak memory
  stays that of PR #978, not the full-vocab `98304×1024` logits of the current `output_router_logits` fallback
  (`components/cross_entropy.py:212-230`, VERIFIED this session).

---

## 6. Smoothing under MF (G7)

**Numbers (VERIFIED this session, `mf_filter.py`; `band_mf_strategy(bands=64, momentum=0.95)`, `n = 2048`, sensitivity
`‖c‖ = 1.0000`, `c[:4] = [0.779, 0.370, 0.264, 0.209]`).** Noise std of a linear filter `F` applied to the released stream
`x̂ = d + σ_h(C⁻¹Z)` is `σ_h·‖row_t(F·C⁻¹)‖`, computed exactly:

| filter | band-MF factor (stationary) | DP-SGD factor | MF/SGD | lag |
|---|---|---|---|---|
| none (single step) | 1.4313 (1.284 at t=0, 1.421 at t=1, stationary from t≈7) | 1.000 | 1.43 | 0 |
| EMA β_f = 0.90 | 0.1482 | 0.2294 | 0.646 | 10 |
| EMA β_f = 0.95 (= strategy momentum) | 0.0824 | 0.1601 | 0.515 | 20 |
| **EMA β_f = 0.99** | **0.0249** | 0.0709 | 0.352 | 100 |
| EMA β_f = 0.995 | 0.0162 | 0.0501 | 0.323 | 200 |
| running mean W = 16 / 64 / 256 | 0.1432 / 0.0479 / 0.0198 | 0.2500 / 0.1250 / 0.0625 | 0.57 / 0.38 / 0.32 | W/2 |

**Reading.** (i) The strategy is optimised for the momentum workload `A = Toeplitz(β^i)`, β = 0.95
(`packages/opaque-dpftrl/src/opaque/api/dpftrl/noise/_band_mf.py:35-60,120-138`, VERIFIED this session); the EMA with
`β_f = β` is `(1−β)·A`, so it *inherits the strategy's optimality statement* scaled by `(1−β)` — that is the sense in which it is
"workload-matched". (ii) But the noise variance of *any* fixed linear filter is a deterministic property of `F·C⁻¹` that needs
no theorem — the table is exact — and every low-pass filter benefits ≈2–3.5× from band-MF's anti-correlated noise, the benefit
*growing* with the filter's memory. (iii) Therefore the filter should be chosen by **lag tolerance**, not by the strategy's
momentum: `β_f = 0.99` (lag 100 steps = 0.64 % of the 15625-step horizon; imbalance drifts on the scale of the router's
adaptation, thousands of steps) gives 0.0249, i.e. **0.83 % of k/E at ρ = 0.02**. The known noise std `s` fed to the shrinkage is
`(σ_h/λ)·0.0249` under MF and `(σ_h/λ)·0.0709` under DP-SGD (burn-in tracked by the variance recursion of the same filter).
(iv) An `lr_schedule` (query weights) changes `A` but not `C⁻¹`'s role in the table; recompute the row norms from the actual
strategy at setup (one `n×n` solve, or the streaming multiply) and store them in `RouterLoadState` — no hand constants.

---

## 7. Scope (G8)

| item | in/out | what is needed |
|---|---|---|
| DP-SFT / causal-LM, attention-only LoRA (both presets) | **in** — §2 as specified | §9 |
| DP-DPO (`mellum2-codesec`) | **in**, mechanism unchanged: the protected unit is the preference pair; `P(x), d(x)` pool the *chosen + rejected* policy forwards (both routed tokens of the same example, `(L·T)`-weighted); the reference forward (precomputed or TR-DPO EMA via `_augment_inputs`, `packages/opaque-transformers/src/opaque/api/transformers/trl/_dpo_trainer.py:807,1065`, VERIFIED this session) contributes **no** aux (HF adds aux only to the policy loss, `modeling_mellum.py:692-700`). TRL converter forwards `router_aux_loss_coef` instead of dropping it (`trl/_convert.py:69-73`, VERIFIED this session) | DPO override of `compute_per_example_loss_and_metrics`; converter change |
| Router z-loss `Z_x = (1/(L·T_x))Σ_{l,t} m_{x,t}(logsumexp_e z^l_{x,t,e})²` (ST-MoE eq. 5, https://arxiv.org/abs/2202.08906 §3.1, VERIFIED phase-1 literature C2) | **in as opt-in**, `ζ = 0` default: per-token separable ⇒ zero DP cost (ordinary clipping); part of Mellum2's pretraining objective (1e-3) but absent from HF's; its numerical motivation (bf16 round-off) is already removed by §3 | one reduction on the captured fp32 logits |
| Experts / router trainable | **in for the mechanism** (nothing in §2 depends on which leaves exist; `opaque_moe` emits per-example expert grads when `needs_input_grad` is set, `kernels/moe.py:536-539`); **out for presets** until M5 runs: PEFT `target_parameters` LoRA on stacked `experts.gate_up_proj/down_proj` under `functional_call`+vmap is untested (primitives §5). Full expert training needs `B·E·2I·H` per-example buffers per layer (0.53 GB/example/layer bf16); LoRA-on-experts / ESFT subsets are the practical forms; ESFT-style selection from private data is itself a query to account (literature D.2). Recommended ρ = 0.1 or the independent draw with c = 1 | M5 smoke test; per-group clipping preset (§4.1.4) |
| Per-layer `f̃^l` | opt-in, ×5.29 (§2.6) | `router_load_per_layer` flag; probe leaf `R^{L×E}` |
| Public-data `f̃` | opt-in (`router_load_source="public"`): forward-only on a public batch each `m` steps, zero release, no probe group; hybrid (public prior + private release) has the *same* sensitivity as the private release and buys nothing in noise — only the pure-public form is cheaper | a public dataloader |
| Loss-free balancing | opt-in (§7.3) | bias tensor in the adapter + serving patch |
| Per-sequence aux, STE/DenseMixer, HF-Trainer-realised objective | out (different objective / different backward / artefact) | — |

### 7.3 Loss-free balancing under DP — and is it admissible for a checkpoint with no bias tensor?

The rule (Wang et al. 2024 Alg. 1, https://arxiv.org/abs/2408.15664; DeepSeek-V3 eq. 16, https://arxiv.org/abs/2412.19437,
both VERIFIED phase-1 literature C4/C5): selection uses `z + b`, gating weights use `softmax(z)` over the selected set, and
`b_e ← b_e − u·sign(c_e − c̄)` from the **previous** batch's counts.

- **Privacy.** `c − c̄` is exactly the imbalance the design already releases. `b_{t+1} = b_t − u·sign(d̂_t)` (or the dead-zone
  variant `sign(d̂_e)·1{|d̂_e| > 2s}`, or the soft sign `clip(d̂_e/s, −1, 1)`) is **post-processing of the same Gaussian
  release — free.** Randomised response on the *true* sign vector is dominated (§2.5: 12.8–70 ε per step, because one example
  can flip every near-zero coordinate). So the DP version of LFB is "sign of the noised deviation"; the random walk of `b` when
  the true deviation is ≈ 0 (std `u·√t = 0.125` over the horizon at `u = 1e-3`) is what the dead zone prevents.
- **Admissibility.** LFB changes the *forward* (`topk(z + b)`), so the fine-tuned model is `base + LoRA + b`. HF
  `MellumTopKRouter` has no bias field and `nn.Linear(hidden, E, bias=False)` has no constant input channel, so `b` cannot be
  folded into `W`; it must ship with the adapter and be applied by a serving-side patch. Two uses: (a) training-only balance
  control with `b` discarded at export — **inadmissible**: it creates train/inference routing mismatch by construction (the
  very defect Mellum2 TR §5.2 reports); (b) an architecture extension shipped with the adapter — **admissible as opt-in**, not
  faithful to the checkpoint's declared objective, but it is what the Mellum2 authors say they will do next (TR A.2, VERIFIED
  phase-1 literature). It is also the only lever that rebalances loads *directly* when the router is frozen (the aux gradient
  reaches routing only through attention-LoRA-modified hidden states). Decision: `router_bias_balancing=False` by default;
  when on, `u = 1e-3`·(router logit scale, measured in §10), dead-zone `2s`, bias saved in the adapter with a documented
  serving patch; the aux term may be set to 0 in that mode (DeepSeek-V3 keeps a 1e-4 sequence-wise term — not adopted).

---

## 8. Privacy hygiene (G9)

| tensor / state | where | class | logged? | checkpointed? | DDP |
|---|---|---|---|---|---|
| probe parameter `z ∈ R^E` (value ≡ 0) | model `nn.Parameter` in `trainable_params` | public constant (re-zeroed each step in `_augment_inputs`; optimizer state inert) | no | as a zero tensor (harmless) | identical by construction |
| per-example fp32 router logits / probs / executed routes `S` | inside the vmapped per-example function | **private-internal**, never leave the closure | never | never | n/a |
| per-example `P(x), h(x), d(x), Z_x`, per-example aux value | inside the closure; `λd(x)` becomes the probe leaf's per-example gradient | **private-internal** (pre-clip/pre-noise) | **not** added to `loss_aux` telemetry (the existing un-noised `loss`/`loss_aux` means, `_dp_trainer.py:2258-2293`, are a pre-existing posture — F11 — nothing is added to it) | never | n/a |
| forward-only per-example `d(x)` on the independent draw (opt-in) | inside `clipped_fun`'s vmap | private-internal | never | never | summed by `reduce_pytree_` *before* noise, shared noise key |
| noised probe leaf `ŷ_t` / noised side release | output of `noise_fn` | **public** | may be logged | — | `sum_gradients_` pre-noise + shared key ⇒ bit-identical on all ranks (`_dp_trainer.py:1478`, VERIFIED phase-1 primitives) |
| `RouterLoadState(d_ema, s (noise std), f_tilde, step, beta, rho, filter_row_norm)` | trainer-side frozen dataclass | **public post-processing** | `f̃_t` stats logged (`router/load_min`, `router/load_max`, `router/imbalance_l2`, `router/shrink_factor`) | **yes** — sidecar in the runtime checkpoint (§9.5); needed for reproducibility, not privacy | rank-identical; `register_sync_type` asserts equality (`packages/opaque-engine/src/opaque/api/engine/distributed/_state.py:537`) |
| `load_target` batch column `(B, E)` = broadcast `f̃_t` | injected by `_augment_inputs` | public | — | no | identical |
| router bias `b` (opt-in LFB) | model buffer | public post-processing | may be logged | yes (adapter) | identical |
| fp32 router weight view | patch | public model parameter | — | derived | — |
| realised batch size, clip rates, group norms | existing `ClippedGradAux` | pre-existing telemetry (outside this design) | as today | as today | as today |

**Privacy statement addition** (docs + trainer docstring): "Each step releases one Gaussian mechanism on the concatenation of
the clipped per-example gradients and the per-example centred router-load vectors (per-record bounds `C_g`/`R` and `C_h = λΔ_h`);
[opt-in: every `m` steps an independent Poisson-subsampled Gaussian release of the centred router-load vector, sensitivity
`Δ_h`, multiplier `c·nm`, composed with `|`]. The load estimate `f̃_t` consumed by the loss (and the optional router bias) is
post-processing of previous releases. No other quantity derived from private routing is released." Logging `f(B_t)`
un-noised, `h(x)`, or per-example aux values would be an unaccounted release and is not implemented.

---

## 9. Implementation plan in Opaque (G10) and test plan

### 9.1 `opaque-patches` — model-side, DP-agnostic
1. **`components/router.py` (new):** `make_fp32_router_forward(original)` for `MellumTopKRouter` (§3). `_factory.py:185
   make_apply_model_patches` gains a `"router"` role (`router_kind="fp32"`, gated by `compat`, user kwarg `router_fp32`),
   applied through `_router.py:59 _patch_forward`. `models/mellum.py:27-45`: add `"router": "MellumTopKRouter"`,
   `router_kind="fp32"`. Default ON for the mellum family.
2. **`components/cross_entropy.py:212-230`:** keep the `output_router_logits` fallback (preserves the upstream aux contract) and
   add a sibling kwarg `opaque_router_stats=True` on the patched causal-LM forward: call the backbone with
   `output_router_logits=True` (HF's `capture_outputs` collects the fp32 logits per layer via
   `OutputRecorder(MellumTopKRouter, index=0)`, `modeling_mellum.py:42,431-432`, VERIFIED this session; works under vmap —
   VERIFIED phase-1 divergence C10/primitives E3), compute per example from `outputs.router_logits` (tuple of `L` tensors
   `(T, E)`): `P(x)`, `h(x)` (broadcast-compare against `arange(E)`, not `F.one_hot`), `d(x)`, `Z_x`, with the attention mask;
   **never** call `load_balancing_loss_func` (in-place `scatter_add_`, not vmap-safe); return them as extra output fields
   (`router_probs_mean`, `router_load_centred`, `router_z`) while keeping the chunked-CE path for the LM head. This is the
   memory-safe route (critic C10).
3. **`_factory.py:316-322`:** mellum family defaults `grouped_moe=True` when the grouped path is available (§5).

### 9.2 `opaque-engine` — one small addition
`auto_clipped_grad(..., fixed_groups: tuple[str, ...] = ())` in `clipping/_auto.py:117` → `auto_scale_pytree`
(`_pytree.py:350`) applies `min(1, C/‖·‖)` to the named groups and AUTO-S to the rest; `PerGroup` max_norm unchanged
(constant ⇒ MF latch OK). `per_group_noise_stddev`, `gaussian_noise`, `mf_gaussian_noise`, `PerGroup`, `clipped_fun`: **no
change** (all VERIFIED phase-1 primitives as sufficient).

### 9.3 `opaque-transformers` — the mechanism (trainer-side)
`trainer/_router_load.py` (new): `RouterLoadState` (frozen dataclass; serialises via the registry, VERIFIED phase-1 primitives
E2), `RouterLoadMixin` used by `DPTrainer` when `args.router_load_release != "off"`:
- construction: register the probe `nn.Parameter("router_load_probe", zeros(E), requires_grad=True)` (never used by the forward;
  picked up by `make_functional(partition_trainable=True)`); seed the `load_target` column so `_discover_batch_keys` includes it
  (TR-DPO pattern, `_dpo_trainer.py:721-748`); resolve `clipping_norm` into `PerGroup(..., router_load_probe=λΔ_h(1+1e-6))`
  (trainer already coerces dict → `PerGroup`, `_training_arguments.py:288,1044`, VERIFIED this session); compute
  `filter_row_norm` from the actual strategy (§6.iv) or `√((1−β_f)/(1+β_f))`.
- `_augment_inputs` (`_dp_trainer.py:2302`): write `f̃_t` broadcast into `load_target`; zero the probe.
- `compute_per_example_loss` (`:2314`): `out = fmodel(params, **inputs, opaque_router_stats=True)`; return
  `out.loss + α·E·⟨load_target, out.router_probs_mean⟩ + ⟨params["router_load_probe"], λ·out.router_load_centred.detach()⟩ + ζ·out.router_z`.
- `on_pre_optimizer_step` (`:2197-2199`, receives the **noised** pytree keyed by name): read `pytree["router_load_probe"]`,
  post-process (§2.2 step 5), update `RouterLoadState`.
- `_create_grad_fn` (`:4230`): pass `fixed_groups=("router_load_probe",)` in `auto` mode.
- **Independent draw (opt-in, DP-SGD only):** `RouterLoadReleaser` owning a second `PoissonSampler`
  (`packages/opaque-dpsgd/src/opaque/api/dpsgd/sampling/_poisson.py:37`) over the train dataset with key
  `fold_in(key, "opaque.moe.load_release")`, `clipped_fun(stat_fn, clipping_norm=Δ_h, normalize_by=q₂N)` forward-only, own
  `gaussian_noise(c·nm, key=fold_in(key, "opaque.moe.load_noise"))`; DDP: each rank samples its shard, `reduce_pytree_` before
  noise; accounting: `ctx.accounting |= acc.cached(poisson(gaussian(c·nm), q₂))` on release steps via the
  `_account_independent_step` seam (`_dp_trainer.py:296`); rejected with `ConfigurationError` when `mechanism_kind != "gaussian"`.
- **TrainingArguments:** `router_aux_loss_coef: float | None = None` (None ⇒ 1e-4 for mellum), `router_load_release:
  Literal["in_stream","independent","public","off"] = "in_stream"`, `router_load_ratio = 0.02`, `router_load_ema = 0.99`,
  `router_load_shrink = True`, `router_load_release_every = 4`, `router_load_release_multiplier = 2.0`,
  `router_z_loss_coef = 0.0`, `router_fp32 = True`, `router_load_per_layer = False`, `router_bias_balancing = False`.
- **Checkpoint:** `save_dp_runtime_state` has a fixed signature (`trainer/_checkpoint.py:311-323`, VERIFIED this session) ⇒
  sidecar `router_load_state.pt` written in `_save_checkpoint` (`:4910`) and restored in `_apply_runtime_state` (`:5335`);
  resume asserts `rho`, `beta`, `E`, `k` match.
- **TRL:** `trl/_convert.py:69-73` forwards the coefficient; `_dpo_trainer.py:1065` pools chosen + rejected stats.
- **Examples:** presets add `--router-aux-loss-coef 1e-4 --router-load-ratio 0.02 --clipping-mode auto` and `grouped_moe`.

### 9.4 Composition with the rest of the pipeline (G10)
- **Chunked CE:** preserved — stats come from the backbone call inside the patched causal-LM forward, not from
  `output_router_logits` on the causal-LM (which triggers the full-vocab fallback).
- **Gradient checkpointing:** HF `OutputRecorder` is a forward hook; under `opaque.patches.torch` checkpoint recompute it fires
  again. The stats are computed *once*, right after the backbone returns, from that call's `outputs.router_logits`; recompute
  during backward re-captures into the recorder but nothing reads it — test 9.5-T4 asserts bit-equality of `d(x)`, `P(x)` and
  the probe gradient with checkpointing on/off. If the recorder accumulates (list growth) across recompute, replace it with a
  per-layer-index dict in the patched `MellumSparseMoeBlock.forward` (role `"moe_block"`).
- **Microbatching:** `microbatch_size` chunks call the per-example function once per chunk; stats are per call — exact.
- **DDP:** probe leaf reduced with the gradient; state rank-identical (§8).
- **torch.compile:** the recorder/dict is a Python side effect ⇒ graph break; the `_grad_compiler` fullgraph fallback
  (`_dp_trainer.py:4191-4228`) handles it (VERIFIED phase-1 primitives read; not executed — test T7).
- **MF latch:** `PerGroup` constant across the run; `rho`, `λ`, `C_g`, group map fixed at construction and asserted on resume.

### 9.5 Test plan (behavioural, per ARC-006/ARC-012; no docstring-pinning tests)
- T1 (`opaque-patches/tests/transformers/models/test_mellum_router.py`): fp32 router patch — logits equal `F.linear(h.float(),
  W.float())`; executed indices equal HF fp32 top-k; scores dtype/renorm identical to `modeling_mellum.py:335-339`.
- T2: `opaque_router_stats` under `vmap(grad)` with padding — `d(x)`, `P(x)` equal an eager per-example loop; `Σ_e d_e = 0`;
  `‖d(x)‖ ≤ Δ_h`; chunked-CE path still taken (peak memory / `logits is None` check).
- T3 (`opaque-transformers`, tiny Mellum, fp32): surrogate exactness — batch mean of `∇ℓ_x` with `f̃ = f(B)` equals HF's
  `output_router_logits=True` gradient on the same batch to 1e-6 rel-L2 (ragged masks with `w_x` weighting variant).
- T4: probe leaf — noised `ŷ_t/λ` has the closed-form σ (`per_group_noise_stddev`), other leaves' σ unchanged; checkpointing
  on/off equality; `microbatch_size=2` equality.
- T5: post-processing — sum-zero projection, EMA variance recursion equals the closed form, shrinkage ≡ identity when
  `‖d̃‖ ≫ s√63` and ≡ 0 below; clamp; `f̃_0 = k/E` ⇒ aux gradient exactly 0 at step 0.
- T6: accounting — in-stream: `ctx.accounting` process identical to the no-probe run; independent: process equals
  `poisson(g(nm),q)*T | poisson(g(c·nm),q₂)*(T/m)`; `mechanism_kind="band_mf"` + `"independent"` raises.
- T7 (`opaque-engine`): `auto_clipped_grad(fixed_groups=...)` — fixed group uses `min(1, C/‖·‖)`, AUTO-S groups unchanged,
  `max_norm` constant; MF latch accepts it (`opaque-dpftrl` test).
- T8 (`distributed` marker): 2-rank Gloo — `RouterLoadState` identical on both ranks after a step; `sync` assertion passes.
- T9: checkpoint round-trip of `RouterLoadState` sidecar; resume with mismatched `rho` raises.
- T10 (`slow`, optional CUDA): grouped vs dense `opaque_moe` per-example gradient equality under the mellum preset shapes.

---

## 10. Validation plan on the real checkpoint (GPU) — G2 / G3

### 10.1 Oracle and drift metric (G2)
- **Oracle:** `MellumForCausalLM` (bf16, `JetBrains/Mellum2-12B-A2.5B-Base`) in a **separate process**, *with only the fp32-router
  patch applied* (so routing precision is common), eager attention, `output_router_logits=False`, per-example Python loop of
  `loss.backward()` on LoRA r=16 q/k/v/o parameters ⇒ per-example gradients `g_x^{HF}` and per-(layer, token) executed top-8
  sets `S^{HF}`. Second oracle column: *unpatched* HF bf16 (routing in bf16) — the PR #980 reference — reported separately.
- **Candidate:** Opaque `clipped_grad(..., clipping_norm=1e9, return_aux=True)` internals: `vmap(grad)` per-example `g_x^{Op}`
  with the full mellum patch set (grouped MoE on CUDA, chunked CE, fp32 router, SDPA), same batch, plus captured `S^{Op}`.
- **Metrics, reported separately:** (a) per-example rel-L2 `‖g_x^{Op} − g_x^{HF}‖/‖g_x^{HF}‖` (median/max) and per-tensor
  worst case; (b) **route-flip counter**: `#{(l,t): S^{Op} ≠ S^{HF}} / (L·T)` and per-example flip counts; (c) the same rel-L2
  split into flip vs no-flip examples (E1b method); (d) fp32 run of both as the accumulation floor.
- **Acceptance:** with the fp32 router on both sides, flips = 0 on ≥ 99 % of examples and bf16 rel-L2 ≤ 1.5 % (inside HF's own
  batched-vs-loop spread, F8); vs unpatched HF-bf16 report flips and rel-L2 as the deliberate deviation. This reproduces
  PR #980's "29 % → 1.3 %" with a script that is committed (`examples/validate_mellum_oracle.py`, GPU, one microbatch of 8,
  minutes).

### 10.2 Statistics to collect (G3) — one GPU, ≈256 KStack examples at T = 1024, preset partition
1. Per-example gradient-norm quantiles (p10/p50/p90/p99/max) at `C = ∞`, per group when experts/router are trained
   (`aux.grad_norms`, `aux.group_norms`); the bias²+noise² curve of §4.2 and the resulting `C`; AUTO-S clip-rate equivalent.
2. `‖f(B) − k/E‖₂` and per-coordinate RMS `δ` at B = 256 (several batches), and its batch-to-batch sd — these set the SNR of
   §2.6 (note: on genuinely private data this measurement is itself a release; on public KStack it is not).
3. Per-example `‖d(x)‖₂` distribution vs the structural bound 2.646 (tells how loose the bound is; a tight distribution far
   below the bound would justify a *smaller* `Δ_h` only via a *clip*, which biases — not adopted unless the tail is empty).
4. Per-example expert usage at T = 1024 (fraction of experts with zero tokens per example, per layer) — H5.
5. Router logit scale (median `|z|`, median top-8/9 margin, fraction within bf16 rounding) — sets `u` for §7.3 and predicts
   flip rates.
6. Surrogate exactness on the real model in fp32: batch mean of per-example surrogate gradients with `f̃ = f(B)` vs HF
   `output_router_logits=True` gradient (≤ 1e-5 rel-L2 expected).
7. Aux-to-CE gradient ratio at `α = 1e-4`/`1e-3` on real batches (H4 at scale).

### 10.3 End-to-end acceptance (short DP runs, ε = 3, both stacks)
- Load-estimate accuracy after burn-in: `‖d̃⁺_t − d_true,t‖/‖d_true,t‖ ≤ 0.25` where `d_true` is computed (in the validation
  harness only, never in production) from the same batches; shrinkage engages only when `‖d_true‖ < s√63`.
- Utility: eval loss vs the `α = 0` baseline within run-to-run noise (3 seeds); expert-usage entropy on eval data not lower than
  the base model's; no expert with usage < 0.25·k/E after training.
- Privacy bookkeeping: `ctx.accounting.epsilon_at(δ)` identical to the no-probe run (in-stream) / equals the composed process
  (independent).
- Stability: per-example gradient rel-L2 vs oracle tracked every 100 steps stays ≤ 1.5 % with zero flips.

---

## 11. Risks and what would falsify the design

1. **The real imbalance is below the noise floor at ρ = 0.02** (per-coordinate RMS `δ < 0.07·k/E` under MF): then shrinkage
   zeroes the term — which is *also* what the true objective does at balance — but the aux loss then does nothing during
   fine-tuning. Falsifier: §10.2 item 2 shows `δ ≥ 0.2` **and** the router drifts under fine-tuning (usage entropy drops);
   remedy: ρ = 0.1 (×1.049) or the independent draw (DP-SGD).
2. **`OutputRecorder` under vmap + checkpoint recompute** accumulates or misorders layer logits (T4). Remedy: capture in a
   patched `MellumSparseMoeBlock.forward` keyed by layer index.
3. **`torch.compile`** does not tolerate the side channel even with the fullgraph fallback (T7). Remedy: return the stats
   through the model output pytree only (no Python side effects) — the design already prefers this.
4. **fp32 routing moves the fine-tuned model away from the HF-bf16 serving router**: if the serving stack routes in bf16, the
   train/inference route disagreement Mellum2 reports persists at inference regardless; the choice matches *pretraining*.
   Falsifier: §10.1 shows fp32-vs-bf16 flips > 3 % of tokens on the checkpoint; then offer `router_fp32=False` with the
   explicit cost (E1b) and keep `f` from executed routes.
5. **Per-group MF correctness relies on degree-1 homogeneity of the participation sensitivity in the row bound** (math §5(i));
   this holds for `BandMfStrategy.sensitivity` (`‖c‖`, `_band_mf.py:137-139`) and b-min-sep sensitivities computed from the
   Toeplitz coefficients (`_toeplitz.py:448-520`). Falsifier: any strategy whose sensitivity is not homogeneous in the bound —
   test T7's MF variant asserts `mf_gaussian_noise` σ on the load leaf equals `per_group_noise_stddev(...)·‖row_t(C⁻¹)‖`.
6. **Independent-draw sampler plumbing under DDP / dataset sharding** could silently change `q₂` (each rank must sample its
   shard at `q₂` with `N` the global size). Falsifier: T6/T8 compare the realised release count distribution to `Poisson(q₂N)`.
7. **The probe parameter leaking into the optimizer** (Adam moments integrating the noisy load) — harmless numerically (the
   parameter is re-zeroed) but the design masks it anyway; falsifier: probe value non-zero at any `compute_per_example_loss`
   call (assert).
8. **Experts-trainable variant** (M5): PEFT `target_parameters` under `functional_call`+vmap may not work; the mechanism is
   unaffected, the preset is.
9. **Utility claim of AUTO-S** (§4.2) is from the literature, PLAUSIBLE; falsifier: §10.3 shows fixed `C` at p50 beating AUTO-S
   by more than run-to-run noise — then the preset keeps fixed clipping and the mixed-mode change is unnecessary.
10. **DPO pooling of chosen + rejected** doubles `L·T` per example but the structural bound is unchanged (fractions) — if the
    reference-model forward is ever routed through the patched causal-LM with `opaque_router_stats=True`, its stats must be
    discarded (test: reference stats never reach the probe).

**What could not be verified here:** every magnitude on the trained checkpoint (§10 is the plan); the concurrent-composition
theorem numbering for the (rejected) MF side-release; Bu et al. theorem numbers; the exact forward/step compute ratio behind
"+8 %". All accountant and filter numbers in §2.6/§6 were produced by the two scripts in `scratchpad/research/design-optimal/`
this session.
