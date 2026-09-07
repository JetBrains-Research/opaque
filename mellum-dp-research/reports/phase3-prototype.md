# Phase 3 — end-to-end prototype of the v2 router-load release on a tiny Mellum

Agent: `prototype`. Repo `/home/user/opaque` @ `ef1abc5`, `git status` clean (nothing tracked touched). All files under
`scratchpad/research/prototype/`: the script `prototype_load_release.py`, its stdout `prototype_output.txt` (final run; the two
earlier runs are kept as `prototype_output_run1.txt` / `_run2.txt`), the results `prototype_results.json`, and two sizing probes
(`probe_sizes.py`, `probe2.py`). Runtime of the final script: **199 s wall-clock** on 4 CPU cores (training arms in a 2-process pool,
2 threads each) — far inside the 20-minute budget, so no size was reduced for time; the toy sizes below were chosen for signal, not
time. Every number in this report is **VERIFIED** (printed by the script in this session) unless marked PLAUSIBLE.

## 0. What was built (design §1.1, §2.1–2.4, §9 — implemented with the real primitives)

| design element | implementation (script section) | evidence |
|---|---|---|
| model | `build_moe_model('mellum','cpu', num_experts=8, num_experts_per_tok=2, moe_intermediate_size=32, num_hidden_layers=2, hidden_size=64, …)` from `packages/opaque-patches/tests/transformers/models/_test_utils.py:100-150` (applies `apply_model_patches(model, eager_attention=True)`); experts forward `__opaque_patched__` = True | VERIFIED |
| router logits under vmap | class-level replacement of the CausalLM forward with a named `opaque_router_logits=True` kwarg (design §9.1 emulation): `self.model(..., output_router_logits=True)` (HF `OutputRecorder`, `modeling_mellum.py:431-433, 474`) + the real chunked CE `linear_nll_sum_chunked` (`kernels/_linear_ce_chunked.py:239-264`); no HF aux branch | VERIFIED |
| statistics | `router_stats`: fp32 `softmax` + `topk` (the router's own ops, `modeling_mellum.py:333-336`), broadcast-compare one-hot, **binarised** mask, explicit `where(T_x>0, ·, 0)` → `h^{(L,E)}(x)`, `P(x)`, `T_x` | VERIFIED |
| loss (design §1.1) | `ℓ_x = CE_x + α·(S_x − sg[S_x]) + ⟨z, sg[λ·w_x·d^{(L,E)}(x)]⟩`, `S_x = E·w_x·Σ_e (f̃_e − k/E)·P_e(x)`, `w_x = T_x/T̄`, `T̄ = T_max` | VERIFIED; `aux.loss_values == CE` exactly (value-neutral) |
| probe | zero `(L,E)` `nn.Parameter('router_load_probe')` registered **before** `make_functional(partition_trainable=True)` → trainable leaf | VERIFIED (`probe in trainable: True`) |
| clipping | `PerGroup(groups={(leaf,): 'fallback' …, ('router_load_probe',): 'router_load_probe'}, values={'fallback': C_g, 'router_load_probe': C_h})` by direct construction (`types.py:72-110`); `clipped_grad(loss, has_aux=True, clipping_norm=pg, normalize_by=B̄, batch_argnums=(1,2,3), return_aux=True)`; `C_h = λΔ_L(1+1e-3)`, `Δ_L = √(kL(1−k/E)) = √3`, `λ = ρC_g/Δ_L` | VERIFIED |
| noise | `opaque.dpsgd.noise.gaussian_noise(noise_multiplier=nm, key=key(seed))` and `opaque.dpftrl.noise.mf_gaussian_noise(trainable, band_mf_strategy(bands=4, momentum=0.95), n_steps=50, noise_multiplier=nm, key=…)` | VERIFIED |
| sampling | `PoissonSampler(ids, sample_rate=B̄/N, n_steps=300, key=…)` (V5); `BMinSepSampler(ids, bands=4, sampling_prob=B̄/N, n_steps=50, key=…)` (V6) | VERIFIED |
| post-processing (§2.4) | `postprocess`: `ŷ/λ` → layer mean → sum-zero projection → EMA β=0.9 with bias correction → `s_t = σ_h/(λ√L)·φ_t/(1−β^t)` (φ from the closed-form recursion under DP-SGD, from `‖row_t(F·C⁻¹)‖` of the **instantiated** `band_mf_strategy(4, .95).coefficients(n_steps=50)` under MF) → dead zone `c=2` → JS+ shrink → clamp → `f̃_{t+1}`; `D_t`, `D^l_t` monitors | VERIFIED |
| optimizer | probe leaf of the noised pytree zeroed after reading; plain SGD (`lr=0.005`) on the other leaves; `z` asserted zero every step | VERIFIED (`probe_param_final_nonzero = 0` in all arms) |
| accountant | `poisson(gaussian(nm), q)*T` and `mf_gaussian(nm, strategy, n_steps)` — built from `(nm, q, T)` / `(nm, strategy, n_steps)` only | VERIFIED |

Toy regime: `E=8, k=2, L=2, hidden=64, vocab=128`; `T_max=32`, lengths uniform in `[16, 32]` (right-padded, labels −100), `N=1024`
"structured" sequences (phase-1 `empirical/common.py:structured_batch`), held-out split 256 sequences (disjoint seeds), `B̄=32`,
`q=1/32`, `T=300` steps, `δ=1e-5`, ε-target 3 → **`nm = 1.0820`** by `acc.calibrate` on the real accountant (1.9 s).
Trainable: q/k/v/o + router (`mlp.gate.weight`), 197 648 parameters. Induced imbalance: rows 0,1 of both routers ×3 →
`f_heldout = [0.405, 0.336, 0.227, 0.164, 0.194, 0.284, 0.211, 0.178]`, **δ₀ = 0.317** (per-coordinate RMS in k/E units),
**D₀ = 0.62** (above the design's trip threshold τ=0.5). `C_g = 5.7195` = median per-example norm at α=0 on the held-out split
(p10/p50/p90/max 4.87/5.72/6.92/9.32; design §4.4). `T̄_true = 24.01` so the release scale factor `T̄_true/T_max = 0.750`.

## 1. Results table

| check | what | result | numbers |
|---|---|---|---|
| **V1** | surrogate identity on the **patched** model, ragged lengths `[25,21,24,25,30,28,17,25,23,18,20,19,25,31,22,30]`: batch-mean of per-example surrogate grads at `f̃ = f(B)` (HF token-weighted) vs `∇ L_aux^HF` (HF's own `load_balancing_loss_func` outside vmap, and my out-of-place copy of `modeling_mellum.py:540-606`) | **PASS** | cosine **1.000010** (fp32 rounding above 1), norm ratio **0.748046** vs `T_tot/(B·T̄)` = **0.748047** (rel diff 9.6e-7); out-of-place vs HF-own gradient rel-L2 **0.0**, values 2.049564 both |
| **V2** | pre-noise probe leaf `== (λ/B̄)Σ_x w_x d^{(L,E)}(x)`; group norm ≤ C_h for every example incl. the adversarial all-same-token row (routes every token to one top-2 set per layer) | **PASS** | max abs err **3.7e-9**; `Σ_e` leaf −1.5e-8; adversarial `h` = `[[1,0,0,0,0,0,0,1],[1,0,0,0,0,1,0,0]]`, `‖d^{(L,E)}‖ = 1.732051 = Δ_L`; probe group norms max **1.944646** (= λΔ_L exactly) vs `C_h = 1.946591`; examples over C_h: **0** (clip never fires; `aux.group_norms`) |
| **V3** | Mahalanobis identity with the real `per_group_noise_stddev` (through `gaussian_noise`) | **PASS** | σ_g = 0.223903, σ_h = 0.130622 (closed forms identical); `(C_g/B)²/σ_g² + (C_h/B)²/σ_h² = 0.8541229522 = 1/nm²`, ratio **1.000000000000**; gradient-noise inflation 1.157731 = √(1+ρ) |
| **V4** | accountant invariance | **PASS** | `poisson(gaussian(1.082), 1/32)*300`: ε = **2.998599** with and without the probe (identical construction, `type Repeated`, signatures `gaussian(noise_multiplier)`, `poisson(inner, sample_rate, *, truncated_batch_size, dataset_size)`, `mf_gaussian(noise_multiplier, strategy, *, n_steps, min_sep, max_participations)` — no pytree / PerGroup / max_norm argument); `mf_gaussian(1.082, band_mf(4,.95), n_steps=50)`: ε = **3.996299** both (sensitivity 1.0000, un-amplified, 0.1 s) |
| **V5** | 300-step DP-SGD, four arms + three nm=0 ablations (table §2) | **PASS** (6/6 criteria; one is borderline, §2.3) | DP(ρ*=0.34) recovers **74 %** of the ORACLE's imbalance reduction (δ: OFF 0.317→0.420, ORACLE →0.075, DP →0.165); cos to the population aux direction while active **0.825**; cos to the exact *batch* aux gradient **0.64** (§2.3); dead zone 24 % (ρ*) vs **86.7 %** (ρ=0.02); realised pooled probe noise / predicted **0.999**; probe never clipped; z stays 0 |
| **V6** | 50 steps under `mf_gaussian_noise(band_mf_strategy(bands=4, momentum=0.95))` + `BMinSepSampler` | **PASS** | latch accepted the two-group `PerGroup` for all 50 steps; realised `noise_stddev` per group vs `base·‖row_t(C⁻¹)‖`: max rel err **2.9e-16** (probe) / **3.4e-16** (fallback); probe leaf received noise, empirical RMS / predicted **0.993**; row norms t=0..4 `[1.2489, 1.4192, 1.4262, 1.4273, 1.4595]`; EMA-filtered factor φ_MF = 0.1488 vs DP-SGD 0.2146 (×0.69); δ 0.317→0.161 in 50 steps; mean batch 29.9 |
| **V7** | gradient checkpointing on/off (`model.gradient_checkpointing_enable()` under the opaque HF glue, `patches/torch/checkpoint/huggingface.py:31-56`) | **PASS** | flags `[True, True]`; probe leaf rel-L2 **0.0**; max over all clipped leaves **0.0**; identical to the earlier non-ckpt run to 0.0 |

## 2. V5 in detail

### 2.1 Arms (all: `B̄=32`, Poisson `q=1/32`, `C_g=5.72`, α=1.0 where on, lr 0.005, β=0.9, c=2, 300 steps, seed-identical data and sampler)

| arm | α | ρ | nm | δ₀ → δ_T (held-out 256) | CE₀ → CE_T | cosB | cosPop / active | dead % | trkPopR / active | s_∞/(k/E) | clip % |
|---|---|---|---|---|---|---|---|---|---|---|---|
| OFF | 0 | 0.34 | 1.082 | 0.317 → **0.420** | 4.859 → 4.804 | – | – | 0.3 | – | 0.071 | 16.5 |
| ORACLE (exact f(B_t), lab-only) | 1 | 0.34 | 1.082 | 0.317 → **0.075** | 4.859 → 4.804 | 1.000 | 0.48 / 0.57 | 68.0* | 2.24 / 1.79* | 0.071 | 22.2 |
| **DP ρ\* = 0.34** | 1 | 0.34 | 1.082 | 0.317 → **0.165** | 4.859 → 4.804 | 0.636 | **0.80 / 0.825** | 24.0 | 0.81 / 0.77 | 0.071 | 19.1 |
| **DP ρ = 0.02** | 1 | 0.02 | 1.082 | 0.317 → **0.314** | 4.859 → 4.797 | 0.691 | 0.62 / 0.84 | **86.7** | 1.02 / 1.07 | 0.254 | 22.6 |
| OFF, nm=0 (ablation) | 0 | 0.34 | 0 | 0.317 → 0.335 | 4.859 → 4.755 | – | – | 0 | – | 0 | 40.8 |
| ORACLE, nm=0 | 1 | 0.34 | 0 | 0.317 → 0.054 | 4.859 → 4.757 | 1.000 | 0.29 | 0 | 2.70 | 0 | 47.0 |
| DP ρ\*, nm=0 (lag/EMA only) | 1 | 0.34 | 0 | 0.317 → **0.052** | 4.859 → 4.757 | 0.412 | 0.55 | 0 | 1.04 | 0 | 42.5 |
| MF band-MF(4,.95), 50 steps | 1 | 0.34 | 1.082 | 0.317 → 0.161 | 4.859 → 4.842 | 0.592 | 0.72 | 4.0 | 0.80 / 0.75 | 0.049 | 60.3 |

Column key: `cosB` = mean over every 10th step of cos(surrogate aux gradient at the `f̃` actually used, exact batch aux gradient at
`f(B_t)`), both at the pre-update parameters; `cosPop` = same against the held-out (256-example) population load, `/active` =
restricted to steps where the dead zone did not engage; `trkPopR` = `‖(f̃−k/E)·T_max/T̄_true − (f_pop−k/E)‖/‖f_pop−k/E‖`;
`s_∞` = the design's known smoothed-noise std; `clip %` = mean fraction of examples clipped on the gradient group. (*) The ORACLE arm
does not consume `f̃`; its dead-zone/tracking columns describe the post-processing running alongside and are meaningful only as a
monitor: they say the ORACLE drove the population imbalance below the ρ\* noise floor, at which point the DP estimate correctly reads
"balanced".

δ_t on the held-out split every 25 steps (the lab-only "true imbalance"):

```
OFF            0.317 0.339 0.354 0.345 0.352 0.383 0.399 0.418 0.409 0.383 0.375 0.403 0.420
ORACLE         0.317 0.162 0.110 0.083 0.101 0.074 0.092 0.084 0.071 0.074 0.060 0.074 0.075
DP rho*=0.34   0.317 0.191 0.140 0.102 0.131 0.129 0.131 0.141 0.139 0.141 0.115 0.149 0.165
DP rho=0.02    0.317 0.266 0.240 0.225 0.212 0.237 0.256 0.261 0.270 0.269 0.260 0.294 0.314
OFF nm=0       0.317 0.316 0.321 0.322 0.325 0.329 0.332 0.335 0.336 0.335 0.330 0.333 0.335
ORACLE nm=0    0.317 0.143 0.092 0.084 0.073 0.062 0.071 0.062 0.055 0.045 0.045 0.047 0.054
DP rho* nm=0   0.317 0.144 0.103 0.087 0.075 0.073 0.069 0.066 0.056 0.053 0.049 0.052 0.052
MF (50 steps)  0.317 0.131 0.161
```

### 2.2 What the arms show (VERIFIED numbers, interpretation marked)

1. **The mechanism works end to end and does what the design says it does.** With the DP `f̃` at ρ\* the imbalance falls from
   0.317 to ≈0.13 within 75 steps and then hovers at 0.12–0.17, i.e. at the design's dead-zone threshold for this regime
   (`δ < √c·s_∞/(k/E)/(T̄_true/T_max) = 0.133`, printed by the script); the dead zone engages 2 % of the time in the first 50 steps
   and 32–42 % in the last 100, exactly when the population imbalance sits at the floor. The ORACLE (exact non-private `f(B_t)`)
   reaches 0.075. Fraction of the ORACLE's reduction recovered by DP: **0.74**. The noise-free DP ablation (same lag/EMA, nm=0)
   reaches 0.052 = the noise-free ORACLE's 0.054: **the one-step lag and β=0.9 EMA cost nothing**; the whole gap between DP and
   ORACLE is the DP noise on the load release, as priced.
2. **ρ = 0.02 is the noise-floor demonstration the task asked for.** At `B̄=32, nm=1.08, β=0.9` the design's r formula gives a
   single-release per-entry noise of `r₁ = 1.18·(k/E)` and a smoothed noise of `0.254·(k/E)`, vs a released imbalance of
   `δ₀·T̄_true/T_max = 0.238` → ratio 1.07. The dead zone engaged on **86.7 %** of the steps (70 % in the first 50, 100 % in the last
   100); `f̃` was exactly `k/E` — i.e. the surrogate was *switched off*, as designed — on all but ≈40 steps, and those steps (which
   passed the dead zone while the bias-corrected `s_t` was still large, `s_1 = 1.11·k/E`) were early and carried real signal
   (cosPop 0.84–0.97 at steps 20 and 30, then 0.04 at step 100 — a *false pass*, §3). The imbalance drifts back to 0.314 as the
   noise walk on the router pushes it up (OFF goes to 0.420). The v2 default ρ = 0.02 is therefore **not** a usable share at this
   toy `B̄`; the formula that says so is the design's own (§2.6), and at the preset regime (`B̄=256, k/E=0.125, nm=0.56, β=0.99`)
   the same formula gives the 2.35 % of the design table. ρ\* here costs **×1.158** gradient noise (√(1+0.34)); nothing else changes
   (V3, V4).
3. **The batch load `f(B_t)` is itself a noisy estimate at B̄=32** — a lab finding that reframes the "cosine to the exact batch aux
   gradient" metric. The per-step batch-sampling std of the released signal `(1/B̄)Σ_x w_x d(x)` is **0.10–0.13 (k/E) per entry**
   in every arm (static-model OFF/nm=0: 0.110), larger than the ρ\* smoothed DP noise (0.071) and comparable to the residual
   imbalance. Consequently even the *exact* batch gradient has cosine only **0.48** (ORACLE) / 0.29 (ORACLE nm=0) to the population
   aux direction once the imbalance is small, whereas the DP `f̃` (an EMA over ≈10 batches) has **0.80** — the DP estimate is a
   *better* estimate of the population load than the single batch it replaces. The task's metric, cos(DP surrogate, exact batch aux
   gradient) = **0.64** at ρ\* (0.95–0.96 in the first 30 steps while δ ≫ floor, 0.3–0.9 later), is bounded above by this
   batch-sampling noise, not by the DP mechanism (the nm=0 DP ablation gets 0.41 on the same metric while matching the ORACLE's δ
   trajectory). At the preset regime (`T=1024` tokens per example, 64 experts, B̄=256) per-example `h(x)` is far less extreme
   than here (6-token vocabularies, T ≤ 32), so this effect should be much smaller — PLAUSIBLE, not measured (critic G3).
4. **The aux term costs nothing in CE and the DP noise dominates learning at this scale.** CE on held-out is identical across
   OFF/ORACLE/DP to 3 decimals (4.804); the nm=0 arms reach 4.755–4.757. `nm·C_g·√d/B̄ ≈ 86` vs signal ≤ C_g = 5.7 per step: a
   `B̄=32` DP run is noise-dominated on 197 k parameters, as expected; the OFF arm's imbalance *grows* under the noise walk on the
   router (0.317 → 0.420) while it is flat without noise (→0.335).
5. **The realised release noise matches the design's closed form**: empirical pooled probe-noise std / predicted `σ_h/(λ√L)·√((E−1)/E)`
   = **0.999** (DP-SGD arms), **1.033** (MF arm, over 50 correlated steps); `s_t` early values for ρ\*: `0.308, 0.218, 0.178, …`
   (k/E units) — the bias-corrected early estimate is honestly noisier and the dead zone accounts for it (step 1 dead, then open).
6. **Per-example gradient norms move at α = 1**: median 5.72 → 7.07 (×1.24) with the surrogate on at δ₀ = 0.32, and the clip rate
   in the first 50 steps is 60 % (MF arm) vs 19 % over the 300-step DP-SGD run. The design's "< 0.1 %" claim is for α ≤ 1e-3
   (phase-1 E3) and is **not tested** here; at α = 1 the aux term enters the clipping budget materially. See deviation D3.

### 2.3 V5 pass criteria and the borderline one

Criteria (all printed): probe never clipped and `z` stays 0 (True); ORACLE cosB = 1 (True, 1.0000132 — fp32); DP recovers ≥ 50 %
of the ORACLE reduction (0.74, True); DP cosPop while active > 0.8 (**0.825**, True — but **0.774 in run 2 with a 64-example
held-out reference**: the criterion is sensitive to the reference set's own sampling noise and should be read as "≈0.8", not as a
clean pass); ρ=0.02 dead zone > 50 % (86.7 %, True); pooled noise within 15 % of prediction (0.999, True). Run 1 additionally had a
metric bug (cosines computed *after* the SGD update, giving ORACLE cosB 0.975); fixed in run 2, output kept for the record.

## 3. Findings the design should absorb

- **F1 (ρ default).** ρ = 0.02 is a *preset-regime* number. The prototype's r-formula rule ("smallest ρ with smoothed noise below
  30 % of the released imbalance") gave ρ\* = 0.34 at `B̄=32, nm=1.08, β=0.9, δ₀=0.32`. The design should ship the rule (or the
  table) rather than the constant, and the trainer should print `s_∞/(k/E)` and the dead-zone δ at setup so a user sees when the
  floor is above any plausible imbalance. VERIFIED.
- **F2 (false passes with bias correction).** With β = 0.9 the bias-corrected `s_t` is large for the first ~10 steps and pure-noise
  passes happened at ρ=0.02 (steps ≈20–30 carried signal, step 100 did not: cosPop 0.04). The design's 4.2e-6 false-pass rate is for
  the χ²₆₃ tail at E = 64; at E = 8 (χ²₇) the same `c = 2` gives P(χ²₇ > 14) ≈ 5e-2 per step — the dead zone is much weaker at
  small E. PLAUSIBLE (arithmetic on the χ² tail; the observed ≈13 % non-dead steps at ρ=0.02 are consistent with it). Not a Mellum2
  concern (E = 64) but should be stated as E-dependent.
- **F3 (batch-sampling noise of the load).** §2.2 item 3: the "exact" `f(B_t)` is a noisy estimator at small `B̄`; the EMA'd DP
  release is closer to the population direction than the single-batch oracle. The design's "faithfulness to the HF batch formula"
  is exact (V1) but the HF formula's own target moves per batch; the design's window/EMA is a feature even at nm=0. VERIFIED
  at toy scale; the preset-regime magnitude is G3 territory.
- **F4 (α).** At α = 1 the aux term dominates the router gradient (CE→router 0.001 vs aux→router 0.3 at init, probe2) and lifts
  per-example norms ×1.24; at Mellum's 1e-3 the term would be invisible at this toy scale. The design's α-independence of `C_g`
  holds only for α ≪ 1. VERIFIED (toy).
- **F5 (MF).** Under band-MF(4, .95) the EMA-filtered noise factor is 0.69× the DP-SGD one (design: 0.35× for bands 64 / β .99);
  the per-step realised σ is `base·‖row_t(C⁻¹)‖` to 3e-16, the two-group latch holds, the probe gets the correlated noise, and the
  post-processing with the strategy-derived `φ_t` works unchanged. VERIFIED. The b-min-sep MC accountant was **not** run (design
  §6.4: minutes to hours at δ/2 resolution); `mf_gaussian` un-amplified ε = 4.00 at nm = 1.082 for 50 steps is the valid looser bound.

## 4. Deviations from the design (and why)

| # | deviation | why |
|---|---|---|
| D1 | tiny model (E=8, k=2, L=2, hidden 64), `T_max=32`, `B̄=32`, `N=1024`, 300 steps | task spec; CPU |
| D2 | **β = 0.9** instead of 0.99 | task spec: a 300-step run must reach stationarity (lag 10 vs 100 steps); all `φ_t` / `s_t` computed for 0.9 |
| D3 | **α = 1.0** instead of `config.router_aux_loss_coef` (1e-3) / preset 1e-4 | at toy scale the router's CE gradient is 1e-3 and the DP noise per router coordinate ≈ 0.2/step; with α ≤ 1e-3 the arms would be indistinguishable. Consequence: per-example norms ×1.24 (item 6) |
| D4 | **ρ\* = 0.34** (computed by the design's r formula against the *released* imbalance `δ₀·T̄_true/T_max` at a 30 % target), plus the required ρ = 0.02 arm | task spec; design's 0.02 is preset-regime |
| D5 | `opaque_router_logits` forward emulated by a class-level replacement of the CausalLM forward (real `linear_nll_sum_chunked`, backbone `output_router_logits=True`) | the named kwarg on the chunked forward (design §9.1) does not exist in the repo; nothing tracked may be modified |
| D6 | ORACLE arm uses an extra no-grad batched forward to obtain `f(B_t)` at the same step | lab-only arm by definition |
| D7 | V5 sampler Poisson (task); V6 sampler `BMinSepSampler` (design preset); V6 accountant `mf_gaussian` un-amplified | b-min-sep MC accountant infeasible on 4 cores (design §6.4) |
| D8 | plain SGD, lr 0.005, no momentum/weight decay | task allows; keeps the noise walk from destroying the 64-dim model within 300 steps |
| D9 | `C_g`, `δ₀`, `T̄_true` measured on the disjoint held-out split (design §4.4); ρ\* uses `δ₀` (a lab quantity) | in a real run ρ would come from the table / a public prior, not from the protected data |
| D10 | arms executed in a 2-process pool (each process rebuilds the model from the same seed) | wall-clock; per-arm results are process-independent (seeded model, data, sampler, noise keys) |
| D11 | added three nm = 0 ablation arms and population-referenced metrics | to separate lag/EMA effects from DP-noise effects and batch-sampling noise from mechanism noise (§2.2) |
| D12 | the "true imbalance" and CE are evaluated on 256 held-out examples (64 in run 2) | 64 was too noisy to read a 0.05 difference in δ; run 2 numbers kept for comparison |

Not covered (out of scope for a CPU prototype): DDP, checkpoint/resume of the filter state, `torch.compile`, bf16, the fp32 router
opt-in, the second-moment stream, the trainer callback seam (`on_pre_optimizer_step`) — the prototype is a manual functional loop,
the shape the design's §9.2 helper is meant to serve. Whether the E=8 CPU run executed the dense `Opaque_MoE` or the grouped
route (design §5.1: grouped needs E ≥ 16) was not asserted — PLAUSIBLE dense; `config._experts_implementation` reads `grouped_mm`
but that is HF's config attribute, not the executed path.

## 5. How to re-run

```
cd /home/user/opaque && .venv/bin/python \
  /tmp/claude-0/-home-user-opaque/a97cca5e-900f-5617-b2df-d73b55081cac/scratchpad/research/prototype/prototype_load_release.py
```
≈ 200 s on 4 cores; writes `prototype_results.json` (checks, per-arm summaries, per-step histories incl. `f̃_t`, `s_t`, dead-zone
flags, `D_t`, cosines) next to the script.
