# Phase 2 — judge-dp: DP-correctness review of the four Mellum2 designs

Agent: `judge-dp`. Lens: **DP CORRECTNESS** (`.junie/differential-privacy-review.md` applied end to end;
`dp_correctness` weighted ×2 in the total). Repo `/home/user/opaque` @ branch `claude/mellum-dp-representation-r6slaz`,
nothing tracked modified. Inputs read in full: BRIEF, PHASE1-DIGEST, the review protocol, phase1-critic, phase1-math,
the relevant sections of phase1-primitives/literature, and the four designs
(`phase2-design-{faithful,minimal,optimal,skeptic}.md`).

Evidence tags: **VERIFIED** = I read the cited lines or ran the cited check this session;
**VERIFIED (critic R6 / literature)** = theorem text extracted from the primary PDF by phase 1 and quoted there — I
did not re-fetch papers; **PLAUSIBLE** = derived/read, not executed.
My checks (CPU, < 20 s total): `scratchpad/research/judge-dp/check.py` (Opaque accountant + `per_group_noise_stddev`)
and `scratchpad/research/judge-dp/mf.py` (band-MF row norms and filter factors for momentum 0.95 vs 1.0, n = 1024).

---

## 0. Verdict in one paragraph

All four designs converge on the same privacy mechanism — a zero probe parameter carrying the per-example router-load
vector as a second `PerGroup` group of the *same* clipped pytree, noised by the same Gaussian / matrix mechanism under
Opaque's Mahalanobis-optimal allocation, so that the accountant call is literally unchanged under both stacks — and that
mechanism is **correct**: I re-derived and re-ran every DP-relevant claim (sensitivity, whitening identity, subsampling,
adaptive use, MF latch, per-group MF sensitivity) and found no privacy error in the core. The designs differ in
hygiene and in secondary claims, and that is where they separate: **minimal** is the most careful about what leaves the
mechanism (probe group excluded from the un-noised `group_metrics` telemetry, never uses the realised batch size, hard
`ConfigurationError` for AUTO-S/adaptive, explicit that the MF columns are not at ε = 3) and touches no engine code;
**skeptic** has equally clean hygiene plus two things nobody else has (the C-calibration pass on private data is itself a
private query; the fp32-router-removes-flips story is refuted by experiment) but defaults to α = 0, which does not meet
the user's "faithful including the batch-level parts" ask; **faithful** and **optimal** both leave the new probe group
inside the trainer's existing un-noised per-group norm logging (`_dp_trainer.py:2258-2283`, VERIFIED), which contradicts
their own privacy statements, and optimal's independent-draw variant is theorem-backed (ε 3.000 → 3.003, VERIFIED by my
run) but omits the second sampler's RNG state from serialization and does not pin down per-rank coin independence.
**Winner under this lens: minimal**, with mandatory grafts from the other three (centred release, preset-momentum MF
filter, monitor/decision rule, fp32-router correction, second-moment exclusion, independent-draw opt-in).

---

## 1. Protocol walk-through applied to the shared core mechanism (all four designs)

### 1.1 Adjacency and protected unit
Example-level (one collated row; DPO: one preference pair), add/remove — the repo default
(`packages/opaque-engine/src/opaque/api/engine/clipping/_clipped_grad.py:135-145`, VERIFIED read by phase 1). All four
state it and state that replace-one doubles every bound. **Correct.** Note (not an error): for the load vector the
replace-one bound can be tightened to `‖h−h'‖₂ ≤ √(2k) = 4.0` (both histograms in `[0,1]^E`, sum `k`, so
`Σ(h_e−h'_e)² ≤ max|·|·Σ|·| ≤ 2k`) instead of `2√k = 5.66` / `2·2.646 = 5.29`; doubling is safe but loose.

### 1.2 Sensitivity of every release
- Gradient leaves: `C_g` by clipping (as today). ✓
- Load leaf: `‖h_x‖₂² ≤ max_e h_e · Σ_e h_e ≤ k` (uncentred, minimal/skeptic) and
  `‖h−k/E‖² = ‖h‖² − k²/E ≤ k(1−k/E) = 7 ⇒ Δ_h = 2.6458` (centred, faithful/optimal). Both structural; the group
  bound is a *bound*, never an active clip ⇒ unbiased release. VERIFIED arithmetic (math §3; the centred identity:
  `−2(k/E)Σh + k²/E = −2k²/E + k²/E = −k²/E`). ✓
- Both bounds hold for every prefix of previous outputs (they do not depend on `f̃_t` or `θ_t`) — the condition
  adaptive composition needs. ✓

### 1.3 Noise calibration — the Mahalanobis constraint
`per_group_noise_stddev` (`packages/opaque-engine/src/opaque/api/engine/noise_allocation.py:44-110`, VERIFIED read):
`σ_i = nm·√(B_i·Σ_j B_j)`. For two groups `(C, ρC)`: `σ_g = nm·C·√(1+ρ)`, `σ_h = nm·ρC·√(1+1/ρ)` and
`Σ_i (B_i/σ_i)²·nm² = 1/(1+ρ) + ρ/(1+ρ) = 1`. **VERIFIED by running the real function** (`check.py`:
`chk = 1.0000` at ρ = 0.02/0.05/0.1; `σ_g/(nm·C) = 1.0100 / 1.0247 / 1.0488`). The whitened concatenation is a
sensitivity-`1/nm` Gaussian (Dong–Roth–Su Thm 2.7, VERIFIED critic R6; Andrew et al. 2021 Thm 1 form, VERIFIED
literature B1). All four designs use this and none uses the naive `σ·C_g / σ·C_h` allocation (which would be
`gaussian(nm/√2)`). ✓

### 1.4 Composition and amplification
- DP-SGD / Poisson: the gradient and load releases share the sampling coin, so the step is ONE Poisson-subsampled
  Gaussian at multiplier `nm` — `poisson(gaussian(nm), q)*T` unchanged. Feldman–Shenfeld Lemma 3.2 / Thm 3.3 as
  implemented in `packages/opaque-accounting/src/amplification/poisson.rs:15-42` (VERIFIED read: the doc comment cites
  arXiv:2602.17284 Thm 3.3 / Alg. 8–9; exact Gaussian fast path when the base is a pure Gaussian). All four are
  correct that "compose two subsampled Gaussians" would be the *wrong* description (non-fresh coin); the joint-vector
  view is the right one (math §4(c)). ✓
- Accountant numbers (VERIFIED, `check.py`, Opaque accountant, q = 5.12e-4, T = 15625, δ = 1e-6): baseline
  **3.0004**; "pay in ε" `poisson(gaussian(nm)|gaussian(c·nm), q)*T` at ρ = 0.05 (c = 4.583) → **3.4171**, at
  ρ = 0.02 (c = 7.141) → **3.2345**. These match faithful §2.6, minimal §2.5, optimal §2.6 and skeptic §2.5 exactly.
- Adaptive use of `f̃_t = φ(y_{<t})`: same kind of dependence as `θ_t`; Zhu–Dong–Wang Thm 10 (VERIFIED critic R6).
  All four correct, and skeptic's mid-run switch of α (a change of the *row function*, not the bound) is covered by the
  same theorem. ✓
- DP-FTRL / band-MF / b-min-sep: the load group shares the example's participation pattern; constant `PerGroup`
  passes `_validate_constant_max_norm` (`packages/opaque-dpftrl/src/opaque/api/dpftrl/noise/_engine.py:473-517`,
  VERIFIED read: latches by equality of the whole `max_norm`, so a constant `PerGroup` is accepted);
  `mf_gaussian_noise` computes `base = per_group_noise_stddev(max_norm, nm)` then `C⁻¹` streaming, realised
  `σ = base·‖row_t(C⁻¹)‖` (`_mf_gaussian_noise.py:163-192`, VERIFIED read). Per-group MF correctness: for a shared
  participation pattern π the worst case of the whitened Frobenius norm is `s(π)²·Σ_g C_g²/σ_g² = s(π)²/nm²`, so the
  sup over π is `sens(C)²/nm²` — the scalar PLD (Denisov Thm 2.1 with adaptive rows, VERIFIED critic R6). The argument
  needs the pattern to be shared across groups, which it is (same example, same step). ✓ All four.
- `normalize_by = expected_batch_size = a.train_batch_size` (`_dp_trainer.py:1394`, VERIFIED read) and the
  b-min-sep sampler is constructed so `E|B_t| ≈ B̄` (`_b_min_sep.py:1-11`, VERIFIED read: `p = p₀/(1−p₀(b−1))`). ✓

### 1.5 Adaptivity / post-processing claims
EMA, window mean, renormalisation by the *noisy* sum, clamp, sum-zero projection, James–Stein shrinkage (optimal),
decision rule (skeptic), sign of the noised deviation for loss-free balancing (optimal §7.3): all functions of noised
outputs only — free. ✓ The one thing that would break this is renormalising by the private realised `|B_t|`
(`aux.batch_size`); minimal and skeptic say so explicitly; faithful/optimal avoid the issue by centring (Σ d = 0
structurally). ✓

### 1.6 DP-FTRL constant sensitivity
Fixed clipping + structural load bound ⇒ constant. AUTO-S on the gradient group keeps `R` constant and would pass the
latch, but `auto_scale_pytree` applies `R·g/(‖g‖+γ)` to *every* group of a `PerGroup`
(`packages/opaque-engine/src/opaque/api/engine/clipping/_auto.py:87-135`, VERIFIED read: `R: float | PerGroup`, no
per-group fixed exemption), which would rescale every `d(x)` to norm ≈ `C_h` — a different statistic (not a privacy
violation, the bound still holds, but a biased estimator). minimal/skeptic/faithful reject `auto` with the feature;
optimal proposes the `fixed_groups` engine change. All privacy-correct. ✓ Adaptive clipping excluded under MF by all. ✓

### 1.7 Serialization / RNG
Public filter state serialises through the registry (primitives §4.4 E2, VERIFIED phase 1). The gradient-noise key is
shared across ranks (`_dp_trainer.py:1478`, VERIFIED critic R4) ⇒ the noised probe leaf is rank-identical. The in-stream
designs add no new RNG. Optimal's independent draw adds a second Poisson sampler and a second noise key — see §3.

### 1.8 Side channels (logging, checkpoint, DDP)
`DPTrainer` logs, for every group of a `PerGroup` max_norm, `group_norms.mean()` (un-noised per-example norm mean) and
a `clip_rate` (`_dp_trainer.py:2258-2283`, VERIFIED read this session), plus `loss_aux` means un-noised (`:2286-2291`).
A new probe group is therefore logged **unless the design excludes it**. minimal §8 and skeptic §8 exclude it;
faithful §8 and optimal §8 classify `group norms` as "pre-existing telemetry, as today" — which for the *new* group
means the batch-mean of `‖λ d(x)‖` (a routing statistic) is released un-noised, contradicting their own privacy
statement ("no other quantity derived from private routing is released"). The clip-rate entry is harmless (always 0
by structure); the norm mean is a genuine, if low-information, unaccounted release. **Error (faithful, optimal).**

### 1.9 Second-moment streams
`paired_noise_stddevs` sums `Δ¹+Δ²` over all groups (`noise_allocation.py:153-255`, primitives §8(b) VERIFIED phase 1).
With `second_moment=True` the probe group's squared stream would consume budget (allocation still valid ⇒ utility, not
privacy). faithful (T6) and skeptic (§9.3) exclude it; minimal and optimal do not mention it. **Gap (minimal, optimal).**

---

## 2. Design-by-design findings

### 2.1 faithful — `phase2-design-faithful.md`
**Mechanism**: centred `d(x)`, `Δ_h = 2.646`, `C_h = λΔ_h(1+1e-6)`, ρ = 0.05 (×1.025), EMA with `β_f :=` workload
momentum under MF, `f̃_0 = k/E`, sum-zero projection, clamp.
- DP core: correct on every protocol item (§1). The `(1+1e-6)` guard is a nice touch (bound used in accounting is
  slightly *above* the true sup — safe).
- MF filter: the claim "EMA with β = strategy momentum is exactly `(1−β)·A`" is algebraically trivial and true; its
  numbers (`‖row_t(C⁻¹)‖ = 1.432`, `(1−β)‖row_t(A C⁻¹)‖ = 0.0824`) are for `band_mf_strategy(bands=64, momentum=0.95)`,
  which **is** what the preset runs (`examples/train_dpftrl.py:447-460` default `sgd`, `:495-498` momentum default
  0.95, `:1563-1581` `_make_strategy` passes it; VERIFIED read). My run reproduces 1.4309 / 0.0824 (`mf.py`). ✓
- Hygiene: **probe-group `group_norms` telemetry not excluded** (§1.8). Second-moment exclusion ✓ (T6).
- fp32 router: "removes the bf16 tie flips … E1b 12.9 % → 1.7 %" — **misattributed** (see §2.4 / errors E4).
- MF cost columns at `nm = 0.5622` are "equal base σ", not "ε = 3" (the b-min-sep calibration yields a different nm);
  §6.2 says "at equal base σ" for the comparison but the table header implies ε held. Minor presentation error.
- DPO in scope with the correct rule (reference forward contributes no aux; pair = protected unit). ✓
- Serialization: `extra_state` slot added to `save_dp_runtime_state` (signature change) — fine.

### 2.2 minimal — `phase2-design-minimal.md`
**Mechanism**: uncentred `s·h_x`, bound `s√k = λ = ρC`, ρ = 0.1 (×1.049), EMA .95 (DP-SGD) / window 256 (MF),
renormalise by the noisy sum with a `Σu > k/2` guard, opt-in feature.
- DP core: correct; the only design that VERIFIED the Mahalanobis identity by calling the real function *and* runs the
  accountant for every row (`cost_table.out`, VERIFIED read; matches my `check.py`).
- Hygiene: the best of the four on the trainer seams — probe group explicitly excluded from `group_norms` mean; probe
  leaf zeroed in the noised pytree after reading so Adam's update is exactly 0; `aux.batch_size` explicitly forbidden
  as a divisor; `ConfigurationError` for `clipping_mode ∈ {auto, adaptive}`; nothing un-noised about `h`/`f(B)` in
  `loss_aux`; T15 makes "accountant unchanged" machine-checkable (`epsilon_at(δ)` identical with/without).
- Explicit and correct that the MF table is at `nm_MF`, not 0.5622 (`×(nm_MF/0.5622)`). ✓ (unique among the four)
- **Error**: all band-MF numbers (`‖row_t(C⁻¹)‖ → 3.80`, "EMA .95 only reaches parity 0.185 vs 0.160", window-256
  = 0.0216, "workload momentum = 1.0 by default") were computed with the factory default `momentum=1.0`
  (`_band_mf.py:154`), not the preset's 0.95. Under the preset strategy (VERIFIED `mf.py`, n = 1024):
  `‖row_t(C⁻¹)‖ = 1.431` (stationary from t ≈ 7, no n-dependence — the momentum-1.0 row norm grows with n:
  2.26 @ 1024, 2.54 @ 2048, 3.80 @ 15625), EMA .95 = **0.0824** (2× better than iid, not parity), EMA .99 = 0.0249,
  window-256 = 0.0198. So "the MF-consistent filter is the boxcar, not an EMA" is an artifact; under the preset the
  EMA(0.95) is the exactly-matched filter and window-256 / EMA .99 are both ≈ 0.02–0.025. Utility only.
- Gap: second-moment stream exclusion not stated.
- fp32 router misattribution shared with faithful/optimal (E4).
- Recorder-under-checkpointing argument (ContextVar collector reset in `finally`) is the most precise of the four
  (PLAUSIBLE, test T5).

### 2.3 optimal — `phase2-design-optimal.md`
**Mechanism**: centred `d(x)`, ρ = 0.02 (×1.010), EMA .99, sum-zero projection, positive-part James–Stein shrinkage
with known noise std, AUTO-S mixed mode (engine change), opt-in independent forward-only Poisson release for DP-SGD,
opt-in loss-free balancing as post-processing of the same release.
- DP core: correct. Independent draw: two mechanisms with fresh coins and fresh noise composed adaptively — ZDW Thm 10 +
  FS Lemma 3.2 per factor; `|` composes at the process level (`core/_base.py:483 __or__`, VERIFIED read). Numbers
  VERIFIED by my run: c = 2, m = 4 → **3.0030**; c = 1, m = 4 → 3.1284; c = 2, m = 16 → 3.0011; 4× batch → 5.2992
  (all match `cost_table.json`). The "ε strongly convex in 1/σ" explanation is right (primitives E1(g)).
- Concurrent composition for the *rejected* MF side-release correctly marked PLAUSIBLE, not used. ✓
- Loss-free balancing: "sign of the noised deviation is post-processing — free" ✓; RR on the true sign correctly
  shown dominated (`64·ln((1−p)/p)`: 12.8 at p = .45, 70.3 at p = .25 — VERIFIED arithmetic). ✓
- AUTO-S mixed mode: `PerGroup(R, C_h)` constant ⇒ MF latch OK; per-record bounds `R` and `C_h` ⇒ privacy fine. Needs
  an engine change (`fixed_groups`), correctly identified. ✓
- **Hygiene gap**: same `group_norms` telemetry omission as faithful (§1.8).
- **Serialization/RNG gap** (independent draw): the second `PoissonSampler`'s key/position and the second noise key's
  counter are not in the checkpoint plan (§9.3 lists only `RouterLoadState`); under DDP "each rank samples its shard"
  with a key derived only from `fold_in(key, "opaque.moe.load_release")` — the per-example inclusion coins must be
  independent across ranks for the Poisson amplification proof (the mixture decomposition conditions on the sampled
  subset), so the key must be rank-domain-separated and its advancement serialized. Not a mathematical error; a
  condition the implementation must meet before "provably DP as run" holds for that variant.
- Second-moment exclusion not stated (gap).
- MF columns at `nm = 0.5622` (unstated "equal base σ" assumption; §2.4 says "the trainer calibrates nm … ε is held
  automatically", which is true for ε but means the MF *load-error* column is not the ε = 3 number).
- fp32 router misattribution (E4).
- Its MF filter table (momentum 0.95, n = 2048) is for the preset strategy and matches my run to 4 digits. ✓

### 2.4 skeptic — `phase2-design-skeptic.md`
**Mechanism**: α = 0 default (Regime A) + DP-released imbalance monitor (same probe leaf, ρ = 0.02), decision rule
`D_t > 0.5` on two consecutive evaluations, switch to the surrogate (α = 1e-4) with no accountant/latch change.
- DP core: correct; the switch is an adaptive-row change and is theorem-covered. The monitor's false-alarm arithmetic
  (z ≈ 22σ at τ = 0.5, W = 256; 4e-3 at ρ = 0.01, W = 64, τ = 0.25 over 64 coordinates) is VERIFIED arithmetic.
- Hygiene: probe group excluded from `group_metrics` ✓; second-moment excluded ✓; AUTO-S rejected ✓; **uniquely flags
  that the `C`-calibration pass over training data is itself a private query** (public proxy or accounted via the
  adaptive-clipping quantile release) — the protocol's "all releases in the privacy statement" line that the other
  three only half-address.
- **New experiment (VERIFIED, `a_fp32_router_flips.py`, read + output)**: on a bf16 tiny Mellum vs the fp32 model,
  stock bf16 router 46/54 flips per 1024 rows vs fp32-logit router 51/56 (random) and 46/47 vs 39/43 (structured);
  vmap-vs-eager at equal precision 0/1024. Conclusion: flips come from bf16 hidden states, the fp32 GEMM only removes
  output rounding. The logic is sound (bf16 activations have 3 significant digits before the router sees them) and
  it refutes the "fp32 router removes the tie flips / E1b 8× error" rationale of the other three; E1b pinned to the
  fp32 *model's* routes, which no in-vmap router precision change can produce. Toy-scale, but the mechanism is
  precision-generic.
- **Errors** (utility, not privacy): (i) "workload A = prefix sums (momentum 1.0 is the preset default)" — wrong for
  the preset (momentum 0.95, §2.2); the factors 2.26/3.80, 0.0157/0.0216, 0.0253/0.0381 are for momentum 1.0; under
  the preset they are 1.431, 0.0198, 0.0249. (ii) §2.2 step 2: "denominator noise std `√E·σ_h/s = 0.044` … 0.55 %":
  `σ_h/s = nm·√k·√(1+1/ρ)/B̄ = 0.0444` is the *per-entry* std; the 64-entry sum has std `8 × 0.0444 = 0.355`, i.e.
  **4.4 %** of k per single step (≈ 0.3 % after the W = 256 window), 8× the stated figure.
- Faithfulness: α = 0 by default is DP-trivially correct but the user's ask is faithfulness *including* the
  batch-level term; skeptic's own Regime B is the same surrogate as the others, so the machinery is complete — the
  default is the deviation.

---

## 3. Theorem-citation audit
Every theorem the designs rely on is one the critic re-verified from the primary PDF (critic R6, VERIFIED): Zhu–Dong–Wang
Def. 7 / Thm 10 (https://arxiv.org/abs/2106.08567); Feldman–Shenfeld Lemma 3.2 / Thm 3.3 (https://arxiv.org/abs/2602.17284,
as cited by `poisson.rs`); Denisov et al. Thm 2.1 (https://arxiv.org/abs/2202.08312); Dong–Roth–Su Thm 2.7 / Cor. 3.3
(https://arxiv.org/abs/1905.02383); Andrew et al. Thm 1 (https://arxiv.org/abs/1905.03871). Objective sources: Switch eqs.
(4)–(6) (math §0), ST-MoE §3.1 eq. (5) (literature C2), Mellum2 TR §3.6/§5.1.2/appendix (literature F1/F.2). Optimal's
extra sources — Wang et al. 2024 Alg. 1 (literature C5), DeepSeek-V3 eq. (16) (C4), Davody 2020 (B3), Ponomareva §5 (B2) —
are all tagged VERIFIED in `phase1-literature.md`; Bu et al. 2023 (AUTO-S utility) and Vadhan–Zhang concurrent composition
are honestly marked PLAUSIBLE and are not load-bearing for privacy. No design invents a citation. Skeptic explicitly cites
nothing beyond phase 1. ✓

---

## 4. Scores (0–10; total = (2·dp + faith + util + impl + compl)/6)

| design | dp_correctness | faithfulness | utility | implementability | completeness | **total** |
|---|---|---|---|---|---|---|
| faithful | 8.0 | 9.0 | 8.0 | 7.5 | 8.5 | **8.17** |
| **minimal** | 8.75 | 8.0 | 8.0 | 9.0 | 9.0 | **8.58** |
| optimal | 8.0 | 8.0 | 9.0 | 6.5 | 9.0 | **8.08** |
| skeptic | 9.0 | 6.5 | 8.0 | 8.0 | 8.5 | **8.17** |

Rationale for the dp_correctness spread (the core is identical and correct in all four; the spread is hygiene and
completeness of the "as run" argument): skeptic 9.0 (cleanest side-channel story, calibration-pass caveat, nothing new
released beyond the probe); minimal 8.75 (same hygiene, nm_MF caveat, machine-checkable accountant invariance; −0.25 for
the missing second-moment exclusion); faithful 8.0 (telemetry gap contradicting its privacy statement; otherwise exact);
optimal 8.0 (telemetry gap; second sampler RNG/rank-independence unstated; largest surface, each piece individually
correct). Faithfulness: faithful keeps α = config and the executed-route definition; minimal is opt-in (presets on);
optimal defaults α = 1e-4 (not config) and shrinkage alters the term's small-signal behaviour (defensible, but a
modification); skeptic defaults α = 0. Utility: optimal's ρ = 0.02 + EMA .99 under the preset strategy (0.83 % of k/E,
VERIFIED factor) and the independent draw are the best numbers; the others are within a factor 2. Implementability:
minimal touches no engine code and rides existing seams; optimal needs an engine change plus a second sampler and
accounting seam.

**Winner: minimal.**

---

## 5. Grafts the synthesis must keep (from non-winning designs)

1. **Centred release** (faithful/optimal): carry `λ·(h(x) − k/E)` with `Δ_h = √(k(1−k/E)) = 2.646`,
   `C_h = λΔ_h(1+1e-6)`, sum-zero projection of the noised leaf; drops the noisy-sum renormalisation and the
   `k·|B_t|/B̄` scale question (critic M11) entirely, 6.5 % less noise.
2. **MF filter from the instantiated strategy** (faithful §6.1, optimal §6.iv): the preset is
   `band_mf_strategy(bands=64, momentum=0.95)` ⇒ `‖row_t(C⁻¹)‖ = 1.431`, EMA(β = momentum) = `(1−β)·A` factor 0.0824,
   EMA .99 = 0.0249, window-256 = 0.0198; compute the filter row norms from the actual `C⁻¹` (and `lr_schedule`) at
   setup and store them in the state — never hand constants, never the factory-default momentum.
3. **fp32 router is not a flip/drift lever** (skeptic §3, VERIFIED toy): keep it as an opt-in pretraining-faithful
   choice, not as a numerics fix; the oracle must be precision-matched and the drift metric must carry the vmap-vs-eager
   route-flip counter at equal precision.
4. **Monitor mode + decision rule** (skeptic §2.3): `D_t = max_e|f̄_e − k/E|/(k/E)`, trip on two consecutive
   evaluations; α is an explicit argument and can be switched mid-run with no accountant/latch change. Ship as the
   Regime-A option and as the diagnostic in every mode.
5. **Calibration pass is a private query** (skeptic §4.2): the `clipped_grad(C=∞)` grad-norm pass to choose `C`
   must run on a public proxy or be accounted (adaptive-clipping quantile release).
6. **Second-moment exclusion** (faithful T6 / skeptic §9.3): the probe group must be excluded from `second_moment`
   paired streams or the run rejected — one-line exclusion + test.
7. **Probe-group telemetry exclusion and no `aux.batch_size`** (minimal §8 / skeptic §8) — already in the winner;
   must survive any merge with faithful/optimal's hygiene tables.
8. **Independent forward-only release as a DP-SGD opt-in** (optimal §2.3): ε 3.000 → 3.003 (VERIFIED), ×1.0002
   gradient noise; conditions: rank-domain-separated sampler key, sampler/noise RNG state serialized with the
   checkpoint, `ConfigurationError` under any MF mechanism.
9. **Noise-aware shrinkage / dead zone** (optimal §2.2 step 5, §7.3) as optional post-processing with the noise std
   taken from the stored filter row norm.
10. **`fixed_groups` in `auto_clipped_grad`** (optimal §9.2) as the v2 engine change that lifts the AUTO-S restriction.
11. **DPO scope** (faithful §7 / optimal §7): pool chosen + rejected policy forwards, reference forward contributes no
    aux and its stats must never reach the probe; converter forwards `router_aux_loss_coef`.
12. **nm_MF caveat** (minimal §2.5, already in the winner): MF load-error columns are at equal base σ; the ε = 3
    b-min-sep calibration must be run to quote MF numbers at ε = 3.
13. Optional tightening (this review): replace-one load bound `√(2k) = 4.0` instead of `2√k`.

---

## 6. Errors found (factual or mathematical), with corrections

- **E1 (minimal)** — band-MF numbers computed with the factory-default `momentum=1.0` (`_band_mf.py:154`):
  `‖row_t(C⁻¹)‖ → 3.80`, "EMA .95 = 0.185 ≈ parity", window-256 = 0.0216, "workload momentum = 1.0 by default".
  The preset runs momentum 0.95 (`examples/train_dpftrl.py:447-460` default `sgd`, `:495-498`, `:1563-1581`).
  Correction (VERIFIED `mf.py`): 1.431 (stationary from t ≈ 7, n-independent), EMA .95 = 0.0824, EMA .99 = 0.0249,
  window-256 = 0.0198; the EMA(0.95) is the matched filter, the "boxcar not EMA" conclusion is an artifact.
- **E2 (skeptic)** — same wrong strategy: "workload A = prefix sums (momentum 1.0 is the preset default)"; factors
  2.26/3.80, 0.0157/0.0216, 0.0253/0.0381 → 1.431, 0.0198, 0.0249 under the preset.
- **E3 (skeptic §2.2 step 2)** — renormalisation-denominator noise "√E·σ_h/s = 0.044 (0.55 % of k)": `σ_h/s = 0.0444`
  is per entry; the sum's std is `√64 × 0.0444 = 0.355` = 4.4 % of k single-step (8× the stated value).
- **E4 (faithful §3, minimal §3, optimal §3)** — "fp32 router logits remove the bf16 tie flips (E2 ≈ 1 %/layer) and the
  8× excess router/expert gradient error (E1b)": REFUTED at toy scale by skeptic's `a_fp32_router_flips.py` (stock
  bf16 router 46/54 vs fp32-logit router 51/56 flips per 1024 rows against the fp32 model; structured 46/47 vs 39/43).
  Flips originate in the bf16 hidden states; E1b's gain came from pinning to the fp32 *model's* routes. The fp32
  router is a pretraining-faithfulness choice only; vmap-vs-eager flips at equal precision are 0 regardless (F8).
- **E5 (faithful §8, optimal §8)** — hygiene tables mark `group norms` "pre-existing, as today", but
  `_dp_trainer.py:2258-2283` logs `group_norms.mean()` for every `PerGroup` group, so the new probe group's un-noised
  batch-mean `‖λ d(x)‖` would be logged — contradicting both designs' privacy statements. Correction: exclude the probe
  group from `group_metrics` (minimal §8 / skeptic §8).
- **E6 (faithful §2.6, optimal §2.6, skeptic §2.5)** — MF load-error columns are evaluated at `nm = 0.5622`, the
  Poisson/DP-SGD ε = 3 calibration; under `b_min_sep(mf_gaussian(...))` the trainer calibrates a different nm, so those
  columns are "equal base σ" numbers, not ε = 3 numbers (minimal states this correctly).
- **E7 (optimal §9.3, independent draw)** — omission, not a formula error: the second sampler's key/position and the
  second noise key's counter are absent from the serialization plan, and per-rank domain separation of the sampler key
  is not specified; both are required for the Poisson amplification argument to describe the mechanism as run.
- **E8 (minimal, optimal)** — no exclusion of the probe group from `second_moment` paired streams
  (`paired_noise_stddevs` sums over all groups): budget silently spent on a squared load stream (utility; allocation
  remains valid).

Nothing in any design violates a theorem assumption or the implemented privacy model; every ε and every Mahalanobis
identity quoted by the four designs reproduced exactly in my own run.
