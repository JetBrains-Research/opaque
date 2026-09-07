# Phase 2 refutation — lens: adjacency, sensitivity, noise calibration

Agent: `refute-sensitivity`. Target: `phase2-final-design.md` (read in full). Repo `/home/user/opaque` @ `ef1abc5`,
nothing tracked modified. All checks CPU-only via `uv run python`; scripts and raw output in
`scratchpad/research/refute-sensitivity/{costtable.py, adversarial.py, filters.py, edge.py, edge2.py, edge3.py,
nm_mf_small.py, nm_mf_small.out}`. HF = `.venv/lib/python3.11/site-packages/transformers/` (5.16.1).

Tags: **VERIFIED** = I read the cited lines or ran the cited script this session; **PLAUSIBLE** = derived, not executed.
Where a claim rests on a theorem I reuse only citations phase-1 `critic` R6 extracted from the primary PDFs (ZDW Def. 7 /
Thm 10, Feldman–Shenfeld Lemma 3.2 / Thm 3.3, Denisov Thm 2.1, Dong–Roth–Su Thm 2.7, Andrew Thm 1); I fetched no paper.

## 0. Verdict

**sound-with-fixes.** I could not construct a neighbouring-dataset pair that breaks any stated sensitivity bound, the
divisor, the Mahalanobis allocation, or the "accountant unchanged" claim; every number in the cost table reproduces with the
real allocator and accountant. What I did break is a set of *secondary* claims around the bound and its consumers: the
un-noised `grad_norm` telemetry (which the design's own T17 says does not exist), the "structural, never active" clip under
two realistic conditions (recorder double-capture; MPS round-off guard), non-binary masks, the per-layer cost framing (the
design prices per-layer at ×5.29 when it is free for the pooled estimate), the James–Stein "never a random regulariser"
overclaim, the uncorrected EMA warm-up bias, the calibration-data wording, and one pre-existing cross-example coupling
(`all_valid_attention`) that the design inherits and should name in its privacy statement. None of these changes ε or the
add/remove bound; all have one-line fixes. No blocking finding.

---

## 1. The cost table, recomputed (VERIFIED, `costtable.py`, 9.3 s)

Preset regime `nm = 0.5622, q = 256/5e5, T = 15625, δ = 1e-6, B̄ = 256, k = 8, E = 64, C_g = 0.9, L = 28`. Every ρ row is
built with the real `PerGroup` + `per_group_noise_stddev` (with bounds already divided by `B̄`, exactly as `clipped_grad`
stores them, `_clipped_fun.py:633, 803`) and the real accountant `opaque.dpsgd.accounting`.

```
Delta_h centred = 2.6458; uncentred sqrt(k)=2.8284; replace-one tight sqrt(2k)=4.000; 2*Delta_h=5.292; per-layer sqrt(kL(1-k/E))=14.000
baseline eps=3.0004
EMA factors: .95=0.1601 .99=0.0709; window256=0.0625
 rho  |   c    | grad_infl | mahal    | eps_nm_held |  r1   | SGD ema99 | SGD w256 | MF single | MF ema99 | JS thr SGD/MF
 0.50 |  1.732 | x1.2247   | 1.000000 | 5.443       |  8.1% |  0.57%    |  0.50%   |  11.5%    |  0.20%   | 0.0057/0.0020
 0.20 |  2.449 | x1.0954   | 1.000000 | 4.219       | 11.4% |  0.81%    |  0.71%   |  16.3%    |  0.28%   | 0.0080/0.0028
 0.10 |  3.317 | x1.0488   | 1.000000 | 3.703       | 15.4% |  1.09%    |  0.96%   |  22.1%    |  0.38%   | 0.0108/0.0038
 0.05 |  4.583 | x1.0247   | 1.000000 | 3.417       | 21.3% |  1.51%    |  1.33%   |  30.5%    |  0.53%   | 0.0150/0.0053
 0.02 |  7.141 | x1.0100   | 1.000000 | 3.234       | 33.2% |  2.35%    |  2.07%   |  47.5%    |  0.83%   | 0.0233/0.0082
 0.01 | 10.050 | x1.0050   | 1.000000 | 3.172       | 46.7% |  3.31%    |  2.92%   |  66.8%    |  1.16%   | 0.0329/0.0115
indep c=2 m=4: eps=3.0030 | indep c=1 m=4: eps=3.1284 | indep c=2 m=16: eps=3.0011
```

Every entry equals the design's §2.6 table (the "r1 closed form" `nm·Δ_h·√(1+1/ρ)/(B̄·k/E)` matches the allocator-derived
value to 1e-9 at every ρ). The band-MF filter factors used in the MF columns were re-derived independently (`filters.py`,
1.5 s): `‖row_t(C⁻¹)‖` for `band_mf_strategy(bands=64, momentum=0.95)` at n = 1024 is 1.2833 (t = 0), 1.4307 (t = 7), 1.4309
(t ≥ 63 — stationary); EMA .95 → 0.0824, EMA .99 → 0.0249, window-256 → 0.0198, against iid 0.1601 / 0.0709 / 0.0625. All
VERIFIED. (One arithmetic slip is mine, not the design's: my script double-counted LoRA A/B and printed d = 13 991 936; the
design's d = 8 257 536, √d = 2874, per-step noise norm 5.68 is correct: q 16·(2304+4096), k/v 16·(2304+512) each, o
16·(4096+2304) = 294 912 per layer × 28.)

The `nm_MF` caveat (§2.6 †, §6.4) remains **uncomputed by me too**: a bounded probe at horizons n ∈ {256, 1024}
(`nm_mf_small.py`, 240 s wall clock) had produced no line by the time this report was written — see §5 for the final state.
Nothing in this lens depends on it (all ratio claims are nm-free).

---

## 2. Attacks and outcomes

### 2.1 The structural bound on the load leaf — neighbouring pairs through the real clipper (VERIFIED, `adversarial.py`)

Setting: `E = 64, k = 8`, synthetic routes `(L, T, k)`, `λ = ρC_g/Δ_h = 0.006803`, `C_h = λΔ_h(1+1e-6) = 0.018000`, a probe
leaf `z ∈ R^64` with loss term `⟨z, λ·d(x).detach()⟩`, a dummy gradient leaf, real `per_group(params,
router_load_probe=C_h, fallback=0.9)` and real `clipped_grad(..., normalize_by=256, return_aux=True)`.

| attack | construction | result |
|---|---|---|
| A1 sup example | every (l, t) routes to experts 0..7, full mask | `group_norms["router_load_probe"] = 0.018000 = 0.99999896·C_h`; **not clipped**; released leaf `== λ·Σ_x d(x)/B̄` **bit-exactly** (max diff 0.0) — the scale is exactly 1 |
| A2 one valid token (`T_x = 1`) | random routes, mask with one 1 | norm 0.0093 ≤ C_h; unclipped; exact |
| A3 neighbouring pair `D` vs `D ∪ {sup}` | A3 vs A1 | `‖M(D') − M(D)‖` on the probe = 7.031250e-05 = **0.99999906 · C_h/B̄** — the add/remove sensitivity is attained and never exceeded |
| A4 replace-one | sup on {0..7} vs sup on {8..15} | `‖d − d'‖ = 4.0000 = √(2k)` exactly (design §2.1: tight replace-one bound; `2Δ_h = 5.29` is the loose one) |
| A5 segment-id mask | weights 1,1,…,2,2,…,3,3 (packing-style) | `Σ_e h = 8.000`, `max h = 1.000`, `‖d‖ = 2.6458 ≤ Δ_h` — the bound survives any **non-negative** weighting |
| A9 `h` computed in bf16 | sup example | unclipped, exact: 0 and 1 are exactly representable; and for any `h ∈ [0,1]^E` with `Σh ≤ k + 8·2⁻⁹` the rounded `‖d‖²` stays < 7 because rounding moves mass onto entries far from the 1-vertex (checked analytically, PLAUSIBLE for the general case, VERIFIED at the sup) |
| ties | 1000 all-equal bf16 logit rows → fp32 softmax → `topk(8)` | 0/1000 rows with duplicate indices; 8 distinct experts — `h_e ≤ 1` holds under exact ties (`edge3.py`) |

The derivation the design gives (`‖h‖² ≤ max_e h_e · Σ_e h_e ≤ k`, centred `‖h‖² − k²/E ≤ 7`) is correct and I could not
break it with any admissible input. Upheld.

### 2.2 Where the "structural, never active, unbiased" claim does break (VERIFIED)

- **A8 — recorder double-capture.** With `2L` router-logit tensors normalised by the *configured* `L` (the design's
  definition of `h` uses `1/(L·T_x)`, §1.1), the sup example's probe norm is `0.036637 = 2.035·C_h` → **the clip fires** on the
  probe group and the released statistic is biased 2×; the random example's `Σ_e h = 2k`. Normalising by the *captured*
  count instead restores `‖λd‖ = 0.018000`, unclipped, exact. Privacy is intact either way (the clip bounds it). See R2.
- **A10 — the `(1+1e-6)` guard vs `_guard_scale`.** `clip_pytree` shrinks every ratio by `2(u_store + norm_roundoff)`
  before `min(1, ·)` (`_pytree.py:111-131, 135-148`), and `norm_roundoff` is computed from the **widest leaf of the whole
  pytree** and the number of leaves (`_pytree.py:214-224`, `_reduction_terms`). For the preset (≈113 LoRA leaves, widest
  4096×16) the shrink is 1.788e-7 on CPU/CUDA (fp64 sum-of-squares) — the design's `1e-6` guard beats it and the scale is
  exactly 1 (A1) — but **6.539e-5 → 1.309e-4 on MPS** (fp32 accumulator, `_sq_accum_dtype`), where `(1+1e-6)(1−1.3e-4) < 1`:
  an example at (or within 1.3e-4 of) the sup gets its probe scaled by ≈ 0.99987. Negligible in size; the claim "never
  rescaled / exactly unbiased" is false on MPS at the sup. See R3.
- **A6 — non-binary masks.** The design passes the attention mask as a *weight* (like HF, `modeling_mellum.py:581` casts it
  to float32 and multiplies). A same-sign additive mask (0 valid / −1e9 pad) keeps the bound (`Σh = 8`, `‖d‖ = 0.53`) but
  computes `h` from the **padding** tokens; a mixed-sign mask (`[1, −0.5, 0, …]`) gives `h ∈ [−0.75, 1.5]`, `‖d‖ = 3.182 >
  Δ_h` — bound broken (`edge3.py`). Nothing in the DP trainer produces such masks by default; the fix is one line. See R4.

### 2.3 `T_x = 0` (fully-masked row) — upheld, with the mechanism named (VERIFIED, `edge.py`, `edge2.py`)

`torch.func.grad` of the probe term for an all-masked row is 64 NaNs (0/0 in `h`), under vmap too. The design does not
mention this case (T2 lists "padded tokens excluded" only). It is nevertheless safe because **`clip_pytree` replaces NaN/Inf
per-example leaves by zero before the norm** (`_pytree.py:525-531`, docstring "NaN and Inf values in the input are replaced
with zeros before clipping … DP-safe"): the real clipper reports `group_norms["router_load_probe"] = 0.0`, releases a finite
leaf, and the same example's finite gradient leaf still contributes normally (`edge2.py`: two-example sum equals
`2·0.1·[1,2,2]/256`, not the one-example value). So an empty row contributes `d = 0` — better than the `clamp(min=1)` guard I
first proposed (which yields `d = −k/E·1`). The existing CE paths guard the analogous 0/0 with `clamp(min=1)`
(`opaque-alignment/.../sft/loss/_nll.py:86, 122`; `patches/.../components/cross_entropy.py:104, 341`). Recommendation
(not a refutation): make the zero-contribution behaviour explicit in `router_load_and_probs` (`T_x.clamp(min=1)` applied to
the *numerator-and-denominator* form gives `h = 0` directly) and add the case to T2, so the property does not depend on the
engine's NaN sanitiser.

### 2.4 Divisor: expected vs realised batch (VERIFIED)

- DP-SGD: `normalize_by = expected_batch_size = a.train_batch_size` (`_dp_trainer.py:1394`, `:4266-4290`); the clipped sum
  is divided once after accumulation (`_clipped_fun.py:802-803`) and the stored bound is `clipping_norm / normalize_by`
  (`:633`) — bound and noise scale together, so the realised `|B_t|` never enters the sensitivity.
- DDP: `train_batch_size` is the **cluster-wide** logical batch (`_training_arguments.py:235-240`), `q =
  train_batch_size/N_total`, each rank Poisson-samples a disjoint shard at that `q` with a **rank-folded** key
  (`_dp_trainer.py:3735-3770`: "each rank draws an independent Bernoulli(q) mask"), so the union is one Poisson(q) draw and
  the global divisor is right. (The comment there records a resume caveat for multi-GPU — pre-existing, out of scope.)
- b-min-sep: `p = p₀/(1−p₀(b−1))` (`_b_min_sep.py:6-11`), and the warm start (Algorithm 2 lines 1–2, `:119-129`) puts the
  cooldown chain in its stationary distribution at step 0, so `E|B_t| = p₀N = B̄` from the first step (steady state:
  `r = p/(1+p(b−1)) = p₀`, checked by hand). My initial worry (a 3.3 % transient at t < b) is closed by the warm start.
- The centred release needs no renormalisation: `Σ_e d(x) = 0` per example makes `Σ_e d̂_t = 0` exactly, and the realised
  `|B_t|/B̄` (sd 6.25 % under Poisson(256)) multiplies the imbalance only. Upheld.

### 2.5 Mahalanobis allocation and "accountant unchanged" (VERIFIED)

`per_group_noise_stddev` returns `nm·√(B_i Σ_j B_j)` (`noise_allocation.py:103-110`); with `S = C_g + C_h`:
`(C_g/σ_g)² + (C_h/σ_h)² = (C_g + C_h)/(nm² S) = 1/nm²` — printed `1.000000` for all six ρ. `gaussian_noise` takes this
branch for a `PerGroup` max_norm (`_gaussian.py:320-322`); `mf_gaussian_noise` computes `base_stddev` the same way and
multiplies by `row_l2(t)` (`_mf_gaussian_noise.py:166-167, 186-188`). The whitening argument (Dong–Roth–Su Thm 2.7 applied to
the whitened statistic; sup attained by an example saturating both bounds — A1 shows the load half is attainable) is correct;
the replace-one joint sensitivity `√((2C_g)²/σ_g² + (4λ)²/σ_h²)` is ≤ `2/nm` because `(4λ)² < 4C_h²` (ratio 4/7), so "the
doubling covers it" holds. Noise samples are drawn per leaf from one generator (`_gaussian.py:311-313`), so the two groups'
noises are independent as the diagonal-covariance argument needs. Upheld.

### 2.6 Places where the per-example loss could depend on other examples

- `f̃_t`: a function of `y_{<t}` only — `_augment_inputs` runs before the grad transform (`_dp_trainer.py:2302-2313`), the
  callback consumes the **noised** pytree (`:2198-2206`, `grads=noisy_grads`), and the loss for step t is fixed before
  `B_t`'s gradients exist. Adaptive-composition-covered. Upheld.
- `T_tot` / `num_items_in_batch`: **absent** from the DP path. `compute_per_example_loss` calls `fmodel(params, **inputs)` with
  the collator's per-example inputs (`_dp_trainer.py:2381`); the only occurrence of `num_items_in_batch` in
  `opaque-transformers/src` is a docstring forbidding it (`trl/_sft_trainer.py:570`); HF's `ForCausalLMLoss` falls back to
  per-example `mean` when it is `None` (`loss/loss_utils.py:39`). Upheld.
- `N̄`: not used (design §1.4 builds the example-mean statistic; a public-`N̄` variant is representable but not built). Upheld.
- **`all_valid_attention` (pre-existing, inherited).** `runtime/masking.py:195-216` picks the attention path (`None` mask →
  SDPA `is_causal` fast path vs a materialised mask) from `physical_mask.all()` over the **whole physical microbatch**
  (functorch-unwrapped). Under add/remove, adding one padded example to an all-valid microbatch changes the kernel for
  *every* example in that microbatch; each per-example gradient is still clipped to `C_g`, but `g_x` is no longer a function
  of `x` alone — the sum can move by more than one clipped vector (by the bf16 accumulation-order difference, 0.45–0.5 %
  rel-L2 per F8, or by `O(C_g)` per example if a route flips). This is the standard "DP up to floating-point" caveat, but here
  the dependence is *systematic* (a deterministic function of another example's padding), not random. The packed `T = 1024`
  presets never flip the switch (all masks all-ones). The design does not introduce it but (i) states "everything else in the
  forward is per-token/per-example, so vmap is exact up to floating point", (ii) relies on the switch for requirement (d), and
  (iii) omits it from the privacy statement. See R9.

### 2.7 Composition direction for the v2 `|` row (VERIFIED)

`poisson.rs:40` builds the subsampled PLD as `new_asymmetric(pmf_remove, pmf_add)`; `pld/mod.rs:461-490` `compose` convolves
remove-with-remove and add-with-add (and uses the symmetric `pmf_remove` for a symmetric factor's add side), so
`poisson(g(nm),q)*T | poisson(g(c·nm),q₂)*(T/m)` is direction-consistent and `epsilon_at` (metrics.rs:42-50) takes the worse
direction. The 3.0030 / 3.1284 / 3.0011 numbers stand. Upheld (v2 only).

### 2.8 Post-processing arithmetic (VERIFIED, `filters.py`)

- Sum-zero projection: per-coordinate variance ×63/64 (trivial).
- EMA .99 warm-up: signal gain `1−β^t` = 0.634 / 0.866 / 0.951 / 0.990 at t = 100 / 200 / 300 / 460 while `φ_t` reaches 0.0660
  / 0.0702 / 0.0708 / 0.0709 — the uncorrected EMA underestimates the imbalance by 37 % at t = 100 and 13 % at t = 200. See R6.
- Per-layer → pooled: per-entry noise on the pooled `d̂` is **0.04149 either way** (direct pooled release vs per-layer release
  at the same ρ then mean over L; ratio 1.0000), while per-layer entries carry ×5.29. See R7.
- JS+ at `s = 0.0235·k/E` (ρ = 0.02, DP-SGD, EMA .99), 2000 draws per δ: zeroed 52.3 % / 19.1 % / 0 % at δ = 0 / 0.01 / ≥ 0.023;
  median relative error of `d̃⁺` 0.97 / 0.73 / 0.43 / **0.231** / **0.078** at δ = 0.01 / 0.023 / 0.05 / 0.1 / 0.3 — the design's
  "24 % / 8 % at δ = 0.1 / 0.3" reproduces. But at δ = 0 nearly half the steps pass a shrunk noise vector. See R5.
- Monitor: τ = 0.5 against smoothed per-entry noise 2.35 % / 0.83 % of `k/E` is 21.3σ / 60σ. Upheld.

---

## 3. Refutations (each with why and fix)

**R1 — T17 / hygiene table: the logged `grad_norm` and `clipped_grad_norm` are functions of un-noised `h(x)`.** *(minor)*
Why: `clipped_fun` builds `aux["norms"]` from `norm.norm` — the norm over **all** leaves, probe included
(`_pytree.py:471-472` `orig_norm = sqrt(_sq_norm(leaves…))`; `_clipped_fun.py:673-675`), and the trainer logs
`aux.grad_norms.mean()` and `aux.clipped_grad_norms.mean()` (`_dp_trainer.py:2251, 2257`). VERIFIED numerically: total norm
0.30054 for the sup example vs 0.30001 for a random one (`adversarial.py`, "aux.grad_norms(total, includes probe)"). The
design's §8 table excludes only `group_norms["router_load_probe"]`, and T17 asserts "no logged key is a function of un-noised
h/f(B)" — false as written. Magnitude ≤ `C_h²/(2‖g‖)` ≈ 1.8e-4 absolute (a pre-existing un-noised channel, F11, made slightly
more informative). Fix: when the feature is on, log `grad_norm` from the non-probe groups (`group_norms["fallback"].mean()`
under the two-group `PerGroup`), or have `clipped_fun` compute `norms`/`clipped_norms` over an excluded-group set; restate T17
accordingly and add the row to §8.

**R2 — the "structural, never active" bound is not invariant to the recorder capture count.** *(minor)*
Why: `h` is normalised by the configured `L` (§1.1); any duplicate capture (checkpoint recompute if T5's PLAUSIBLE claim fails,
a second `MellumTopKRouter` call per layer, a future HF change) yields `Σ_e h = 2k`, `‖λd‖` up to `2.035·C_h`, and the probe
clip **fires** — VERIFIED (A8). Privacy holds; the release is biased and the "unbiased / clip_rate 0" claims fail silently.
Fix: normalise by `len(router_logits)·T_x` (the bound then holds for *any* capture count, verified: back to 0.018000,
unclipped) and additionally assert `len(router_logits) == config.num_hidden_layers` so a double capture is loud.

**R3 — the `(1+1e-6)` guard is below the engine's round-off shrink on MPS.** *(minor)*
Why: `_guard_scale` applies `ratio·(1 − 2(u_store + norm_roundoff))` before `min(1,·)` (`_pytree.py:111-131, 135-148`);
`norm_roundoff` scales with the leaf count and the widest leaf (`:214-224`) and with the fp32 sum-of-squares accumulator on MPS
(`_sq_accum_dtype`). For the preset partition: shrink 1.788e-7 (CPU/CUDA — guard OK, A1 scale exactly 1) vs **1.309e-4 (MPS)**
> 1e-6 — VERIFIED (A10). Fix: set `C_h = λΔ_h·(1 + 4·(u + roundoff))` computed from `_norm_roundoff` at setup (or a flat
`1e-3`, which changes ρ by 0.1 %); document that the guard is device-dependent.

**R4 — the mask contract is under-specified; the bound needs non-negative weights, HF semantics need 0/1.** *(minor)*
Why: the design says "attention mask" and multiplies (§1.1, §9.1); HF does the same after `to(float32)`
(`modeling_mellum.py:581`). Any non-negative weighting keeps `0 ≤ h ≤ 1, Σh = k` (A5), a same-sign additive mask keeps the
bound but scores the **padding** tokens (A6), and a mixed-sign mask breaks it: `h ∈ [−0.75, 1.5]`, `‖d‖ = 3.182 > 2.646`
(`edge3.py`). Fix: `m = (attention_mask != 0)` (2-D, boolean) inside `router_load_and_probs`, assert `attention_mask.ndim
== 2`, and add a non-binary-mask case to T2. (For the DP trainer's default 0/1 collator masks nothing changes.)

**R5 — "the shrinkage guarantees the design degrades to zero, never to a random regulariser" overclaims.** *(minor)*
Why: JS+ zeroes only when `‖d̃‖² < (E−1)s²`; for pure noise `‖d̃‖²/s² ~ χ²₆₃` sits above that threshold ≈ 48 % of the time
(VERIFIED 52.3 % zeroed at δ = 0), and those steps inject a shrunk noise direction (median factor small, but not 0). The
attenuation is ~10×, not ∞. Also a dof nit: after the sum-zero projection the noise has total variance `63·σ²` with
`s² = (63/64)σ²`, so the unbiased threshold is `E·s²`, not `(E−1)·s²` (1.6 % low). Fix: a dead zone `‖d̃‖² < c·E·s²` with
`c = 1.5` (χ²₆₃ 2.8σ → false-pass 0.3 %) or `c = 2` (5.6σ → 0 %), then JS+ above it; or reword to "attenuated by ≈ 10×".
Utility/wording only.

**R6 — the EMA is used without bias correction; `f̃_t` is biased low for ≈ 300 steps.** *(minor)*
Why: `d̃_t = β d̃_{t−1} + (1−β) d̂_t` from `d̃_0 = 0` has mean `(1−β^t)·d_true`: 0.634 at t = 100, 0.866 at t = 200
(VERIFIED). `s_t` uses the exact variance recursion so the shrinkage is self-consistent, but the surrogate coefficient is
37 % / 13 % too small at 100 / 200 steps, and acceptance 10.3(a) ("≤ 10 % of k/E after 200 steps") passes only because the
absolute imbalance is small. Fix: Adam-style correction `d̃_t/(1−β^t)` with `s_t/(1−β^t)` (one line each; under MF apply the
same factor to the filtered stream); or start the EMA from the first release.

**R7 — the per-layer row of the cost table is priced wrongly for the design's own consumer.** *(minor)*
Why: the design prices per-layer at ×5.29 relative noise and says "never free / usable under MF only at ρ ≥ 0.2" (§1.3(c),
§2.6, §7). VERIFIED: a per-layer release at the **same ρ** (`λ_L = ρC_g/√(7L)`) pooled over L gives the pooled `d̂` with
**identical** per-entry noise (0.04149 vs 0.04149, ratio 1.0000) — because the √L smaller λ is exactly compensated by the
√L averaging — plus per-layer entries at ×5.29 for free. Per-layer therefore *dominates* pooled at equal privacy and equal
gradient-noise cost for the pooled surrogate; the ×5.29 is the price of using per-layer `f̃^l` in a per-layer (Megatron-style)
surrogate, not of the release. Fix: make the `(L, E)` probe the carrier, derive `f̃` by pooling, keep the pooled surrogate as
default, and restate the ×5.29 as the accuracy of per-layer *entries*. Cost: 1792 floats of probe/optimizer/filter state.
(The candidate-(c) rejection in §1.3 should be re-argued on faithfulness grounds only, not cost.)

**R8 — the calibration-data wording contradicts the design's own hygiene row.** *(minor)*
Why: §4.4–4.5 say the `C_g` calibration pass runs "on a public proxy (KStack is public; `JetBrains/KStack` is the preset's
own dataset)". If the protected training set *is* KStack, a non-DP pass over 256 of its rows is a query on the protected data
— the protocol's "hyperparameter searches … included in the privacy statement" — which §8's own row ("a private query if run
on training data") acknowledges. Fix: require the calibration split to be **disjoint** from the protected set (a held-out
KStack shard never fed to the DP run), or route it through `adaptive_clipped_grad` + `adaclip` accounting; state which in §8.

**R9 — pre-existing cross-example coupling via `all_valid_attention`, inherited and unstated.** *(minor, pre-existing)*
Why: see §2.6. `runtime/masking.py:195-216` selects the attention kernel from the whole physical microbatch's mask; under
add/remove one padded example changes every microbatch-mate's kernel, so `g_x` depends on other examples through bf16
accumulation order (and, if a route flips, by `O(C_g)`). The clip still bounds each contribution, but the sum's sensitivity
is `C_g` only up to this systematic floating-point term. The preset (packed 1024, all-ones masks) is unaffected. The design
inherits it while claiming per-example exactness "up to floating point". Fix: under DP training, derive the path from a
**public** property — e.g. `all_valid := args.packed_sequences` (collator guarantee) or "always materialise the mask when the
collator can pad" — and add one sentence to the privacy statement naming floating-point kernel selection as the residual
non-per-example effect. Not a defect of this design; it belongs in its privacy statement.

---

## 4. Upheld (tried and could not refute)

1. Add/remove sensitivity of the load release: `sup_x ‖λ d(x)‖ = λ√(k(1−k/E)) = C_h/(1+1e-6)`; attained (A1, A2) and never
   exceeded by any admissible input; neighbouring-pair difference on the probe `= 0.99999906·C_h/B̄` (A3).
2. Replace-one tight bound `λ√(2k) = 4λ` (A4, attained), `< 2C_h`; the generic doubling covers the joint mechanism.
3. Centred vs uncentred: `Δ_h = 2.6458` vs `2.8284`; per-layer `√(7L) = 14.0`.
4. Mahalanobis identity with the real allocator at ρ ∈ {.5,.2,.1,.05,.02,.01}: `1.000000`; gradient inflation `√(1+ρ)`;
   probe multiplier `√(1+1/ρ)`; independent per-leaf noise draws.
5. "Accountant unchanged": `gaussian(nm)` / `mf_gaussian(nm, strategy)` with the two-group `PerGroup`; the naive
   `σ·C_g, σ·C_h` allocation is never used (`_gaussian.py:320-322`, `_mf_gaussian_noise.py:166-167`).
6. Cost table §2.6: every number reproduces (baseline ε 3.0004; ε-if-nm-held 3.234 at ρ = 0.02; r₁ 33.2 %; 2.35 % / 0.83 %;
   independent-draw 3.0030 / 3.1284 / 3.0011); MF factors 1.4309 / 0.0824 / 0.0249 / 0.0198.
7. Divisor: public `B̄ = train_batch_size` (cluster-wide), bound and noise scaled together; DDP shards with rank-folded keys;
   b-min-sep warm start gives `E|B_t| = B̄` from step 0.
8. No `T_tot`, `num_items_in_batch`, or `N̄` in the per-example loss path.
9. `f̃_t` is post-processing of `y_{<t}` (ordering in `training_step` verified); `monitor_then_surrogate`, clamp, projection,
   EMA, JS, `D_t` are all functions of noised outputs and public constants.
10. Empty batch and fully-masked row: `_empty_batch_response` zeros every leaf; NaN/Inf per-example leaves are zeroed by
    `clip_pytree` before the norm (`_pytree.py:525-531`) — no NaN propagation, no membership signal.
11. Exact ties: `torch.topk` returns distinct indices; `h_e ≤ 1` holds.
12. The v2 `|` accounting composes add/remove direction-wise (`pld/mod.rs:461-490`).
13. Poisson realised-batch scale multiplies the signal only; sum-zero projection removes the need for renormalisation.
14. The surrogate's dependence on `f̃` cannot enlarge the gradient bound (clipped); the aux term's effect on `‖g_x‖` is
    `α·O(1)` (phase-1 E3, not re-run).
15. DP-DPO pooling `L·(T_c + T_r)` keeps `h` a masked mean → same bound; reference forward contributes nothing (design).
16. Monitor threshold τ = 0.5 is 21σ / 60σ of the smoothed noise.

Attempted and abandoned as non-refutations: mixed-dtype probe leaf (fp32 by design, sum in fp32 — `_MicrobatchAccumulator`
promotes to the output dtype, `_clipped_fun.py:139-148`); MF latch equality with the `/B̄` bounds (constant); the JS/EMA
state as a side channel (public constants only); second-moment streams (v1 raises).

---

## 5. Notes and status

- **nm_MF — still uncomputed.** `nm_mf_small.py` (b-min-sep + band-MF(64, .95) vs Poisson-DP-SGD at n ∈ {256, 1024},
  nm ∈ {0.5622, 1, 2}, 240 s wall-clock bound) was killed at the bound (`nm_mf_small.out`: `exit 124`) before printing even
  the n = 256, nm = 0.5622 line — the `b_min_sep(mf_gaussian(·))` accountant does not finish on 4 CPU cores at a horizon
  60× smaller than the preset's, consistent with the synthesizer's 570 s failure at n = 15625 (design §6.4). VERIFIED that
  it did not run; the value is unknown. The design's †-caveat ("every absolute band-MF load-error number scales by
  `nm_MF/0.5622`, PLAUSIBLY several × larger") therefore stands exactly as stated; nothing in this lens depends on it, and the
  design's plan to record `nm_MF` from the trainer's own `_calibrate_noise` on the GPU run (§10.2 last row) is the right way
  to close it.
- The design's `f` from executed routes (fp32 softmax → `topk` on the same tensor) is the definition under which
  `Σ_e h_e = k` is exact; HF's bf16-softmax recomputation (critic R9) would still satisfy the bound (any top-k does) — only
  faithfulness to HF's *number* differs, ≤ 1e-3·k/E.
- Severity summary: 0 blocking, 0 major, 9 minor. Verdict **sound-with-fixes**.
