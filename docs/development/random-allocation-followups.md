# What the random-allocation machinery unlocks elsewhere in Opaque

**Status:** survey / research note. Nothing here is implemented. It follows on from
`random-allocation-accounting.md`, which covers the work that shipped.

**Question asked:** now that the FS26 random-allocation transform, `disc_dist`
(stochastic domination) and `GeomPmf` (geometric-grid exp-PLD convolution) exist in the
tree, can this line of work help the rest of the library — DP-FTRL in particular?

**Method.** A multi-agent survey: five mapping agents ground-truthed the accounting
core, the DP-FTRL strategies, the new primitives, the unused paper results and the
non-accounting surface; five proposal lenses generated 39 distinct candidates; triage
shortlisted 10; each shortlisted candidate got one adversarial verification pass
(precondition / soundness / value), and a completeness critic swept for gaps. Verdicts
below marked **PROVEN** were checked against code or measured; **PLAUSIBLE** means
argued but not measured.

**Independently re-verified before publishing.** The two live defects in §2.1 and §2.4
are the load-bearing claims here, so they were reproduced by hand rather than taken on
the agents' word:

| Check | Result |
|---|---|
| `C = α·I`, normalized Gram | `G[0,0] = E = 2.0` for α ∈ {0.5, 1, 2} — invariant, as claimed. ε identical at 4.24584 for all three, while the true mechanism gives 1.21093 / 4.24584 / 12.75634. |
| Real BLT, b=16, E=4 | Gram diagonal: shipped (normalized) 5.01128 vs raw 11.34671. ε at δ=1e-5: shipped **3.96993** vs correct **6.31958** — the shipped path **under-reports by 1.59×** (σ=2); 1.56× at σ=5. |
| MC ε vs δ, `G = E·I`, b=16 E=4 σ=1 | MC saturates at **9.064** from δ=1e-8 through 1e-14, while the exact identity path climbs 9.98 → 12.21 → 14.12. A hard cap, not a convergence rate. At production δ=1e-8 the MC path under-reports by 9.2%. |

Both reproduce. §2.1 is a soundness bug on a path users can reach today; §2.4 means the
surviving Monte Carlo accountant is not an upper bound at production δ, on a path
`calibrate()` documents as privacy-safe.

---

All paths below are relative to `/home/user/opaque` (branch `claude/opaque-fs26-followups-analysis`). Verdicts marked **PROVEN** were verified by reading code or by measurement on this box; **PLAUSIBLE** means argued but not measured.

---

# Can the random-allocation line of work help the rest of Opaque?

## 1. Answer

Yes for the accounting core, yes for DP-SGD, and **less than one would hope for DP-FTRL** — but the survey found more value in DP-FTRL than the transform itself delivers.

The FS26 allocation transform is structurally locked to `G = c·I_b`, i.e. `IdentityStrategy` alone, and that migration already shipped (`packages/opaque-dpftrl/src/opaque/api/accounting/dpftrl/amplification/_balls_in_bins.py:169-179`). What generalises to the correlated strategies is not the transform but a *wrapper* around it: Loewner domination `λ_min(G)·I ⪯ G ⪯ λ_max(G)·I` turns the shipped primitive into a deterministic two-sided bracket for BandMF/BSR/BiSR/λ-CGD/BLT. That bracket is real and is the first non-trivial deterministic bound those strategies have ever had — but it is **regime-gated**: at σ=10 it cuts ε by 65-77% versus no amplification, at σ=3 it is a wash for wide bands, and at σ=1 it loses to the trivial unamplified bound for every strategy. It must ship as `min(bracket, unamplified)`, not as a replacement.

The larger DP-FTRL prize — a deterministic accountant for the dense-Gram strategies (BLT, BiSR at p≥8, λ-CGD at λ≥0.9) — is reachable only through SK26 conditional composition, which is weeks of work, entirely un-prototyped, and whose own paper reports the bound *degrading* in the multi-epoch regime Opaque runs.

The highest-value findings in this survey are not the new machinery at all. They are two live soundness defects the machinery *exposed*: BLT's balls-in-bins ε is under-reported by 1.6-1.9× (**PROVEN**), and the surviving Monte Carlo accountant returns an ε that is hard-capped below any valid bound at production δ=1e-8 (**PROVEN**) on a path `calibrate()` documents as privacy-safe.

---

## 2. What survived

Ranked by value/effort. Every entry passed an adversarial verification pass; where the reviewer materially corrected the proposal, I state the correction rather than the original claim.

### Tier 1 — do these regardless

#### 2.1 `blt-gram-column-normalization` — a live under-report of ε on a shipped public path

**What.** `BltStrategy` computes its balls-in-bins Gram *column-normalized* while deploying the *unnormalized* `C`.

**Mechanism (PROVEN).** `packages/opaque-dpftrl/src/opaque/api/dpftrl/noise/_blt.py:116` passes `True` for `normalized` to `_native.toeplitz_gram_matrix`, and `_balls_in_bins.py:198` repeats it on the pinned K-prefix path — but `BltStrategy.streaming_matrix` (`_blt.py:168-174`) and `BltStrategy.sensitivity` (`_blt.py:176-192`) are both raw. `bnb_mc_pld` applies no compensating rescale (`packages/opaque-accounting/src/amplification/balls_in_bins/monte_carlo.rs:170-176`: raw `G`, raw σ, `diag_terms = -G_kk/(2σ²)`), and shrinking `G` by `s²` is equivalent to inflating σ by `1/s`. Contrast `_bsr.py:117-119`, which passes `False`.

**Decisive test (PROVEN).** With `C = α·I`, the normalized Gram is `E·I` independent of α, so `bnb_mc_pld` returns ε = 6.73038 for α ∈ {0.5, 1, 2} at fixed σ=1, while the true mechanism has ε = 1.26610 / 6.53782 / 20.92133. Measured under-report on real BLT: **1.76×/1.64×/1.63×** at n=128, b=32, E=4, σ ∈ {2,5,10}; **1.94×/1.87×/1.83×** at n=512, b=64, E=8.

**Precondition.** Holds. The fix direction is forced — `normalized=False` — because normalizing the deployment would change the mechanism.

**Cost.** Two boolean literals plus a cross-strategy invariant test. The reviewer found **no pinned golden ε values for BLT BnB** (`test_per_step_invariants.py:301` checks only monotonicity in K and boundedness), so the "golden-value churn" in the original estimate does not exist. Under 2 hours.

**One correction.** The proposed invariant `gram[0,0] == sensitivity()²` is stronger than provable — `sensitivity()` maximises over all min-sep patterns while the BnB Gram diagonal covers only the strictly periodic one. Assert `max_i gram[i,i] <= sensitivity()²` with an approximate-equality check for the current strategy set.

**Blocks:** every Gram-based candidate below reads the same Gram.

#### 2.2 `geom-conv-index-table` — 8.6-11.7× on the transform's dominant cost

**What.** Replace the `log_add` in `GeomPmf::conv`'s O(n²) core with a precomputed integer index table.

**Mechanism (PROVEN, derived and reproduced).** `packages/opaque-accounting/src/pld/pmf/geom.rs:185` computes `pos = (log_add(la, b.log_value(j)) - log_v_min)/log_ratio` — one `exp` + one `ln_1p` per pair. Both operands are forced to share `log_ratio` (`geom.rs:113-118`) and `log_value(i) = log_v_min + i·log_ratio` (`geom.rs:55`). With `d = i−j`, `pos(i,j) = j + C(d)` where `C(d) = (B − M + softplus((A−B)+dR))/R`; `j` integer gives `ceil(j+C) = j+ceil(C)`. So `idx(i,j) = j + K[i−j]` for a table of `2n−1` integers. `dC/dd = sigmoid(·) ∈ (0,1)` proves `K` non-decreasing with steps in {0,1}, so the scatter is sequential, not random.

**Measured in the real crate (PROVEN):** 8.6-11.7× end-to-end on `random_allocation_gaussian_pld`, all 351 Rust tests passing unmodified, ε drift ≤5e-5. Against 60-dps mpmath truth over 603,513 pairs the table commits **0** rounding-direction violations; the *current* expression commits 2,730.

**Named consumer (PROVEN).** `calibration.py:84` bisects σ at tolerance 1e-6 and the `lru_cache`s at `_random_allocation.py:77` / `_balls_in_bins.py:112` miss on every probe because σ is in the key. A ~30-probe calibration of a `RandomAllocation` or identity `BallsInBins` process goes from ~3.5 minutes to ~21 seconds. That is the gap phase 2 opened when it traded a 0.04 s MC point estimate for a 6.8-13 s bound.

**Preconditions (corrected).** The proposal's rounding argument is stated at the wrong layer and its empirical claim reverses by operand geometry:
- In the `A==B` self-conv geometry (`geom.rs:267`) the current code rounds one bin too far at `i==j`; in the `A≠B` geometry (`geom.rs:262`, `random_allocation.rs:100`) it is the *table* that rounds high on 6,618 of 8,194 diagonal entries. `d=0` must be special-cased on the exact identity `C(0)=0`, not handled by a widened guard.
- The directional guard must bound the whole `(B−M+softplus)/R` expression (measured error 2.3-3.0e-12 index units, dominated by the f64 error of the anchor `M` at `geom.rs:125`, not by `softplus`).
- Provable domination additionally requires rounding the *anchor* at `geom.rs:125` safely — a separate one-line fix neither version makes today.
- `geom.rs:197`'s `acc[(idx.max(0.0)) as usize]` clamps negative indices to 0 in *both* directions, which rounds **up** under `Rounding::Down`. Dead today; the table makes it reachable (`K_down[0] = −1` observed). Must route to `zero_mass` like `geom.rs:145`.

None of these are live privacy defects — both residuals sit ~13 orders below the measured sandwich width — but they are the difference between "faster" and "provably safe".

**Cost.** ~80 lines in one file, no API change. The merged `raise-MAX_CONV_GRID` claim rides on it: `MAX_CONV_GRID = 8192` (`packages/opaque-accounting/src/amplification/random_allocation.rs:44`) is the binding accuracy constraint and `discretization` is inert for σ ≲ 25 (**PROVEN**: refining 1e-3 → 1e-5 moves the σ=2 sandwich only 3.19% → 2.90%).

### Tier 2 — the actual DP-FTRL answer

#### 2.3 `bnb-psd-gram-bracket` — deterministic two-sided bound for correlated MF via Loewner domination

**Mechanism (PROVEN, math verified).** The Choquette-Choo Lemma 3.2 pair is `P_G = (1/b)Σ N(m_i, σ²I)`, `Q = N(0,σ²I)`; because `Q` is isotropic the pair depends on the means only through `G`. If `G' = G + NᵀN`, set `m'_i = [m_i; n_i]`; coordinate projection maps `(P_{G'},Q')` to `(P_G,Q)`, so data processing gives `H_γ ≤` in **both** argument orders — a genuine dominating pair, hence composable. A pair with Gram `c·I_b` is exactly the k=1 allocation pair at `t=b`, `σ_eff = σ/√c`, so the bracket comes straight from `random_allocation.rs:164`.

The refinement `G' = c·I + s·J` is **exact, not approximate**: the second-block mean `v` is the same for every `i`, so `P'` is a product measure and its PLD is `random_allocation_gaussian_pld(σ/√c, b, 1, ·).compose(gaussian_pld(σ/√s))`.

**Measured (PROVEN, b=32, E=4, δ=1e-8, σ=10, disc=1e-3):** BandMF(2) [0.2246, 0.2461] vs MC 0.2330; BSR(2) [0.2497, 0.2643] vs MC 0.2470; λ-CGD(0.9) [0.7032, 0.7682] vs MC 0.6765; BLT(4) [0.8102, 0.8792] vs MC 0.7870. Unamplified baseline for BandMF(2): 1.0212. For `G = E·I` the bracket collapses exactly onto the shipped identity path.

**Preconditions (three real, two proposal claims refuted).**
1. BLT is blocked until §2.1 lands — every BLT number above is around the wrong mechanism.
2. The lower leg must use `random_allocation_gaussian_pld(nm, 1, 1, false, cfg)` (the t==1 early return at `random_allocation.rs:90,:115` is the down-rounded Gaussian), **not** `gaussian_pld` — connect-the-dots is a valid *upper* bound on δ and is uncertified inside a lower-bound leg. One-line substitution.
3. **REFUTED:** the proposal's headline risk, that `PrivacyLossDistribution::compose` re-inflates the lower leg, does not exist — `compose` passes `tail_mass_truncation = 0.0` (`packages/opaque-accounting/src/pld/mod.rs:389`). The tail-budget artifact measures 2.7e-9 in ε and only via `self_compose`. The claimed "real code change in the composition path" is not needed.
4. Regime gate is mandatory (**PROVEN**): at σ=1 it loses for every strategy (BandMF(2) 15.65 vs 12.75 unamplified); at σ=3 it loses for BandMF(4); even at σ=10 the naive λ_max leg loses for λ-CGD (1.5098 vs 1.0492) — the `sJ` refinement is load-bearing whenever `λ_max/G[0,0]` is large (16.5 for λ-CGD, 18.3-51 for BLT).
5. Claim wording: this brackets the **dominating pair's** ε, not the mechanism's. Upper leg is a mechanism bound; lower leg may be published only as an MC diagnostic.

**What it buys.** The first deterministic, thread-independent, composable bound for BandMF/BSR/BiSR/λ-CGD balls-in-bins, currently served only by an uncorrected MC point estimate (`monte_carlo.rs:134`, `samples_to_pmf` at `:270-345`). 5.6-13.6% over the MC estimate at high σ, and 65-77% below unamplified for banded strategies.

**Cost.** ~40 lines of Python around `numpy.linalg.eigvalsh` plus existing `_native` calls. But: the advertised 5.8-12% bracket width is **search-effort dependent** and mostly the transform's own discretisation slack — the identity case with zero Loewner gap is already 4.5% wide at disc=1e-3; an independent 6-point `s`-grid gave 14.8% for BandMF(2) where the proposal reports 5.8%. Each `s` probe is one full transform call (12-32 s measured at b=32 *before* §2.2). At production `b` ≈ 1000-2000 both the cost and the width are **unmeasured**.

**Dependency note.** §2.2 is what makes the `s`-search affordable.

#### 2.4 `bnb-mc-oracle-diagonal-gram` — the MC accountant's ε is a hard cap, not a noisy estimate

**Mechanism (PROVEN).** `bnb_mc_pld` takes an arbitrary flattened Gram (`monte_carlo.rs:134-150`), so feeding `E·I_b` runs it on exactly the mechanism the deterministic bracket certifies. I verified the reduction analytically in both directions against the sampler (`monte_carlo.rs:69-83`, `:106-119`).

**The finding is stronger than the proposal states.** `samples_to_pmf` builds its grid from `min_sample`/`max_sample` (`monte_carlo.rs:299-301`), which makes the overflow branch at `:323` unreachable, so `infinity_mass` is identically 0 and `epsilon_at` returns the empirical maximum for every δ below resolution. Measured at (b=16, E=4, σ=1): MC ε is **flat from δ=1e-8 to δ=1e-14** while the bracket climbs 41%. This is a **cap**, not a convergence rate — no sample budget fixes it. The design note's framing ("cost scales with 1/δ") understates it.

Measured shortfall vs the bracket's lower endpoint at δ=1e-8: −9.1% to −38.1% at n=1e5, −9.2% to −21.4% at n=1e6. At δ=1e-3 the same estimator is fine (+0.3% to +2.8%).

**Thread-count defect, same runs (PROVEN):** identical `seed=42`, `RAYON_NUM_THREADS` 4 vs 1 gives 9.063 vs 8.102 at (16,4,1.0,n=1e5) — 11.9% apart. Cause: `monte_carlo.rs:188` `rayon::current_num_threads()`, per-thread seeds at `:201`/`:231`.

**Why it matters.** `calibration.py:341-344` (`is_safe`: `achieved <= budget.value`) treats `epsilon_at` as a bound, and `calibration.py:106-110` documents the result as "privacy-safe". `b_min_sep/mc.rs:315-316,:418-419` import the same `samples_to_pmf`, so `BMinSep` inherits the identical cap.

**Corrections.** "Nothing in the tree can detect it" is false — `test_identity_mf.py:121-161` already feeds `E·I_b` to `bnb_mc_pld` and compares against the deterministic path, pinned at δ=1e-5 (`:16`) with a 10% tolerance where the gap is −1.04%. At δ=1e-8 the same config measures −19.30%. So the deliverable is a **δ retarget on an existing test** (as `xfail(strict=True)`, since a red test cannot merge) plus docstring warnings on `BallsInBins.pld` and the b-min-sep entry points — roughly 10 lines, not 40. Also: no diagonal Gram reaches `bnb_mc_pld` from Python any more (`_identity.py:33-40` raises), so the measurement is synthetic w.r.t. shipping code; it characterises the estimator, not a live identity-path bug.

#### 2.5 `bandmf-bnb-gram` — split it in two; only half is worth doing

**Half A — the sensitivity defect (HIGH value, independent of everything else).** `BandMfStrategy.sensitivity` is `(self, *, n_steps: int, **_)` returning `float(coefs.norm())` (`packages/opaque-dpftrl/src/opaque/api/dpftrl/noise/_band_mf.py:137-139`), and `_toeplitz.py:570` normalises to unit norm, so it is identically 1.0 — silently dropping `min_sep` and `max_participations`. `mechanisms/_mf_gaussian.py:77-86` reads exactly this method. Measured under-report on the bare `MfGaussian.pld` path: **90.7×** in ε (4.3772 vs 397.2353) at the participation context the shipped integration tests use — not the 2× the proposal claims.

But this is a **contract decision, not a one-line fix**: `_poisson.py:160` and `_b_min_sep/__init__.py:146` both need the k=1 value, four shipped tests assert `sensitivity == 1.0`, and `types.py:39` documents the drop as intended. Decide the contract (probably: keep `sensitivity()` single-participation, make `MfGaussian.pld` reject or correctly price a multi-participation BandMF), then land the invariant test.

**Half B — BnB Gram plumbing (LOW/medium value).** Mechanically verified: `_native.toeplitz_gram_matrix(coefs, n, min_sep, max_participations, False)` reproduces the exact Lemma 3.2 Gram to machine zero (max abs diff 0.0) across bands ∈ {2..64}. But three things deflate it:
- The claimed route `bnb-psd-gram-bracket` does not exist yet; landable today, BandMF becomes the **sixth MC consumer**, which is the opposite of the stated goal.
- "Smallest λ_max/diag of any strategy" is **false** — BSR is tighter at matched bandwidth (1.797 vs 1.966 at bw=2), and BandMF's ratio grows monotonically to 17.84 at bands=32.
- The realistic BandMF-under-BnB config is `bands = min_sep = num_bins`, where the Gram is fully dense and the banding buys nothing.
- `toeplitz_gram_matrix` uses **signed** column inner products (`bisr.rs:583-593`) while the Lemma 3.2 mean is the entrywise-`|C|` sum. BandMF coefficients are never validated non-negative (`_toeplitz.py:565-569` runs L-BFGS-B unbounded, unlike BSR at `_bsr.py:98-102`). A negative coefficient under-reports `G` in the unsafe direction. Needs an explicit assertion.

Also: because BandMF retunes per `n_steps` (`_band_mf.py:84-98`), it must use the pinned-coefficient BLT branch (`_balls_in_bins.py:186-200`), which calls `_native.toeplitz_gram_matrix` directly — so the proposed `BandMfStrategy.gram_matrix` would end up with no BnB caller at all.

#### 2.6 `subsample-pld-transform` — the first amplification primitive not hardwired to the Gaussian

**Mechanism (PROVEN analytically, not just numerically).** FS26 Thm 3.3. Both branches were independently re-derived and collapse to the textbook subsampled-Gaussian formulas to <1e-14 relative at 60 decimal digits, and reproduce the structural fact `δ_add = 0` for `ε ≥ ln(1/(1−λ))`. The domination-transfer identity `D_g(P_λ‖Q) = λ·D_{(g−1+λ)/λ}(P‖Q)` holds exactly, so hockey-stick domination transfers through the transform.

**Two corrections that change the design.**
1. **The add branch does not need the dual and must not use the proposed flip.** Since `negD = −L_add`, we have `L_add_sub = −ln(1−λ+λe^{−L_add}) =: g(L_add)`, which is *increasing*. Applying `g` directly to the stored `pmf_add` needs no dual, no direction flip, and is tighter. The proposal's version, read naturally, is **invalid** — it needs `disc_dist(negD, Rounding::Down)` on the *input*, and the natural reading undershoots true `δ_add` by up to 2.4% (σ=2, λ=0.5, ε=0.4: 1.15350e-3 vs 1.18194e-3), with error growing without bound in α. Both precedents the proposal cites (`geom.rs:337-340`, `random_allocation.rs:121-124`) flip a *different* rounding.
2. **Value claim (b) — "legal input to the allocation transform" — is not delivered by the stated scope.** `alloc_remove` (`random_allocation.rs:80-88`) takes `&L: LossRealization` for both arguments and calls `disc_dist` itself; it cannot consume a `Pmf`. So the two downstream candidates it claims to unblock are not unblocked by this ticket as written.

**Where it genuinely wins (PROVEN, and large).** Subsampling an externally supplied `(ε,δ)` mechanism cannot be expressed today (`_poisson.py:20`, `_random_allocation.py:39` accept Gaussian | AdaClip | NonPrivate only; `_eps_delta.py:15` has no amplification path). The classic amplified pair puts all mass on one atom and composes linearly; the exact 3-atom subsampled PLD puts ~95% of its mass at `ln(1−λ) < 0` and composes with cancellation. Measured at ε₀=3.0, δ₀=1e-6, λ=1e-3, 8192 steps, δ=1e-5: **ε = 1.74 vs 154.88**, a ~90× reduction.

For the Gaussian it **ties** `poisson_gaussian_pld` (1e-10 relative). Do not sell it as a tightening there.

**Cost.** ~180 lines Rust + ~60 Python, plus the `map_monotone` substrate extracted from `geom.rs:275-356`. Note `into_pmf_log` hardcodes zero-atom → `negative_infinity_mass` (`geom.rs:321`) and cannot represent `h_λ(−∞) = ln(1−λ)`, a finite interior atom.

### Tier 3 — large, un-derisked, or hygiene

#### 2.7 `sk26-conditional-composition-per-step` — the only route for dense-Gram strategies

**Status:** applicable-conditional, preconditions hold-with-work, value medium, **weeks**, least-derisked item in the whole note.

Paper text verified: Lemma 4.1, Alg 4's mixture-Gaussian pairs, Lemma 4.3's reverse-hazard dominance and Thm 4.8's Normal variational bound all exist as cited, and the direction is correct (single additive `δ_E` on an upper bound on `H_γ`).

**Corrections that matter.**
- The real gap is not the mixture PLD. Alg 4 consumes mixture means `m ∈ ℝ₊^{b×N}`; Opaque computes only the `b×b` Gram (`gram_matrix.rs:4-6`, `monte_carlo.rs:13`, `_balls_in_bins.py:210-215`). A new mixture-means path plus an O(bN) working set (~122 MB at b=1953, E=4 if materialised) is required first.
- Composition needs only **hockey-stick** dominating pairs, so `disc_dist`/`Rounding::Up` is *not* a hard dependency — connect-the-dots suffices and is tighter. Effort estimate should drop accordingly, and `mixture-normal-realization` (§2.10) is not on the critical path.
- The exclusivity claim is **false for BiSR and λ-CGD**: measured `τ₃₂/diag` is 0.016 (BiSR p=8), 0.113 (BiSR p=16), 0.035 (λ-CGD 0.9) — SK26's Rényi τ-truncation route is viable for all three. Only BLT is genuinely dense (0.468).
- Quoted value numbers do not reproduce: BiSR p=16 mid-row/diag measures 0.066 not 0.42; BLT λ_max/diag measures 51.4 not 18.75.
- The N per-step pairs are all distinct, so `self_compose` does not apply — N−1 sequential `compose` calls (`pld/mod.rs:383`) with coarsening at each step; N ≈ 7800 at production. Conservative, but the accumulated slack is an unmeasured tightness risk.
- `δ_E` is per-direction and Opaque's PLD is asymmetric; folding a single `δ_E` under-counts.

**The honest value proposition** is not tightness — it is **δ-independent cost** against an MC accountant whose binding constraint is `1/δ`, plus determinism. SK26's own empirical finding is that the bound degrades for multi-epoch DP-SGD and improves for non-trivial matrix mechanisms.

**Recommendation:** prototype Alg 4 in Python against the b=1 Gaussian closed form and the identity Gram *before* committing Rust.

#### 2.8 `lower-bound-composition` — hygiene, value LOW (downgraded)

`random_allocation.rs:234-237` applies `with_tail_budgets(tail, tail)` regardless of `upper`. The per-direction budget rule is correct and the k=1 precondition is genuine. But **the headline value is already delivered by shipped code**: the 19.4% bracket at (σ=2, t=64, E=64) was reproduced by calling the existing `.so` with no changes. The fix itself moves ε by **2.1e-9 to 2.6e-7**, with a sign at E=64 that contradicts its own directional argument (FFT residue dominates). The "runtime signal for `discretization`" is refuted by its own measurement (1e-3 → 1e-5 moves the gap 3.19% → 2.90%). Real work remaining: reject `k>1` (currently accepted, `random_allocation.rs:180-185`), and suppress coarsening on the Dominated side at ~1 GB peak RSS per evaluation.

#### 2.9 `alloc-prefix-horizon` — mathematically sound, no caller, value LOW

The measure identity `P_K = (K/b)·P̄_K + (1−K/b)·Q_K` is exact and the three-way rounding assignment survives independent re-derivation. But: (a) the merged identity-BnB half is **invalid past the first epoch** — the K-prefix Gram is `diag(E+1,…,E,…)`, not `c·I_b`; (b) the claimed consumer does not exist — `DPTrainer` has no allocation sampling mode (`_training_arguments.py:132-134`) and its DP-SGD branch always builds `dpsgd_acc.poisson` (`_dp_trainer.py:3989`), which is already 1-step granular; (c) DP-SGD's `DpProcess` (`core/_base.py:69-123`) has no `_pld_at_horizon`/`atomic_unit`/`per_step` machinery — those are dpftrl-only, so this is a new API surface, not ~120 lines. Genuine residual value: a library user with a partial final epoch currently overcharges ~15% at E=1, ~3% at E=8.

#### 2.10 `mixture-normal-realization` — sound math, refuted consumers, value LOW

The privacy math holds and is in fact stronger than claimed (`g` is a log-sum-exp of affine functions, hence convex for *any* signs of μᵢ, so monotonicity is not load-bearing and the proposed constructor assert would wrongly hard-fail computable configurations). But all three named consumers fail:
- **Phase 7:** conditional composition is a hockey-stick statement, so CTD suffices and is *tighter* — `disc_dist`'s round-up inflation accumulates linearly across N-fold composition (~0.5 additive ε at N=1e4, disc=1e-4). `realization.rs:11-13` explicitly disclaims this use.
- **`parallel_poisson`:** the "un-directional bisection inverse" (`parallel_poisson.rs:225-277`) is **not a slack** — δ is stationary in `x_cutoff`, so a 1e-15 bisection error yields ~1e-30 δ error. (There *is* a real 5-line bug there: no failure path if 20 bracket doublings don't bracket, `:249-261`.)
- **`opaque-engine`:** `noise_allocation.py:33-50` is an anisotropic-Gaussian whitening argument over a *single* Gaussian; a mixture PLD is not the object.

A ~30-line generalisation of `parallel_poisson.rs:135,:168-170` (which hardcodes collinear integer means and binomial weights) delivers the doc's prescribed `mixture_gaussian_get_delta` through the existing `discretize_asymmetric_mechanism` — cheaper *and* tighter. **Do that instead if phase 6 is needed.**

### 2.11 Not shortlisted, but PROVEN in ground truth and cheaper than anything above

Flagging because it is directly responsive to the question and was never adversarially reviewed:

**The landed work is unreachable from `DPTrainer`.** `_training_arguments.py:132-134` (`_SAMPLING_MODES`) has no `random_allocation`; `:143` and `:157` pin `mf_identity` to `poisson`. So both the deterministic DP-SGD allocation path *and* the new deterministic identity-BnB path (7-27% tighter than the Rényi route) are reachable only by hand-constructing accountants from the raw library API. Three lines in the allow-list plus a `build_sampler`/`build_amplifier_factory` branch.

**`calibrate()` on MC-backed processes.** `calibration.py:106-110` promises a privacy-safe result; `:341-344` implements a bare `achieved <= budget.value`. For `BallsInBins`-on-correlated and `BMinSep` that is a point estimate whose value depends on core count. No test covers it (`test_calibration.py:159-344` is all deterministic). Related: `calibration.py:266-267` clears the b-min-sep transcript corpus in a `finally` block at the exact moment it becomes useful (**PROVEN**, up to 4 GiB of σ-independent work discarded).

---

## 3. What died, and why

### 3a. Structurally impossible — do not re-propose

| Proposal | Why it is closed |
|---|---|
| FS26 allocation transform as a drop-in for BLT/BSR/BiSR/λ-CGD/BandMF | Requires `G = c·I_b`. Only `IdentityStrategy` satisfies it. **PROVEN** by measurement across all six strategies. |
| Change of basis to make a correlated Gram diagonal | The pair is invariant under `O(d)` acting on `ℝ^d`, which leaves `G` fixed; the only index-side symmetry is a bin permutation, which cannot alter the off-diagonal multiset. |
| Block refinement `G' = c·I + s·B_g` (coarser g-way allocation) | `m'_i = [√c e_i; √s u_{g(i)}]` has **both** blocks indexed by `i`, so `P'` does not factorise and there is nothing to compose. (Contrast `c·I + s·J`, which does work — §2.3.) |
| SK26 τ-truncation for BLT | Measured `τ₃₂/diag = 0.468`; there is no band to truncate. |
| `neg-dual-on-the-trait` / a numeric `pld_dual` on a discretised `Pmf` | FS26 Claim B.2: stochastic domination is **not** preserved under the dual transform. The dual must come from the raw analytic loss. |
| `mc-confidence-envelope` | Its own precondition disqualifies it: at 1e5 samples the certified δ floor is ~3e-5 against production 1e-8, so a confidence-corrected accountant would have to refuse every shipped configuration. |
| `heterogeneous-allocation` | Requires diagonal `G`, i.e. Identity only — explicitly does not touch the correlated-MF gap. The useful residue is a one-line `__post_init__` validation. |

### 3b. Sound but not worth a slot

| Proposal | Reason |
|---|---|
| `mf-gaussian-loss-realization` | Self-admits it does not tighten ε (expects a *rise* of ~`discretization`); blocked on the `pmf_beta_asymmetric` reconciliation. A guarantee relabel. |
| `poisson-gaussian-realization` | Concedes the `Up` bound is looser than the shipped CTD number and the ~`N·α` bracket is unusable at production composition depth. |
| `epsilon-bracket-audit-join`, `audit-three-number-ladder` | Both entirely downstream of a composable lower bound (§2.8), which was itself downgraded to LOW. Nothing to verify until that exists. Correctly identify their own hard precondition (valid only where the pair is tight — k=1 allocation, identity BnB, per Chua Thm 3.1). |
| `calibrate-bracket-and-early-stop`, `bracket-linearity-regression`, `alpha-from-one-cheap-probe` | All downstream of the same bracket; the α-linearity is already measured at ×1.95-2.01, and the cost curve they optimise is changed by §2.2. |
| `dual-mass-gate` | 7.6× violation fully explained (ulp quantisation of a second difference amplified by `e^{|l_min|}`) and **unreachable today** — `pmf_delta`/`pmf_epsilon`/`pmf_beta` never touch the contaminated buckets. Ships as asserts plus a documented negative result. |
| `disc-dist-survival-tail` | Real defect (931 zeroed right-tail bins, 2.8e-15 dropped mass, `infinity_mass` computing to exactly 0.0) with ~7 orders of margin at δ=1e-8. 40-line accuracy fix, changes no shipped number. |
| `emit-pld-at-transform-resolution` | Admitted ε penalty (+0.13% to +0.56%); loosens the reported number to save composition time. |
| `progressive-alpha-calibration`, `chernoff-orders-from-distribution-scale` | Both non-regressing by construction (coarse α over-noises; a bad Chernoff order gives a weaker bound). Performance only. |
| `clean-probs-swallows-nan`, `disc-dist-unasserted-realization-preconditions`, `cyclic-cholesky-residual-blind-to-dropped-pattern` | Defensive hardening on unreachable branches, or a reported quantity that changes no output. Do them; don't staff them. |

---

## 4. Sequencing

```
STAGE 0 (independent, ship first)
  [A] blt-gram-column-normalization        ~2h    §2.1
  [B] trainer allow-list for random_allocation + mf_identity BnB   ~1d   §2.11
  [C] BandMF sensitivity contract decision + invariant test        ~2d   §2.5A

STAGE 1 (engine, unblocks everything below)
  [D] geom-conv-index-table + anchor rounding + zero_mass clamp    ~3d   §2.2
        └─ then raise MAX_CONV_GRID; re-measure the σ=2 sandwich

STAGE 2 (the DP-FTRL deliverable)
  [E] bnb-psd-gram-bracket, gated as min(bracket, unamplified)     ~2d   §2.3
        needs: [A] (BLT), [D] (s-search affordable)
  [F] bnb-mc-oracle: δ retarget on test_identity_mf.py + docstrings ~1d  §2.4
        needs: nothing; sharpens the case for [E]
  [G] BandMF BnB plumbing                                          ~1d   §2.5B
        needs: [C], [E] — pointless before [E]

STAGE 3 (new capability, DP-SGD / core)
  [H] map_monotone substrate + subsample_pld (add branch via g)    ~1w   §2.6

STAGE 4 (research, gate on a Python prototype)
  [I] parallel_poisson generalisation → mixture-Gaussian δ(ε)      ~2d   §2.10
  [J] SK26 Alg 4 Python prototype vs b=1 closed form               ~1w   §2.7
  [K] SK26 conditional composition in Rust                         weeks — only if [J] beats MC

DEPRIORITISED
  lower-bound-composition (§2.8), alloc-prefix-horizon (§2.9),
  mixture-normal-realization as specified (§2.10)
```

**Do [A] first.** It is two boolean literals, it is a live under-report of ε by 1.6-1.9× on a path users can reach today (`_balls_in_bins.py:284` accepts `BltStrategy`), it has no golden-value churn, and every Gram-based item downstream reads the same number. Nothing else on this list is both a correctness fix and a precondition for the main deliverable.

**Do [D] before [E].** The bracket's `s`-search costs one full transform call per probe; at 12-32 s each (b=32) the search is the dominant cost and at production `b` it is unmeasured. An 8.6-11.7× cut turns a research script into something a calibration loop can call.

---

## 5. Open questions

1. **Does the bracket survive production `b`?** Every measurement is at b=32, E=4. `cyclic_cholesky.rs`'s own docstring cites b=1953. Both the bracket *width* and the `s`-search *cost* at b ≈ 1000-2000 are unmeasured. **Evidence needed:** run §2.3 at b ∈ {256, 1024, 1953} for BSR(p=8) and λ-CGD(0.9) after [D] lands, and record the width against `λ_max/G[0,0]`.

2. **How much of the bracket width is Loewner slack vs discretisation slack?** The identity case (zero Loewner gap) is already 4.5% wide at disc=1e-3, and two independent `s`-searches disagreed 5.8% vs 14.8% for BandMF(2). **Evidence needed:** re-run the identity bracket at `MAX_CONV_GRID` ∈ {8192, 32768} post-[D]; the residual after that is the true Loewner gap.

3. **Is the σ regime gate stable across `E` and `n_steps`?** All gate measurements are at E=4, n=128. The crossover with the unamplified bound is what decides whether the bracket ships as a default or as an opt-in. **Evidence needed:** a σ × E × bandwidth sweep recording `sign(ε_bracket − ε_unamplified)`.

4. **Does SK26 conditional composition beat the (biased-low) MC estimate in Opaque's actual multi-epoch regime?** The paper reports the bound degrading for multi-epoch DP-SGD and improving for BSR/BiSR; Opaque runs multi-epoch matrix mechanisms, which is neither case cleanly. This cannot be settled from the paper. **Evidence needed:** [J] — Python Alg 4 at b=32, E ∈ {1,4,8}, compared against the §2.3 bracket (a valid bound) rather than against MC.

5. **Can a BandMF L-BFGS solution produce a negative coefficient?** `_toeplitz.py:565-569` runs unbounded; none was found across bands ∈ {2,…,64} and eight momentum values, but the search was not exhaustive and `bisr.rs:583-593` uses signed inner products where Lemma 3.2 requires `|C|`. **Evidence needed:** either a proof of positivity for the optimiser's fixed point, or a runtime assertion. A negative coefficient under-reports `G` in the **unsafe** direction.

6. **Is BLT's `C` non-negative?** `_blt.py:187-193` explicitly bypasses the `sensitivity.rs:405-410` sign check with `skip_checks=True`, and `output_scale` is sign-unconstrained throughout `_blt_math.py`. Structurally `C ≈ (1−x)^{−1/2}` has all-positive exact coefficients, but torch was unavailable in the verification environment so this was **not** confirmed numerically. **Evidence needed:** one `torch` run over the BLT coefficient grid before any constructor assertion is added — an unproven assert would hard-fail BLT at runtime.

7. **`pmf_beta_asymmetric` drops `negative_infinity_mass`** (`pld/metrics.rs:268`, `:287-298`) while the symmetric path (`:204-211`) keeps it, with the codebase's own comment calling the latter "conservative". Since every amplification transform returns an asymmetric PLD and `disc_dist(Down)` / `into_pmf_log` both *produce* −∞ mass, `beta_at()` on a random-allocation PLD already reads differently from `delta_at()`. **This is live, not hypothetical**, and it is unresolved. **Evidence needed:** decide whether `cal.beta_budget` / `cal.risk_budget` are currently non-conservative, and by how much, before any dual-direction transform ships.

---

# Appendix — completeness critique

An adversarial pass over the survey itself, looking for what was missed rather
than what was wrong. Recorded unedited; its gaps are not resolved above.

# COMPLETENESS CRITIQUE — gaps in the FS26 follow-ups survey

All paths relative to `/home/user/opaque`. Measurements below were run this session in `.venv/bin/python` against the built `target/release/libopaque_accounting.so`.

---

## A. THE SINGLE LARGEST GAP: the survey skipped its own design doc's Phase 4

**Gap A1 — the Rényi accountant for banded Grams produced ZERO candidates, and it is the cheapest route the doc has.**

`docs/development/random-allocation-accounting.md:884-891` is the phase table. The survey mined phases 3 (§2.6), 6 (§2.10) and 7 (§2.7). **Phase 4 — "Rényi DP accountant for banded Grams (`p = 1` first); add Gram support to BandMF", dependencies "—", state "proposal" — is absent from all 11 candidates and from the entire draft.** Confirmed: `grep -rn "renyi|Renyi|rdp|RDP" packages/opaque-accounting/src --include=*.rs` returns **nothing** — there is no RDP machinery in the crate at all.

This is not an untested idea. `docs/.../random-allocation-accounting.md:630-655` (§5.4) records a **completed, validated prototype**: brute force, cyclic-banded DP, add-direction closed form, RDP→(ε,δ); `b=1` reduces to `R_α = α‖m‖²/(2σ²)` to 10 dp; the DP reproduces brute force to <1e-8 on six `(b,p,α)` configurations; λ-CGD truncation stayed above exact in every configuration tested. §5.4 conclusion 2 puts the cost at `b·α·C(α+p−1,α)²` ≈ 1.5e7 ops at `b=1000, p=64, α=2` — "seconds, not eons".

And §5.2's strategy table (`:545-560`) assigns **all six strategies** a route (`p` is a free *effective* bandwidth decoupled from `C`'s true bandwidth). That is strictly broader coverage than §2.3's bracket (which the draft itself gates off at σ≤3) and orders of magnitude cheaper than §2.7's SK26 (weeks, un-prototyped).

Worse, the survey's own §2.7 verdict **rediscovered Phase 4's viability and filed it as a correction instead of a candidate**: "the exclusivity claim is false for BiSR and λ-CGD: measured τ₃₂/diag is 0.016 (BiSR p=8), 0.113 (BiSR p=16), 0.035 (λ-CGD 0.9) — SK26's Rényi τ-truncation route is viable for all three." Those measurements *directly contradict* the doc's own pessimistic §5.4 table (which was measured at `b=8, E=2` where τ decays slowly). Nobody reconciled the two.

**What closes it:** one candidate — "Rényi banded-Gram accountant (doc phase 4)" — scoped as: port the §5.4 prototype's cyclic-banded DP, validate against the `b=1` closed form and the `E·I_b` identity Gram, then run the sweep doc Open Question 5 (`:940-945`) explicitly asks for: effective `p` × α × strategy, recording `ε_RDP` vs the §2.3 bracket vs MC. Decide it against §2.3 and §2.7 *before* staffing either. Reconcile the b=8 doc table with the b=32 τ measurements in the §2.7 verdict.

---

## B. CODE NEVER EXAMINED BY ANY LENS

**Gap B1 — `amplification/b_min_sep/` (663 lines) is measured-unsound and got one clause.**

The draft mentions b-min-sep exactly twice, both as "inherits the identical cap" (§2.4), and never measures it. It is a **public factory** (`packages/opaque-dpftrl/src/opaque/api/accounting/dpftrl/amplification/__init__.py:7`, `__all__ = ["b_min_sep", ...]`) and one of only two amplification routes BandMF has (`balls_in_bins` rejects BandMF per §2.5).

Measured this session, `bandmf_b_min_sep_warm_mc_pld(coef=[1,.5,.3,.2], n=64, p=0.1, σ=1, seed=42)`:

| n_mc | δ=1e-3 | 1e-5 | **1e-8** | 1e-10 | 1e-14 |
|---|---|---|---|---|---|
| 1e5 | 5.713 | 7.716 | **8.130** | 8.131 | 8.131 |
| 1e6 | 5.745 | 8.274 | **10.586** | 10.596 | 10.596 |

Flat from 1e-8 to 1e-14 at both budgets — the `samples_to_pmf` cap the draft proves in §2.4. But the 10× sample increase moves ε by **+30.2%**, so at 1e5 samples the reported ε is 23% below the 1e6 value and *neither is converged*. Unlike §2.4's balls-in-bins measurement — which the draft's own correction concedes is **synthetic** ("no diagonal Gram reaches `bnb_mc_pld` from Python any more; `_identity.py:33-40` raises") — this one is on a live, Python-reachable, sole-accountant path.

**What closes it:** move the §2.4 deliverable to `BMinSep`, where it is not synthetic. Add the same δ-sweep as a strict-xfail plus a `BMinSep.pld` docstring warning. Separately: the design doc has a dedicated section **§7.2 "Deterministic replacement for b-min-sep"** (`:843-848`) with zero candidates against it — scope it after phase 4 or §2.7, whichever lands.

**Gap B2 — `amplification/truncated_poisson.rs` (294 lines) was never opened.** It is the shipped truncated-batch DP-SGD path, reachable through `dpsgd_acc.poisson(..., truncated_batch_size=, dataset_size=)` (`packages/opaque-dpsgd/src/opaque/api/accounting/dpsgd/amplification/_poisson.py:23-56`) and referenced by 20 files. Measured cost vs plain Poisson at δ=1e-8: `σ=1,q=0.01,B=150,n=1e4` → 1.009× at 1 step, **1.362× at 1000 steps**; `σ=2,q=0.02,B=250,n=1e4` → **1.965×** at 1 step, 1.144× at 1000.

The concrete, cheap, unproposed item inside it: `truncated_get_delta` collapses both adjacencies to `d_add.max(d_rem)` for component 1 (`truncated_poisson.rs:139-147`), and `truncated_epsilon_bounds` does the same for the bounds (`:174-186`) — deliberately discarding the asymmetric PLD that `discretize_asymmetric_mechanism` is built to carry. Every other amplification path in the crate is asymmetric. **What closes it:** a candidate "truncated-Poisson asymmetry" — pass `d_rem` for `Remove` and `d_add` for `Add`, measure the ε recovered at production `(q, B_max, n)`. Note also `:90` repeats the unconditional `with_tail_budgets(tail, tail)` that §2.8 flags in `random_allocation.rs:234-237`, so any fix there must cover this site too.

**Gap B3 — the entire non-ε metric surface (`pld/metrics.rs` β/risk/advantage + `_budgets.py`) was never evaluated, and the bracket is 4–7× wider there.**

`DpProcess` exposes `beta_at`, `risk_at`, `advantage` (`packages/opaque-accounting/src/opaque/api/accounting/core/_base.py:169,189,211`) and `calibrate()` accepts `AdvantageBudget`/`BetaBudget`/`RiskBudget` (`_budgets.py:139,171,205`; factories `:287,308,330`). **Every value claim in all 11 candidates and the entire draft is stated in ε.** Measured on `random_allocation_gaussian_pld(σ,t,1,upper,cfg)`, disc=1e-4:

| config | ε@1e-8 bracket | advantage bracket | β@α=0.3 bracket |
|---|---|---|---|
| σ=1, t=16 | [3.0469, 3.0527] **0.19%** | [0.12009, 0.12567] **4.6%** | [0.5699, 0.6014] **5.2%** |
| σ=2, t=64 | [0.3247, 0.3345] **3.0%** | [0.02435, 0.02903] **19.2%** | [0.6273, 0.7261] **13.6%** |

The discretisation slack the draft measures as 3% is **19%** in the metric `advantage_budget` calibrates against. This changes the priority ordering: it is a second, independent and much larger motivation for §2.2 + raising `MAX_CONV_GRID`, and it means anyone calibrating a `RandomAllocation` or identity-`BallsInBins` process against `advantage_budget`/`beta_budget` today is over-noising by up to 19%.

Related and unclosed: the draft's **Open Question 7** (`pmf_beta_asymmetric` at `pld/metrics.rs:268` dropping `negative_infinity_mass` at `:294-299`, while `pmf_beta` adds it at `:204-211`) is stated as "live, not hypothetical" and then appears **nowhere in the candidate pool and nowhere in the §4 sequencing plan**. Partial good news I can supply: I found **no direction violation** on the shipped `upper=true` path — β_up ≤ β_lo and advantage_up ≥ advantage_lo at every α tested, both configs. So the concern is not currently a soundness bug on the dominating side; it is unquantified on the `upper=false` side. **What closes it:** a candidate that (a) pins the β/advantage/risk bracket alongside the ε bracket in the transform's own tests, (b) resolves whether `pmf_beta_asymmetric` should adopt `pmf_beta`'s `negative_infinity_mass` treatment, and (c) re-runs §2.2's value case in β-space.

**Gap B4 — modules with no lens, ranked by whether that matters.** Never opened by any verdict: `numerics/fft.rs` (377), `numerics/special.rs` (411), `numerics/truncation.rs` (272, cited but not read), `discretization/connect_the_dots.rs`, `matrix_factorization/{lambda_cgd.rs (479), sensitivity.rs (750), mf_gaussian.rs, gram_matrix.rs}`, `mechanisms/{identity,gaussian,non_private}.rs`, `python/*.rs` (the whole PyO3 boundary), `transformations/adaclip.rs`, `_b_min_sep/_transcript_cache.py`, `core/_accountant.py`, `core/_process_codec.py`, `core/_serialization.py`, `core/composition/*`.

Most are legitimately out of scope. Two are not: **`matrix_factorization/sensitivity.rs` (750 lines)** is directly load-bearing for §2.5A's contract decision and for the draft's Open Questions 5 and 6 (coefficient non-negativity) — the draft cites `sensitivity.rs:405-410` second-hand via `_blt.py`'s `skip_checks=True` but nobody read the module. And **`numerics/truncation.rs`** is the mechanism behind §2.3's refuted precondition and §2.8's entire subject; it was cited (`:104-107`) but not audited.

**Gap B5 — `opaque-transformers` is outside the stated scope yet hosts the draft's own top-priority Stage-0 item.** The task named dpsgd/dpftrl/engine/auditing. `_training_arguments.py` and `_dp_trainer.py` — the sole citations for §2.11, which the draft itself flags as "PROVEN in ground truth and cheaper than anything above" and **"never adversarially reviewed"** — live in `packages/opaque-transformers/src/opaque/api/transformers/trainer/`. So the one item the draft admits is unreviewed is also the one item in the one package no lens was pointed at. **What closes it:** run the adversarial pass on §2.11 specifically, in `opaque-transformers`, before it ships as Stage 0 [B].

**Genuinely complete (no gap):** `opaque-alignment`, `opaque-optimizers`, `opaque-patches`, `opaque-base`, `opaque-engine`'s clipping/scheduling/distributed/precision subtrees. I checked these — they carry no accounting surface, and the one engine file with a privacy claim (`noise_allocation.py`) *was* examined and correctly refuted in the §2.10 verdict. Ignoring them was right, not an omission.

---

## C. PAPERS IN THE BRIEF THAT PRODUCED NO CANDIDATE

**Gap C1 — Dong & Ganesh, b-min-sep for BandMF (arXiv:2602.09338): zero candidates.** It is one of five papers in the brief. It is implemented (`b_min_sep/mc.rs`, header cites §5 Eq (2) and the post-Thm-5.1 warm-start correction), it is public API, and it is measured-unconverged (B1). Its likelihood ratio is an **exact, deterministic backward DP** (`mc.rs:47-77`), i.e. unlike balls-in-bins the loss is analytically computable — which is exactly the precondition a `LossRealization`/`disc_dist` route needs, and nobody asked whether it has an analytic or bracketable CDF. **What closes it:** one scoping candidate — "is `likelihood_ratio_warm` a `LossRealization`?" — checking whether the disjoint-block structure (`column_mu` pads to `bands`; `log_gaussian_ratio_block` reads `y[i..i+bands]`) admits a `GeomPmf`-style block convolution, and if not, whether §2.3's Loewner argument has any analogue for the renewal-process pair.

**Gap C2 — Feldman & Shenfeld 2502.08202 (the *analytic/RDP* paper) produced no candidate.** The brief lists it separately from 2602.17284 precisely because it is the analytic/RDP result; every candidate draws only on the 2026 PLD-transform paper. This matters because §2.2's **entire headline value proposition is that calibration is too slow** ("~3.5 minutes → ~21 seconds for a 30-probe bisection"). An O(1) closed-form RDP upper bound on random allocation would let `calibration.py:84`'s bisection skip most probes entirely — bracket with the analytic bound, run the 6.8-13 s transform only on the final few. That is a larger speedup than 8.6×, is independent of §2.2, and nobody proposed it. **What closes it:** a candidate "analytic RDP pre-filter for allocation calibration", cross-validated against the shipped transform as a free correctness oracle (it is also the only independent check the deterministic transform has besides the MIT reference impl).

**Gap C3 — `arXiv:2410.06266`, Choquette-Choo et al., *Near-exact privacy amplification for matrix mechanisms*, is in the doc's own reference list (`:951`) but absent from the brief and from every candidate.** This is the paper whose title is the exact problem the draft concludes has no good answer ("less than one would hope for DP-FTRL"). The survey used only the 2024 ICLR paper's Lemma 3.2. **What closes it:** read 2410.06266 and check whether its near-exact construction subsumes §2.3's Loewner bracket or §2.7's SK26 route, before either is staffed.

Also cited-and-unused, lower priority: Zhu/Dong/Wang characteristic-function accounting (`:953`) — a plausible alternative to the O(n²) `GeomPmf` convolution that §2.2 optimises; Chua et al. AISTATS 2025 is used only as a tightness citation for §2.8's precondition (b).

---

## D. KILLS THAT LOOK SHAKY

**Gap D1 — `epsilon-bracket-audit-join` and `audit-three-number-ladder` were killed on a precondition the survey's own evidence shows is already met.** §3b: "Both entirely downstream of a composable lower bound (§2.8), which was itself downgraded to LOW." But the §2.8 verdict states the opposite: "**The headline value is already delivered by shipped code** — the 19.4% bracket at (σ=2,t=64,E=64) was reproduced by calling the existing `.so` with no changes", and "the fix itself moves ε by **2.1e-9 to 2.6e-7**". A lower bound correct to 7 significant figures is a usable audit floor. So the audit candidates' stated blocker does not exist, and they were killed by transitivity through a downgrade whose own reasoning contradicts the transitive step.

This is compounded by B3: audits compare against **empirical ROC / β / advantage**, not ε — `opaque-auditing`'s `OneRunEstimate.beta_at` / `attack_beta_at` (`one_run/_estimate.py:208,298`) and `GdpMethod.beta_at` (`one_run/_gdp.py:149`) mirror `Pld`'s β surface deliberately. In β/advantage space the bracket is **4.6–19.2% wide**, not 0.19–3.0%. An audit estimate landing outside a 19%-wide bracket is a far stronger bug signal than the ε framing suggests, and `opaque-auditing` — one of the four packages the task names — otherwise received **zero candidates**. **What closes it:** revive the audit join scoped to β/advantage rather than ε, against the shipped `upper=false` PLD with no §2.8 dependency; measure whether the one-run GDP μ̂ lands inside the β bracket for the identity-BnB configuration.

**Gap D2 — `heterogeneous-allocation` killed as "Identity only — explicitly does not touch the correlated-MF gap."** Identity is the shipped deterministic BnB path (`_balls_in_bins.py:169-179`) and the one path the draft says is 7-27% tighter than the Rényi route. "Identity only" is a limitation everywhere else in this survey and a virtue here. The kill reasoning is a category error even if the conclusion is right. **What closes it:** state the actual reason (no caller wants heterogeneous bin weights) or reinstate.

**Gap D3 — `mixture-normal-realization` was downgraded to LOW on three refuted consumers, but two consumers it never considered exist in-tree.** The §2.10 verdict's own finding — `g` is a log-sum-exp of affine functions, hence convex *for any signs* of μᵢ — makes the object strictly more general than proposed. `truncated_poisson.rs` (B2) and `b_min_sep/mc.rs` (B1/C1) are both mixture-structured and were never checked as consumers. **What closes it:** re-score §2.10 against those two before deprioritising.

---

## E. UNPROPOSED APPLICATIONS OF THE THREE PRIMITIVES

**E1 — the two-sided bracket as a *validity gate*, not a replacement.** §2.4 uses the bracket to *characterise* MC bias offline. Nobody proposed the cheap online use: on every MC-backed `epsilon_at(δ)` call, assert the result lies inside a deterministic bracket where one exists, and **raise** otherwise. That converts §2.4's finding from a docstring warning into an enforced invariant on `calibration.py`'s `is_safe` path (`:341-344`) at the cost of one cached transform call per configuration.

**E2 — the bracket as a `discretization`/`MAX_CONV_GRID` auto-tuner in β-space.** §2.8's "runtime signal for `discretization`" was refuted *in ε* (3.19% → 2.90% under a 100× refinement). B3 shows the same knob has 4–7× more leverage in advantage/β. The refutation does not carry over and was never re-tested in the metric where it matters.

**E3 — `GeomPmf` exposed beyond random allocation.** The brief describes `geom.rs` as exponentiated-PLD convolution on a geometric grid with exponentiation by squaring — a general primitive. Every candidate treats it as private plumbing for one transform. It is the natural engine for the `k>1` improvement (currently a loose block bound, `random_allocation.rs:129-147`, which §2.8 proposes to *reject* rather than fix, and which the doc's own text at `:900-908` says "whoever gives it a Python caller should decide first whether the looseness is acceptable, or compute k-out-of-t directly") and for any mixture-over-participation-counts law including b-min-sep. Nobody proposed either.

**E4 — `disc_dist` on the `Adjacency::Replace` direction.** `grep -rn "Adjacency::Replace"` shows it used only in `poisson.rs`, `truncated_poisson.rs`, `parallel_poisson.rs` — **never** in `random_allocation.rs`, `balls_in_bins/`, or `b_min_sep/`. The entire random-allocation line is add/remove only. Whether replace-adjacency accounting is wanted for these paths was never asked.

---

## F. CLAIMS THE DRAFT OVERSTATES RELATIVE TO ITS OWN VERDICTS

**F1 — §1: "the first non-trivial deterministic bound those strategies have ever had".** The verdict corrects exactly this: "the trivial unamplified `gaussian_pld(sigma/sqrt(G00))` already existed, so 'first deterministic bound of any kind' is overstated." The draft moved the word "non-trivial" in but kept the superlative framing, then two paragraphs later concedes the bracket **loses to that trivial bound at σ=1 for every strategy**. Given Gap A1, "first" is also wrong in a second way: the §5.4 Rényi prototype is a deterministic bound for the same strategies and already validated.

**F2 — §2.4's header and §1 assert a live defect that the section's own corrections retract.** Headline: "the surviving Monte Carlo accountant returns an ε that is hard-capped below any valid bound at production δ=1e-8 (**PROVEN**) on a path `calibrate()` documents as privacy-safe." The verdict and the draft's own corrections say the measurement "is entirely synthetic with respect to the shipping code" because `_identity.py:33-40` raises and no diagonal Gram reaches `bnb_mc_pld` from Python. The finding is real (see B1 — on `BMinSep`, where it is *not* synthetic and I measured it), but as written §1 claims a live path that §2.4 then withdraws.

**F3 — §2.5A's "90.7×" is a headline number attached to a contract the draft does not claim is wrong.** The verdict establishes that `types.py:39` documents the participation drop as *intended*, four shipped tests assert `sensitivity == 1.0`, and two correct consumers (`_poisson.py:160`, `_b_min_sep/__init__.py:146`) require the k=1 semantics. A 90.7× "under-report" against a documented single-participation contract is a mis-pairing in `MfGaussian.pld`, not a 90.7× under-report. Contrast §2.1, where the 1.6-1.9× is a genuine mechanism/accountant mismatch and the framing is correct.

**F4 — §1's "yes for DP-SGD" is thinner than stated.** The only DP-SGD deliverable that survives to a stage is §2.6 (Stage 3), whose sole verified new consumer is `EpsDelta` — a library-only path with no trainer route; §2.9 (the other DP-SGD item) is deprioritised with a refuted consumer. The honest DP-SGD answer is "yes, one new capability on a library-only path", which is closer to the "less than one would hope" the draft reserves for DP-FTRL.

**F5 — §2.6 inherits an uncorrected doc error.** Design doc §7.1 (`:838-841`) names **AdaClip** as the concrete Opaque beneficiary of the subsampling transform ("arguably the highest value-per-line item in this note"). The §2.6 verdict refutes it: "AdaClip is not a new consumer; it reduces to an effective Gaussian noise multiplier already served by the native path." I confirmed — `transformations/adaclip.rs:30-34` is a scalar `z̃ = sqrt(1/z² + K/(4σ_b²))` rescale, nothing more. The draft records the correction for its own candidate but never flags that **the design doc's stated headline motivation for Phase 3 is wrong**, so anyone reading the doc will re-derive the same bad justification.

---

## G. SUGGESTED MINIMAL ADDITIONS TO §4 SEQUENCING

```
STAGE 0  (add)
  [L] BMinSep delta-sweep xfail + docstring  ~1d   §B1  — replaces §2.4's
      synthetic target with a live, Python-reachable one
STAGE 0.5 (new, gates Stage 2)
  [M] Renyi banded-Gram accountant: port the validated §5.4 prototype,
      run doc Open Question 5's sweep, reconcile vs the §2.7 tau measurements
      ~3d prototype  §A1  — decide [E] and [J] AGAINST this, not before it
STAGE 1 (add to [D]'s value case)
  [N] re-measure the bracket in advantage/beta/risk space  ~2h  §B3
      (19.2% vs 3.0% — changes [D]'s and [E]'s payoff)
UNSTAGED (cheap, currently absent)
  [O] truncated-Poisson asymmetry: drop max(d_add,d_rem)   ~1d  §B2
  [P] audit join scoped to beta/advantage, no §2.8 dependency  ~2d  §D1
```
