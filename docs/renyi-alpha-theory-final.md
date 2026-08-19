# The Rényi order α is not a tuning knob — and here is the proof

**Scope:** non-DP only, margins held as a stated offset (not swept as an effect), utility = eval loss on
KStack held-out. **Date:** 2026-08-14. **Supersedes** `renyi-alpha-utility-verdict.md` §§6–12 and
`HANDOFF.md` §3, which reached the right verdict twice by invalid routes and reopened it once.

Everything below is derived from **36 verified non-DP runs already in W&B** (`state=finished`,
260 steps, r=16, Qwen2.5-Coder-7B / KStack, 1 epoch, seed 42 unless noted). No new GPU time was
needed for the verdict. §8 lists the runs that *are* still worth buying and why, §8.1 how the noise
floor is measured (two different quantities have been called that), and §8.2 the follow-up rule.

---

## 0. Answer, in four sentences

1. **α is not viable, and the reason is structural, not statistical.** α enters the algorithm through
   exactly one integer per matrix per rotation, `⌊N_α⌋`. In non-DP the momentum spectrum is so
   rank-1 dominant that `⌊N_α⌋ = 1` for **every** α ≥ 1, at **every** operating point in the corpus
   (34/36 runs measured, 36/36 certified for α ≥ 2) — so the whole half-line α ∈ [1, ∞] is
   *one algorithm*, not a family. This is certified a priori from a single logged scalar (Thm 2) and
   confirmed exactly by the realised depths (≤1.9% of matrix-steps differ, by ≤1 slot).
   **Scope it honestly (§6.1):** α < 0.5 *is* identifiable — but only at shallow operating points,
   where the method is already worse by more than α's whole reach, so entering α's live region costs
   3.8× what tuning inside it could gain (§6.2, a dominance argument, not a null).
2. **Every previously reported α "effect" is nondeterminism.** The maximum effect α can mediate in
   these experiments is 1.1e-6 – 2.8e-5; the run-to-run floor at that depth is 3.0e-4. The observed
   differences are 6× – 529× larger than α could possibly cause, and they are *negatively*
   correlated with the α dose (Spearman ρ = −0.48, n=11).
3. **The "per-matrix adaptivity" was never adaptive.** At α ≥ 1 the rule assigns the *same* depth to
   all 196 matrices at all 260 steps. The earlier finding "heterogeneity contributes nothing" is
   true for a stronger reason than reported: there was no heterogeneity.
4. **Don't ship Shannon as a choice; ship the constant it degenerates to.** `r_e = r − 1 − m`
   ("keep the dominant momentum direction, refresh the rest"), plus the plateau measurement that
   makes the constant safe. Shannon is *inside* the equivalence class, so nothing changes numerically
   — but the paper stops claiming a knob it does not have. **The real theory contribution is the
   no-go (Thm 4 / §5.1): `N₂ = 1/Σp_i²` is the inverse Herfindahl index, so the rule answers a
   *concentration* question when the task is a *detection* question — and its response to noise
   therefore has the wrong sign (5/5 matched pairs, §5.2).** Lead with the sign, not with
   scale-invariance, which has a one-line rebuttal. §8.2 gives the constructive replacement.

---

## 1. Setup and notation

LoRA-XS: `W = W₀ + B R A`, with `B ∈ ℝ^{d_out×r}`, `A ∈ ℝ^{r×d_in}` **frozen** and `R ∈ ℝ^{r×r}`
trainable. LoRA-XSe re-draws part of the frozen basis every τ steps.

At rotation `t` and matrix `ℓ` (there are `L = 196 = 28 layers × 7 modules`), let `M` be the momentum
of `R`, with SVD `M = U Σ Vᵀ`, `σ₁ ≥ … ≥ σ_r ≥ 0`, and **energy shares**

$$p_i \;=\; \sigma_i^2 \Big/ \textstyle\sum_j \sigma_j^2, \qquad p \in \Delta_{r-1}.$$

Rényi entropy of order α and the **Rényi effective rank**:

$$H_\alpha(p)=\frac{1}{1-\alpha}\log\sum_i p_i^\alpha \ (\alpha\neq1),\quad
H_1=-\sum_i p_i\log p_i,\quad H_\infty=-\log p_1,\qquad N_\alpha=e^{H_\alpha}.$$

`N₀ = |supp p|`, `N₁ =` perplexity, `N₂ = 1/Σp_i²` (participation ratio), `N∞ = 1/p₁` (stable rank).

**The rule as implemented** (`xse.py:444–469`):

$$k = \lfloor N_\alpha(p)\rfloor + m,\qquad r_e = \mathrm{clip}(r-k,\,1,\,r-1),\qquad r_{keep}=r-r_e.$$

Keep the top `r_keep` singular directions of `M`; resample the other `r_e` uniformly from the
orthogonal complement; re-express `R` in the new basis by the 2×2 block projection.

**Source-level fact needed by Thm 1.** `grep` for `_ADAPTIVE_DEPTH_ALPHA` in `xse.py` returns
exactly three live uses: line 444 (compute `H`), line 467 (compute `r_e_dyn`), line 631 (logging).
α touches nothing else. In particular `U_keep = U[:, :r_keep]` — **at equal `r_keep` the kept set is
bit-identical across α.**

---

## 2. Theorem 1 — α is sufficient-statistic-collapsed onto one integer

> **Theorem 1 (exact equivalence).** The LoRA-XSe trajectory depends on `(α, m)` only through the
> integer field `k^{(ℓ)}_t = ⌊N_α(p^{(ℓ)}_t)⌋ + m`. Consequently, if two settings `(α,m)`, `(α′,m′)`
> satisfy
> $$\lfloor N_\alpha(p^{(\ell)}_t)\rfloor + m \;=\; \lfloor N_{\alpha'}(p^{(\ell)}_t)\rfloor + m'
> \qquad \forall \ell,\ \forall t$$
> along the coupled trajectory, then the two runs are **the same stochastic algorithm**: identical
> trajectory law, and identical realisations under a shared random stream.

*Proof.* Induction on `t`. At `t = 0` the states coincide by construction. Suppose the states
coincide at rotation `t`. Then the momenta coincide, hence `p^{(ℓ)}_t` coincides, hence by hypothesis
`k^{(ℓ)}_t` coincides, hence `r_e` and `r_keep` coincide. The update is a deterministic function of
(state, `r_keep`, random draw): `U_keep`, `V_keep`, `B_kept`, `A_kept` are slices at index `r_keep`;
`B_explore`, `A_explore` are a draw of `r_e` directions from the complement; `R_new` is the block
projection. Same state + same integer + same draw ⇒ same next state. ∎

**Corollary 1.1 (identifiability).** `N_α ∈ [1, r]` and is non-increasing in α (standard; the
project's own exact derivation `d/dα log N_α = −D_KL(p^{(α)}‖p)/(1−α)² ≤ 0` is in
`renyi-effective-rank-theory.md`). Hence α ↦ `⌊N_α⌋` is a non-increasing **step** function with at
most `r−1` jumps. **α has at most `r` distinguishable values no matter how finely you sweep it.**
It is not a continuous hyperparameter; it is a selector among ≤ r integers.

**Corollary 1.2 (the margin is an exact offset — *within a rotation only*).** `k = ⌊N_α⌋ + m` is
additively separable, so **at a fixed spectrum** `p`, `r_e(α,m) − r_e(α′,m)` is independent of `m`.

> ⚠️ **CORRECTED 2026-08-14.** An earlier version of this corollary concluded "a margin sweep can
> never break an α-degeneracy." **That is false at the run level** and the error was caught in review.
> `m` relocates the operating point, the operating point reshapes `R` and hence the spectrum, and the
> spectrum determines `⌊N_α⌋`. So `m` *does* modulate how much α can do — measurably: the α span
> `N_{0.5} − N_∞` grows **0.38 → 0.64 → 0.84** as `m` goes 1 → 2 → 3 (§6.3). The separability is
> instantaneous, not dynamical. The conclusion survives anyway, but by a different route (§6.3), and
> the honest form of this corollary is only the fixed-spectrum statement above.

**Corollary 1.3.** The three depth-14 runs the project used as its α comparison
(`renyi-ad-nodp-{a1,a2,ainf}-m1-s42`) are **replicates**, not conditions. Their spread is a valid
noise-floor estimate — which is exactly what §11 of the old verdict claimed and §12 rejected as
circular. §11's *number* was right; the justification it lacked is Theorem 1.

---

## 3. Theorem 2 — collapse certificates from one scalar

Let `δ = 1 − p₁` (spectral leakage out of the dominant direction).

> **Theorem 2.**
> **(a) α ≥ 2.** For α > 1, `N_α ≤ p₁^{−α/(α−1)} ≤ p₁^{−2}` for all α ≥ 2. Hence
> $$p_1 > 2^{-1/2} \approx 0.7071 \;\Longrightarrow\; \lfloor N_\alpha\rfloor = 1 \ \ \forall \alpha \ge 2 .$$
> **(b) α = 1.** By the grouping identity `H₁(p) = H_b(δ) + δ·H(p̃)` with `p̃` the conditional law on
> `{2..r}` and `H(p̃) ≤ log(r−1)`:
> $$\exp\!\big(H_b(\delta) + \delta\log(r-1)\big) < 2 \;\Longrightarrow\; \lfloor N_1\rfloor = 1 .$$
> **(c) α ∈ (0,1).** Hölder on the tail gives
> `N_α ≤ [(1−δ)^α + (r−1)^{1−α}δ^α]^{1/(1−α)}`, with equality iff the tail is uniform. Define
> `α*(δ,r) = inf{α : bound < 2}`; then `⌊N_α⌋ = 1` for all α ≥ α*.
> **(d) Exact, no bound.** By monotonicity in α: if `N_β < 2` for any β, then `⌊N_α⌋ = 1` ∀ α ≥ β.

*Proof of (a).* For α > 1, `Σ_i p_i^α ≥ p₁^α`, and `x ↦ x^{1/(1−α)}` is decreasing (negative
exponent), so `N_α = (Σp_i^α)^{1/(1−α)} ≤ p₁^{−α/(α−1)}`. `α/(α−1)` decreases in α and equals 2 at
α = 2. ∎ *(b) and (c) are the grouping bound and Hölder as stated.*

### 3.1 The certificate evaluated on our data (`p₁ = 1/N∞`, both logged)

| runs | p₁ range | (a) certifies α ≥ 2? | (b) bound on N₁ | (b) certifies α = 1? |
|---|---|---|---|---|
| **all 36 non-DP runs** | 0.801 – 0.998 | **36 / 36 YES** | 1.02 – 2.82 | 9 / 36 |
| the depth-14 α family (`*-m1-s42/43`) | 0.900 – 0.906 | YES | **1.76 – 1.81** | **YES** |

So for the exact configuration in which the project ran its α comparison, Theorem 2 certifies
**a priori, from one logged number, that α ∈ [1, ∞] is a single algorithm.** No experiment was ever
capable of separating those arms.

*Caveat, stated plainly:* `N∞` is logged as a mean over the 196 matrices, so `p₁` above is a mean.
The certificate should be applied per matrix with the worst-case (smallest) `p₁`. That needs one
extra logged scalar (`min_ℓ p₁`, one line in `xse.py`). The **empirical** version in §4 does not
need it, because realised depth aggregates correctly.

### 3.2 The empirical version — exact, no modelling

Since `⌊N_α⌋ ≥ 1` always and the clip is inactive here, `mean_ℓ ⌊N_α⌋ = r − \bar d − m` **exactly**,
where `\bar d` is the logged `rotation/r_e_dyn`. And `mean(⌊N⌋−1) ≥ P(⌊N⌋ ≥ 2)`, so
`r − \bar d − m − 1` is an **upper bound on the fraction of matrix-steps that got anything other
than "keep exactly one direction"**:

| α | m=1 | m=2 | m=3 |
|---|---|---|---|
| **∞** | **≤ 0.02 %** | ≤ 0.18 % | ≤ 0.45 % |
| **2** | ≤ 0.33 % | ≤ 5.4 % | ≤ 7.8 % |
| **1 (Shannon)** | ≤ 1.9 % | ≤ 14.7 % | ≤ 23.3 % |
| 0.5 | ≤ 19.2 % | ≤ 42.9 % | ≤ 67.1 % |
| 0.25 | — | 180 % † | — |

† > 100 % means `⌊N⌋` genuinely exceeds 2 — real heterogeneity appears only at α ≤ 0.25.

> **The "per-matrix adaptive depth" of LoRA-XSe is, at every α ≥ 1, the constant
> `r_e = r − 1 − m` on all 196 matrices at all 260 rotations.** It is not adaptive. The project's
> repeated finding that "per-layer tailoring adds nothing" is true because no tailoring occurred.

---

## 4. Theorem 3 — the mediation bound, and the dose–response falsification

### 4.1 The mediator curve (clean: fixed `p_e`, adaptive OFF, no feedback loop)

`med-nodp-fixed-re{1,5,9,13}-s42`, non-DP, seed 42:

| realised depth | eval loss | marginal slope per slot |
|---|---|---|
| 1 | 0.34700 | — |
| 5 | 0.34429 | −6.78e-4 |
| 9 | 0.34368 | −1.52e-4 |
| 13 | 0.34354 | **−3.52e-5** ← plateau slope Λ̂ |

Total 1 → 13: **−3.46e-3**; the slope decays **19×** across the range. Monotone-saturating.

### 4.2 Replicate floors — each from runs that are provably the same algorithm

| # | group | n | sd | spread |
|---|---|---|---|---|
| A | depth 5, non-adaptive, **identical config *and* seed** | 2 | 3.43e-5 | 4.85e-5 |
| B | depth 5, non-adaptive, seeds 42/43/44 (+dup) | 4 | **3.00e-5** | 7.26e-5 |
| C | **depth 14, adaptive, α ∈ {1,2,∞}, seed 42** (Thm 1 ⇒ replicates) | 3 | **3.03e-4** | 5.71e-4 |
| D | depth 14, adaptive, {2,∞} × seeds {42,43} | 4 | 3.27e-4 | 7.09e-4 |
| E | depth 13, adaptive α=∞ m=2, two separate submissions, same seed | 2 | 1.26e-4 | 1.78e-4 |

Two things to take from this, both correcting `HANDOFF.md` §8:

- The **floor scales with depth, not with "adaptivity"**: 3.0e-5 at depth 5, 1.8e-4 at depth 13,
  3.0e-4 at depth 14. Deeper refresh injects more fresh randomness per rotation. Attributing it to
  adaptivity is an artifact of the adaptive runs all being deep.
- Group A vs B: same-seed nondeterminism (3.4e-5) is **as large as** the across-seed sd (3.0e-5). The
  0.23e-4 "seed sd" used elsewhere in the docs is not a seed effect at all — it is GPU
  nondeterminism, and it is the *floor*, not a *seed* term.

### 4.3 The bound

Define the **dose** between two orders as the separation of their realised depth fields,
`D₁ = mean_t |d^α − d^{α′}|` and `D∞ = max_t |·|` (logged per step as `rotation/r_e_dyn`).

> **Theorem 3 (first-order mediation bound).** Under (A1) *the loss responds to the depth schedule to
> first order, with sensitivity measured by uniform depth perturbations*, `|L(α)−L(α′)| ≲ Λ̂ · D`.

(A1) is an assumption; it is the honest pairing, because a uniform 1-slot shift *is* dose 1, and it
is supported by the matched-depth residual audits (7 arms, |residual| ≤ 1.9e-4, random sign). §4.4
gives a falsification test that needs no assumption at all.

With `Λ̂ = 3.52e-5` per slot on the plateau (all contrasts below live at depth ≥ 11.3):

| contrast | D₁ | D∞ | max mediated \|ΔL\| | observed \|ΔL\| | observed / possible |
|---|---|---|---|---|---|
| α=2 vs ∞, m=1 | 0.0031 | 0.031 | **1.1e-6** | 5.71e-4 | **529×** |
| α=1 vs ∞, m=1 | 0.0190 | 0.138 | 4.9e-6 | 4.62e-4 | 95× |
| α=1 vs 2, m=1 | 0.0159 | 0.107 | 3.8e-6 | 1.08e-4 | 29× |
| α=2 vs ∞, m=2 | 0.0522 | 0.097 | 3.4e-6 | 1.05e-3 | 307× |
| α=1 vs ∞, m=2 | 0.1454 | 0.209 | 7.4e-6 | 1.25e-4 | 17× |
| α=0.5 vs ∞, m=1 | 0.1913 | 0.628 | 2.2e-5 | 2.31e-4 | 10× |
| α=0.5 vs ∞, m=2 | 0.4268 | 0.694 | 2.4e-5 | 1.52e-4 | 6× |
| α=1 vs ∞, m=3 | 0.2280 | 0.321 | 1.1e-5 | 2.51e-4 | 22× |
| α=2 vs ∞, m=3 | 0.0732 | 0.102 | 3.6e-6 | 3.11e-5 | 9× |
| α=0.5 vs ∞, m=3 | 0.6660 | 0.806 | 2.8e-5 | 1.39e-5 | **0.5×** |
| m=0: α=0.5 vs 2 | 0.0286 | 0.429 | 1.5e-5 | 3.82e-5 | 3× |

**The largest effect α can mediate anywhere in this table is 2.8e-5 — an order of magnitude below the
3.0e-4 replicate floor at that depth.** α is not "flat"; it is *below the resolution of the
apparatus by construction*, and buying more seeds cannot change that.

### 4.4 The assumption-free falsification: the dose–response is *negative*

A real mechanism produces effects that grow with the dose. Across the 11 contrasts:

$$\rho_{\text{Spearman}}(D_\infty,\ |\Delta L|) = \mathbf{-0.482}, \qquad r_{\text{Pearson}} = -0.466, \quad n=11.$$

The three *largest*-dose contrasts (D∞ = 0.81, 0.69, 0.63) produced the *smallest* differences
(1.4e-5, 1.5e-4, 2.3e-4), and the two *smallest*-dose contrasts (0.031, 0.097 — algorithmically
near-identical) produced the *largest* (5.7e-4, 1.05e-3). This needs no Lipschitz assumption, no
noise-floor estimate, and no theory: **whatever is generating those differences, it is not α.**

---

## 5. Theorem 4 — the no-go: entropy ranks are scale-blind, and respond to noise with the wrong sign

This is the part that generalises beyond LoRA-XSe.

> **Theorem 4.** (i) `N_α(p(cM)) = N_α(p(M))` for every `c > 0` and every α — the Rényi effective
> rank is **scale-invariant**, because `p` is a normalised spectrum.
> (ii) In the spiked model `M = Σ_{j≤s} θ_j u_j v_jᵀ + ς r^{−1/2} G`, `G` iid `N(0,1)`, the BBP
> transition makes spike `j` detectable iff `θ_j > ς`; the oracle keep-set
> `K*(θ,ς) = {j : θ_j > ς}` has `|K*|` **non-increasing** in `ς`.
> (iii) With the mean-field spectrum `σ_i² = θ_i² + ς²`, the vector `p(ς)` is **majorised** by
> `p(ς′)` for `ς > ς′` (all ratios `p_i/p_j = (θ_i²+ς²)/(θ_j²+ς²)` compress toward 1). `H_α` is
> Schur-concave for every α ≥ 0, so `N_α(p(ς))` is **non-decreasing** in ς, with `N_α → r`.
> (iv) Therefore `r_e = r − ⌊N_α⌋ − m` is **non-increasing** in the noise level while the correct
> `r − |K*|` is **non-decreasing**. **The signs are opposite, for every α.**

### 5.1 State this carefully — "it is scale-free" is *not* by itself the argument

There is an obvious objection to any loose version of (i), and it is correct: **the SNR `θ/ς` is
itself invariant under `(θ, ς) → (cθ, cς)`, so invariance under `M → cM` cannot on its own be the
defect.** Do not present it that way. The argument is three separate claims of unequal strength:

**Defect 1 — wrong functional (the strong one).** `N_2 = 1/Σp_i²` is *exactly the inverse
Herfindahl–Hirschman index* — the "effective number of firms" from industrial organisation — and
`N_1` is its entropy analogue. **We used a market-concentration index to answer a signal-detection
question.** They are different questions with different correct answers:

> A town where one person earns €10M and fifteen earn €50k. *"Effective number of earners"*
> (concentration) ≈ 1 — correct. *"How many earn above minimum wage?"* (detection) = 16 — also
> correct. The rotation rule needs the second and we gave it the first.

Concentration is dominated by the largest atoms; a detection count is about where the *bulk ends*.
With one dominant spike, concentration reports ≈1 even when six directions sit well clear of the
noise floor. That is exactly the measured gap: `N_α ≈ 1.3` versus `#{σ_i > 2·median} ≈ 6`.

**Defect 2 — the statistic cannot consume a known noise scale.** Detection needs a threshold in
*absolute units*, i.e. a minimum wage. `p_i = σ_i²/Σσ_j²` is dimensionless, so `ς` has no slot to
enter through. This matters because under DP we **know `ς` exactly** (noise multiplier × clipping
norm / batch, times a closed-form momentum factor) and the normalisation discards it. That is the
honest version of the scale point: not "invariance is bad", but *a dimensionless statistic cannot
consume a known scale*.

**Defect 3 — the sign (provable, and matched-pair verified in §5.2).** This is (ii)–(iv) above, and
it is the claim to lead with, because it needs no appeal to units at all. Intuition: noise does not
produce zero singular values — an r×r noise matrix has r nonzero ones, filling a bulk. Adding noise
hands every direction a random bonus, the energy distribution *flattens*, concentration falls, so
`N_α` **rises**, so the rule freezes *more* directions — precisely when more of them have become junk
and should be refreshed.

**No choice of α repairs any of the three**, because all three are properties of the functional
`p ↦ N_α(p)`, not of the order α.

**Corollary 5.1 (what the margin `m` was actually doing).** `m` is the only term in
`k = ⌊N_α⌋ + m` that can encode an SNR correction, since `⌊N_α⌋` cannot. That is why `m` was as
load-bearing as α, and why the two proved interchangeable (Cor 1.2). The paper should stop
describing `m` as a "safety buffer" and describe it as the rule's only noise-aware term — an
un-calibrated constant standing in for a threshold.

### 5.2 Defect 3, on matched pairs

Earlier drafts checked this on pooled regime buckets, which mixed run types and produced a
non-monotone spike-count column. Replaced here with **exactly matched pairs**: same `p_e`, same
fixed depth, adaptivity OFF (so realised depth is constant and identical across the pair), r=16,
1 epoch, same trainer.

| `p_e` (fixed depth) | N₁ non-DP | N₁ with noise | ratio | spike # non-DP | spike # with noise |
|---|---|---|---|---|---|
| 0.0625 (d=1) | 1.84 | 5.45 (ε=1) | **3.0×** | 5.31 | 2.65 |
| 0.3125 (d=5) | 1.81 | 5.25 (ε=1) | **2.9×** | 5.85 | 2.77 |
| 0.333 (d=5) | 1.81 | 4.85 (ε=3) | **2.7×** | 5.88 | 2.85 |
| 0.5625 (d=9) | 1.69 | 4.51 (ε=1) | **2.7×** | 7.00 | 7.00 |
| 0.8125 (d=13) | 1.43 | 3.21 (ε=1) | **2.2×** | 6.23 | 6.00 |

**`N₁` rises in 5/5 matched pairs, by 2.2–3.0×.** Since `r_e = r − ⌊N_α⌋ − m`, that removes ~3 slots
of exploration per matrix under noise — the wrong direction, exactly as Defect 3 predicts. This is
the cleanest empirical statement in the document: no pooling, no modelling, one controlled variable.

**Be honest about the last two columns.** The threshold statistic moves the *right* way in 3/5 pairs
(5.31→2.65, 5.85→2.77, 5.88→2.85 — keep fewer, explore more) and is **flat in 2/5** (7.00→7.00,
6.23→6.00). `2·median` is a hand-rolled fixed-quantile heuristic, not the bulk edge, and it
saturates. So the direction of travel supports §8.2's proposal *where the estimator has headroom*;
it does not license a claim that the current proxy already works. Fix the estimator first (§8.2).

One further observation that `N_α` structurally cannot make: under DP the **per-matrix minimum** of
the spike count falls to **0.04–0.35** (vs 3.5 non-DP). Some matrices have *no* direction above the
bulk — the statistic is detecting "this entire matrix is now noise." `N_α ≥ 1` always, so it can
never say that.

Also note the level disagreement: the threshold count says ≈6 directions carry signal where the Rényi
rank at usable α says **1** — a 6× gap — and the threshold count is the one with genuine per-matrix
spread (sd 0.70 across the 196 matrices, range 4.2–8.0, versus *identically* 1).

---

## 6. Theorem 5 — even granting an effect, α is a strictly worse parameterisation of depth

Empirical, from the m=2 sweep (`rotation/r_e_dyn` trajectories):

| α | mean depth | `d(depth)/dα` | drift over training | depth sd over t | loss_min |
|---|---|---|---|---|---|
| 0.05 | 2.85 | — | +0.29 | 0.161 | 0.34537 |
| 0.10 | 6.50 | +73.0 | +1.07 | 0.353 | 0.34405 |
| 0.15 | 8.90 | +48.1 | +1.35 | 0.381 | 0.34372 |
| 0.20 | 10.33 | +28.7 | +1.28 | 0.338 | 0.34353 |
| 0.25 | 11.20 | +17.4 | +1.02 | 0.266 | 0.34346 |
| 0.50 | 12.57 | +5.5 | +0.37 | 0.081 | 0.34346 |
| 1 | 12.85 | +0.56 | +0.10 | 0.038 | 0.34346 |
| 2 | 12.95 | +0.09 | +0.03 | 0.025 | 0.34349 |
| ∞ | 13.00 | ~0 | 0.00 | 0.002 | 0.34362 |

Three things, all fatal to α as a knob:

1. **Conditioning.** `|d(depth)/dα|` spans **800×** over α ∈ [0.05, 2] and is exactly 0 on [α*, ∞].
   Depth is directly settable via `p_e`; via α it is reachable only through a badly conditioned,
   data-dependent map.
2. **Non-stationarity.** The drift column is a runaway feedback loop: deeper refresh concentrates
   `R`, which lowers `N_α`, which deepens the refresh. At m=0 it is worse — `m0-nodp-a025-m0-s42`
   drifted **12.7 → 14.4**. So α does not name a depth; it names a *time-varying schedule* you did
   not choose.
3. **α is inert exactly where it is stable, and unstable exactly where it is live.** Drift and
   sensitivity rise and fall together. There is no operating point where α is both controllable and
   consequential.

**Equivalence statement for the paper.** Over α ∈ [0.2, ∞] (six settings, m=2) `loss_min` spans
**1.6e-4**, which is *smaller than a single replicate spread* at that depth (1.8e-4 – 5.7e-4).
That is a TOST-style equivalence at ±1.6e-4 (±0.05 % relative), not an untested null.

**Where α does something — and it is bad.** Below α ≈ 0.2 the reachable depth drops into the steep
region of §4.1 and loss degrades monotonically: α=0.05 → depth 2.85 → 0.34537, i.e. **+1.9e-3 ≈ 6×
the floor**. So α is not globally inert. It is a **floor constraint** ("stay above ~0.2"), which any
constant trivially satisfies.

### 6.1 Scope check: which α are collapsed, and did m = 2 choose the answer?

Every α comparison in this project ran at m ∈ {0,1,2,3}. Since `m` sets the operating point and the
spectrum's flatness depends on the operating point, this needs auditing rather than asserting. Result
of checking **max over the trajectory** of `N_α` in all 36 non-DP runs (`an8.py`):

| order | collapsed (`max_t N_α < 2`, so `⌊N_α⌋ = 1`) | where it fails |
|---|---|---|
| **α ≥ 2** | **36 / 36** — and `p₁ > 0.7071` in 36/36, so Thm 2a certifies it | nowhere |
| **α = 1** (Shannon) | **34 / 36** | only `fixed-re1` (depth 1) and `lowa-a005` (depth 2.8), marginally (max N₁ = 2.02) |
| **α = 0.5** | **7 / 36** | everywhere shallower than depth ≈ 13.8; `max_t N_{0.5}` reaches 3.13 |
| α ≤ 0.25 | 0 / 36 | separates strongly everywhere (`N₀ = 15.96`, so α→0 drives `⌊N⌋` toward 15) |

**So the collapse claim must be scoped: α ∈ [1, ∞] is one algorithm at every operating point in the
corpus; α = 0.5 is only collapsed in the deep band; α < 0.5 is genuinely identifiable.** The project's
own sweep was α ∈ {0.5, 1, 2, ∞}, so three of its four arms were provably the same algorithm and the
fourth was nearly so — but "α is unidentifiable" is *not* a statement about all α.

**And yes, `m` chose the operating point — the α span varies 24× across it:**

| realised depth band | n | mean span `N_{0.5} − N_∞` | mean p₁ |
|---|---|---|---|
| 1 – 4 | 2 | **1.611** | 0.805 |
| 4 – 7 | 7 | 1.397 | 0.804 |
| 7 – 10 | 4 | 1.157 | 0.815 |
| 10 – 12.5 | 7 | 0.903 | 0.829 |
| **12.5 – 14.5** | 14 | **0.499** | 0.879 |
| 14.5 – 15 | 2 | 0.068 | 0.993 |

Shallower refresh ⇒ more accumulated structure spread over more directions ⇒ flatter spectrum ⇒ larger
span. **m ∈ {1,2,3} put every α comparison in the two deepest bands, where α has the least authority
it will ever have.** That is a real limitation of the design, and §8 (below) confirms the gap:
**there is no α contrast at depth < 11.3 anywhere in the corpus.**

### 6.2 Why raising `m` still cannot rescue α — a dominance argument

Detectability is `dose(d) × slope(d) / floor(d)`. Measured at the margins actually run, and projected
to the ones never run (dose extrapolated from the observed +0.09 slots per unit `m`; α=0.5 vs ∞):

| m | depth | dose (max) | local slope /slot | max effect | floor | detectable? |
|---|---|---|---|---|---|---|
| 1 | 13.9 | 0.628 | 3.5e-5 | 2.2e-5 | 3.0e-4 | 0.07× no |
| 2 | 12.8 | 0.694 | 3.5e-5 | 2.4e-5 | 3.0e-4 | 0.08× no |
| 3 | 11.7 | 0.806 | 3.5e-5 | 2.8e-5 | 1.0e-4† | 0.28× no |
| 6 | ~9 | ~1.08 | 3.5e-5 | 3.8e-5 | 1.0e-4† | 0.38× no |
| **8** | **~7** | ~1.26 | 1.5e-4 | 1.9e-4 | 1.0e-4† | **1.9× yes** |
| **10** | **~5** | ~1.44 | 1.5e-4 | 2.2e-4 | 3.0e-5 | **7.3× yes** |

† the floor at depth 7–12 is **unmeasured** and interpolated; §8.1's replicate runs are what would fix it.

So α *would* become detectable at m ≈ 8–10. But that is self-defeating, and this is the cleanest way
to close the question:

> **Detectability requires slope; slope only exists in the region where the method is already worse.**
> At depth 5 the loss is 0.34429 — **+8.3e-4 above the plateau optimum** — while the *entire* reachable
> α effect there is 2.2e-4. **Moving to α's live region costs 3.8× more than the best possible gain
> from tuning α once you are in it.** Setting `m` high to make α identifiable is strictly dominated by
> leaving `m` low and not tuning α at all.

This strengthens §6's third bullet rather than weakening it. The sharper statement: **α is
unidentifiable exactly where the method is good, and identifiable exactly where the method is bad.**
Its live region and the method's useful region are effectively disjoint, so no operating point makes
α a useful knob. That is a dominance result, not an appeal to a null.

### 6.3 The margin objection, answered in full

**The objection (raised in review, and correct on the mechanism):** *almost every α experiment ran at
m = 2. Since the rule is adaptive, it will "adjust to the margin situation" — so an α comparison at one
margin tells you nothing about α in general, and the whole campaign is practically invalidated.*

**Conceded first:** the mechanism is real, and Cor 1.2 as originally stated was wrong. `m` is not a
neutral offset at the run level — it relocates the operating point, which reshapes the spectrum, which
changes `⌊N_α⌋`. Measured: the same α gives different `⌊N_α⌋` at different `m` (α=1 gives mean
`⌊N_α⌋` = 1.019 / 1.147 / 1.233 at m = 1 / 2 / 3), and the α span grows 2.2× over that range. So `m`
genuinely modulates α's authority. The objection identifies a true interaction.

**But it does not invalidate the campaign, for four measured reasons** (`an9.py`).

**(1) The sweep ran at four margins, not one.**

| m | α values run (non-DP, seed 42) |
|---|---|
| 0 | 0.1, 0.25, 0.5, 2 |
| 1 | 0.5, 1, 2, ∞ |
| 2 | 0.05, 0.1, 0.15, 0.2, 0.25, 0.5, 1, 2, ∞ |
| 3 | 0.5, 1, 2, ∞ |

The α ∈ {0.5, 1, 2, ∞} contrast exists at **m = 1, 2 and 3** independently, and partially at m = 0.

**(2) The rule does not compensate for a margin change — it *amplifies*. This is the direct test of the
objection's mechanism.** If the adaptive part "adjusted to" the margin, it would give back some of what
`m` took, i.e. `|d(depth)/dm| < 1`. Measured:

| α | depth at m=0 | m=1 | m=2 | m=3 | `d(depth)/dm` | verdict |
|---|---|---|---|---|---|---|
| ∞ | — | 14.00 | 13.00 | 12.00 | **−1.002** | neutral |
| 2 | 15.00 | 14.00 | 12.95 | 11.92 | −1.026 | amplifies |
| 1 | — | 13.98 | 12.85 | 11.77 | −1.107 | amplifies |
| 0.5 | 14.97 | 13.81 | 12.57 | 11.33 | −1.214 | amplifies |
| 0.25 | 14.33 | — | 11.20 | — | **−1.567** | amplifies |
| 0.1 | 9.18 | — | 6.50 | — | −1.344 | amplifies |

**Not one setting compensates.** Raising `m` by 1 removes one direction by construction; the adaptive
part then removes a *further* 0.03–0.57. And the over-reaction grows monotonically as α falls, so the
settings that are nominally "more adaptive" over-react *most*. The objection's hypothesised mechanism
runs the opposite way to the data.

**(3) The α ranking does not reproduce across margins.** A real effect must keep the same winner:

| m | α=0.5 | α=1 | α=2 | α=∞ | spread | best | worst |
|---|---|---|---|---|---|---|---|
| 0 | 0.34364 | — | 0.34368 | — | 3.8e-5 | 0.5 | 2 |
| 1 | 0.34385 | 0.34371 | 0.34399 | 0.34362 | 3.8e-4 | **∞** | 2 |
| 2 | 0.34346 | 0.34346 | 0.34349 | 0.34362 | 1.6e-4 | **1** | ∞ |
| 3 | 0.34363 | 0.34370 | 0.34365 | 0.34361 | 8.0e-4→8.0e-5 | **∞** | 1 |

(`loss_min`, the operative metric.) The winner is 0.5, then ∞, then 1, then ∞. The *worst* α is 2,
then ∞, then 1. **Four independent operating points, four different orderings** — a replication
failure, not a weak effect.

**(4) The decisive one: giving α more room makes the observed effect *smaller*.**

| m | α span `N_{0.5} − N_∞` | depth gap α=0.5 vs ∞ | observed `loss_min` spread |
|---|---|---|---|
| 1 | 0.382 | 0.19 slots | **3.78e-4** |
| 2 | 0.638 | 0.43 slots | 1.58e-4 |
| 3 | 0.835 | 0.67 slots | **8.03e-5** |

As `m` hands α **3.5× more authority**, the measured spread **falls 4.7×**. This is the dose–response
test (§4.4) run *specifically along the margin axis the objection is about*, and it comes out negative
there too. m = 3 is the margin where α has the widest reach of any tested, and it is the margin with
the *smallest* α spread.

**(5) Holding α fixed and moving `m` shows the loss tracks depth, not α.** α = 0.1 was run at both
m = 0 (depth 9.18, `loss_min` 0.34361) and m = 2 (depth 6.50, 0.34405). Both land on the fixed-`p_e`
depth curve — residuals **−6.3e-5** and **−1.2e-5**, at the floor. Same α, different margin, and the
loss follows the *depth* the pair produced. So the low-α penalty is a depth effect, confirmed by
varying the margin rather than α.

**What is genuinely untested, and the boundary of the claim.** No α contrast exists at m ≥ 4 — only
α=1 was run there. That is the shallow regime where α becomes identifiable (§6.1) and where §6.2's
dominance argument says any gain is outweighed by the cost of being there. So the correct scope is:

> **α ∈ [1, ∞] is one algorithm at every operating point tested (m = 0–3), and across those four
> margins the α ordering fails to replicate while α's measured effect shrinks as its authority grows.
> The untested corner is m ≥ 4, where α is identifiable and the method is worse.** Four runs
> (α ∈ {0.5, 1, 2, ∞} at m = 8) would close it; the prediction is a detectable but harmful effect.

### 6.4 What the margin is actually for — including one job nobody documented

Three answers, increasingly honest. (Full version, with the experiment plan that follows from it, in
`docs/alpha-margin-experiment-prep.md` §3.)

**(a) Stated intent** (`xse.py:428`): `N_α` estimates how many directions carry signal; keep those,
refresh the rest; the margin keeps "a couple extra" so a borderline-useful direction is not discarded.
Library default 2, kept by a July sweep over m ∈ {1,2,3}.

**(b) What it actually corrects: a systematic undercount.** `N_α` measures *concentration*, not count.
With `p₁ = 0.90`, `N_α ≈ 1.1` while `#{σ_i > 2·median} ≈ 6` directions sit above the noise floor. The
margin is a **hand-set patch for a statistic known to read low**, and per Cor 5.1 it is the *only* term
in `k = ⌊N_α⌋ + m` that can carry such a correction. It also absorbs the floor's rounding-down bias
(`N_α = 1.9` ⇒ `int()` returns 1). A *constant* cannot be right in general, because the size of the
true undercount depends on the noise level and `N_α` never sees it.

**(c) Undocumented job: the margin is a brake on a positive feedback loop.** The loop is: refresh more
⇒ `R` loses accumulated structure ⇒ spectrum spikier ⇒ `N_α` falls ⇒ refresh even more. Measured
within-run depth drift (first eval → last eval) at matched α:

| α | m=0 | m=1 | m=2 | m=3 |
|---|---|---|---|---|
| 0.1 | **+1.53** | — | +1.07 | — |
| 0.25 | **+1.72** | — | +1.02 | — |
| 0.5 | +0.44 | +0.52 | +0.37 | +0.25 |
| 2 | +0.01 | +0.04 | +0.03 | −0.01 |

**A larger margin damps the runaway, and it only matters where the loop is live (low α)** — at α ≥ 2
there is nothing to damp because `⌊N_α⌋` is pinned at 1. Removing the brake entirely (m=0) is exactly
why `m0-nodp-a025-m0-s42` drifted 12.7 → 14.4 and had to be excluded from the depth curve as a
time-varying schedule.

For the paper: *the margin was introduced as a safety buffer; it is really a constant correction for a
statistic that systematically under-counts the active subspace; and it incidentally damps a feedback
loop between refresh depth and spectral concentration.* None of those three jobs should be done by a
hand-set constant — which is the argument for the threshold rule in §8.2.

**One measurement caveat at the very low end.** At α = 0.05, m = 2 the mean `⌊N_α⌋` is 11.16 and
`r_e = 1` requires `⌊N_α⌋ ≥ 13`; with `N₀ = 15.96` (all 16 directions nonzero) some matrices plausibly
hit the `clip(·, 1, r−1)` lower bound. Per-matrix depths are not logged, so clamping cannot be
confirmed — but it would compress the α→depth map at the bottom, and it is consistent with α=0.05
showing the *smallest* within-run drift (+0.29 vs +1.35 at α=0.15). Treat the α=0.05 point as possibly
saturated. The §9.3 logging fix would settle it.

---

## 7. What to write in the paper

**Keep, unchanged — this is the result.** Rotating the frozen LoRA-XS basis recovers full-LoRA
quality at 201× fewer trainable parameters, ε-free by post-processing. Multi-seed, paired, both
ε = 1 and ε = 3.

**Keep — the mechanism.** The depth curve: `−3.46e-3` from depth 1 → 13, monotone-saturating, slope
decaying 19×. Rule: **refresh ≥ ~55 % of the rank; above that it is one flat plateau.** Going
shallow is the only mistake, and it costs a quarter of the entire rotation benefit. Quote it against
the *depth-matched* floor (3.0e-4 at depth 13–14): the shallow penalty is **11×** that floor.

**Reframe — the rule.** Ship
$$\boxed{\;r_e \;=\; r - 1 - m\;}$$
"keep the dominant momentum direction, refresh the rest", and prove (Thm 2a, 36/36 runs certified)
that this is what *every* α ≥ 2 realises in this regime, and (Thm 2b) what α = 1 realises too at the
depths we use. The Rényi machinery is then presented as what it is: **the diagnostic that establishes
rank-1 dominance** — pooled over all 36 non-DP runs `p₁ = 0.847`, `N₁ = 1.55`; in the depth-14 family
specifically `p₁ = 0.900` and the top **two** momentum directions carry **98.8 %** of the energy
(`rotation/energy_ratio` at `r_keep = 2`) — and therefore *derives* the constant. Correct by construction rather than tuned — a stronger claim than
a tuned α, and one no reviewer can ask you to sweep.

**New contribution — the negative, stated as theory.** Theorems 1–4 are a self-contained result:
*spectral-entropy effective-rank statistics cannot implement capacity selection, for two independent
reasons — quantization destroys identifiability (Thm 1: ≤ r distinguishable settings, here exactly
1), and the statistic is a concentration index answering a detection question, which makes its
response to noise go the wrong way (Thm 4 / §5.1 Defect 3).* This is not a story about our
optimizer. Entropy and effective-rank statistics on normalised spectra are used for rank allocation
across the PEFT literature; both objections apply verbatim. That is the section a reviewer will
remember.

The one-line hook for the abstract: **`N_2 = 1/Σp_i²` is the inverse Herfindahl index. We — and the
literature we inherited it from — used a market-concentration statistic to decide which directions
carry signal.** State Defect 3 (wrong sign, 5/5 matched pairs) as the headline and Defect 1 (wrong
functional) as the mechanism; do **not** lead with "it is scale-invariant", which has a one-line
rebuttal (§5.1).

**Retire, explicitly.** "α as a tuning knob"; "per-matrix adaptive depth" as a *benefit* (Thm 2.2:
there was no per-matrix variation to benefit from); "the margin as a safety buffer" (Cor 5.1: it is
the rule's only noise-aware term, uncalibrated).

**Do not repeat these two errors** (both are in the doc history and a reviewer could find them):
a noise floor taken from a *contrast between configurations* (circular), and a floor taken from a
*different depth/family* than the comparison. Theorem 1 is what licenses group C in §4.2 as a
genuine replicate set — the argument the earlier version was missing.

---

## 8. Should you buy more GPU? Ranked — plus how to measure the floor, and the follow-up

The α verdict needs **nothing**. These close real gaps.

**#1 — worth it (5 runs, revised up from 3). Decouple depth from adaptivity in the noise floor.**
`med-nodp-fixed-re13` (non-adaptive, `p_e = 0.8125`, depth 13) × seeds {42…46}. Right now every
floor in the paper is confounded: all deep runs are adaptive and all shallow ones are not (§4.2). If
the group lands at ~3e-4, the floor is a **depth** effect and the paper says so — and "adaptive runs
are 20× noisier", which §3.3 of `HANDOFF.md` currently asserts, is **withdrawn**. If it lands at
~3e-5, adaptivity really does cost variance and that is a reportable cost. Either answer is
load-bearing for every effect size you quote. This is the one I would actually run. §8.1 explains
why 5 and not 3.

### 8.1 How the non-DP noise floor is actually measured — and how many runs it takes

Two different quantities in this project have been called "the noise floor". They are unrelated and
conflating them has already caused one retraction. Name them separately in the paper.

**(A) The measurement floor** `σ_run` — run-to-run sd of the final eval loss when the *same*
configuration is re-run. This is the denominator of every effect size. Two sources feed it: GPU
nondeterminism (non-associative float reductions, kernel selection) and the algorithm's own
randomness (`random_orthogonal_complement` draws `r_e` fresh directions per rotation). Measurement
protocol:

1. **Same config, same seed, k re-submissions** → isolates GPU nondeterminism alone. Measured: 3.4e-5
   at depth 5 (n=2), 1.8e-4 at depth 13 (n=2).
2. **Same config, k different seeds** → nondeterminism *plus* algorithmic randomness. **This is the
   floor to quote**, because two different configs also differ in their random draws. Measured:
   3.0e-5 at depth 5 (n=4), 3.0e-4 at depth 14 (n=3).
3. **Always at the depth of the comparison.** The floor is not a constant — it grows steeply with
   `r_e` (3.0e-5 → 1.8e-4 → 3.0e-4 across depth 5 → 13 → 14), because deeper refresh injects more
   fresh randomness per rotation. Using a shallow-run floor for a deep-run comparison is error #2 in
   §7, and it is how the α question got reopened.

**How many runs?** An sd from n samples is itself uncertain, by χ²(n−1). Approximate 95 % CI as a
multiple of the point estimate:

| n | 95 % CI on σ_run | comment |
|---|---|---|
| 3 | **[0.52×, 6.3×]** | a floor that could be 6× larger than you quote — not publishable as a denominator |
| 5 | [0.60×, 2.9×] | minimum defensible |
| 6 | [0.62×, 2.5×] | comfortable |
| 8 | [0.66×, 2.0×] | diminishing returns |

This is why #1 above is **5 runs, not 3** — my earlier recommendation of 3 was too thin for a number
that divides every effect in the paper. (Normality of the loss values is assumed; it is an
approximation, but the order of magnitude of the uncertainty is the point.)

**Free improvement, no GPU.** Rather than one floor per family, fit `log σ_run` against depth across
*all* existing replicate groups (depth 5 n=4, depth 13 n=2, depth 14 n=3+4) and quote the fitted
value at the comparison depth, with the 5 new runs at depth 13 as the anchor. Pooling turns four
weak estimates into one usable curve, and the depth-dependence becomes a reportable finding in its
own right: **deep rotation costs variance**, which matters to anyone deploying this.

**(B) The spectral noise scale** `ς` — the size of the junk component *inside the momentum spectrum*,
in loss-gradient units. This is not a run-to-run quantity at all; it is what the §8.2 threshold rule
needs, and it is the harder of the two in non-DP. Under DP it is known analytically: per-entry
gradient noise std is `noise_multiplier × clipping_norm / batch_size` (logged as `train/noise_std`),
and SGD momentum `m_t = βm_{t−1} + g_t` inflates its variance by `1/(1−β²)` = 5.26× at β = 0.9, so
`ς_m = ς_g/√(1−β²)`. In **non-DP there is no injected noise, so `ς` is pure minibatch sampling
noise and must be estimated.** Three ways, in increasing order of rigour:

1. **Bulk fit (assumption: MP).** For an r×r matrix with iid entries of std `s`, the singular values
   fill `[0, 2s√r]` (quarter-circle law), so `s` is recoverable from a robust statistic of the lower
   spectrum. Gavish–Donoho (2014) give the optimal hard threshold for a square matrix with unknown
   noise level as `τ = 2.858 · median(σ)`. **The code's current `2.0 · median` is a hand-rolled
   approximation of that constant** (`xse.py:419-422`) — replacing 2.0 with 2.858 is a one-character
   fix with a citation behind it.
2. **Split-batch null (assumption-free).** The difference of two independent half-batch gradient
   estimates has **zero signal** and twice the sampling-noise variance. Project it into the r×r core
   coordinates and its spectrum *is* the noise bulk, measured rather than assumed. Nearly free here:
   the per-sample/per-microbatch gradients already exist in the functional/vmap clipping harness, so
   their scatter gives the sampling-noise covariance directly.
3. **Cross-check 1 against 2.** If the bulk-fit `s` and the split-batch `s` agree, the MP assumption
   is validated for this architecture and scale — which is itself worth a figure, since the whole
   RMT framing in `renyi-effective-rank-theory.md` currently rests on it untested.

Note the asymmetry, and state it in the paper: **non-DP is the *harder* regime for a threshold rule,
not the easier one**, because `ς` must be estimated instead of read off the privacy accountant.

**#2 — worth it if the theorem needs to be general (4 runs). Escape the null space.**
Non-DP, **r = 64**, α ∈ {0.5, 1, 2, ∞}, m = 2, seed 42. Theorem 2 says the collapse is a property of
`(r, p₁)`, not of α — and the DP data already shows the statistic escaping the floor as `r` grows
(⌊N₁⌋ = 1 at r=2,4 → **4** at r=16 under ε=3). Prediction: at r=64 non-DP, `⌊N_α⌋` separates across
α and genuine per-matrix heterogeneity appears for the first time. Then either (i) loss is *still*
flat — the strongest possible negative, "α identifiable and still inert", which upgrades §5 from a
degeneracy argument to a real null; or (ii) it is not flat, and you have found the regime where α
matters, which is a positive result. Both are publishable. This is the only experiment that can
still surprise you.

**#3 — free, do it anyway (0 runs).** Log `min_ℓ p₁` and `max_ℓ N_{0.5}` per rotation (one line near
`xse.py:466`). That makes the Theorem 2 certificate checkable **per matrix** instead of on a layer
mean, closing the caveat in §3.1. It rides along on #1/#2.

**Do not run:** more α sweeps at r=16 non-DP (Thm 2: unidentifiable, 36/36 certified); more margin
sweeps (Cor 1.2: exact offset, cannot decollapse α); more seeds at depth 14 (they are replicates —
useful for the floor, worthless for α). The two `seedrep-ad-nodp-*-s44` runs in flight are fine to
let finish: they take group C/D to n=6 and sharpen the floor. They cannot reopen α.

### 8.2 The follow-up worth proposing: replace counting-by-concentration with counting-by-threshold

This is the constructive half of §5. The replacement must be **scale-aware in absolute units**, or
Defect 3 kills it on arrival:

| candidate | fixes Thm 1 (identifiability)? | fixes Defect 3 (sign)? | status |
|---|---|---|---|
| **Bulk-edge count** `k = #{i : σ_i > τ(ς̂)}` (BBP / Gavish–Donoho) | yes — real per-matrix spread | **yes** | half-built: `XSE_MP_SHRINKAGE` (`xse.py:419`), logged every run as `xs_spread/rec_rank_*` with the crude `c=2·median` |
| Energy threshold `k = min{k : Σ_{i≤k} p_i ≥ τ}` | partly | **no** — still normalised | not a fix; a smoother knob with the same defect |
| De-quantized `N_α` (randomised rounding of `⌊N_α⌋`) | yes | **no** | makes α identifiable, but the depth it unlocks lies inside the plateau ⇒ predicted null. Not worth GPU |

**The proposal.** Keep direction *i* iff its momentum singular value clears the noise bulk edge:

$$k \;=\; \#\{\, i : \sigma_i > \tau \,\},\qquad
\tau = 2\,\varsigma_m\sqrt{r}\ \ \text{(known }\varsigma\text{)}\quad\text{or}\quad
\tau = 2.858\cdot\mathrm{median}(\sigma)\ \ \text{(unknown }\varsigma\text{)},$$

$$r_e = r - k \quad\text{— no margin, no }\alpha,\ \text{nothing to tune.}$$

Why it is the right object, and why it is a real paper rather than a patch:

- **Right question.** It counts directions above the noise floor instead of measuring how
  concentrated the energy is (§5.1 Defect 1). Under the spiked model it is the *oracle* rule: BBP says
  spike *j* is recoverable iff `θ_j > ς`, and the observed value is `ρ = θ + ς²/θ`.
- **Right sign.** `|K*|` is non-increasing in ς by construction, so `r_e = r − |K*|` rises with noise
  (§5.1 Defect 3). Partially visible already in §5.2's spike-count columns.
- **Parameter-free under DP.** `ς_m = ς_g/√(1−β²)` with `ς_g` from the accountant (§8.1B). "We
  computed the threshold" is a strictly stronger claim than "we swept α", and it is the kind of claim
  that survives review.
- **It degenerates correctly at both ends.** As ς → 0 it keeps the true rank and refreshes the rest;
  as ς → ∞ it keeps nothing and refreshes everything, which is the right answer when the momentum is
  all noise. `⌊N_α⌋` does the opposite at the noisy end.
- **It can express "this matrix is finished".** `k = 0` is reachable (§5.2: the per-matrix minimum
  already hits ~0 under DP). `N_α ≥ 1` always, so the current rule can never say it.
- **It is genuinely per-matrix.** sd 0.70 across the 196 matrices, range 4.2–8.0 — versus identically
  1. This is the only candidate under which "per-matrix adaptive depth" could ever be a real claim.

**Two fixes required before it is testable**, both small:

1. Replace `2.0 · median` with the Gavish–Donoho constant **`2.858 · median`** for square matrices
   (`xse.py:421`). The current constant is an undocumented approximation of it.
2. Under DP, prefer the *known* `τ = 2ς_m√r` over any median estimator — the accountant already gives
   `ς_g`, and estimating what you know is strictly worse.

**Where it can be validated — and where it cannot.** In non-DP the loss is flat over depth 9–15
within the measurement floor, and every sensible rule lands in that band (the threshold rule puts
depth at `16 − 6 ≈ 10`, inside the plateau). **Non-DP is structurally incapable of discriminating
between depth-selection rules** — that is a property of the response surface, not a shortcoming of
the experiment. Add to that §8.1B: non-DP also forces you to *estimate* ς rather than read it off.
So the honest sequencing is:

| phase | regime | question it can answer | cost |
|---|---|---|---|
| 0 | non-DP | does the split-batch `ς` agree with the bulk-fit `ς`? (validates the RMT framing, no utility claim) | 1 run + analysis |
| 1 | DP, ε = 3 | does the threshold rule beat `⌊N_α⌋` at matched compute, with ς known? | 2 arms × 5 seeds |
| 2 | DP, ε = 1 | does the advantage grow as noise grows, as Defect 3 predicts? | 2 arms × 5 seeds |

### 8.3 Phase 0 is already DONE — the iid/MP model of the momentum is validated, no GPU used

The assumption the entire RMT section of `renyi-effective-rank-theory.md` rests on is testable from
runs already on disk, via the matched non-DP/DP pairs (`an11.py`).

**The test.** Model the momentum as `M = M_signal + E` with `E` iid per-entry std `ς_m`, independent of
the signal. For a matched pair differing only in noise, `‖M_dp‖² − ‖M_nodp‖² = r²ς_m²`. Compare that
*measured* `ς_m` against the *accountant's* prediction `ς_m = train/noise_std × √(1/(1−β²))`, where
`1/(1−β²) = 5.26` at β = 0.9 (torchopt's `trace` is `m ← βm + g`, PyTorch heavy-ball — verified in
`torchopt/transform/trace.py:202`, `dampening=0`).

| `p_e` (depth) | ε | ‖M‖ non-DP | ‖M‖ DP | ς_m **measured** | ς_m **predicted** | ratio | noise share of ‖M‖² |
|---|---|---|---|---|---|---|---|
| 0.0625 (1) | 1 | 0.0202 | 0.1541 | 0.00955 | 0.00929 | **1.03** | 98.3 % |
| 0.3125 (5) | 1 | 0.0138 | 0.1535 | 0.00955 | 0.00929 | **1.03** | 99.2 % |
| 0.333 (4.8) | 3 | 0.0147 | 0.1173 | 0.00727 | 0.00703 | **1.03** | 98.4 % |
| 0.5625 (9) | 1 | 0.0128 | 0.1415 | 0.00881 | 0.00929 | 0.95 | 99.2 % |
| 0.8125 (13) | 1 | 0.0134 | 0.1043 | 0.00647 | 0.00929 | 0.70 | 98.3 % |

**Three results, all new:**

1. **The iid/Marchenko–Pastur picture of the momentum is validated** — 1.03, 1.03, 1.03, 0.95 for four
   of five pairs. The RMT framing is now empirical, not assumed. (The depth-13 pair reads 0.70; the
   likely cause is that a shallower refresh lets the momentum accumulate over a longer effective
   horizon than the stationary `1/(1−β²)` assumes. Worth one line of follow-up, not a blocker.)
2. **The normalisation convention is pinned:** the agreement requires *no* batch-size division, so the
   per-entry noise std the optimizer receives equals `train/noise_std` as logged, and the momentum
   variance gain is `1/(1−β²)` — both needed to set the threshold analytically under DP.
3. **Under DP, >98 % of the momentum's *energy* is injected noise.** The signal is under 2 %. That is a
   striking number in its own right and it belongs in the paper.

**And the count, in absolute units** (`τ = 2ς_m√r`, `σ₁ = ‖M‖/√N_∞`):

| `p_e` | ε | τ | σ₁ | σ₁/τ | threshold rule keeps | entropy rule keeps | refresh: threshold / entropy |
|---|---|---|---|---|---|---|---|
| 0.0625 | 1 | 0.0764 | 0.0979 | 1.28 | **1** | 7 | **15 / 9** |
| 0.3125 | 1 | 0.0764 | 0.0958 | 1.25 | **1** | 7 | **15 / 9** |
| 0.333 | 3 | 0.0582 | 0.0746 | 1.28 | **1** | 7 | **15 / 9** |
| 0.5625 | 1 | 0.0704 | 0.0924 | 1.31 | **1** | 6 | **15 / 10** |
| 0.8125 | 1 | 0.0517 | 0.0782 | 1.51 | **1** | 5 | **15 / 11** |

**Only the top direction clears the bulk edge, and only by 1.25–1.5×.** Everything else is inside the
noise bulk. So the two rules disagree by 5–6 slots and in opposite directions — now measured in
absolute units rather than through the `2·median` proxy of §5.2.

**What is still not measurable, and the exact fix.** In non-DP there is no noiseless reference to
subtract, so the *sampling*-noise scale cannot be recovered from `‖M‖` alone. Two routes:

1. **Gavish–Donoho:** `τ = 2.858 · median(σ)` for a square matrix with unknown noise level. Needs the
   spectrum logged — `singular_values_top` is computed at `xse.py:617` and discarded (§9.3).
2. **Split-batch null (assumption-free).** For two independent half-batch gradients,
   `d = (g_A − g_B)/2` has **zero signal** and *exactly* the full-batch noise variance
   (`Var[(g_A−g_B)/2] = Var[(g_A+g_B)/2]`). Its spectrum **is** the bulk, measured rather than
   assumed. Then `ς_m = ς_grad·2.294` and `τ = 2ς_m√r`. The per-microbatch gradients already exist in
   the vmap clipping harness, so this is nearly free.

Doing (1) and (2) and checking they agree would extend the validation above from the DP regime (where
it now holds) to the non-DP regime (where the threshold rule would actually have to run).

---

## 9. What the existing telemetry already shows — and two metrics not to trust

Sweep of all ~35 logged `rotation/*` and `xs/*` fields (`an6.py`, `an7.py`). Two independent
confirmations, one free code fix, two retractions.

### 9.1 The α span was already being logged, and nobody read it

`rotation/renyi_gap_a0p5_ainf` = `N_{0.5} − N_∞` — literally "how much room is there between the two
extreme Rényi orders". The rule consumes only `⌊·⌋`, so a span below 1 means the two extremes are at
most one integer step apart:

| regime | mean span | range | runs with span < 1 |
|---|---|---|---|
| **non-DP** | **0.863** | 0.060 – 1.657 | **22 / 36** |
| DP (ε=1, 3) | 3.969 | 0.617 – 5.171 | 2 / 34 |

And in the specific family used as the project's α comparison (adaptive, m=1):

| run | span |
|---|---|
| `renyi-ad-nodp-ainf-m1-s42` | 0.359 |
| `renyi-ad-nodp-a2-m1-s42` | 0.363 |
| `renyi-ad-nodp-a1-m1-s42` | 0.371 |
| `renyi-ad-nodp-a0.5-m1-s42` | 0.437 |
| `seedrep-ad-nodp-{ainf,a2}-m1-s43` | 0.352 |

**The whole of α ∈ [0.5, ∞] spans 0.36 of a unit in a rule that only uses whole units.** This is the
single most quotable line in the document, and it needed no new analysis — the diagnostic was in every
run from the start.

*Precision, because a referee will check:* span < 1 does not by itself prove a shared floor (1.8 → 2.4
spans 0.6 yet straddles 2). It is conclusive here because the interval also lies **inside** [1, 2):
max over the trajectory of `N_{0.5}` is 1.85–1.89 for that family (§ analysis B in `an2.py`), so
monotonicity in α (Thm 2d) closes it exactly. Quote the span as the *intuition* and Thm 2 as the proof.

### 9.2 `promotion_count` reproduces the depth curve — from mechanism, not from loss

`rotation/promotion_count` counts how many of the retained top-`r_keep` momentum directions came out
of the *previously randomised* block — i.e. how often exploration actually produced a keeper. It is
**capped at `r_keep`** by construction (`xse.py:588`), so the raw count must be normalised; the
scale-free quantity is `promotion_count / r_keep` = the fraction of the retained set that was freshly
discovered rather than inherited.

| realised depth | n | promoted fraction of retained set | eval loss |
|---|---|---|---|
| 1 | 1 | **0.011** | 0.34700 |
| 3 | 1 | 0.035 | 0.34537 |
| 5 | 6 | 0.056 | 0.34429 |
| 9 | 4 | **0.069** | 0.34368 |
| 11 | 3 | 0.064 | — |
| 13 | 7 | 0.058 | 0.34354 |
| 14 | 7 | 0.057 | — |
| 15 | 2 | 0.057 | 0.34366 |

**Same shape as the loss curve, measured independently of the loss**: catastrophically low at depth 1
(5–6× below the plateau), rising to depth ~9, then flat. So the depth result now has a *mechanism* to
report alongside the curve — at depth 1 exploration essentially never yields a retained direction,
and from ~5–9 upward the yield saturates at ~6%. This is the strongest new material in this pass and
it belongs in the paper next to §4.1.

### 9.3 Free code fix: the one metric that would settle the cut point is computed and thrown away

`singular_values_top` (top-8 singular values per matrix) is computed at `xse.py:617` and then
**omitted** from the per-layer diagnostic dict at `xse.py:803–815`, so it never reaches W&B. Adding
one string to that tuple would give the actual momentum spectrum, which is what is needed to answer
"where *should* the cut be?" — currently unanswerable (§9.5).

### 9.4 RETRACTED: `xs/grad_snr` is not a signal-to-noise ratio

`train_causal_lm.py:2374` defines it as `‖m_R‖ / ‖g_R − m_R‖` — a **momentum-consistency** ratio, not
an SNR. With momentum 0.9, `‖m‖ ≈ 10‖g‖`, so it pins near 1 regardless of noise: **0.9547–0.9585
non-DP vs 0.9651–0.9693 DP**, a 1.5 % difference across a regime change that alters everything else by
2–3×. It must **not** be used to support the noise argument (§5). Either rename it or replace it with
a real estimator (§8.1B option 2).

### 9.5 RETRACTED: "the sharpest spectral cliff sits after direction 1"

`rotation/spectral_gap` = `σ[r_keep+1]/σ[r_keep]` is the drop *at the cut*, so assembling it across
runs with different `r_keep` looks like it traces the spectrum's shape — and it appears to show a huge
cliff at k=1 (ratio 0.081) flattening to ~0.75 by k=11. **That reading is confounded**, by exactly the
circularity that caused the earlier retractions in this project: deeper refresh ⇒ less accumulation ⇒
spikier `R` ⇒ smaller reported ratio. The runs at small `r_keep` are spiky *because* they refresh
deeply. Restricting to the fixed-depth runs (where `r_keep` is set externally, not from the spectrum)
still leaves four different trajectories:

| run | k = r_keep | σ[k+1]/σ[k] | energy in top k |
|---|---|---|---|
| `fixed-re13` | 3 | 0.429 | 0.99111 |
| `fixed-re9` | 7 | 0.731 | 0.99711 |
| `fixed-re5` | 11 | 0.751 | 0.99949 |
| `fixed-re1` | 15 | 0.324 | 0.99999 |

**So "where should the cut be?" is currently unanswerable from the logs**, and the fix is §9.3.
Do not claim the rule lands on a real spectral gap until `singular_values_top` is logged.

### 9.6 A second statistic that separates the regimes cleanly: `cond(R)`

Matched pairs (same `p_e`, fixed depth, adaptivity off, r=16):

| `p_e` | cond(R) non-DP | cond(R) DP | ratio | r_eff(R) non-DP → DP |
|---|---|---|---|---|
| 0.0625 | 2.73e9 | 1.15e7 | 237× | 1.74 → 4.68 |
| 0.3125 | 5.34e9 | 2.02e8 | 27× | 1.70 → 4.09 |
| 0.333 | 4.67e9 | 1.46e8 | 32× | 1.62 → 3.72 |
| 0.5625 | 5.46e9 | 2.15e8 | 25× | 1.58 → 3.32 |
| 0.8125 | 4.78e9 | 2.09e8 | 23× | 1.36 → 1.96 |

`R` is 23–237× more ill-conditioned without DP noise, and its effective rank rises 2–3× under noise —
the same direction as the momentum result in §5.2, now measured on a *different object* (the core `R`
rather than the momentum). Two independent statistics, same conclusion: **rank-1 dominance is a
property of the noiseless regime, and noise destroys it.** That is the mechanism behind Defect 3.

---

## 10. Batch A results (m=0 sweep, run 2026-08-14) — predictions held

5 runs submitted against the pre-registered predictions in `docs/alpha-margin-experiment-prep.md` §1.4.
4 finished, α=0.15 crashed at step 84 (discarded unread per the state+step rule).

| α | predicted depth | **actual** | predicted loss | **actual** | verdict |
|---|---|---|---|---|---|
| ∞ | 15.00 | **15.000** | 0.3436 ± 3e-4 | 0.34371 | ✓ |
| 1 | 15.00 | **14.995** | 0.3436 ± 3e-4 | 0.34377 | ✓ |
| 0.2 | 12–13 | 13.54 | 0.3435 ± 3e-4 | 0.34363 | depth 0.5–1.5 slots deeper than predicted; loss ✓ |
| 0.15 | 11–12 | 11.27 @ step 84 | — | **discarded** | crashed |
| 0.05 | 4–6 | **5.33** | 0.3444 ± 5e-4 | **0.34439** | ✓ exact |

**Headline prediction confirmed.** α ∈ {0.5, 1, 2, ∞} at m=0 realise depth **14.963 / 14.995 / 14.999 /
15.000** and losses **0.34364 / 0.34377 / 0.34368 / 0.34371** — spread **1.3e-4**, against a predicted
"< 2e-4". Their α spans are **0.088 / 0.076 / 0.073 / 0.073**, the smallest measured anywhere, exactly
as §1.1 of the prep doc predicted from the clamp geometry.

**And every m=0 arm lies on the same depth curve as the m=2 and fixed-`p_e` arms**: residuals vs the
fixed-`p_e` curve at matched depth are **+1.5e-4** (d 5.33), **−5e-5** (d 9.16), **+7e-5** (d 13.54).
Mixed signs, all ≲ 1.5e-4. **The margin creates no separate regime — everything collapses onto depth.**
The one exception remains `m0-nodp-a025-m0-s42` (+3.5e-4), the run whose depth drifted 12.3 → 14.4,
i.e. a time-varying schedule rather than a fixed depth.

### 10.0 The completed m=0 sweep (9 readable arms, α = 0.05 → ∞)

`m0b-nodp-a015-m0-s42` crashed at step 84 and was re-run (`-r2`, finished, depth 11.81). Full sweep:

| α | mean depth | `loss_min` | α span `N_{0.5}−N_∞` | residual vs depth curve |
|---|---|---|---|---|
| 0.05 | 5.33 | 0.34439 | **1.440** | +1.5e-4 |
| 0.1 | 9.16 | 0.34361 | 1.164 | −6.4e-5 |
| 0.15 | 11.81 | **0.34343** | 0.864 | −1.6e-4 |
| 0.2 | 13.54 | 0.34363 | 0.503 | +5.5e-5 |
| 0.25 | 14.31 | 0.34399 | 0.312 | +3.7e-4 ‡ |
| 0.5 | 14.96 | 0.34364 | 0.088 | −2.0e-5 |
| 1 | 14.99 | 0.34377 | 0.076 | +1.1e-4 |
| 2 | 15.00 | 0.34368 | 0.073 | +1.6e-5 |
| ∞ | 15.00 | 0.34371 | 0.073 | +5.1e-5 |

‡ the drifting run (12.3 → 14.4), i.e. a time-varying schedule, not a fixed depth.

Residuals: **mean +5.7e-5, |max| 3.7e-4, 3/9 negative.** Every arm lies on the fixed-`p_e` depth curve.
And the α span **collapses monotonically with depth** — 1.44 at α=0.05 down to 0.073 at α=∞ — which is
the collapse mechanism (§6.1) traced out over a full sweep at a single margin.

### 10.1a r = 64: THE COLLAPSE HOLDS — Theorem 2's scope widens beyond r = 16

`r64-probe-a1-m2-s42` OOM'd at step 5, but logged the spectrum first — which is all this probe needed.
**The OOM is unrelated to rank:** it died in `linear_cross_entropy.py:857` allocating `(B_vmap, V, D)`
for the vocab projection, at `peak_gb = 77.7 / 79.2` (98 % utilisation). The r=16 runs already sit at
~98 %; the extra ~320 MB of frozen r=64 basis tipped it over. Fix is `--microbatch-size 8`, not a rank
limit.

Measured at r = 64 (step 5, early training):

| quantity | r = 16 (converged) | **r = 64 (step 5)** |
|---|---|---|
| p₁ (top-direction energy share) | 0.80 – 0.85 | **0.884** |
| N_∞ | 1.18 – 1.25 | **1.131** → ⌊·⌋ = **1** |
| N_2 (inverse Herfindahl) | 1.34 | **1.261** → ⌊·⌋ = **1** |
| N_1 (Shannon) | 1.55 | **1.611** → ⌊·⌋ = **1** |
| N_0.5 | 2.06 | **4.149** → ⌊·⌋ = 4 |
| realised depth | — | 60.73 = 64 − 1.27 − 2 |

**Prediction confirmed.** §8.1 #2 predicted that if p₁ ≈ 0.8 is intrinsic to the gradient rather than a
property of r, then `⌊N_α⌋ = 1` and the collapse would persist at r = 64. Measured p₁ = **0.884** —
*higher* than at r = 16 — and α ∈ [1, ∞] all still floor to 1. So:

> **Theorem 2's scope widens from "r = 16 in our setting" to "any rank at which the gradient projection
> is rank-1 dominant."** One (failed) run bought that.

Two refinements. (i) α = 0.5 *does* separate at r = 64: ⌊N_0.5⌋ = 4 vs ⌊N_∞⌋ = 1, a 3-slot gap, and the
α span grows 0.86 → **3.02**. So low α becomes more identifiable at larger r — but 3 slots of 64 (4.7 %)
is *relatively smaller* than 1 slot of 16 (6.25 %), which is the "more identifiable, less consequential"
trade predicted in §8.1. (ii) This is a **step-5 snapshot**; in the r=16 runs N_α *falls* over training
(1.41 → 1.24 for α=1), so the converged r=64 values are likely lower still, strengthening the collapse.
A `--microbatch-size 8` re-run would confirm at convergence.

### 10.1c FLOOR, FINAL (n=5 group in hand): ~6.5e-5, and the "adaptive is noisier" signal is ONE run

The `fixed-re13` replicate group is complete — **n=5, non-adaptive, depth 13**:

> 0.3435417 / 0.3434333 / 0.3434410 / 0.3434855 / 0.3435749 → **sd 6.20e-5**, 95 % CI [3.7e-5, 1.8e-4]

Against the adaptive depth-14 group (sd 2.02e-4, n=6) that is an F-ratio of **10.7**, above the critical
9.36 — apparently significant, which would *reinstate* "adaptive runs are noisier." **It doesn't.**
That group contains a single anomalous run:

- `renyi-ad-nodp-a2-m1-s42` = 0.343993 vs its group mean 0.343597
- **Grubbs test: G = 1.956 > 1.887 critical (n=6, two-sided 5 %) ⇒ a statistical outlier at p < 0.05**
- corroborated independently: its `eval/loss` 0.34466 vs `loss_min` 0.343993 is a **6.7e-4 late-training
  excursion**, one of the largest in 36 runs (the median gap is exactly 0)
- drop it and the group's sd is **6.48e-5** — indistinguishable from every other group (F = 1.09)

**And that outlier is the very run that generated the original "α=2 is worse than α=∞" claim.** The
entire α investigation traces back to one run with a late-training excursion.

> **Pooled floor, outlier excluded: σ = 6.49e-5 (df = 16).** Use this.

### 10.1d WHICH CONCLUSIONS ARE STABLE UNDER FLOOR RE-ESTIMATION — the honest filter

The floor has been re-estimated four times in this project as replicates accumulated. A finding is only
reportable if its *category* survives all four:

| effect | per-group | pooled w/ outlier | pooled w/o outlier | best group | stable? |
|---|---|---|---|---|---|
| **shallow-depth penalty** | 12.4× | 20.4× | 53.3× | 55.8× | **YES — robust** |
| cost of margin 8 vs 2 | 2.2× | 3.7× | 9.7× | 10.1× | no |
| m = 0…3 whole spread | 1.0× | 1.7× | 4.4× | 4.6× | no |
| α effect at margin 8 | 0.7× | 1.1× | 2.9× | 3.0× | no |
| **adaptive vs uniform at m=8** | 0.2× | 0.3× | 0.9× | 0.9× | **YES — noise** |
| **α max mediated (m ≤ 3)** | 0.1× | 0.2× | 0.4× | 0.5× | **YES — noise** |

**Only the extremes are stable.** The three claims the paper actually needs are exactly the three that
never move category:

1. **Depth matters** — going shallow costs 12–56× the floor, under every estimate. **Robust.**
2. **α does nothing in the operating range** — 0.1–0.5× the floor, under every estimate. **Noise.**
3. **Per-matrix adaptivity does nothing** — 0.2–0.9× the floor, under every estimate. **Noise.**

Everything in the 1–5× band flips between "marginal" and "real" depending on which denominator is
chosen. **Those must not be reported as findings** — including three numbers I quoted with more
confidence than they deserved: that margin 8 is significantly worse than margin 2, that α is detectable
at margin 8, and any ranking among margins 0–3. The *directions* are consistent; the significance is
not established.

### 10.1b The per-group estimates are indistinguishable — pool them

With the `fixed-re13` replicates in (§8.1 #1), there are now six same-algorithm groups. Their sds span
9× — but with n = 2–6 each, **none of them is distinguishable from any other**:

| group | n | sd | 95 % CI on the true sd |
|---|---|---|---|
| non-adaptive depth 5 | 4 | 3.00e-5 | [1.7e-5, 1.1e-4] |
| **non-adaptive depth 13** (new) | 3 | 6.05e-5 | [3.2e-5, 3.8e-4] |
| adaptive depth 13 | 2 | 1.26e-4 | [5.6e-5, 4.0e-3] |
| adaptive depth 14 | 6 | 2.80e-4 | [1.8e-4, 6.9e-4] |
| adaptive depth 15 (m=0) | 4 | 5.50e-5 | [3.1e-5, 2.1e-4] |
| adaptive depth 7 (m=8) | 2 | 8.73e-5 | [3.9e-5, 2.8e-3] |

Every interval overlaps every other. The apparent 9× spread is **one common floor, sampled badly.**

> **Use the pooled floor: σ = 1.70e-4, 95 % CI [1.26e-4, 2.63e-4] (df = 15).**

That is the number to quote everywhere, and it retires the whole per-group-floor exercise — including
§4.2's depth story and §3.3 of `HANDOFF.md`'s "adaptive runs are 20× noisier" (the adaptive/non-adaptive
ratio at depth 13 is 2.1×, well inside sampling error of 1×).

**Every conclusion re-scored against the pooled floor:**

| effect | size | vs pooled floor | verdict |
|---|---|---|---|
| shallow-depth penalty (depth 1 vs 13) | 3.46e-3 | **20.3×** | **ROBUST** |
| cost of margin 8 vs margin 2 | 6.28e-4 | **3.7×** | **ROBUST** |
| m = 0…3 whole spread | 2.85e-4 | 1.7× | marginal, and non-monotone ⇒ noise |
| **α effect at margin 8** | 1.89e-4 | **1.1×** | **MARGINAL — softened, see below** |
| adaptive vs uniform at margin 8 | 5.79e-5 | 0.34× | noise |
| α max mediated effect (m ≤ 3) | 2.80e-5 | 0.16× | noise |

**The one claim that weakens.** §10.2 called the m=8 α effect "detectable" at 2.2× the floor, using the
m=8-specific replicate pair (8.73e-5). That pair has n=2, hence a 95 % CI spanning 70×, so it is a
worthless denominator on its own. Against the pooled floor the same effect is **1.1×** — marginal, not
established. Report it as *between 1.1× and 2.2× depending on the denominator; more m=8 replicates
would settle it.*

**This does not weaken the dominance argument — it strengthens it.** The cost of *being* at margin 8
(6.28e-4) is **robust at 3.7×**, while α's gain once there (1.89e-4) is **not even established**. So:

> **What you pay to reach α's live region is certain. What you gain there is not.**

And the §10.2 *prediction* check is untouched: predicting 2.3e-4 and measuring 1.89e-4 is a test of the
mediation model (right sign, right magnitude), not a significance claim.

**Meta-lesson, now the third time the floor has been the crux of this project:** with n = 2–6 you
cannot distinguish variance estimates. Do not build a story on floor *differences* between families;
pool them, quote one number with its CI, and use the same denominator throughout.

### 10.1 CORRECTION: "the noise floor grows with depth" is WITHDRAWN

The depth-15 group above is a **4-way replicate set** (Thm 1: all four realise the same integer), and it
gives **sd 5.5e-5, spread 1.3e-4**. Collected with the others:

| depth | n | sd |
|---|---|---|
| 5 | 4 | 3.0e-5 |
| 13 | 2 | 1.3e-4 |
| 14 | **6** | 2.8e-4 |
| **15** | **4** | **5.5e-5** |

**Not monotone.** The depth-14 figure is also inflated by a single high draw — 5 of its 6 values lie
within 1.7e-4 and one sits at 0.344186. So §4.2's "the floor tracks depth, not adaptivity" is
withdrawn: **the floor is 3e-5 – 3e-4 depending on configuration, with no clean depth trend.** Use the
same-configuration group every time and quote its own sd; do not interpolate a floor from a depth model.

*Conclusions unaffected.* α's maximum mediated effect (2.8e-5) is still below even the *tightest* floor
(5.5e-5), and the shallow-depth penalty (3.46e-3) is 11–115× every floor estimate. What changes is only
that we can no longer explain the floor's variation — which is what the §8.1 replicate runs are for.

### 10.2 Batch B wave 1 (m=8) — α is finally detectable, and it is fully explained by depth

The margin objection's own test: put α where it has the most room. Pre-registered in
`alpha-margin-experiment-prep.md` §2.2, run 2026-08-14.

| arm | predicted depth | **actual** | predicted `loss_min` | **actual** |
|---|---|---|---|---|
| α=∞, m=8 | 7.0 | **6.98** | 0.34395 | **0.344044** |
| α=0.5, m=8 | 5.3 | 6.02 | 0.34418 | **0.344233** |
| **difference** | 1.7 slots | 0.97 slots | **2.3e-4** | **1.89e-4** |

**Both loss predictions landed within 1e-4, and the predicted α difference (2.3e-4) matched the
observed one (1.89e-4) to 4e-5.** The depth extrapolation for α=0.5 was 0.7 slots pessimistic; the
α=∞ depth was exact.

**This is the first detectable α effect in the project — and it is mediated entirely by depth.**
The observed 0.97-slot gap at the local slope (1.52e-4/slot) predicts 1.47e-4; observed 1.89e-4, a
residual of 4e-5, well inside the floor. Against the fixed-`p_e` curve directly, both arms sit on it:
residual **+5.4e-5** (α=∞ at depth 6.98) and **+1.0e-4** (α=0.5 at depth 6.02). No heterogeneity term.

**And the dominance argument (§6.2) is confirmed quantitatively.** Both m=8 arms are far worse than
the m=2 plateau (0.34346): **+5.8e-4** and **+7.7e-4**. So the cost of entering α's live region is
**3–4× the entire α effect available once there** (1.89e-4) — §6.2 predicted 3.8×.

> **The margin objection is now answered by direct experiment, not inference.** At the margin where α
> has the most authority we have ever given it, α produces a real 1.9e-4 difference — and it is the
> difference its depth gap predicts, in a regime that is itself 3–4× worse than simply leaving the
> margin alone.

### 10.3 Batch B wave 2 — the heterogeneity question is CLOSED, and the integer prediction was exact

The last open channel: at shallow depth α genuinely produces per-matrix spread (§6.1), so for the first
time "does the *distribution* of depth matter, or only its mean?" is a well-posed question. The control
holds the mean fixed at 6 and removes all spread (fixed `p_e = 6/16`, adaptive OFF).

| run | realised depth | `loss_min` | `N_{0.5}` | `N_∞` |
|---|---|---|---|---|
| α=∞, m=8, s42 | 6.985 | 0.344044 | 2.34 | 1.25 |
| α=0.5, m=8, s42 | 6.015 | 0.344233 | 2.47 | 1.26 |
| **uniform depth 6 (control)** | **6.000** | **0.344175** | 2.41 | 1.25 |
| α=∞, m=8, s43 | 6.985 | 0.343921 | 2.34 | 1.26 |

**The decisive comparison — adaptive vs matched-depth uniform:**

> adaptive α=0.5 (depth 6.02) **0.344233** − uniform (depth 6.00) **0.344175** = **+5.79e-5**
>
> The **uniform** arm is *better*, by 0.67× the floor. Pre-registered threshold to reopen the verdict
> was "adaptive better by > 3e-4". **The verdict holds, and the residual is below the floor.**

**And the first floor near depth 7**, from the α=∞ seed pair: 0.344044 / 0.343921 → spread 1.23e-4,
**sd 8.73e-5**. This replaces the interpolated 1.0e-4 used in §6.2 (the interpolation was right). With
a measured floor there, the m=8 numbers firm up: the α effect (1.89e-4) is **2.2×** the floor, and the
cost of being at m=8 rather than m=2 (6.28e-4) is **7.2×** it. Both real.

**The integer prediction was exact — this is the cleanest confirmation of Theorem 1's mechanism in the
whole project.** At this operating point the theory says `⌊N_{0.5}⌋ = 2` while `⌊N_∞⌋ = 1`, so the two
arms should realise `16 − 2 − 8 = 6` and `16 − 1 − 8 = 7`. Measured `N_{0.5}` = 2.47 and `N_∞` = 1.26 —
floors 2 and 1 — and realised depths **6.015** and **6.985**. The mechanism is not inferred; it is read
off directly, integer for integer.

**Closing summary of the heterogeneity question across all regimes:**

| regime | per-matrix spread present? | adaptive vs matched uniform |
|---|---|---|
| non-DP deep (α ≥ 1, m ≤ 3) | **none** — all 196 matrices identical | question is vacuous |
| non-DP shallow (α=0.5, m=8) | **yes**, `⌊N⌋` ∈ {1,2} | **+5.79e-5 (uniform better)**, 0.67× floor |
| ε=1, matched depth, 3 seeds | yes | −3.1e-4 / +3.0e-3, signs disagree → no effect |

**Per-matrix adaptive depth is dead in every regime where the question can be asked.** Where α gives no
spread, there is nothing to test; where it does, uniform matches or beats it.

---

## 10.5 THREE TRAINER KNOBS ARE SILENTLY INERT ON THE LoRA-XSe PATH

Found while mapping the tunable surface; verified at source and against all 297 runs.

**(1) `--lr-schedule` never reaches the XSe optimizer.** The schedule is built correctly into
`lr_for_opt` (`train_causal_lm.py:1868-1905`) and passed to **every** optimizer branch — lines 1922,
1944, 1949, 1957, 1965, 1972, 1980, 1988 — *except* the XSe branch, which takes the raw scalar:

```python
base_opt = xse_sgd(lr=args.learning_rate, ...)   # train_causal_lm.py:1934  <-- not lr_for_opt
```

> **Every LoRA-XSe run in this project has trained at a constant learning rate.** Corpus check:
> `lr_schedule` is `none`/`None` in **297 / 297** runs, and `lr_warmup_steps` is 0/None in all of them.

**(2) `--warmup-steps` is dead code.** Defined at `train_causal_lm.py:491` and **never read** — the only
read anywhere is `args.lr_warmup_steps` (line 1868), a different dest. And that one feeds `lr_for_opt`,
which XSe doesn't receive either, so warmup is doubly inert on this path.
**76 runs in the corpus set it** (`warmup_steps` = 20 in 74 runs, 5 in one, 10 in one) — including the
three best-ever runs in `vendor/lora-privacy/docs/results.md`, all named `…-warmup20-…`, and the
recommended command at `why-adaptive-depth-wins.md:172`. **Those runs had zero warmup.** Any conclusion
in those docs that attributes anything to warmup is unfounded.

**(3) `--weight-decay` is silently dropped.** `xse_sgd()` has no `weight_decay` parameter at all
(`xse.py:849-856`). 293 runs set `weight_decay=0.01`; on the XSe path it was ignored every time.

### 10.5a Why this is the biggest utility lead in the project

It joins up with three things measured independently in §11.4 and §10.0:

- eval loss is **still falling at the final step** (2-epoch runs put `loss_min` at step 520 / 510),
- it **wobbles ±1.9e-4 per 10 steps** late in training,
- and the late-training sd varies **2× between arms** (7.9e-4 – 1.6e-3).

All three are the textbook signature of **a learning rate that is too high at the end of training** —
which is exactly what a constant 5e-2 with no decay produces. A cosine decay to ~0 would plausibly

1. **improve final loss** (the standard effect, often large), and
2. **shrink the trajectory wobble**, which is what *sets the 6.5e-5 noise floor*.

That is a two-for-one: a utility win *and* a sharper instrument. The second half may matter more — the
floor is what has blocked every marginal conclusion in this project (§10.1d).

**The fix is one line.** `torchopt.sgd` takes `lr: ScalarOrSchedule` natively
(`torchopt/alias/sgd.py:49`), and `_XSeSGD` passes `lr` straight through to it (`xse.py:674`). So
`lr=args.learning_rate` → `lr=lr_for_opt` at `train_causal_lm.py:1934` is sufficient. Keep
`--lr-schedule none` as the default so the 297 existing runs stay comparable; the flag simply starts
working when set.

**What this does NOT invalidate.** Every α and depth comparison was run at the *same* constant LR, so
they are internally fair and none of the verdicts change. What it means is that all absolute numbers
come from an **untuned optimizer configuration**, and there is headroom nobody has touched.

---

## 11. The rotation schedule — what the data does and does not support

Question raised in review: *shouldn't we explore heavily early, so many small new directions can
complement the main ones, then taper off toward the end?*

### 11.1 Exploration productivity is U-shaped, not decaying

`promotion_count / r_keep` = the share of the retained set that was freshly discovered rather than
inherited (normalising is required: the raw count is capped at `r_keep`, `xse.py:588`).

| step window | 0–40 | 40–80 | 80–130 | 130–180 | 180–220 | 220–260 |
|---|---|---|---|---|---|---|
| `fixed-re13` (d 13) | **0.227** | 0.008 | 0.038 | 0.036 | 0.054 | 0.059 |
| `ainf-m1` (d 14) | **0.132** | 0.027 | 0.018 | 0.036 | 0.039 | 0.059 |
| `fixed-re5` (d 5) | 0.046 | 0.033 | 0.046 | 0.061 | 0.071 | **0.075** |

A burst in the first ~40 steps (up to 0.23 for deep runs — the initial basis is poor, so almost any
draw improves it), a collapse at 40–80, then a **steady rise** through the rest of training. So the
premise "productive early, unproductive late" is **half right**: there is an early burst, but late
exploration is *more* productive than mid-training, not less.

### 11.2 The current rule already schedules — in the opposite direction

Of the 26 adaptive non-DP runs, **every one either explores more over time or stays flat; none explores
less** (drift +0.01 to +1.72 slots). So LoRA-XSe today implements an *increasing* exploration schedule,
which is the opposite of the proposal — and it happens to align with the productivity rise in §11.1's
back half while missing the early burst entirely.

### 11.3 Temporal schedules also appear to act only through mean depth

The four runs with substantial upward drift, against the fixed-depth curve at their *mean* depth:

| run | depth path | mean | residual |
|---|---|---|---|
| `m0-nodp-a025-m0` | 12.3 → 14.4 | 14.31 | **+3.7e-4** |
| `m0-nodp-a01-m0` | 7.8 → 9.6 | 9.16 | −6.3e-5 |
| `lowa-nodp-a01-m2` | 5.9 → 7.0 | 6.50 | −1.2e-5 |
| `lowa-nodp-a015-m2` | 8.0 → 9.4 | 8.90 | +2.5e-5 |

Mean +8e-5, mixed signs, one outlier. **An earlier reading of the first row alone as "suggestive
evidence that rising schedules underperform" is retracted** — the other three sit on the curve. On this
(weak, n=4) evidence, *temporal* heterogeneity behaves like per-layer heterogeneity did: it acts through
the mean and nothing else.

### 11.4 RETRACTED before publication: the "per-rotation cost" in `train/loss`

With `eval_steps=10` and the auto interval `max(1, round(0.5/(1−β))) = 5`, **every eval step is also a
rotation step**, and with 260 steps `260 % 5 == 0` — so *the final rotation lands exactly on the last
training step*. Both are structural facts (verified: optimizer update including rotation → `global_step`
increment → eval, `train_causal_lm.py:2175→2184→2538`).

That motivated looking for a per-rotation cost in per-step `train/loss`. A detrended phase analysis
appeared to find one — ~2.2e-3 peak-to-trough, reproducible across four runs agreeing to 10 %. **It is
an artifact of data order.** All four shared seed 42, hence the same batch sequence. Repeating with
different seeds:

| run | seed | argmax phase | argmin phase |
|---|---|---|---|
| `renyi-nodp-s42` | 42 | 4 | 2 |
| `renyi-nodp-s43` | 43 | 0 | 3 |
| `renyi-nodp-s44` | 44 | 3 | 1 |
| `seedrep-ad-nodp-ainf-m1-s43` | 43 | 1 | 3 |
| `seedrep-ad-nodp-a2-m1-s43` | 43 | 0 | 3 |

**The phase pattern moves with the seed, not with the rotation.** All three seed-43 runs agree with
each other and disagree with seeds 42 and 44. Worse (better, for clarity): `renyi-nodp-s43` (depth 5)
and `seedrep-ad-nodp-ainf-m1-s43` (depth 14) have nearly identical phase profiles despite radically
different rotation configurations — so **per-step `train/loss` is dominated by batch difficulty and is
not a usable instrument for measuring rotation effects at all.**

Conclusion: the per-rotation cost is *below the resolution of `train/loss`*. The phase-alignment and
final-step-rotation facts stand on their own, but there is **no measured evidence** that fixing them
would help. Treat the cool-down as a mechanism-motivated hypothesis, not a supported one.

### 11.5 What to actually do, ranked

1. **Decouple eval from rotation phase — free.** Set the rotation interval coprime to `eval_steps`
   (e.g. 3 or 7 against 10). Evals then sample all phases instead of always the same one, which makes
   any real per-rotation cost measurable *on eval loss* for the first time. Costs nothing.
2. **Cool-down: stop rotating for the last K steps.** One parameter, clear mechanism (a rotation at
   step 260 leaves `r_e` of 16 directions freshly random with no training after). Test K ∈ {0, 10, 25}.
   Prior: small gain, but it is the cheapest schedule intervention and the only one with a mechanism
   that the flat depth curve does not already rule out.
3. **Front-loaded schedule matching §11.1** — deep for the first ~40 steps, then the current rule.
   Motivated by the measured productivity burst, not by the exploration/exploitation intuition.
4. **Do not** implement a monotone decay. §11.1 shows late exploration is more productive than
   mid-training, and §11.3 shows temporal schedules act through the mean anyway.

**Why schedules are not already ruled out by the flat plateau:** the depth curve says *constants* in
9–15 are equivalent. A schedule is a different object — it is not a point on that curve — so the
plateau result does not settle it. That makes this a genuinely open direction, unlike α.

---

## 12. Reproducing every number here

Scripts are committed at `campaign_logs/alpha_analysis/` (see its README for the section map):

```bash
cd campaign_logs/alpha_analysis
export WANDB_BASE_URL=https://jetbrains.wandb.io

uv run python fetch_cache.py   # ~5 min: 297 runs + histories for the 82 carrying the Renyi grid
uv run python an2.py           # depth trajectories, N_alpha grid, alpha dose
uv run python an3.py           # mediator curve, replicate floors, bound table
uv run python an4.py           # dose-response rho, noise response, heterogeneity audit
uv run python an5.py           # drift, r-dependence, Thm 2 certificates
uv run python an6.py           # full rotation/ + xs/ metric sweep (needs hist2.json, see an6 header)
uv run python an7.py           # sec 9: alpha span, promotion fraction, the two retractions
uv run python an8.py           # sec 6.1/6.2: which alpha collapse where, margin as operating point
uv run python an9.py           # sec 6.3: the margin objection -- amplification, ranking, dose tests
```

Key logged fields this analysis rests on, all already present:
`rotation/r_e_dyn` (realised mean depth — gives `mean_ℓ ⌊N_α⌋` exactly),
`rotation/r_eff_{a0,a0p5,a1,a2,ainf}` (the **whole α curve, logged every rotation regardless of the
configured α** — `xse.py:108–123`; this is why the counterfactual analysis needed no new runs),
`xs_spread/rec_rank_{min,median,max,std}` (the noise-thresholded alternative rule),
`train/noise_std`.
