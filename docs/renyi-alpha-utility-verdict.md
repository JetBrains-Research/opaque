# Does the Rényi order α affect utility? — audit, metrics, and the decisive test

> **SUPERSEDED 2026-08-14 by `docs/renyi-alpha-theory-final.md`** for everything about α and depth
> (§§6–12 here). Keep this file for the code-defect audit (§2) and the metrics/statistics plan (§3),
> which still stand. The α sections chased a noise floor through four revisions; the new doc replaces
> the statistical argument with an identifiability one — α enters the algorithm through a single
> integer `⌊N_α⌋`, which is provably 1 for all α ≥ 2 in **36/36** non-DP runs, so those arms were
> never separable by any experiment. §1's observation that "the non-DP α arms are arithmetically
> vacuous" was the right instinct; the new doc turns it into a theorem with certificates.

**Date:** 2026-08-11. Sources: 3 independent agents (adversarial theory review, full W&B
data audit over 256 runs, metrics/literature research). This supersedes looser claims
in `renyi-campaign-results.md` and Part VIII of the theory doc.

## 1. Verdict on the existing evidence: NO α→utility signal, and two arms were vacuous

- **No ε=1 α-sweep has ever been run.** Every α comparison in the project is at ε=3 or
  non-DP. The only ε=1 runs touching α are W0-allocation (α=∞), which *lose* to uniform
  (0.36269/0.36346 vs 0.36175 loss; 0.590 vs 0.675 MBPP+). The pre-registered gate
  (`renyi-utility-experiments-plan.md` §7.3) is **unrun**.
- **The non-DP α arms are arithmetically vacuous.** Non-DP `r_eff` ∈ [1.1, 2.1] for every
  α, and `r_e = r − int(r_eff) − m` collapses them to the same integer: at m=1 the layer-mean
  `r_e` is 13.89 / 13.98 / 14.00 / 14.00 for α = 0.5/1/2/∞. `renyi-ad-nodp-{a1,a2,ainf}-m1`
  are **the same run three times**. Their flatness is guaranteed by truncation, not physics —
  so the docs' description of non-DP as "the clean control" is WRONG and is retracted here.
- **Under DP the dial is live but the outcome is inert.** At ε=3, r=16, α 0.5→∞ moves mean
  exploration depth **7.81 → 12.49 of 16** (~1.8×) while eval/loss moves **0.00017** —
  against a **0.00037** spread between three *computationally identical* runs.
- **Mediator gain is ~1e-4/slot**, from three independent estimates (margin sweep;
  adaptive on/off; DP α-sweep). α's full reach ≈ 4 slots ⇒ ceiling ≈ 4e-4 ≈ the noise floor.
- **α ≈ a uniform shift in `r_e`** (sd 0.47–0.91 vs mean 1.3–4.2) ⇒ α is a re-parametrization
  of the margin `m`, already swept and flat.
- **The only real α effect is the degenerate endpoint:** α=0 ⇒ `r_eff=r` ⇒ `r_e=1` ⇒ no
  exploration ⇒ 0.34630 vs ~0.3449 elsewhere. "Exploration helps", not an α-ordering.
- **Two reported positives died with added seeds:** α=∞ alloc ≥ uniform on MBPP+ is 3/4 (not
  3/3) and loses on loss 4/4; the α=1 arms of that follow-up never ran.

## 2. Code defects found (must fix before any further α claim)

| # | defect | location | consequence |
|---|---|---|---|
| 1 | doc says `ceil`, code does `int()` (floor) | `xse.py:39` vs `:454` | every run explored 1 slot deeper than the published formula |
| 2 | `N_α` computed on **core R**, but the keep/explore basis is sliced from the **momentum** SVD | `xse.py:425` vs `:401,482` | the spiked/MP theory is about the noised momentum; the estimand is measured on the wrong object |
| 3 | `min(r-1, ·)` is dead code (Prop 2 ⇒ `N_α ≥ 1`) | `xse.py:454` | none, but misleading |
| 4 | `allocate_ranks` repair loop dumps the whole rounding deficit on the 1–2 lowest-weight layers | `allocation.py:112-116` | penalises the wider-spread arm (α=1) — a non-α explanation for "α=∞ > α=1" |
| 5 | `config.lora_xs_alloc_alpha` reads `inf` on **all** allocation runs incl. α=1 arms | `train_causal_lm.py:1277` | grouping by that key silently merges arms |
| 6 | `eval/loss_min` is a min over checkpoints | `train_causal_lm.py:2493` | downward-biased order statistic, bias grows with trajectory noise (which differs by arm) |

## 3. Metrics to adopt (from the literature review)

**Our loss metric is not the bottleneck** — a 0.001 effect vs seed sd 2–5e-4 is 2–5σ. What was
missing is per-example variance, an interpretable unit, and paired seeds.

1. **Per-example bits-per-byte (BPB)** on held-out KStack (NLL ÷ UTF-8 bytes, per file).
   *Signal and Noise* (arXiv 2508.13144): SNR **5.5→42.0** on code, **2.0→41.8** on MBPP vs
   pass@1. Primary metric. Bootstrap over (example, seed).
2. **`JetBrains/Kotlin_HumanEval`** (161 tasks, MXEval) — in-domain pass@1, plus decomposed
   **compile / syntax / runtime error rates** (they move independently of pass@1).
3. **KStack FIM next-line exact match + edit similarity** — JetBrains' own protocol
   (Kotlin ML Pack, arXiv 2405.19250: 630-example holdout, first-line EM, built explicitly to
   detect fine-tuning degradation). Scale 630 → ~10k (binomial sd 1.95pp → 0.49pp).
4. **Python HumanEval+/MBPP+ relabelled as a RETENTION metric.** Base model = 0.680 MBPP+ and
   every fine-tune ≤ it; report as catastrophic-forgetting, not utility.
5. Free variance reduction: score the **average of the last 5 checkpoints**.

**Statistics:** paired seeds (common random numbers) → per-seed deltas → BCa bootstrap CI +
sign-flip permutation; claim only if CI>0 AND p<0.05. **k≥6** (nonparametric floor at k=5 is
p=0.0625). One **monotonic-trend test** across α, not pairwise. Report **TOST equivalence**
if null. Precedent for a published null in this exact area: arXiv 2512.11482 (DP code LMs)
headlines paired-Wilcoxon p>0.05.

## 4. The decisive experiment

**E1 — mediator dose–response at ε=1.** Fixed `r_e ∈ {2,4,6,8,10,12}` via `p_e`, adaptive
depth OFF, uniform r=16 (no rank variation ⇒ no scaling confound), 6+ **paired** seeds.
Metric: BPB primary, final loss + last-5 average secondary. OLS trend with seed fixed effects
+ **TOST** against δ=5e-4. **If |β|·4 < 5e-4, the α→utility hypothesis is dead by mediation
for every α at every ε** — publish as an equivalence result.

**E2 — only if E1's slope ≠ 0.** α ∈ {0.5, ∞} at ε ∈ {1, 0.5}, **at r=64** (r=16 non-DP is
arithmetically inert), plus a **matched-depth control** replaying α=∞'s mean depth uniformly,
to separate "α picks better per-layer depths" from "α just explores deeper". Fix defects
1, 2, 6 first. 10 paired seeds.

**Prior probabilities (adversarial review):** effect ≥3e-4 at ε=1: **12%**; surviving the
matched-depth control: **4%**.

## 5. What remains true and publishable regardless

- Rotation (LoRA-XSe) beats base LoRA-XS by **0.0151** at identical params/ε — ~75× the seed
  sd, reproduced across campaigns. This is the method result.
- The **DP rank-inflation mechanism** (low-α effective rank inflated ~3.8× vs non-DP, tight
  CIs) is solid *as a statement about measurement* — with the caveat of defect 2 (measured on
  R, not the momentum).
- The **two measurement traps**: out-of-domain pass@1 measures forgetting; `alpha/r` scaling
  silently perturbs per-layer effective LR by up to 4× in any variable-rank method.

---

## 6. RESOLVED (non-DP): α cannot produce a detectable utility effect — with a bound

Experiment (2026-08-12, 7 runs, non-DP, seed 42, r=16, 1 epoch, image `7e97389`):
`r_e` fixed uniformly at {1,5,9,13} with adaptive depth OFF, plus adaptive depth ON
at α ∈ {0.25, 0.5, ∞}. α enters training only through the integer `r_e`, so
measuring `L(r_e)` + composing with the measured α→`r_e` map answers the α question
for all α without an α sweep.

### 6.1 The mediator curve — depth matters, and saturates

| `r_e` | eval/loss | marginal slope |
|---|---|---|
| 1 | 0.34700 | — |
| 5 | 0.34429 | −6.8e-4 /slot |
| 9 | 0.34368 | −1.5e-4 /slot |
| 13 | 0.34354 | −3.5e-5 /slot |

Total 1→13 = **−0.00346**; slope falls **19×** across the range. Exploration depth
is the dominant mechanism and it saturates by `r_e`≈13.

### 6.2 Measured noise floor (free internal control)

α=∞ yields `r_e_dyn` = **13.00 uniformly** — i.e. it *is* the fixed-13 configuration.
Run separately, the two scored 0.34344 vs 0.34354 ⇒ **run-to-run floor = 1.0e-4.**
(Supersedes the earlier 3.7e-4 estimate.)

### 6.3 Heterogeneity contributes nothing

| adaptive arm | depth | loss | uniform curve @ depth | Δ |
|---|---|---|---|---|
| α=0.25 | 11.49 | 0.34346 | 0.34359 | −1.3e-4 |
| α=0.5 | 12.68 | 0.34357 | 0.34355 | +0.2e-4 |
| α=∞ | 13.00 | 0.34344 | 0.34354 | −1.0e-4 |

|Δ| ≤ 1.3e-4 with **inconsistent sign** ⇒ no evidence that per-layer tailoring beats
a matched uniform depth. Any effect is bounded by ~1.3e-4 (n=1/arm; a real claim
either way needs seeds).

### 6.4 The bound

α's reachable depth range in non-DP is **11.49–13.00 (1.51 slots)**; the tail slope
there is **3.5e-5/slot**; hence

> **max possible α effect = 5.3e-5 = 0.53× the measured noise floor.**

So α is not merely "flat" in non-DP — it is *provably undetectable*, because its
whole operating range lies in the saturated tail of the depth curve. Reaching the
steep region (`r_e` ≲ 9) would require α ≲ 0.1; α=0.25 only reaches 11.49.

### 6.5 What this means for the paper

- **Keep:** exploration depth is a real, large mechanism (−0.00346, ~35× the floor),
  and rotation itself remains the headline (−0.0151 vs frozen-basis LoRA-XS).
- **Reframe α:** it is the *derivation* of a depth rule that works and needs no
  tuning, not a performance lever. The insensitivity is now quantified, not assumed.
- **Retire:** per-layer heterogeneity as a claimed benefit (non-DP).
- **Caveat (corrected):** the earlier claim that three configs reported *identical*
  loss was a ROUNDING ARTIFACT of our own reporting — full precision gives
  0.3434580 / 0.3434635 / 0.3434576, all distinct. Also `loss_min == loss_final` in
  4 of 5 runs, so order-statistic bias was minor here. The real limitation is
  statistical power: one scalar per run cannot resolve 1e-5–1e-4 effects against a
  1.0e-4 nondeterminism floor. Fix = per-example BPB + paired bootstrap/TOST; see
  `renyi-story-and-evaluation.md` Part VI.
- **Open:** ε=1 (high noise) is untested; there α's reachable depth range is much
  wider (7.8–12.5 at ε=3), so the same bound argument would give a larger ceiling.

---

## 7. non-DP COMPLETE (11 runs): α is a floor constraint, not a tuning knob

Low-α sweep (α ∈ {0.05,0.1,0.15,0.2}, non-DP, m=2, seed 42) added to §6, giving
7 adaptive arms spanning realized depth 3.1 → 13.0.

### 7.1 α → loss (the practical question)

| α | realized `r_e` | loss | vs best |
|---|---|---|---|
| 0.05 | 3.13 | 0.34537 | +0.00193 |
| 0.1 | 7.01 | 0.34405 | +0.00061 |
| 0.15 | 9.38 | 0.34372 | +0.00028 |
| 0.2 | 10.78 | 0.34353 | +0.00009 |
| 0.25 | 11.49 | 0.34346 | +0.00002 |
| 0.5 | 12.68 | 0.34357 | +0.00013 |
| ∞ | 13.00 | 0.34344 | — |

- **Full span 0.00193 = 19× the noise floor** ⇒ α is NOT globally inert.
- **Plateau α ≥ 0.2: range 0.00013 = 1.3× floor** ⇒ Shannon, α=2, α=∞ interchangeable.
- Monotone: lower α ⇒ shallower ⇒ worse. **No interior optimum.**

**CORRECTION to §6.4.** The "max α effect = 0.53× floor" bound was computed over
α ≥ 0.25 only (reachable depth 11.5–13.0). It holds *on the plateau*; it does not
hold once α < 0.2, where the reachable depth extends to 3.1 and the span reaches 19×.

### 7.2 Heterogeneity: still nothing (now n=7 arms)

Adaptive arms vs the fixed-depth curve at their realized depth:
residual **mean −5.3e-5, |max| 1.87e-4, 4/7 better** — scatter at the noise floor
with no consistent sign. **Per-matrix tailoring adds nothing beyond the average
depth it selects.**

### 7.3 The complete non-DP mechanism

> α → average exploration depth → loss, and nothing else. The depth curve is
> monotone-saturating, so any α ≥ 0.2 is equivalent, and below that α monotonically
> destroys quality. **α is a constraint to respect, not a parameter to tune.**

Paper framing: state both numbers — the plateau width (1.3× floor, hence no tuning
needed) and the penalty for leaving it (19× floor at α=0.05, hence the rule is not
vacuous). The derived rule (any α ≥ 0.2, e.g. Shannon or the stable rank) is
*correct by construction* rather than tuned.

---

## 8. eps=1 — RETRACTED (analysis used unfinished runs); see §9

> **RETRACTION (2026-08-13).** The paired analysis below was computed while 3 of the
> 4 seed runs were STILL TRAINING (26 of 27 eval points). Their values moved by
> ~1e-3 afterwards — larger than the ~6e-4 effect under test — so every number and
> the "no effect" conclusion in this section are INVALID. The methodological point
> in §8.2 (shared-seed fan-out is not replication) still stands on its own.
> Corrected analysis in §9 once all runs finish.

### 8.1 The decisive paired test

adaptive(α=1, r_e≈9.3) − fixed(r_e=9.00), eps=1, matched depth, `last5_avg`:

| seed | adaptive | fixed | delta |
|---|---|---|---|
| 42 | 0.34805 | 0.34869 | −6.4e-4 |
| 43 | 0.34905 | 0.34864 | **+4.1e-4** |
| 44 | 0.34970 | 0.34974 | −0.4e-4 |

**mean −9.1e-5, sd 5.3e-4, 2/3 negative, paired t = −0.30** (need |t|>4.30 at n=3).
**No effect.**

### 8.2 Why the n=1 result looked so convincing — a trap to avoid

The earlier "3/3 adaptive arms below the curve (mean −6.0e-4)" used α ∈ {0.5,1,∞}
**all at seed 42**. Those are not three independent confirmations; they are ONE seed's
draw measured at three depths. A shared-seed fan-out mimics replication. Only the
paired multi-seed test is diagnostic, and it returns zero.

### 8.3 Noise floors, measured

| regime | paired-difference noise |
|---|---|
| non-DP | ~1.0e-4 (identical-config control) |
| **eps=1** | **sd 5.3e-4** (paired, n=3) |

At eps=1 any single-seed effect below ~1e-3 is uninterpretable. This retroactively
explains several earlier "signals" in DP runs.

### 8.4 Final cross-regime verdict on per-matrix adaptivity

| regime | residual vs matched-depth control | verdict |
|---|---|---|
| non-DP (7 arms) | mean −0.5e-4, 4/7 negative | no effect |
| eps=1 (paired, 3 seeds) | mean −0.9e-4, sd 5.3e-4, t=−0.30 | **no effect** |

**Per-matrix adaptive depth provides no measurable utility benefit over a matched
uniform depth, in either regime.** Combined with §7: α acts only through average
depth, and is a floor constraint rather than a tuning knob.

### 8.5 What still stands

- **Rotation:** −0.0151 vs frozen-basis LoRA-XS (~150× the non-DP floor). Unaffected.
- **Depth matters in non-DP** (−0.0035, saturating); at eps=1 the uniform depth curve
  is flat/non-monotone (0.34839 / 0.34869 / 0.34844 at r_e=5/9/13), i.e. under heavy
  noise refreshing more stops helping too.
- **DP rank inflation** (~3.8× at eps=3; r_eff 1.35→3.61 at eps=1, matched depth) —
  a measurement result, independent of utility.

---

## 9. eps=1 FINAL (all 6 runs verified `finished`, 27 evals each)

Paired: adaptive(α=1, r_e≈9.3) − fixed(r_e=9.00), matched depth, eps=1, seeds 42/43/44.

| seed | Δ `loss_min` | Δ `last5` |
|---|---|---|
| 42 | −8.3e-4 | −6.4e-4 |
| 43 | −2.8e-4 | +7.9e-3 |
| 44 | +1.7e-4 | +1.7e-3 |
| **mean** | **−3.1e-4** (2/3 neg, t=−1.09) | **+3.0e-3** (1/3 neg, t=+1.17) |

n=3 needs |t|>4.30 for p<0.05. **Neither metric is significant, and they disagree in
SIGN** — the signature of no effect rather than a weak one.

**Verdict: the eps=1 heterogeneity effect does NOT replicate.** (Same conclusion as
the retracted §8, but on complete data; §8's numbers were computed mid-training and
were wrong.)

### 9.1 Metric choice at eps=1

`loss_min` and `last5` diverge badly here because DP trajectories destabilise late.
`loss_min` is the operationally correct primary: the trainer runs
`--restore-best-checkpoint` by default, so **the deployed model IS the best
checkpoint**. `last5` is reported as a stability indicator, not a utility metric.

### 9.2 Side observation: adaptivity costs late-training stability

Mean gap between `loss_min` and `last5` (bigger = more late blow-up):

| arm | gap |
|---|---|
| adaptive α=1 | **0.0052** (0.0011 / 0.0122 / 0.0024) |
| fixed r_e=9 | 0.0019 (0.0009 / 0.0039 / 0.0008) |

Adaptive arms destabilise ~2.7× more late in training at eps=1. Plausible mechanism:
re-randomising directions from an increasingly noise-dominated spectrum. n=3 and
one run dominates ⇒ hypothesis, not a finding — but it is a *cost* of adaptivity, and
DP practitioners care about stability.

### 9.3 FINAL CROSS-REGIME VERDICT

| regime | per-matrix adaptivity vs matched uniform depth |
|---|---|
| non-DP (7 arms, 11 runs) | mean −0.5e-4, 4/7 neg → **no effect** |
| eps=1 (paired, 3 seeds) | loss_min −3.1e-4 (t=−1.09) / last5 +3.0e-3 (t=+1.17), signs disagree → **no effect** |

**α affects utility only through average exploration depth, in both regimes. It is a
floor constraint (stay above ~0.2), not a tuning knob, and per-matrix tailoring adds
nothing measurable.**

### 9.4 Process note

This section exists because §8 was published from runs that were still training
(26/27 evals). Lesson: **verify `state == "finished"` explicitly; do not infer
completion from eval count.** Four candidate effects in this project have now died
under replication (non-DP heterogeneity, α=∞ allocation on MBPP+, the retention
effect, eps=1 heterogeneity) — plus one that died from being measured too early.

---

## 10. The depth curve, consolidated (margin & m=0 sweeps)

Motivation: in non-DP the rule keeps `⌊N_α⌋ + m` directions, and with `⌊N_α⌋`=1 the
constant margin supplies **two thirds** of that count. So `m` — held at the library
default 2 for every earlier experiment — was the dominant term, and α the minority
one. m=0 had never been tested.

### 10.1 Algebra (verified empirically)

`r_e = r − ⌊N_α⌋ − m` is **additively separable** in α and m, so
`r_e(α₁,m) − r_e(α₂,m)` is independent of m: the margin shifts every depth by a
constant and **cannot alter differences between α values**.
Confirmed: at m=0, α=0.5 and α=2 both realized depth **15.00**, and scored
0.34364 vs 0.34368 (agreeing to 0.4e-4). Removing the margin does NOT decollapse α.

### 10.2 The consolidated depth→loss curve (non-DP, seed 42, all runs verified finished)

| realized depth | loss | lever |
|---|---|---|
| 1.00 | 0.34700 | fixed p_e |
| 5.00 | 0.34429 | fixed p_e |
| 8.5 | (pending) | α=1, m=6 |
| 9.00 | 0.34368 | fixed p_e |
| 9.56 | 0.34361 | α=0.1, m=0 |
| 10.6 | (pending) | α=1, m=4 |
| **12.7–12.9** | **0.34346** | α=0.5/1, m=2 ← minimum |
| 13.00 | 0.34354 | fixed p_e |
| 14.0 | 0.34371 | α=1, m=1 |
| 15.00 | 0.34364 / 0.34368 | α=0.5/2, m=0 |

- **Shallow interior optimum near depth ~12.8 (≈80% of r).**
- Degradation: **+1.8e-4 at depth 15**, **+1.5e-4 at depth 9** — i.e. everything from
  9 to 15 lies within 2e-4 (≈2× the noise floor). A **forgiving plateau**.
- Excluded as a curve point: α=0.25/m=0 (loss 0.34399). Its depth *drifted*
  12.3→14.4 during training (at m=0 the feedback loop — deeper refresh concentrates
  R, lowering `N_α`, deepening further — runs away), so it is a time-varying
  schedule, not a fixed depth.

### 10.3 The three levers are interchangeable

At matched depth, fixed `p_e`, margin, and α give the same loss:
depth 9.00 → 0.34368 vs 9.56 → 0.34361 (Δ 0.7e-4); depth 15 via α=0.5 vs α=2 →
0.34364 vs 0.34368 (Δ 0.4e-4). ⇒ state the recommendation as a **kept-count rule**,
not in terms of any single parameter.

### 10.4 Answer to "why margin = 2 in every experiment?"

It is the library default (documented as a buffer against discarding
borderline-useful directions), and a July sweep over m ∈ {1,2,3} selected it. m=0 was
untested — a real gap, now closed:

> **m=2 beats m=0 by ~1.8e-4** because it places depth at ~13 rather than 15.
> Justified, but modest (≈2× noise floor), and for a mundane reason: the margin is an
> additive offset that happens to land near the optimum. It has no independent
> mechanism.

### 10.5 Crash accounting (7 crashed runs, all discarded)

marg-m{0,4,6}, marg2-m{4,6}, m0-ainf and others crashed at **non-deterministic**
steps (122–166 of 260). Correlated with batch co-location on one node
(`...-g9bk`) ⇒ likely **node-level memory pressure from packing multiple 200Gi pods**,
made more likely by over-subscribing the namespace quota (10 runs against ~5 slots).
Every partial run was discarded unread: at step ~157 they showed 0.34509/0.34555,
which would have implied "deeper margins are much worse" — a pure artifact of 40%
less training. **Rule adopted: check `state == "finished"` AND step count before
reading any number; cap concurrency at ~3 pods.**

---

## 11. FINAL non-DP depth curve (29 verified runs) — RETRACTS §10.2/§10.4

Assembled every non-DP run with `state == "finished"` and ≥25 evals (29 runs), keyed
on realized depth `rotation/r_e_dyn`.

### 11.1 Within-depth scatter is 2–4e-4, not 1e-4

Three runs at **identical depth 14.00**: 0.34362 / 0.34371 / 0.34399 ⇒ spread
**3.7e-4**. Seven runs at depth 12.67–12.99: 0.34344–0.34362 ⇒ spread 1.8e-4.
**The true run-to-run scatter at fixed depth is 2–4e-4.** The earlier 1.0e-4 estimate
(from the α=∞ / fixed-13 pair) was one lucky pair, not the distribution.

### 11.2 What survives

| effect | magnitude | verdict |
|---|---|---|
| shallow depth (≤5) penalty vs best | **+3.6e-3** | **ROBUST (~10× scatter)** |
| plateau spread depth 9→15 (24 runs) | 5.6e-4 | ≈ scatter ⇒ flat |
| "interior optimum at ~12.8" | 1.8e-4 | **RETRACTED — below scatter** |
| "m=2 beats m=0 by 1.8e-4" | 1.8e-4 | **RETRACTED — below scatter** |

### 11.3 The honest rule

> **Refresh at least ~55% of the rank (depth ≥ 9 of 16). Above that it does not
> matter — depth 9 through 15 is one flat plateau within run-to-run noise. Going
> shallow is the only mistake: depth ≤5 costs ~3.6e-3, roughly 10× scatter and a
> quarter of the entire rotation benefit.**

So α, the margin, and fixed `p_e` are three interchangeable ways to land somewhere in
that plateau, and **none of them needs tuning** — the only requirement is not landing
below ~9. This also answers "why margin=2": any m ∈ {0,1,2,3} puts depth in 12–15,
all equivalent. m=2 is fine, and so is m=0.

### 11.4 Correction record

I claimed an interior optimum twice (§10.2, and a "4.5× turnover" before that) on
1.8–4.5e-4 differences, without first measuring the scatter at fixed depth. The
depth-14 triplet was already in the data and refutes both. Lesson: **measure
replicate scatter at a fixed setting before interpreting differences between
settings** — the noise floor must come from the same distribution as the comparison.

---

## 12. §11 IS SUPERSEDED — its noise floor was invalid

§11 declared the depth 9–15 plateau "flat within scatter" using a **3.7e-4 within-depth
scatter** computed from runs at the same depth but **different α**. That is a contrast
between configurations, not a replicate scatter — so it silently assumed α has no
effect in order to conclude α has no effect. Circular.

The correct same-config replicate floor in non-DP is **2.3e-5** (`renyi-nodp-s42/43/44`,
plus a duplicate `renyi-nodp-s44` pair differing by 3e-5). Against that floor:

- the depth 9→15 plateau spread (5.6e-4) is **~24×** the floor, not noise;
- three configs at **identical mean depth 14.00** differing only in α span **5.7e-4 ≈ 20×**
  the floor (α=∞ 0.34362, α=1 0.34408, α=2 0.34419).

So the α question is **re-opened**, not closed. Mechanistically this is consistent with
the mediation argument, which constrains α to act through the integer **vector**
`(r_e,ℓ)_ℓ` across 196 matrices — *not* through its mean. Equal means with different
distributions are permitted.

**Caveat that prevents claiming it:** the whole non-DP depth curve is seed 42, and the
2.3e-5 floor comes from a *non-adaptive, fixed-p_e, depth-5* family. Adaptive-depth runs
may be intrinsically noisier (realised depth depends on random rotation draws). Seed
replicates of an adaptive config at matched depth are required, and are running
(`seedrep-ad-nodp-{ainf,a2}-m1-s43/44`).

What survives from §11 unchanged: the **shallow-depth penalty** (depth 1 → +3.6e-3,
≈155× the corrected floor). That was never in doubt.

See `docs/renyi-status-summary.md` for the current consolidated position.
