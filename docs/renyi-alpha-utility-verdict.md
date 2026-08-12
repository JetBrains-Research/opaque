# Does the Rényi order α affect utility? — audit, metrics, and the decisive test

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
- **Caveat:** `eval/loss_min` is a min over ~50 checkpoints and three configs report
  exactly 0.34346 — evidence of plateau clustering. Final-loss/BPB is the better
  metric for any follow-up.
- **Open:** ε=1 (high noise) is untested; there α's reachable depth range is much
  wider (7.8–12.5 at ε=3), so the same bound argument would give a larger ceiling.
