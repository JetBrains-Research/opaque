# Prepared, not submitted: full m=0 sweep + a small m=8 probe

**Status: BATCH A SUBMITTED 2026-08-14** (5 runs, image `...-7e97389`, tracked by
`campaign_logs/m0b/wait_m0b.py`). Batch B queued behind it. **Phase 0 of §2.3 is already DONE and
PASSED with no GPU** — see §8.3 of the theory doc.
Companion to `docs/renyi-alpha-theory-final.md` §6.1–6.3.

Two batches. **Predictions are written down below before submission** — this project has retracted
four conclusions that were formed after seeing the numbers, so every arm here carries a
pre-registered expectation and a statement of what would falsify it.

---

## 0. Do this first — it is free and both batches need it

Add two things to the per-layer diagnostic tuple at `xse.py:803–815`:

1. `"singular_values_top"` — already computed at `xse.py:617`, currently thrown away. Without it we
   cannot say where the momentum spectrum's real cut point is (§9.5 of the theory doc).
2. per-matrix **min/max of `r_e`** — needed to detect **clamping**, which is a live risk in both
   batches (§1.1, §2.1). Currently only the layer *mean* is logged, which hides it.

One image rebuild. Both batches are much weaker without it.

---

## 1. Batch A — complete the m=0 sweep (5 runs, +2 optional)

### 1.1 Why m=0 is the *least* informative margin — read before interpreting

`r_e = clip(16 − ⌊N_α⌋ − m, 1, 15)`. At m=0:

| ⌊N_α⌋ | raw | realised depth | |
|---|---|---|---|
| 1 | 15 | **15** | ← at the `r−1` ceiling, zero headroom above |
| 2 | 14 | 14 | |
| 3 | 13 | 13 | |

Since `⌊N_α⌋ ≥ 1` always, **every α that yields ⌊N_α⌋ = 1 lands on depth 15 and is indistinguishable
from every other such α.** That is the largest α-collapse of any margin. Confirmed by the measured
span: the depth 14.5–15 band has α-span **0.068**, the smallest of any band and 24× smaller than the
shallow bands.

**So m=0 cannot test the review objection** (which is about giving α *more* room — that means *larger*
m, batch B). Run it for three other reasons:

1. **Completeness for review** — "we swept the boundary margin, all of it," with no gaps.
2. **It closes two literal holes**: `m0-nodp-ainf-m0-s42` crashed at step 124 and
   `marg-nodp-a1-m0-s42` crashed at step 166. Both are unread.
3. **The genuinely informative part is α ≤ 0.2**, where m=0 removes the brake on the feedback loop
   (§3 below). This maps the runaway boundary, which is a real stability finding.

### 1.2 What already exists at m=0

| α | run | state | depth | loss |
|---|---|---|---|---|
| 0.1 | `m0-nodp-a01-m0-s42` | finished | 9.18 (drifted 8.04 → 9.56) | 0.34361 |
| 0.25 | `m0-nodp-a025-m0-s42` | finished | 14.33 (drifted 12.72 → **14.44**) | 0.34399 |
| 0.5 | `m0-nodp-a05-m0-s42` | finished | 14.97 | 0.34364 |
| 2 | `m0-nodp-a2-m0-s42` | finished | 15.00 | 0.34368 |
| 1 | `marg-nodp-a1-m0-s42` | **crashed** @166 | — | — |
| ∞ | `m0-nodp-ainf-m0-s42` | **crashed** @124 | — | — |

Missing entirely: **α ∈ {0.05, 0.15, 0.2}**. Crashed and needing a re-run: **α ∈ {1, ∞}**.

### 1.3 The runs (5)

```bash
cd /Users/david.stanojevic/PycharmProjects/opaque
git checkout david-stan/zenml-training

# VPN check FIRST — this has cost whole batches
host zenml.labs.jb.gg && \
  curl -s -o /dev/null -w '%{http_code}\n' --max-time 10 https://zenml.labs.jb.gg/api/v1/info   # expect 200

export OPAQUE_DOCKER_REGISTRY=europe-west4-docker.pkg.dev/gke-dev-dws-jbr/ml
export OPAQUE_DOCKER_TAG=<tag of the image WITH the §0 logging fix>
export WANDB_MODE=online

# --dry-run each one first and read the resolved argv.
# Cap at 3 concurrent (quota is ~5; over-subscribing crashed 7 runs in July).
for A in 0.05 0.15 0.2 1 inf; do
  XSE_ADAPTIVE_DEPTH=1 XSE_ADAPTIVE_DEPTH_ALPHA=$A XSE_ADAPTIVE_DEPTH_MARGIN=0 \
    .zenml-client/bin/python deploy/zenml/run.py nodp \
      --run-name m0b-nodp-a${A/./}-m0-s42 --seed 42 \
      --extra --lora-xse-p-e 0.333 --lora-r 16 --num-epochs 1
done
```

### 1.4 Pre-registered predictions

| α | predicted ⌊N_α⌋ | predicted depth | predicted loss | falsified if |
|---|---|---|---|---|
| ∞ | 1 | **15.00** | 0.3436 ± 3e-4 | depth ≠ 15.00 exactly |
| 1 | 1 | **15.00** | 0.3436 ± 3e-4 | depth < 14.9 |
| 0.2 | 3–4 | 12–13 | 0.3435 ± 3e-4 | depth > 14 |
| 0.15 | 4–5 | 11–12 | 0.3436 ± 3e-4 | depth > 13.5 |
| 0.05 | 10–12 | 4–6 | **0.3444 ± 5e-4** | depth > 8 |

**Headline prediction: α ∈ {0.5, 1, 2, ∞} all land at depth 15.00 and their losses span < 2e-4.**
Together with the existing α=0.5 and α=2 runs that gives a **4-way replicate group at depth 15**
(licensed by Theorem 1, since all four realise the same integer) — which is the *by-product worth
having*: it measures the run-to-run floor at the deepest operating point, currently estimated from a
single pair (α=0.5 vs α=2 differ by **4e-5**).

**What would overturn the α verdict:** any of {1, 2, ∞} landing at a depth other than 15.00, **or**
the four depth-15 losses spanning more than ~5e-4 with a *reproducible* ordering.

**Expected drift warning:** α ∈ {0.05, 0.15, 0.2} at m=0 will likely drift upward during training
(§3). If drift exceeds ~1 slot, that run is a *time-varying schedule*, not a fixed depth, and must not
be placed on the depth curve — the same reason `m0-nodp-a025-m0-s42` was excluded.

### 1.5 Optional +2 (seed floor)

`m0b-nodp-a1-m0-s43` and `-s44`. The 4-way α group in §1.4 gives a floor from *algorithmic*
replicates; these give one from *seed* replicates. Worth it only if a reviewer disputes Theorem 1.

---

## 2. Batch B — the m=8 probe (2 runs now, 2 after reading them)

This is the batch that actually addresses the review objection: it puts α where it has the most room.

### 2.1 Clamping risk — check before trusting anything

At m=8, `r_e = clip(16 − ⌊N_α⌋ − 8, 1, 15)`, so **⌊N_α⌋ ≥ 7 pins `r_e` at 1**. At depth ≈ 6 the
measured `N_{0.5}` runs 2.4–2.6 on the layer mean, but individual matrices could exceed 7. If the
α=0.5 arm clamps on a meaningful fraction of matrices, its realised depth will not follow the
extrapolation and the contrast is degraded. **This is exactly what the §0 min/max logging detects.**
Do not run batch B without it.

### 2.2 Wave 1 (2 runs)

```bash
for A in inf 0.5; do
  XSE_ADAPTIVE_DEPTH=1 XSE_ADAPTIVE_DEPTH_ALPHA=$A XSE_ADAPTIVE_DEPTH_MARGIN=8 \
    .zenml-client/bin/python deploy/zenml/run.py nodp \
      --run-name marg8-nodp-a${A/./}-m8-s42 --seed 42 \
      --extra --lora-xse-p-e 0.333 --lora-r 16 --num-epochs 1
done
```

**Pre-registered predictions** (extrapolated from the measured `d(depth)/dm` of −1.002 for α=∞ and
−1.214 for α=0.5 over m = 1,2,3):

| arm | predicted depth | predicted loss (from the fixed-`p_e` curve) |
|---|---|---|
| α=∞, m=8 | **7.0** | 0.34395 |
| α=0.5, m=8 | **5.3** | 0.34418 |
| difference | 1.7 slots | **α=∞ better by ≈ 2.3e-4** |

So for the first time we expect a **detectable** α difference (≈2–7× the local floor). Note what that
does and does not mean:

- **Expected and uninteresting:** α=∞ wins by roughly what the depth gap predicts. That confirms the
  mediation model — α acts through depth and nothing else — and confirms §6.2's dominance argument
  (both arms are 5–8e-4 *worse* than the m=2 plateau, so the whole regime is a loss).
- **Would be a real finding:** α=0.5 **beats** α=∞ despite being shallower, or the gap is much larger
  or smaller than 2.3e-4. Either would mean α does something beyond mean depth.

### 2.3 Wave 2 (2 runs, after reading wave 1)

The scientifically important pair, and the only remaining open question in the whole project:

1. **Matched-depth control** — fixed `p_e` (adaptive OFF) at exactly the depth α=0.5 realised. Set
   `--lora-xse-p-e <realised_depth/16>` and drop the `XSE_ADAPTIVE_DEPTH*` env vars. This separates
   *"α=0.5 is worse because it went shallower"* from *"α=0.5 distributed depth unevenly across the 196
   matrices, and that mattered."* At shallow depth α produces genuine per-matrix spread for the first
   time (§6.1), so this is the first real test of the heterogeneity hypothesis.
2. **Seed replicate** — `marg8-nodp-ainf-m8-s43`, to get *any* floor at depth ≈ 7, which is currently
   unmeasured (§8.1 flags the depth 7–12 floor as interpolated).

**Pre-registered prediction for the control:** it matches the α=0.5 arm to within the floor
(|residual| < 2e-4, random sign), as every previous matched-depth audit found. **If α=0.5 beats its
matched-depth control by more than ~3e-4, per-matrix heterogeneity is real at shallow depth and §6.1's
"identifiable but harmful" verdict needs revisiting.** Prior: low, but this is the one place it could
still happen.

---

## 3. Why the margin exists at all — three answers, increasingly honest

**(a) The stated intent** (`xse.py:428`): `N_α` estimates how many directions are carrying signal;
keep those and refresh the rest. The margin is "keep a couple extra so you don't throw away a
borderline-useful direction." It is the library default (2) and a July sweep over m ∈ {1,2,3} kept it.

**(b) What it is actually correcting: a systematic undercount.** `N_α` measures *concentration*, not
count. With one direction holding 90 % of the momentum energy, `N_α ≈ 1.1` even though ~6 directions
sit above the noise floor (`#{σ_i > 2·median} ≈ 6`). So the margin is a **hand-set patch for a
statistic that is known to read low** — and it is the only term in `k = ⌊N_α⌋ + m` that *can* carry
such a correction, since `⌊N_α⌋` structurally cannot see the noise scale (§5.1, Cor 5.1). It also
absorbs the floor's rounding-down bias: `N_α = 1.9` means "nearly two active" and `int()` returns 1.
(Related documented defect: the design comment says `ceil`, the code does `int`, so every run explored
one slot deeper than the published formula — the margin was partly compensating for that too.)

This is why a *constant* margin cannot be right in general: the size of the true undercount depends on
the noise level, and the statistic never sees the noise level.

**(c) The reason nobody wrote down — the margin is a brake on a runaway feedback loop.**

The loop: refresh more directions → `R` loses accumulated structure → its spectrum gets spikier →
`N_α` falls → the rule refreshes *even more*. Positive feedback. Measured within-run depth drift
(first eval → last eval) at matched α:

| α | drift at m=0 | drift at m=1 | drift at m=2 | drift at m=3 |
|---|---|---|---|---|
| 0.1 | **+1.53** | — | +1.07 | — |
| 0.25 | **+1.72** | — | +1.02 | — |
| 0.5 | +0.44 | +0.52 | +0.37 | +0.25 |
| 2 | +0.01 | +0.04 | +0.03 | −0.01 |

**A larger margin means less runaway, and it only matters where the loop is live (low α).** At α ≥ 2
there is nothing to damp because `⌊N_α⌋` is pinned at 1 regardless. m=0 removes the brake entirely,
which is exactly why `m0-nodp-a025-m0-s42` drifted 12.7 → 14.4 and had to be excluded from the depth
curve as a time-varying schedule rather than a fixed depth.

So the honest three-line answer for the paper: *the margin was introduced as a safety buffer; it is
really a constant correction for a statistic that systematically under-counts the active subspace; and
it incidentally damps a positive feedback loop between refresh depth and spectral concentration. None
of those three jobs is one a hand-set constant should be doing — which is the argument for the
threshold rule in §8.2.*

---

## 4. Budget and expected value

| batch | runs | closes | expected to change the α verdict? |
|---|---|---|---|
| §0 logging | 0 | makes the cut point and clamping observable | no, but both batches need it |
| A: m=0 sweep | 5 (+2) | 3 missing α, 2 crashed runs; gives a 4-way depth-15 replicate group | **no** — mechanically the least informative margin |
| B wave 1: m=8 | 2 | the review objection, at α's widest reach | expected to *confirm* mediation |
| B wave 2 | 2 | the heterogeneity question at shallow depth | **the only one that could** |

Total 9 GPU runs, 3 concurrent, ~2 batches. **If only two runs can be afforded, run B wave 1** — it is
the one that answers the objection on the table. Batch A is completeness insurance, not new science.
