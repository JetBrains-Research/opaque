# LoRA-XSe — Status Summary

Audience: project lead (mathematician). Scope: what is proven, what is measured,
what is still open, and what a reviewer will attack.
Date: 2026-08-14. Branch `david-stan/zenml-training`.

---

## 1. Bottom line

We have **one strong, defensible result**, and it is not the one we set out to prove.

> **Rotating the frozen basis of LoRA-XS recovers full-LoRA quality under DP-SGD at
> 201× fewer trainable parameters, at zero additional privacy cost.**

The Rényi-entropy machinery, which was the original theoretical thrust, turned out to
be a **diagnostic rather than a lever**: it correctly predicts *that* you must rotate
deeply, but the choice of the order α is at best a second-order effect, and one part of
that question is still open (§5).

The honest paper leads with the rotation mechanism. The Rényi analysis is the tool that
explains *why* it works and *how deep* to go — not the headline.

---

## 2. The defensible result

Setting: Qwen2.5-Coder-7B, JetBrains/KStack (Kotlin), 50k train / 1k eval, 2 epochs
(520 steps), DP-SGD with per-sample clipping, SGD+momentum 0.9, lr 5e-2, batch 192.
All 7 projection modules in all 28 layers.

### 2.1 Head-to-head at ε = 1, paired on seeds

| method | trainable params | eval loss (mean) | n seeds |
|---|---|---|---|
| LoRA r=16 | 40,370,176 | 0.34556 | 3 |
| **LoRA-XSe r=32** (ours) | **200,704** | **0.34559** | 2 |
| LoRA-XS r=32 (frozen basis) | 200,704 | 0.35454 | 3 |

Paired differences on common seeds:

- **XSe − LoRA-XS = −8.72e-3** (per-seed: −9.14e-3, −8.30e-3 — same sign, consistent)
- **XSe − LoRA = +2e-5** (per-seed: +1e-5, −4e-5 — a tie)

Against the ε=1 seed noise of the LoRA-XS arm (sd 5.8e-4), the −8.7e-3 gap is ≈ 15σ.

### 2.2 Why this is a clean claim

The ablation is **exactly two config fields**. `cmp-basexs-*` and `cmp-xse-*` are
byte-identical except:

| field | LoRA-XS | LoRA-XSe |
|---|---|---|
| `lora_xse_p_e` | 0 | 0.5 |
| `lora_xse_rotation_step_interval` | null | 3 |

Same r, same lr, same seed, same data order, same clipping, same ε. With `p_e = 0`
there is no explore block, so rotation is a structural no-op — the baseline is genuine
frozen-basis LoRA-XS, not a hobbled variant.

### 2.3 The parameter count (computed, not logged)

`trainable_params` was never logged, so this is computed from the architecture
(hidden 3584, intermediate 18944, 28 layers, 4 KV heads → kv width 512):

- LoRA r=16: `Σ_modules r(fan_in + fan_out) × 28` = **40,370,176**
- LoRA-XS/XSe r=32: `r² × 7 × 28` = **200,704**
- **ratio = 201×**

Under DP-SGD the injected noise scales as √d, so this is a **14.2× reduction in the
noise the optimiser must fight** — which is the mechanism behind the result, and the
reason the gap grows as ε shrinks.

> **Correction to earlier reporting.** I had been quoting "448× fewer parameters" and
> "0.3466 vs 0.3618". Both were wrong: 448× conflated r=16-vs-r=16 (804×) with the
> actual r=16-vs-r=32 comparison (201×), and 0.3618 does not exist in the data. The
> corrected numbers are above. The direction and significance of the result are
> unchanged.

---

## 3. What is mathematically proven

### 3.1 Monotonicity of the Rényi effective rank — proven, exact

For a spectrum `σ₁ ≥ … ≥ σ_r` with `p_i = σ_i²/‖M‖_F²`, define
`N_α = (Σ_i p_i^α)^{1/(1−α)}`. Then

```
d/dα log N_α  =  − D_KL( p^(α) ‖ p ) / (1−α)²   ≤   0 ,
      where   p^(α)_i = p_i^α / Σ_j p_j^α        (escort distribution)
```

so `N_α` is non-increasing in α, with removable singularity at α=1 giving the
Shannon/Roy–Vetterli rank. This makes α=0 (rank), α=1 (Roy–Vetterli), α=2
(participation ratio), α=∞ (stable rank) a single monotone family, and it is what
licenses "α selects how conservatively we count directions."

Status: exact, and independently reproduced by three separate proof attempts that were
asked to refute each other. This is the solid mathematical core.

### 3.2 The privacy claim — proven, and it is the load-bearing one

The rotation is computed **from the DP momentum buffer only**, which is already the
output of a DP mechanism. By post-processing immunity of differential privacy, applying
any data-independent-given-that-output map costs nothing:

> If `M` is (ε,δ)-DP and `f` is any (possibly randomised) map, `f(M)` is (ε,δ)-DP.

Hence the adaptivity — basis rotation *and* per-matrix depth selection — is **ε-free**.
Nothing in the accounting changes. This is what makes §2 a fair comparison rather than
a privacy-budget trade.

### 3.3 Mediation / additive separability — proven, and it is a *limitation*

The algorithm consumes α only through the integer per-matrix depth

```
r_e,ℓ  =  max(1, min(r−1,  r − ⌊N_α(ℓ)⌋ − m))
```

Two consequences follow immediately, and both were confirmed empirically:

1. **The margin `m` is a pure additive offset.** It cannot change differences between
   α values at fixed `m`. Verified: at `m=0`, α=0.5 and α=2 both realise mean depth
   15.00 and score 0.34364 vs 0.34368 (agree to 4e-5). Removing the margin does *not*
   decollapse α. — This answers the "why was margin 2 everywhere?" question: it is an
   offset, so it was never hiding an α effect.
2. **The rounding staircase.** α affects behaviour only via `⌊N_α⌋`, so every α
   interval mapping to the same integer is *behaviourally identical*. For our spectra
   `⌊N_α⌋` barely moves for α ≳ 0.8, which is exactly why the α sweep looked flat
   there, and why pushing α down to 0.05–0.25 finally produced visible movement.

**Important caveat I previously got wrong:** mediation says α acts only through the
integer **vector** `(r_e,ℓ)_ℓ` over all 196 matrices — *not* only through its mean. Two
α values can produce the same mean depth while distributing depth differently across
matrices. Collapsing the vector to its mean is an extra assumption, and it is the one
currently under test (§5).

### 3.4 Retracted theorems

Two results in the original draft did **not** survive adversarial review and have been
removed rather than patched:

- **T3** was claimed to converge; it diverges as Θ(r).
- **T4** was claimed to be MSE-optimal in general; it is not — optimality holds only
  under a spectral-decay assumption we cannot justify here.

### 3.5 Incomplete

The random-matrix results we rely on for intuition (Marchenko–Pastur bulk, BBP
threshold, Benaych-Georges–Nadakuditi spike map `ρ = θ + σ²/θ`) are **asymptotic**. A
rigorous finite-r statement — which is what a genuine "optimal α" theorem would need —
is not done. This is the part where a collaborator with RMT depth would add real value.

---

## 4. What is measured

### 4.1 DP inflates the apparent rank — mechanism confirmed

Effective-rank gap (`r − r_eff`), 3 seeds each:

| regime | gap | sd |
|---|---|---|
| DP, ε=3 | 4.72 | 0.002 |
| non-DP | 1.24 | 0.010 |

**≈3.8× inflation.** DP noise raises the noise floor of the spectrum, so entropy-based
rank estimates read high under DP. This is the predicted effect and it is solid.

### 4.2 Noise floors are strongly regime-dependent — and this matters a lot

Same config, different seed:

| regime | seed sd | vs non-DP |
|---|---|---|
| non-DP | **2.3e-5** | 1× |
| DP ε=3 | **3.7e-4** | 16× |
| DP ε=1 (1 epoch) | **1.5e-3 – 9.9e-3** | 65–420× |

Two consequences, both of which invalidate earlier claims of mine:

- **Single-seed ε=1 comparisons are worthless.** The ε=1 α sweep spanned 5.1e-4
  against a seed sd of 1.5e-3. Any α ordering read off it — including "α=∞ is worst at
  ε=1" — is unsupported. One ε=1 family (`fixed-re9`) spans **1.9e-2** across three
  seeds, larger than every effect we are chasing.
- **The 2-epoch `cmp-` family is the trustworthy one** (arm sds 7e-6 … 5.8e-4), which
  is why §2 uses it exclusively.

### 4.3 Depth matters, and shallow depth is genuinely bad

Non-DP, 29 verified runs (`state=finished`, ≥25 evals), keyed on realised depth:

| depth | eval loss | vs best |
|---|---|---|
| 1.00 | 0.34700 | +3.56e-3 |
| 3.13 | 0.34537 | +1.93e-3 |
| 5.00 | 0.34429 | +8.5e-4 |
| 9.00 | 0.34368 | +2.4e-4 |
| **13.00** | **0.34344** | — |
| 15.00 | 0.34364 | +2.0e-4 |

Against the non-DP floor of 2.3e-5, the depth-1 → depth-13 improvement of **3.6e-3 is
≈155σ**. Unambiguous: you must refresh a substantial fraction of the basis, and
refreshing ~1 direction is nearly as bad as not rotating.

**Practical rule:** refresh at least ~55% of the rank (≥9 of 16).

---

## 5. The one genuinely open question (experiments running now)

**Does α matter at matched mean depth?**

Three configs sit at **identical realised mean depth 14.00**, same seed, same margin,
differing *only* in α:

| α | eval loss |
|---|---|
| ∞ | 0.34362 |
| 1 | 0.34408 |
| 2 | 0.34419 |

Spread **5.7e-4 ≈ 20×** the non-DP seed floor. If that is real, α *does* act through
the per-matrix distribution of depth (consistent with §3.3's caveat), and α=∞ is
nominally best here.

**Why I cannot yet claim it.** The entire non-DP depth curve is **seed 42 only**. My
2.3e-5 floor came from `renyi-nodp-*`, a *non-adaptive, fixed-p_e, depth-5* config.
Adaptive-depth runs may simply be noisier, since the realised depth is itself driven by
random rotation draws. Using a floor from one config family to judge another is not
valid.

**Running now:** `seedrep-ad-nodp-{ainf,a2}-m1-s43` (2 more at s44 queued) — direct
seed replicates of the two extremes at matched depth 14.00. If the 5.7e-4 gap survives
across seeds, α has a real effect beyond mean depth. If it washes out, adaptive runs
are just noisy and the plateau is flat.

Also running: **`cmp-basexs-eps3-s{42,43,44}`** — the ε=3 frozen-basis baseline was
never run multi-seed, so "beats LoRA-XS under DP" currently rests on ε=1 only. These 3
runs complete the main table.

### 5.1 A methodological error worth recording

In an earlier pass I computed a "within-depth scatter" of 3.7e-4 by taking the spread
among runs at the *same depth but different α*, then used it as a noise floor to
conclude that α does not matter. That is circular — it assumes the conclusion. A noise
floor must come from **replicates of one configuration**, not from a contrast between
configurations. Correcting this is what re-opened §5.

---

## 6. What a reviewer will attack, and our answer

| attack | status |
|---|---|
| "n=1, no seeds" | Fixed for the headline (paired, n=2–3). The α curve is being replicated now. Must not ship single-seed α claims. |
| "1 epoch flatters the efficient method" | **Real and serious.** LoRA scores 0.399 at 1 epoch vs 0.3456 at 2 — it is undertrained at 1 epoch precisely because DP noise ∝ √d and its d is 201× larger. Every headline number uses 2 epochs. We must never quote a 1-epoch LoRA comparison. |
| "eval loss is not utility" | **Unresolved.** Python pass@1 measures *forgetting* here — the base model scores 0.680 and KStack is Kotlin, so fine-tuning can only move it down. We need a Kotlin-appropriate downstream metric, or we drop the downstream claim. BPB is implemented but has never been run. |
| "Rényi is your title but it does nothing" | Partly conceded. Reframe: rotation is the contribution; Rényi is the diagnostic that sets the depth. Do not oversell α. |
| "ε=3 baseline missing" | Being filled now (3 runs). |
| "why margin=2 everywhere?" | Answered analytically: additive offset, provably cannot change α contrasts, and verified at m=0 (§3.3). |

---

## 7. Inventory

**Solid.** Rotation win at ε=1 (multi-seed, paired, 2 epochs); the depth requirement
(155σ); DP rank-inflation mechanism (3.8×); noise-floor characterisation across
regimes; monotonicity proof; post-processing/ε-free adaptivity proof; margin
separability (proved and verified).

**In flight.** `cmp-basexs-eps3` ×3 (completes main table); matched-depth seed
replicates ×2 submitted, ×2 queued (settles §5).

**Not done.** Vision tasks. BPB in a real run. AdaLoRA-miscalibration analysis (no GPU
needed). Finite-r RMT (needs a collaborator). A Kotlin downstream metric — this is the
most important scientific gap after §5.

**Dead ends, recorded.** Per-matrix rank *allocation* (probe- and W0-based) was null in
both regimes. peft's AdaLoRA is structurally incompatible with our functional/vmap
per-sample-gradient harness (it needs `.grad` on module params), so the baseline was
realised as same-allocation-with-naive-score instead.

---

## 8. How I would pitch it

> Under differential privacy, the noise you must add grows with the number of
> parameters you train, so parameter-efficient fine-tuning is not merely convenient —
> it is what makes private fine-tuning viable. LoRA-XS is the extreme point: it trains
> an r×r core inside a frozen SVD basis, ~200× smaller than LoRA. But freezing the
> basis costs real quality, because the right subspace is not known in advance.
>
> We let the basis rotate, driven entirely by the already-privatised momentum, and
> therefore for free in privacy terms. That recovers full-LoRA quality at 201× fewer
> trainable parameters at ε=1. The Rényi effective rank tells us how much of the basis
> to refresh, and the answer is: at least half — refreshing one direction is almost as
> bad as freezing everything.

That claim is supported, the ablation is two lines of config, and the privacy argument
is a one-line appeal to post-processing. The α-order question is a secondary analysis
we should report honestly, including the parts that came out flat.
