# The Rényi / LoRA-XSe project: what we're doing, what we found, how to publish it

**Audience:** anyone joining cold, or a reviewer who wants the story before the
equations. Part I is deliberately non-technical. Parts II–IV are the method, the
results, and the conference framing. Part V is what still needs proving. Part VI is
the evaluation protocol — how to measure findings properly, which is where our
current weakness is.

---

# Part I — The problem, intuitively

## I.1 Why fine-tune with a tiny adapter at all?

You have a 7-billion-parameter code model and you want it to write Kotlin better,
**without leaking the training code**. Differential privacy (DP) gives that
guarantee: during training you clip each example's gradient and add random noise.

The catch: **the amount of noise you must add grows with the number of parameters
you train.** Train a billion parameters privately and the noise swamps the signal.
So DP practically forces you to train something small.

That is why adapters matter here. Standard **LoRA** trains two thin matrices per
layer (~115,000 parameters for one Qwen projection). **LoRA-XS** goes much further:
it freezes a basis taken from the pretrained weight's SVD and trains only a tiny
`r × r` core — **256 parameters** at r=16. That's ~450× fewer, so ~450× less noise
for the same privacy budget.

## I.2 The catch with LoRA-XS: a cage

LoRA-XS gets its cheapness by freezing the basis. But that basis is chosen from the
**pretrained weights, before seeing your task**, and never changes. Every update the
adapter can ever make is confined to that fixed subspace — a cage. If what your task
needs lies outside it, you cannot get there.

Empirically that hurts: frozen-basis LoRA-XS reaches held-out loss **0.3618**, while
standard LoRA (with 450× more parameters) reaches ~0.3455. You saved noise and paid
in quality.

## I.3 Our method: rotate the cage

**LoRA-XSe** un-freezes the *direction* of the basis without adding parameters.

Picture each layer as having **16 pots**. Plants that thrive keep growing; the
others get pulled and replaced with fresh seeds. Concretely, every ~5 steps, for each
layer independently:

1. Look at the momentum of the gradient — which of the 16 directions are actually
   receiving signal.
2. **Keep** the top ones and *rotate* the basis to align with them.
3. **Refresh** the rest: replace them with brand-new random directions, orthogonal to
   what's kept — including directions outside the original cage.
4. Rewrite the tiny core so the kept part computes exactly what it did before, and
   the new directions start contributing nothing (so nothing is disrupted) but can
   now receive gradient.

Two properties make this almost free:
- **No new parameters** — it's a change of coordinates absorbed into the frozen
  factors, so the DP noise is unchanged.
- **No privacy cost** — every input to the decision is the *already-noised* momentum.
  By DP's post-processing property, you can compute anything you like from data that
  has already been privatised. **Zero extra ε.** (This is the one thing our adversarial
  proof panel confirmed outright.)

Result: **0.3466** vs **0.3618** for the frozen basis — recovering ~93% of the gap to
full LoRA at 1/450th of the trainable parameters.

## I.4 The question that generated all the theory

Step 2 above requires a number: **how many directions are actually working?** Keep
that many; refresh the rest.

That's a rank-estimation problem, and it has a classic answer — "effective rank," the
exponential of the spectrum's entropy. But there isn't *one* entropy. **Rényi entropy
is a family indexed by α**, and α controls how much weak directions count:

| α | what it becomes | attitude to weak directions |
|---|---|---|
| 0 | matrix rank | every nonzero direction counts equally |
| 1 | Shannon effective rank | honest average |
| 2 | participation ratio | favours strong ones |
| ∞ | stable rank | only the loudest matters |

These four are treated as separate tools in the literature; they are one curve read
at four points. **On the same matrix they disagree by 4×** (for spectrum (10,1,1,1):
4.00 / 1.18 / 1.06 / 1.03). So "how much rank is here" is ill-posed until you fix α.

## I.5 Why privacy makes the α choice interesting

Without noise, weak directions are *real but small*. With DP noise, **weak directions
are largely fabricated** — the noise fills every direction with a little energy. So:

- **Low α** (generous) counts the noise as signal → overestimates the rank.
- **High α** (strict) ignores the noise floor → tracks the real signal.

We measured this on the real model: DP inflates the low-α estimate **3.8×** relative
to the identical non-private run (tight CIs, 3 seeds each). The raw spectra show it
too — DP flattens the singular-value decay from 0.50 to 0.74 and raises the
numerical rank from 8 to 11.

That gives a genuinely counterintuitive story: the Rényi family is the ecology
*diversity index*, whose natural regime is α ≤ 1 precisely because you want to catch
rare species. Under DP, "rare" means "fabricated," so the classical instinct is
backwards.

---

# Part II — How we approached it (and what went wrong on the way)

## II.1 The chain we had to test

```
α  →  effective rank estimate  →  r_e (an integer: how many directions to refresh)  →  model quality
```

We verified in the code that **α appears nowhere else**. That single fact shaped the
whole investigation, because it means you never have to sweep α: measure the middle
link once, and compose.

## II.2 Four wrong turns worth documenting

1. **We swept α and got flat results — twice.** Interpreted as "α doesn't matter."
2. **We tried per-layer rank *allocation*** (give important layers bigger adapters).
   It consistently *lost* to uniform. Later diagnosis: the layers only use 1–3 of
   their 16 directions, so the rank budget was never binding — allocation just moved
   slack around.
3. **We evaluated with Python coding benchmarks (HumanEval+/MBPP+) while training on
   Kotlin.** Then measured the un-finetuned base model: **0.680 MBPP+, and every
   fine-tuned model scored below it.** The benchmark was measuring *how little the
   model had changed* — i.e. forgetting — not task quality. The method that learned
   Kotlin best scored worst.
4. **Three "silent" bugs** that produced no error but invalidated results:
   `WANDB_MODE=disabled` propagating into pods (a whole batch trained and recorded
   nothing); `evalplus` missing from the image (downstream eval failed inside a
   try/except); and the effective rank being computed on the *wrong matrix* (the core
   `R` instead of the momentum whose basis the decision actually slices).
5. **A confound that would have made the paper wrong:** LoRA-XS scales each layer by
   `alpha/r`, so giving layers different ranks silently changed their **learning
   rates by up to 4×**. Every allocation comparison was measuring rank placement
   *plus* LR distortion.

## II.3 The move that actually settled it: test the mediator

Instead of sweeping α, **fix `r_e` by hand** and measure quality: refresh 1, 5, 9, or
13 of 16 directions uniformly. Then run the real adaptive method and compare it
against the fixed curve *at the depth it actually chose*.

Analogy: rather than trying twenty thermostat settings, first check whether the heater
does anything by running it manually at minimum and maximum.

---

# Part III — What we found

## III.1 Exploration depth is the real mechanism (non-DP, measured)

| refresh depth `r_e` | held-out loss | marginal slope |
|---|---|---|
| 1 | 0.34700 | — |
| 5 | 0.34429 | −6.8e-4 /slot |
| 9 | 0.34368 | −1.5e-4 /slot |
| 13 | 0.34354 | −3.5e-5 /slot |

Total **−0.00346** (~35× our noise floor), and the curve **saturates** — the slope
falls 19× across the range. More refreshing is better, with strongly diminishing
returns.

## III.2 α provably cannot matter in non-DP — with a bound

A free control fell out of the design: **α=∞ selects `r_e`=13.00 uniformly, so it *is*
the fixed-13 configuration.** Run separately they differ by **1.0e-4** — a direct
measurement of run-to-run nondeterminism.

α's entire reachable depth range in non-DP is **11.5–13.0** (1.5 slots), and the slope
there is 3.5e-5/slot. Therefore:

> **max possible α effect = 5.3e-5 = 0.53× the noise floor.**

Not "we found nothing" — **α cannot produce a detectable effect, and here is why:**
it only ever moves along the saturated tail of the depth curve.

## III.3 Per-layer tailoring adds nothing (non-DP)

Adaptive arms vs the uniform curve at matched depth: α=0.25 −1.3e-4, α=0.5 +0.2e-4,
α=∞ −1.0e-4. **Inconsistent signs, all ≈ the noise floor** ⇒ no evidence that
choosing depth per layer beats one good global depth.

## III.4 What survives as positive results

1. **Rotation works:** 0.3466 vs 0.3618 frozen-basis, at identical parameters and ε —
   ~75× the noise floor, reproduced across independent campaigns.
2. **Depth is the mechanism:** −0.0035 across the range, saturating.
3. **DP inflates rank estimation 3.8×** — a measurement fact about *any* method that
   scores importance from a noisy spectrum.
4. **Two methodological traps** (out-of-domain pass@1 measures forgetting; `alpha/r`
   scaling confounds variable-rank comparisons).
5. **Adaptivity costs no privacy** (post-processing; proven).

---

# Part IV — Packaging for a conference

## IV.1 The paper

**Title shape:** *Un-caging LoRA-XS: basis rotation for differentially private
fine-tuning, and the rank estimator that calibrates it.*

**Lede = the method, not α.** Rotation is the result with a 75σ effect. α is the
principled derivation of the rule the method needs, plus the diagnosis of a failure
mode that affects other methods.

**Contributions, in order:**
1. **Method:** LoRA-XSe. Recovers ~93% of the frozen-basis→LoRA gap at ~450× fewer
   trainable parameters, provably zero extra ε.
2. **Mechanism:** exploration depth is what matters, with a measured dose–response.
3. **Theory:** the Rényi rank family unifies four classical measures; the optimal
   order shifts with noise (α\*(σ)); DP inflates the naive estimate 3.8×, which
   mis-calibrates AdaLoRA-style importance scoring.
4. **A quantitative insensitivity result:** the depth rule needs **no tuning**, and we
   bound the tuning parameter's maximum possible effect at 0.53× the noise floor.
5. **Measurement methodology:** two traps that invalidate naive evaluations.

## IV.2 Why "α doesn't matter" is a feature, not a hole

Presented as a tuning curve, a flat sweep means "your hyperparameter is useless."
Presented as a **derivation**, flatness means "**the method is robust and needs no
tuning**" — provided you can explain *why* it's flat. We can, three ways: the
integer collapse (α=2 and α=∞ agree on 94% of layers), the saturated tail (the bound
above), and the theory's own prediction that the error surface is shallow near the
crossover. And the theory says where insensitivity should *end* — higher noise, lower
α — which is falsifiable.

## IV.3 The reviewer objection to prepare for

*"If α doesn't change performance, why is the Rényi theory here?"* Answer: because it
(a) explains a 3.8× miscalibration that affects other people's methods, (b) yields the
rule we use, and (c) predicts the regime boundary. If that doesn't persuade, the α
content becomes a section rather than a headline — and the paper still stands on
rotation.

## IV.4 Figures

1. **Loss vs epoch** for LoRA / LoRA-XS / LoRA-XSe at fixed ε, annotated with
   parameter counts. The money shot.
2. **Depth dose–response** with the saturation, plus the adaptive arms plotted at
   their realised depth (shows they land *on* the curve).
3. **α\*(σ)**: optimal Rényi order vs noise, with our measured non-DP and DP points.
4. **DP rank inflation** (3.8×) with CIs.

---

# Part V — What still needs mathematical proof

Status from the adversarial proof panel (42 agents, 2 proofs + 2 adversarial referees
per theorem):

**Proven:** the four-ranks-are-one-curve identities; monotonicity of `N_α` in α (via
the escort/KL derivative); the exact two-level bias law; Eckart–Young; DP
post-processing (ε-freeness).

**Corrected — earlier claims were false:** "the stable rank recovers the true rank as
r→∞" (it *diverges*, `N_∞ = Θ(r)`), and "α=∞ is universally MSE-optimal" (false at
strong signal; a conditional version survives). Both hold only in the high-noise,
finite-r regime — which is the DP regime the method operates in.

**Open, and needs a random-matrix collaborator:**
1. Finite-`r` expansion of `E[N_∞]` **including singular-value repulsion** (the toy
   model gets the sign wrong at strong signal).
2. A tail bound locating the regime boundary `P(N_∞ ≥ k) → 1` — this is where the
   conditional optimality theorem applies.
3. Finite-`r` variance/MSE per regime.
4. **The link we now know is missing:** a rigorous statement connecting
   rank-estimation error to *utility*. Our own measurement shows this link is
   **weak by construction** — the mediator's gain is ~1e-4/slot and α spans ~1.5
   slots, so the composition is bounded below detectability. The honest theoretical
   contribution is therefore about **estimation**, with a *measured* bound on how
   little it propagates to loss.

---

# Part VI — How to evaluate findings properly (our current weakness)

The effects we chase are **1e-5 to 1e-4**. Our measurement apparatus was one scalar
per run with no variance estimate. That is the binding constraint, not the metric
definition.

## VI.1 What was actually wrong (and what wasn't)

- **Not wrong:** `loss_min` ≈ `loss_final` in 4 of 5 runs (the min occurred at the last
  checkpoint), so order-statistic bias was minor in practice.
- **Also not wrong:** "three runs reported identical loss" was a **rounding artifact of
  ours** — full precision shows 0.3434580 / 0.3434635 / 0.3434576, all distinct.
- **Genuinely wrong:** one number per run cannot support any inference. With a
  measured nondeterminism floor of **1.0e-4** and target effects of 1e-5–1e-4, single
  scalars can never resolve them.

## VI.2 The protocol we should adopt

1. **Per-example scoring, not one scalar.** Compute per-example loss / **bits-per-byte**
   (NLL ÷ UTF-8 bytes) on a fixed held-out set and keep the whole vector. Byte
   normalisation makes it tokenizer-independent and interpretable as compression.
2. **Paired comparisons.** Two models' per-example losses correlate ~0.99, so
   `Var(X−Y) ≪ Var(X)`. Comparing the *paired difference vector* is dramatically more
   powerful than comparing two means — this is where a 1e-4 effect becomes resolvable.
3. **Bootstrap + permutation.** BCa bootstrap CI over examples (and over seeds when
   available); sign-flip permutation test. Claim only if CI excludes 0 **and** p<0.05.
4. **Equivalence tests (TOST)** when the answer is "no difference." "Utility is
   statistically equivalent across α, within ±δ" is a publishable claim; "we failed to
   reject" is not.
5. **Common random numbers.** Same seed → same init/data order for both arms, so the
   difference isolates the treatment.
6. **≥6 seeds if using a nonparametric test** (at k=5 the minimum two-sided p is
   2/32 = 0.0625 — you *cannot* reach 0.05).
7. **Average the last k checkpoints** rather than taking one — free variance reduction.
8. **Report which variance source the error bars capture** (seed / data order /
   nondeterminism), as the NeurIPS checklist requires.
9. **In-domain evaluation.** For Kotlin training use Kotlin evaluation:
   `JetBrains/Kotlin_HumanEval` (161 tasks) and the KStack FIM next-line exact-match
   protocol from JetBrains' own Kotlin ML Pack report — which was built *specifically*
   to detect fine-tuning degradation. Keep Python pass@1 explicitly relabelled as a
   **retention/forgetting** metric.
10. **Reduce the noise floor itself.** The 1.0e-4 floor between computationally
    identical runs is GPU nondeterminism. Deterministic kernels and fixed data order
    would shrink it; worth paying the throughput cost on measurement runs.

## VI.3 Status of implementation

- `--eval-bpb` is implemented and verified (exact on synthetic; end-to-end smoke
  tested). It logs the **per-example vector** to W&B, which is what enables 1–4.
- It has **not yet been used in a real experiment** — the image containing it did not
  finish building before the non-DP runs went out, so those used the scalar metric.
- **Any follow-up (starting with ε=1) should use it**, with paired seeds and the
  bootstrap/TOST analysis, so conclusions come with intervals instead of point
  differences.

## VI.4 Honest read on the non-DP conclusion under this lens

The non-DP result does **not** depend on resolving 1e-5 effects, which is why it
stands: the mechanism effect is **−0.0035 (35× the floor)**, and the α conclusion is a
**bound** (α's range × the measured slope < the floor) rather than a claimed small
difference. Where better metrics matter is the *next* question — whether α does
anything at ε=1, where its reachable range is much wider and the effect could be real
but small.
