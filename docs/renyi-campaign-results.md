# Rényi Campaign — Results (Qwen2.5-Coder-7B / KStack, ZenML)

**Date:** 2026-07-18. **Runs:** W&B `federated-compute/opaque-lora-xs`. 1 epoch,
r=16, seed 42 unless noted. Two families: an **adaptive-depth α×margin sweep**
(non-DP + DP ε=3) and a **DP-vs-non-DP seed sweep** (adaptive-depth OFF, seeds
42/43/44). Metric = `eval/loss_min` (lower better; downstream pass@1 not measured
in these runs). All runs tracked online (after the `WANDB_MODE=disabled` fix).

---

## Headline: the noise-inflation mechanism is confirmed with tight CIs

Seed sweep (adaptive-depth OFF), `rotation/renyi_gap_a0p5_ainf` (low-α minus
stable-rank effective rank of the core):

| | eval/loss_min | renyi_gap (a0.5 − a∞) |
|---|---|---|
| **DP (ε=3)** | 0.3469 ± 0.0004 (n=3) | **4.72 ± 0.002 (n=3)** |
| **non-DP** | 0.3444 ± 0.0001 (n=3) | **1.24 ± 0.01 (n=3)** |

**DP inflates the low-α effective rank ~3.8× vs non-DP**, with very tight CIs
(±0.002 / ±0.01), n=3 both arms. This is the multi-seed confirmation of the
earlier single-seed result (2.9×) and the synthetic prediction (~2.7×). The core
thesis — *DP noise inflates the low-α (diversity) effective rank while the stable
rank stays robust* — holds robustly on the real 7B model.

The DP privacy cost on the task is small and clean: eval/loss 0.3469 (DP) vs
0.3444 (non-DP), ≈ +0.0025.

---

## Adaptive-depth α-sweep — eval/loss_min (the honest utility result)

Non-DP:

| α \ margin | m=1 | m=2 | m=3 |
|---|---|---|---|
| 0.5 | 0.3438 | 0.3435 | 0.3436 |
| 1 | 0.3437 | 0.3435 | 0.3437 |
| 2 | 0.3440 | 0.3435 | 0.3436 |
| ∞ | 0.3436 | 0.3436 | 0.3436 |

DP ε=3 (m=2): α=0.5 → 0.3459, α=1 → 0.3458, α=2 → 0.3458, α=∞ → 0.3459
(all final; flat within noise, range 0.0001).

**Reading it honestly:**
- **Non-DP: completely flat** (0.3435–0.3440, range 0.0005 = within noise). This
  is exactly what the theory predicts — *without noise, the α-choice doesn't
  matter* — and it is the clean **control**: any α-effect elsewhere must come from
  noise, not from α per se.
- **DP: also flat on eval/loss** (≈0.346 across α, within noise; α=2 marginally
  worse). So the **adaptive-depth α-choice does NOT produce a utility win on
  eval/loss under DP.** The mechanism (α changes the measured rank) is real, but it
  does not move this task metric.
- Adaptive depth (any α) ≈ 0.3435–0.346 vs plain rotation (seed sweep) 0.3444–
  0.3469 — a small (~0.001) improvement in both regimes, but **single-seed on the
  adaptive arms**, so suggestive not conclusive.

## Secondary confirmation: higher α ⇒ cleaner spectrum

`rotation/renyi_gap_a0p5_ainf` (final) by adaptive-depth α, m=2:

| α | non-DP gap | DP gap |
|---|---|---|
| 0.5 | 0.59 | 4.66 |
| 1 | 0.55 | 4.21 |
| 2 | 0.56 | 3.83 |
| ∞ | 0.54 | **3.76** |

Under DP, raising the adaptive-depth α (more aggressive re-exploration of noise
slots) **reduces** the residual spectral gap (4.66 → 3.76) — i.e. it leaves a
less noise-inflated kept spectrum. This is a clean *mechanistic* confirmation that
high α filters noise, even though it doesn't translate into an eval/loss gain.

---

## What this means (honest)

- **Strong, publishable:** the noise-inflation *mechanism* — DP inflates the
  low-α effective rank ~3.8× (tight CIs, multi-seed, real 7B), non-DP flat
  (control), high-α filters the noise floor. This anchors the analysis/critique
  angle ("adaptive-rank importance scores are DP-miscalibrated") + the theory
  (rescoped, `renyi-effective-rank-theory.md`).
- **NOT yet shown:** a *utility win* from the Rényi order. The adaptive-depth
  α-choice is flat on eval/loss (DP and non-DP). Two untested routes to a real
  win remain: (1) the **per-layer rank allocation** method (implemented, not yet
  run — needs the probe→allocate pipeline), and (2) **downstream metrics**
  (HumanEval+/MBPP+ pass@1), which these loss-only runs didn't measure and which
  can differ from eval/loss.

## Caveats / next steps
- α-sweep arms are **single-seed (42)** — add 43/44 before any α-ordering claim.
- **eval/loss only** — rerun the best arms with `--eval-humaneval --eval-mbpp`.
- **Per-layer allocation arms** (the actual method vs AdaLoRA-style naive score)
  still to run — image `f677706` is built; see `renyi-zenml-campaign-plan.md`.
- All 22 runs finished (the last two, `renyi-ad-dp3-a2-m2-s42` → 0.3458 and
  `renyi-nodp-s44` → 0.3443/gap 1.248, are folded in above; they left the
  conclusions unchanged — DP α-sweep flatter, non-DP seed n=3).

*Bottom line: the mechanism is confirmed and tight; the utility payoff is still
open and now has a precise, falsifiable next experiment (per-layer allocation +
downstream metrics + seeds).*

---

# Part II — Allocation campaign (2026-08-06), Qwen2.5-Coder-7B / KStack

Setup: `qwen-coder-kstack-lora` preset (Qwen2.5-Coder-7B, JetBrains/KStack Kotlin,
50k samples, 1 epoch, r=16, 7 target modules, SGD lr 5e-2, batch 192), DP-SGD at
ε∈{1,3}, δ=6.8e-6. Downstream = HumanEval+/MBPP+ pass@1 (evalplus).

## ⚠ Methodological finding: Python pass@1 is NOT a utility metric here

We train on **Kotlin** but HumanEval+/MBPP+ measure **Python**. The method that
fits the task best scores *worst* on pass@1:

| ε=3 | KStack eval/loss (in-domain) | MBPP+ pass@1 (Python) |
|---|---|---|
| base LoRA-XS uniform | 0.36180 | 0.675 |
| LoRA-XSe (rotation) uniform | **0.34663** | 0.635 |

So Python pass@1 largely measures **retention of pretrained ability** (inverse
forgetting), not task utility — it rewards *not learning*. In-domain held-out loss
is the valid utility metric; pass@1 is reported as a secondary
(retention) axis. HumanEval+ is additionally unreliable here: only ~44–58% of its
completions parse (`syntax_valid_rate`), vs ~99% for MBPP+, plausibly because a
Kotlin-tuned model drifts when completing a Python signature.

## Result 1 (settled): data-free W0 allocation HURTS

Allocating per-layer rank by the stable rank of the **frozen** W0, at matched
budget Σr_ℓ²:

| arm | ε | uniform loss | W0-alloc loss | uniform MBPP+ | W0-alloc MBPP+ |
|---|---|---|---|---|---|
| base LoRA-XS | 1 | **0.36175** | 0.36346 | **0.675** | 0.590 |
| base LoRA-XS | 3 | **0.36180** | 0.36362 | **0.675** | 0.603 |
| LoRA-XSe | 3 | **0.34663** | 0.34742 | 0.635 | 0.643 |

Worse on loss in all three pairs (4 including the ε=1 XSe pair). Mechanism:
clean-weight rank has no reason to track where task gradient signal lives, and
funding some layers starves others toward r_min.

## Result 2 (diagnosis): at r=16 the rank budget is NOT binding

The probe (`util3-probe-eps3`, 60 steps, noised cores, 196 layers) gives per-layer
**stable rank 1.01 / 1.52 / 3.42** (min/median/max) out of r=16. Every layer uses
1–3 of its 16 directions, so allocation only shuffles slack — which explains both
the null (adaptive-depth α) and negative (W0) results. Motivates the binding-budget
experiment (r=2: 37 layers→rank 1, 137→2, 22→3; uniform r=2 costs 0.36792 vs
0.34663 at r=16, so rank is scarce there).

## Result 3 (REPLICATED, n=3): the single-seed α-ordering does NOT survive

Probe-based allocation from the **noised** core spectra, LoRA-XSe, ε=3, matched
budget, seeds 42/43/44:

| arm | eval/loss mean±sd | MBPP+ mean±sd |
|---|---|---|
| **uniform** | **0.34665 ± 0.00013** | 0.609 ± 0.075 |
| α=∞ (stable rank) | 0.34704 ± 0.00036 | **0.661 ± 0.032** |
| α=1 (Shannon) | 0.34742 ± 0.00051 | 0.616 ± 0.070 |

Paired per-seed:
- α=∞ beats α=1 on **loss in 3/3 seeds** (+0.00075/+0.00006/+0.00032) — the
  predicted direction, but the effect is ~0.0004 and one seed is a tie.
- α=∞ ≥ uniform on **MBPP+ in 3/3 seeds** (+0.048/+0.008/+0.100) with much lower
  variance (0.032 vs 0.075; on s44 uniform collapsed to 0.524 while α=∞ held
  0.624). Suggestive robustness effect, t≈1.9, **not significant at n=3**.
- **uniform beats both on loss in 3/3 seeds** — allocation does not win.

**The single-seed "+5.3 MBPP+ for α=∞ over α=1" was noise**: MBPP+ seed spread is
±0.07, far larger than the effect. Single-seed pass@1 comparisons are unreliable
in this setup.

## Result 3b: binding budget does not rescue allocation

| r | uniform loss | probe-alloc loss |
|---|---|---|
| 2 | **0.36792** | 0.37090 |
| 4 | **0.34918** | 0.36272 |

Even where rank is scarce, allocation loses. (Note r=4 uniform ≈ 0.34918 is close
to r=16's 0.34663 — 4 directions per layer nearly suffice, consistent with the
measured signal rank of 1–3.)

## ⚠ CONFOUND: variable rank changes per-layer scaling

LoRA-XS sets `scaling = lora_alpha / r` **per layer**. Varying r therefore also
varies each layer's effective scale: with lora_alpha=16, r=23 → 0.70, r=9 → 1.78,
and α=1's rank-3 layers → **5.3×**. Every allocation arm is thus confounded —
"more rank" is entangled with "different effective LR" — and the arm with the most
extreme spread (α=1) is penalised most. This plausibly explains both (a) why all
allocation variants lose on loss and (b) why α=∞ > α=1, without any appeal to
signal-tracking quality.

**A valid allocation test requires holding scaling fixed** (per-layer
`lora_alpha ∝ r`, or rank-stabilized `alpha/√r`). Until then the allocation
results should be read as *inconclusive-to-negative*, not as a refutation of the
theory.

## Result 4: rotation (LoRA-XSe) is the real method-level win

ε=3, matched rank/params/steps: LoRA 0.39924 → base LoRA-XS 0.36180 → **LoRA-XSe
0.34663**. The XSe-over-base-LoRA-XS margin (−0.015) is large and reproduced by the
earlier 3-seed campaign (0.3469 vs 0.3617).

**Caveat on "beats LoRA":** at 1 epoch LoRA is undertrained (0.399, reproduced
twice); the earlier tuned 2-epoch LoRA reached ~0.3455 while XSe is already
converged after 1 epoch (~0.3466). So the honest claim is a **convergence-speed**
advantage at matched steps, not a better final optimum.

## Pending
Seed replication (6), binding-budget r=2/r=4 probe-alloc (3), wikipedia
domain-shift arms (5).
