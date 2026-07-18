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
| **non-DP** | 0.3444 ± 0.000 (n=2*) | **1.24 ± 0.010 (n=2*)** |

**DP inflates the low-α effective rank ~3.8× vs non-DP**, with very tight CIs
(±0.002 / ±0.010). This is the multi-seed confirmation of the earlier single-seed
result (2.9×) and the synthetic prediction (~2.7×). The core thesis — *DP noise
inflates the low-α (diversity) effective rank while the stable rank stays robust*
— holds robustly on the real 7B model. (*non-DP n=2; the 3rd seed was still
running at write time.)

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

DP ε=3 (m=2): α=0.5 → 0.3459, α=1 → 0.3458, α=2 → 0.3464*, α=∞ → 0.3459
(*a2 was still running; preliminary).

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
- 2 runs (`renyi-ad-dp3-a2-m2-s42`, `renyi-nodp-s44`) were finishing at write
  time; they only firm up the a2 DP cell and non-DP seed n → 3.

*Bottom line: the mechanism is confirmed and tight; the utility payoff is still
open and now has a precise, falsifiable next experiment (per-layer allocation +
downstream metrics + seeds).*
