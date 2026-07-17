# Experiment Pre-Registration — Does Rényi/stable-rank allocation win on utility under DP?

**Goal.** Show a *credible, falsifiable* privacy–utility win from allocating LoRA
rank by a **noise-robust (high-α) effective rank** under DP-SGD — and prove the
win comes from the Rényi-order insight itself, not just "adaptive is good."

This is a pre-registration: fix the method, hypotheses, metrics, and analysis
*before* the big runs, and gate the expensive campaign behind cheap go/no-go
checks. Theory: `vendor/lora-privacy/docs/renyi-effective-rank-theory.md`.
Confirmed mechanism (DP inflates low-α rank ~2.9× vs non-DP): W&B runs
`o2kev43t` (DP) / `odivaztt` (non-DP).

---

## 0. One-line claim

*Under DP-SGD, allocating a fixed rank/parameter budget across layers in
proportion to each layer's **stable rank** (`r_eff^(∞)`) — computed on the
already-noised state, so ε-free — beats uniform LoRA-XS and AdaLoRA on the
privacy–utility frontier, with the gap widening as ε shrinks.*

---

## 1. Method under test

**Rényi-allocated LoRA-XS (static, primary).**
1. Fix total budget `Σ_ℓ r_ℓ² = B` (⇒ fixed trainable params ⇒ fixed DP noise).
2. Warm up briefly at a uniform probe rank `r_probe`.
3. Measure per-layer `k̂_ℓ = r_eff^(α)(R_ℓ)` from the **DP-noised** core/momentum
   (post-processing ⇒ **0 extra ε**).
4. Allocate `r_ℓ = round(sqrt(B · k̂_ℓ / Σ_j k̂_j))`, clamp to `[r_min, r_max]`,
   renormalize to hit `B`. Freeze; train to completion.

**Rényi-allocated LoRA-XS (dynamic, extension).** Re-allocate every `T_realloc`
steps from the current noised spectrum (directly comparable to AdaLoRA's schedule).

**DP-safety invariant (must hold everywhere):** allocation decisions read ONLY the
noised state; never clean gradients. State it as a lemma (post-processing).

---

## 2. Hypotheses (pre-registered, falsifiable)

> **Framing update after the proof panel** (`vendor/lora-privacy/docs/renyi-proof-status.md`).
> The stable-rank prescription is **not universal**: it is MSE-optimal only in the
> **high-noise regime** where $N_\infty\ge k$ (Theorem 4′), and can *lose* at
> strong signal / low noise. So the experiments must **map the regime**, not chase
> a blanket "α=∞ always wins." The theory's sharp, falsifiable prediction is a
> *2-D regime map over (ε, r)*: the advantage concentrates in the **high-noise
> (low ε) / small-r corner** and fades toward low noise. That is harder to fake
> than a single win and is the headline result.

- **H1 — Frontier win in-regime.** Rényi(α=∞)-allocation ≥ AdaLoRA and ≥ uniform
  LoRA-XS on the privacy–utility frontier for ε ∈ {1, 3, 8}, **strongest at ε=1**.
- **H2 — Regime map (the theory fingerprint).** The advantage over AdaLoRA/uniform
  *grows as ε decreases and as r decreases* (more noise / less headroom ⇒ more
  mis-allocation to fix). Equivalently: the win tracks $P(N_\infty\ge k)$ across
  the (ε, r) grid. ← the compelling, hard-to-fake result.
- **H3 — Order ablation.** In the high-noise cells, utility is ordered
  α=∞ ≥ α=2 ≥ α=1 ≥ α=0.5. In low-noise cells the ordering *may invert toward a
  finite α* (predicted by the interior-optimum finding) — a bonus confirmation if
  seen. ← proves the win is the Rényi-order insight and its regime-dependence.
- **H4 — Rotation cadence.** `renyi_gap_a0p5_ainf` decreases with rotation interval
  / momentum β; every-step rotation gives the noisiest estimates and *hurts more
  under DP than non-DP*. Optimal interval ≈ the momentum-tied default.

Primary endpoint = downstream task metric (not eval loss). Decide it per task in
§5 before running. NOTE: do **not** pre-register "α=∞ is universally best" — the
panel disproved that; H2/H3 are the regime-aware, defensible versions.

---

## 3. Baselines (who we beat), all at matched param budget

| Baseline | Role |
|---|---|
| Uniform LoRA-XS (same B) | the "no allocation" control (base method) |
| LoRA (matched params) | the standard everyone reports |
| **AdaLoRA (matched params)** | key rival; theory says its score is DP-miscalibrated |
| DoRA (optional, matched params) | extra PEFT rival |
| Full DP fine-tuning (where feasible) | upper reference |
| **Rényi-alloc α∈{0.5,1,2,∞}** | our method + the H3 ablation |

---

## 4. Tasks & models (tiers)

- **Tier 1 — canonical, cheap, high power (DO FIRST):** RoBERTa-large on GLUE
  (SST-2, MNLI, QNLI, QQP). *The* DP fine-tuning benchmark; supports many seeds.
  Metric: task accuracy (MNLI m/mm).
- **Tier 2 — hero (infra ready):** Qwen2.5-Coder-7B + KStack. Metric:
  HumanEval+/MBPP+ pass@1 (+ eval loss secondary).
- **Tier 3 — generality:** Llama-2-7B (or Qwen) on commonsense-8 and/or GSM8K.
  Metric: mean accuracy / exact-match.
- **Tier 4 — optional breadth:** CIFAR-100 + DP-ViT.

Ship Tier 1 + Tier 2 minimum; add Tier 3 for a strong generality claim.

---

## 5. Controls (the credibility machinery)

- **Matched trainable-param budget** across all methods (verify by counting) ⇒
  identical DP noise at fixed ε. Wins come from *placement*, not *amount*.
- **Equal HP-tuning budget** per method (same-size grid on a val split; report on
  test). Pre-empts "tuned ours, defaulted theirs."
- **Identical DP accounting** (ε, δ, clipping, accountant, sampler) across methods.
- **≥5 seeds** (Tier 1), ≥3 (7B tiers). Report mean ± 95% CI; **paired test**
  across shared seeds/splits.
- **Report allocation entropy** — if allocation ≈ uniform, disclose it (explains a
  null honestly).

---

## 6. Figures (what the paper shows)

1. **The regime map (headline).** Heatmap over the **(ε, r) grid** of the utility
   advantage of α=∞-allocation vs AdaLoRA/uniform — hot in the high-noise/small-r
   corner, cooling toward low noise (H2). Overlay the contour where
   $\hat P(N_\infty\ge k)=0.5$ (the theory's predicted boundary from Theorem 4′).
   This single figure *is* the paper: it shows the win AND that it appears exactly
   where the theory says it must.
2. **Privacy–utility frontier**: x=ε, y=metric, line/method, CIs — a 1-D slice of
   figure 1 at fixed r; gap widening at low ε (H1).
3. **Order ablation**: utility vs allocation-α (0.5→1→2→∞), split by high-noise vs
   low-noise cells (H3; ordering may invert in low-noise).
4. **Mechanism→outcome bridge**: AdaLoRA importance ranking on noised vs clean
   spectra (how badly it mis-orders under DP) next to our robust ranking.
5. **Rotation cadence**: `renyi_gap` and utility vs interval, DP vs non-DP (H4).

---

## 7. Go/No-Go gates (cheap; run BEFORE the campaign)

Run these in the **high-noise corner** (worst-noise ε=1, small r), where the
theory says the effect is largest, on one task (GLUE-SST2 or the Qwen setup):

0. **Are we in the valid regime?** Estimate $\hat P(N_\infty\ge k)$ from the
   noised per-layer spectra (Theorem 4′'s hypothesis). If $\approx1$ at ε=1 we are
   in-regime → proceed; if the cores are already strong-signal ($N_\infty<k$
   common) the stable-rank claim doesn't apply here → move to a smaller-r / lower-ε
   cell or reconsider.
1. **Non-trivial allocation?** Compute `k̂_ℓ = r_eff^(∞)` per layer under DP. Flat
   ⇒ allocation ≈ uniform ⇒ no win possible ⇒ **STOP / pivot**. Spread ⇒ go.
2. **Does AdaLoRA mis-rank under DP?** Compare its importance order on noised vs
   clean spectra. Barely changes ⇒ critique weak. Reorders ⇒ strong.
3. **Single-cell H3:** α=∞ vs α=1 allocation, 5 seeds, one task, ε=1. Real margin
   for α=∞ ⇒ **green-light the (ε, r) regime-map sweep.** Within noise even at ε=1
   (the most favorable cell) ⇒ the utility win likely won't materialize ⇒ pivot to
   the **analysis/critique paper** (§10).

Cost: ~5–10 runs. This is the closest thing to "for sure" available honestly, and
it also tells you *which cell* of the regime map to expand into first.

---

## 8. Ablations (in the full campaign)

- Allocation order α ∈ {0.5, 1, 2, ∞} (H3).
- **Rotation interval T ∈ {1, 2, 5, 10, 20} and β ∈ {0.9, 0.95, 0.99}** (H4) —
  measure `renyi_gap` and utility, DP vs non-DP. Prediction: T=1 worst under DP.
- Static vs dynamic allocation.
- Allocation granularity: per-layer vs per-module-type vs global.
- Probe length `r_probe` / warm-up length sensitivity.

---

## 9. Scale & sequencing

1. **§7 gates** (~1 day, ~10 runs) — go/no-go + locate the hot cell.
2. **Tier 1 regime map** — the headline figure: 4–6 methods × **(ε ∈ {1,3,8}) ×
   (r ∈ {4,8,16,32})** × ≥2 GLUE tasks × 5 seeds on RoBERTa-large (hours each).
   Builds the (ε, r) heatmap (H1/H2) and the α-order ablation (H3).
3. **Tier 2 hero** (7B, 3 seeds) — confirm the frontier slice on code.
4. **H4 rotation sub-study** (cheap; reuse Tier 1 task).
5. **Tier 3 generality** if 2–4 are positive.
6. **Theory tightening in parallel** — the finite-r RMT items in
   `vendor/lora-privacy/docs/renyi-proof-status.md` (the panel found the r→∞
   claims false; the finite-r versions are what the regime map validates).

---

## 10. Risk & fallback

The earlier α-sweep on final loss was within noise, so the utility win is NOT
guaranteed. If §7.3 is within noise even at ε=1, do **not** force it — pivot to a
still-publishable **analysis/critique** paper: *"Adaptive-rank fine-tuning is
systematically miscalibrated under DP"* (H-mechanism + AdaLoRA mis-ranking +
the theorem + the confirmed DP-vs-non-DP inflation), targeting a findings/analysis
track or a DP-ML / PEFT workshop. Either way the mechanism result already stands.
