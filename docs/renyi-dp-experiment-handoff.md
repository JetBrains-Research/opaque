# Rényi Effective Rank under DP — Theory + Experiment Guide

A self-contained onboarding + resume guide. It carries the **theory** you need to
understand the work, then shows how to **validate it with no GPU** and how to
**run the confirming experiment on any GPU environment**.

Full write-ups live in the submodule:
- `vendor/lora-privacy/docs/renyi-effective-rank-theory.md` — ICLR-level treatment
- `vendor/lora-privacy/docs/renyi-concepts-primer.md` — every concept from scratch
- `vendor/lora-privacy/docs/renyi_synthetic_validation.py` — no-GPU numerical proof

---

## Part I — The theory

### The problem

In LoRA-XSe we fine-tune with a low-rank correction `ΔW = U_r R V_r^T` and, every
few steps, **rotate** the basis: keep the directions of the trainable core `R`
that carry signal, re-explore the rest. The single decision each rotation makes is

> **how many of the `r` directions are actually doing useful work?**

Call that the *effective rank*. Keep that many; re-explore the remainder. Under
DP-SGD the gradient — and hence `R` — is buried in isotropic noise, so this count
must be estimated from a **noisy** spectrum. Getting it right is the whole game.

### Effective rank and the α dial

Square-and-normalize `R`'s singular values into an energy spectrum
`p_i = σ_i² / Σσ_j²`. The exponential of entropy is an *effective count* of
directions. **Rényi entropy** turns this into a one-parameter family indexed by
α ≥ 0:

$$
r_{\text{eff}}^{(\alpha)} = \exp H_\alpha(p) = \Big(\textstyle\sum_i p_i^\alpha\Big)^{\frac{1}{1-\alpha}}.
$$

α is a **skepticism dial about weak directions**: low α lets faint directions
count; high α trusts only the dominant ones. Concretely, one family unifies four
classical rank surrogates (proof in the theory doc):

| α | measure | reads |
|---|---|---|
| 0 | matrix rank (Hartley) | counts every nonzero direction |
| 1 | Roy–Vetterli effective rank (Shannon) | the honest average |
| 2 | participation number `‖R‖_F⁴/Σσ⁴` | favors strong directions |
| ∞ | **stable rank** `‖R‖_F²/‖R‖_2²` | only the top direction(s) |

`r_eff^(α)` is **non-increasing in α**: raising α can only lower the estimate.
(For the adaptive-depth mechanism: higher α ⇒ smaller kept rank ⇒ deeper
re-exploration.)

### The core result: α as a noise filter

A DP spectrum is a **few signal spikes on a broad noise floor** (the
Marchenko–Pastur bulk; a spike is only visible above the BBP edge). Model it as
`k` signal directions of strength `s` plus `r−k` noise directions of strength `ν`,
and let `c = s/edge` be the spike-to-edge ratio. Then (Theorem 3 in the doc):

- **α → 0:** `r_eff = r` — counts the entire noise floor (bias `r−k`).
- **α → ∞:** `r_eff = k + (r−k)·ν²/s² ≈ k + (r−k)/4c²` — bias decays like `1/c²`
  and `→ k` as spikes strengthen.
- **monotone in between**, so the overcount **bias falls monotonically with α**.

The mechanism in one line: relative to Shannon, order-α weights a direction by
`p_i^{α−1}`. For small (noise) `p_i` this **blows up when α<1** (amplifies the
noise floor) and **vanishes when α>1** (filters it).

### Which α is the best estimator of k?

`MSE(α) = bias² + variance`. Bias falls with α; the variance does *not* grow fast
enough to create an interior optimum (verified numerically), so **MSE is monotone
and the stable rank (α=∞) is the MSE-optimal rank estimator under DP noise.**

> **Estimator vs task.** α=∞ is optimal for *estimating* k. The best α for the
> *training task* is a shallow plateau around α≈2–5 (within eval noise), because
> the estimate feeds an *action* (exploration depth `r_e = r − ⌈r_eff⌉ − m`) whose
> over-aggression has its own cost. That estimator↔task gap is a named open
> problem, not a loose end.

### The headline

The Rényi family is the ecology **diversity index**, where the *interesting*
regime is α ∈ (0,1) — deliberately rewarding rare components so you don't miss
rare-but-real species. DP breaks the premise: the rare components **are the
noise**. So the diversity instinct is exactly backwards, and the optimal order
flips:

$$
\underbrace{\alpha\in[0,1]}_{\text{optimal for noiseless diversity}}
\ \xrightarrow{\ \text{add DP noise}\ }\
\underbrace{\alpha\to\infty\ (\text{stable rank})}_{\text{optimal for signal-rank estimation}}.
$$

**"DP inverts the optimal Rényi order."** This is provable (Theorem 3 + the MSE
result) and explains the empirics: α=0 is decisively worst, higher α is better,
(0,1) is strictly worse than α≥1.

### From estimator to method

The corrected (large-α) estimate drives: (i) the rotation's **keep/explore
partition** — keep confirmed spikes, refresh noise-occupied slots (a spectral
matching-pursuit); (ii) **variable per-layer rank** allocation using a
noise-robust `k̂_ℓ`; (iii) an **α-annealing** schedule as spikes emerge over
training. See Part V of the theory doc.

### What's proven vs empirical vs open

- **Proven / exact:** the four-ranks-are-one-curve identities; monotonicity; the
  two-level bias theorem incl. `N_∞ ≈ k + (r−k)/4c²`.
- **Empirically verified (synthetic):** bias monotone → k; MSE monotone → α=∞;
  the DP-vs-non-DP inflation gap. All reproduced by the no-GPU script.
- **Open (Part VI of the doc):** rigorous random-matrix (deterministic-equivalent)
  versions of the theorem; BBP-consistency of the stable rank; an
  exploration-cost model closing the estimator↔task gap; matching-pursuit
  recovery guarantees.

---

## Part II — Validate the theory now (no GPU)

This is the "does it make sense?" check and needs only numpy:

```bash
python vendor/lora-privacy/docs/renyi_synthetic_validation.py
```

Verified output: identities match to 4 dp; bias falls monotonically toward the
true `k` as α grows; MSE is minimized at α=∞; and the DP-vs-non-DP demo shows the
low-α inflation **gap ~2–3× larger under noise** — a synthetic preview of exactly
what the real runs measure.

---

## Part III — The confirming experiment

Two **identical** LoRA-XSe runs whose **only** difference is DP noise:

| Arm | flag | mechanism |
|---|---|---|
| DP (ε=3) | *(default; noise calibrated)* | Gaussian DP-SGD |
| non-DP | `--noise-multiplier 0` | `acc.nonprivate()` — same clipping, no noise |

Everything else identical: Qwen2.5-Coder-7B, KStack, r=16, `p_e=0.333`, 1 epoch,
adaptive-depth OFF. Each rotation logs the full α-grid.

**Instrumentation** (already in the code): `xse.py::_renyi_rank_grid` computes
`r_eff^(α)` for α ∈ {0, 0.5, 1, 2, ∞} on the core spectrum every rotation,
independent of the configured adaptive-depth α. `train_causal_lm.py` logs the
layer-means to W&B as `rotation/r_eff_a0 … r_eff_ainf` and the headline
`rotation/renyi_gap_a0p5_ainf` (low-α minus stable rank).

**Prediction:** the DP run has `r_eff_a0p5 ≫ r_eff_ainf` (large gap); the non-DP
run's α-curve is much flatter (small gap). The decisive signal is the
**interaction**: gap_DP ≫ gap_non-DP. If the gaps are similar, the
noise-inflation mechanism is wrong and the theory needs revision.

---

## Part IV — Run it on any GPU

Needs ~1 GPU with ≥40 GB (80 GB comfortable) for the 7B model.

```bash
# setup
git submodule update --init --recursive
uv sync --group examples --all-packages
export HF_TOKEN=... WANDB_API_KEY=...
export WANDB_BASE_URL=https://jetbrains.wandb.io
export WANDB_ENTITY=federated-compute WANDB_PROJECT=opaque-lora-xs

# DP arm (ε=3)
RUN_NAME=renyi-alpha-dp-eps3 uv run python examples/train_causal_lm.py \
  --preset qwen-coder-kstack-lora --lora-method lora-xs \
  --lora-xse-p-e 0.333 --num-epochs 1

# non-DP arm (identical except no noise)
RUN_NAME=renyi-alpha-nodp uv run python examples/train_causal_lm.py \
  --preset qwen-coder-kstack-lora --lora-method lora-xs \
  --lora-xse-p-e 0.333 --num-epochs 1 --noise-multiplier 0
```

**Cheap smoke test first** (validate the pipeline + that `rotation/r_eff_*`
metrics appear, without paying for the 7B): swap in a tiny CausalLM, e.g.
`--preset custom --model-name sshleifer/tiny-gpt2 --num-train-samples 512
--num-epochs 1 --lora-method lora-xs --lora-xse-p-e 0.333 --optimizer sgd
--sgd-momentum 0.9`. Confirm the run logs `rotation/renyi_gap_a0p5_ainf`, then
launch the real pair.

On a different orchestrator (e.g. ZenML), wrap each command above as a step
parameterized by `noise_multiplier` (None=DP, 0=non-DP); W&B logging is
orchestrator-independent.

---

## Part V — Reading the result

In W&B (`federated-compute/opaque-lora-xs`), overlay the two runs:
- `rotation/r_eff_a0p5` vs `rotation/r_eff_ainf` — DP's a0p5 sits well above its
  ainf; non-DP's two lines nearly coincide.
- `rotation/renyi_gap_a0p5_ainf` — the one-number headline; DP ≫ non-DP.

Note `rotation/r_eff_a0 ≡ 16` in both (Hartley counts all directions) — a
degenerate control, not a discriminator. The discriminating pair is **a0p5 vs
ainf**. For the paper figure, average the gap over late-training rotations per run
and report gap_DP / gap_non-DP with a CI over ~3 seeds.

---

## Next steps

1. Run the two-arm experiment (Part IV); confirm gap_DP ≫ gap_non-DP.
2. If confirmed, produce the theory figure: full α-curve, DP vs non-DP overlay,
   ~3 seeds, CIs.
3. Formalize the open proofs (Part VI of `renyi-effective-rank-theory.md`).
4. Wire the stable-rank estimator into variable per-layer rank + α-annealing
   (theory doc Part V).
