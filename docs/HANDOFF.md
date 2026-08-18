# LoRA-XSe — Handoff / Resume Guide

Everything needed to pick this up in a fresh session.
Date: 2026-08-14. Branch `david-stan/zenml-training`. HEAD at time of writing: `1d90cce6`.

---

## 0. TL;DR — where the project actually stands

**One robust result, and two honest negatives.**

| claim | status |
|---|---|
| **Rotating the frozen LoRA-XS basis recovers full-LoRA quality at 201× fewer trainable params, ε-free** | **CONFIRMED**, both ε=1 and ε=3, multi-seed, paired |
| Rényi order α changes utility | **REFUTED** by direct replication (§3) |
| Adaptive (per-matrix) depth beats fixed depth | **NO EVIDENCE**, and it costs ~20× seed variance (§3.4) |

So the paper is about **rotation**. The Rényi machinery is the diagnostic that says
*rotate deeply* (≥ ~55% of the rank); the specific α is a reported negative.

---

## 1. The headline result (finished, defensible)

Qwen2.5-Coder-7B, JetBrains/KStack (Kotlin), 50k train / 1k eval, **2 epochs (520
steps)**, DP-SGD, per-sample clipping, SGD momentum 0.9, lr 5e-2, batch 192, all 7
projection modules × 28 layers. Family prefix `cmp-`.

### 1.1 Main table (all cells now filled)

| method | params | ε=3 | ε=1 |
|---|---|---|---|
| LoRA r=16 | 40,370,176 | 0.34556 (n=2) | 0.34556 (n=3) |
| **LoRA-XSe r=32** | **200,704** | **0.34527 (n=3)** | **0.34559 (n=2)** |
| LoRA-XS r=32 (frozen) | 200,704 | 0.35196 (n=3) | 0.35454 (n=3) |

Paired differences (same seeds, common-random-numbers):

| contrast | ε=3 | ε=1 |
|---|---|---|
| **XSe − LoRA-XS** | **−6.69e-3** (per-seed −6.39, −6.69, −6.98 e-3) | **−8.72e-3** (−9.14, −8.30 e-3) |
| XSe − LoRA | −2.3e-4 (−0.04, +0.49 e-3) | +2e-5 (tie) |

**The gap grows as ε shrinks** (6.69e-3 → 8.72e-3), which is exactly the predicted
√d-noise mechanism: fewer trainable parameters ⇒ less injected noise ⇒ bigger advantage
in the noisier regime. That monotonicity in ε is a nice sanity check to put in the paper.

### 1.2 Why the ablation is clean

`cmp-basexs-*` and `cmp-xse-*` differ in **exactly two config fields**:

| field | LoRA-XS | LoRA-XSe |
|---|---|---|
| `lora_xse_p_e` | 0 | 0.5 |
| `lora_xse_rotation_step_interval` | null | 3 |

With `p_e=0` there is no explore block, so rotation is a structural no-op. Confirmed
empirically: the `basexs` runs log **no** `rotation/r_e_dyn` metric at all.

### 1.3 Parameter counts (computed — never logged, so recompute if challenged)

hidden 3584, intermediate 18944, 28 layers, 4 KV heads (kv width 512):
- LoRA r=16: `Σ_modules r(fan_in+fan_out) × 28` = **40,370,176**
- LoRA-XS/XSe r=32: `r² × 7 × 28` = **200,704** → **201×**
- √d noise reduction = **14.2×**

---

## 2. Depth: the one real lever (finished)

Non-DP, 29 verified runs (`state=finished`, ≥25 evals), keyed on realised mean depth
`rotation/r_e_dyn`:

| depth | eval loss | vs best |
|---|---|---|
| 1.00 | 0.34700 | +3.56e-3 |
| 3.13 | 0.34537 | +1.93e-3 |
| 5.00 | 0.34429 | +8.5e-4 |
| 9.00 | 0.34368 | +2.4e-4 |
| **13.00** | **0.34344** | — |
| 15.00 | 0.34364 | +2.0e-4 |

**Robust conclusion:** the depth-1 → depth-13 gain of **3.6e-3 is ≈7× the adaptive-run
noise floor** (§3.3). Refreshing ~1 direction is nearly as bad as not rotating.

**Rule: refresh ≥ ~55% of the rank (≥9 of 16). Above that it is flat.**

---

## 3. The α question — RESOLVED (refuted), and how

> **SUPERSEDED 2026-08-14 by `docs/renyi-alpha-theory-final.md`.** The verdict below (α refuted) is
> correct, but the route is weaker than it needed to be and two numbers here are now wrong:
> - §3.1–3.2 treat α ∈ {∞,1,2} at m=1 as three *conditions*. They are **replicates**: α enters only
>   via `⌊N_α⌋`, which is 1 on ≥98.1 % of matrix-steps for all three, so the runs execute the same
>   algorithm. Their 5.7e-4 spread is the **noise floor**, not an effect that "vanished".
> - §3.3's framing "adaptive runs are ~20× noisier" is **confounded**: the floor tracks *depth*
>   (3.0e-5 at depth 5, 1.8e-4 at 13, 3.0e-4 at 14), and every adaptive run here is deep. Untangling
>   it needs a `fixed-re13` seed triple — the one experiment still worth buying.
> - §3.4's "adaptive depth buys nothing" is true for a stronger reason: at α ≥ 1 the rule assigns the
>   *same* depth to all 196 matrices, so there was no adaptivity to buy anything.
>
> Read the new doc for the theorems (quantization ⇒ ≤ r distinguishable α; scale-blindness ⇒ wrong
> sign of response to noise) and the mediation bound (max possible α effect 2.8e-5 vs 3.0e-4 floor).

This went through three stages. The full history matters because the first two were
wrong in *different* ways.

### 3.1 The test

Three non-DP configs land at **identical realised mean depth 14.00**, same seed 42,
same margin m=1, differing only in α:

| α | s42 |
|---|---|
| ∞ | 0.34362 |
| 1 | 0.34408 |
| 2 | 0.34419 |

Spread 5.7e-4 — apparently 20× the non-DP floor I was using (2.3e-5). So I replicated
the two extremes at seed 43.

### 3.2 The result — the effect vanishes

| seed | α=∞ | α=2 | gap |
|---|---|---|---|
| 42 | 0.34362 | 0.34419 | **+5.7e-4** |
| 43 | 0.34353 | 0.34348 | **−5e-5** |

**The gap flips sign and collapses 11×.** Per-config seed spread: α=∞ moves 9e-5, α=2
moves **7.1e-4**. So α=2's seed-42 value was simply a high draw.

**Conclusion: α does not affect utility at matched depth.** Not a real effect.

### 3.3 The key by-product: adaptive runs are ~20× noisier

| family | same-config seed sd |
|---|---|
| non-adaptive, fixed `p_e`, depth 5 (`renyi-nodp-*`) | **2.3e-5** |
| **adaptive depth** (`seedrep-ad-nodp-*`) | **~5.0e-4** |

This is the number to use for every adaptive-depth comparison. Against it:

| effect | size | vs floor |
|---|---|---|
| depth-14 α spread | 5.7e-4 | 1.1× → noise |
| plateau 9–15 spread | 5.6e-4 | 1.1× → noise |
| **shallow-depth penalty** | 3.56e-3 | **7.1× → real** |

### 3.4 New negative: adaptive depth buys nothing and costs variance

At matched mean depth 13: **fixed** `r_e=13` scores 0.34354, **adaptive** α=∞ m=2 scores
0.34344 — a 1.0e-4 difference, **0.2× the floor**. No evidence adaptivity helps the
mean, while it demonstrably raises variance ~20×.

That is a genuine, reportable negative on the "adaptive depth" half of LoRA-XSe. (n is
small — 2 configs × 2 seeds; the s44 pair in §4 firms it up.)

### 3.5 Error history — worth reading before trusting any noise claim here

1. **§11 of `renyi-alpha-utility-verdict.md`**: declared the plateau flat using a
   "within-depth scatter" of 3.7e-4 computed from runs at the same depth but *different
   α*. That is a **config contrast used as a noise floor** — circular, since it assumes
   α has no effect to conclude α has no effect. *Right answer, invalid derivation.*
2. **§12**: correctly attacked that, but substituted the floor from the *non-adaptive,
   fixed-`p_e`, depth-5* family (2.3e-5) — the wrong family. Concluded α mattered and
   reopened the question. *Valid critique, wrong floor, wrong conclusion.*
3. **Now**: direct seed replicates of *adaptive* configs give 5.0e-4. Plateau is flat.
   *Right answer, valid derivation.*

**Lesson:** a noise floor must come from replicates of *the same configuration family*
you are comparing. Neither a cross-config spread nor a different family will do.

---

## 4. PENDING — the only runs in flight

Two runs, auto-submitted, tracked by `campaign_logs/final_waiter.sh`:

- `seedrep-ad-nodp-ainf-m1-s44`
- `seedrep-ad-nodp-a2-m1-s44`

**Purpose:** third seed for §3.2/§3.4, taking each config to n=3.
**Expected:** gap stays ≈0 and within ±5e-4. Would only be surprising if α=2 again came
in ~7e-4 above α=∞, which would reopen §3.
**Where:** W&B `federated-compute/opaque-lora-xs` (base URL `https://jetbrains.wandb.io`).

Check them:

```bash
cd /Users/david.stanojevic/PycharmProjects/opaque
WANDB_BASE_URL=https://jetbrains.wandb.io uv run python - <<'PY'
import os; os.environ.setdefault("WANDB_BASE_URL","https://jetbrains.wandb.io")
import wandb; api=wandb.Api(timeout=60)
for r in api.runs("federated-compute/opaque-lora-xs",
                  filters={"display_name":{"$regex":"^seedrep-ad-nodp-"}}):
    print(r.name, r.state, r.summary.get("_step"),
          r.summary.get("eval/loss"), r.summary.get("rotation/r_e_dyn"))
PY
```

Then update §3.2/§3.4 tables and `docs/renyi-status-summary.md`.

---

## 5. Environment / how to run

```bash
cd /Users/david.stanojevic/PycharmProjects/opaque
git checkout david-stan/zenml-training

# VPN must be up — verify BEFORE submitting (this has bitten repeatedly)
host zenml.labs.jb.gg && \
  curl -s -o /dev/null -w '%{http_code}\n' --max-time 10 https://zenml.labs.jb.gg/api/v1/info
# expect 200; anything else = VPN down, submissions will fail with NameResolutionError

export OPAQUE_DOCKER_REGISTRY=europe-west4-docker.pkg.dev/gke-dev-dws-jbr/ml
export OPAQUE_DOCKER_TAG=david-stan-zenml-training-7e97389
export WANDB_MODE=online          # see gotcha 1
```

Submit (`nodp` | `dp` | `both`; `dp` arm = ε=3 from the preset):

```bash
# non-DP adaptive-depth run
XSE_ADAPTIVE_DEPTH=1 XSE_ADAPTIVE_DEPTH_ALPHA=inf XSE_ADAPTIVE_DEPTH_MARGIN=2 \
  .zenml-client/bin/python deploy/zenml/run.py nodp --run-name NAME --seed 42 \
    --extra --lora-xse-p-e 0.333 --lora-r 16 --num-epochs 1

# the cmp-family DP baseline (frozen basis, 2 epochs, r=32)
.zenml-client/bin/python deploy/zenml/run.py dp --run-name NAME --seed 42 \
    --extra --lora-r 32 --lora-alpha 32 --lora-xse-p-e 0 --num-epochs 2 --microbatch-size 16
```

Always `--dry-run` first and read the resolved argv.

### Gotchas that have each cost a full batch of runs

1. **`WANDB_MODE=disabled` in your shell silently propagates into the pod** and drops
   all metrics. `run.py` now overrides it to `online` for GPU submits (escape hatch
   `OPAQUE_ALLOW_WANDB_DISABLED=1`). Don't defeat this.
2. **Put `run.py` flags BEFORE `--extra`.** `--extra` is `argparse.REMAINDER` and will
   swallow `--dry-run`, causing an accidental real submission.
3. **Never pass a joined shell variable as one arg** (`$DS`) — argparse dies before
   `wandb.init`, so the run fails with no W&B record at all.
4. **Duplicate flags are fine** — `--extra` values come after the arm defaults and
   argparse takes the last occurrence. That is how the overrides above work.
5. **Cluster capacity ≈ 5 concurrent** (1Ti/200Gi quota, namespace
   `zenml-workload-common-gpus`). Exceeding it gets us co-located and runs crash
   mid-training. **Cap at 3–5.**
6. **Verify `state=="finished"` AND step count before reading any result.** 25 runs have
   crashed mid-training at non-deterministic steps; a partial run reads as a plausible
   but wrong number. I published a conclusion from a partial batch once and had to
   retract the commit.
7. **`--microbatch-size 4`** for plain LoRA under DP or it OOMs at step 0 (450× more
   params under per-sample clipping).
8. ZenML client 0.94.6 vs server 0.96.1 version-mismatch warning is benign.

---

## 6. Open questions, ranked

1. **A Kotlin downstream metric.** *The most important scientific gap.* Right now
   utility = eval loss only. Python pass@1 measures **forgetting**, not gain: the base
   model scores 0.680 and KStack is Kotlin, so fine-tuning can only push it down. BPB is
   implemented (`--eval-bpb`, `--eval-bpb-samples`, `--eval-bpb-microbatch`) but **has
   never been run** — the image containing it (`8140447`) was never built. Build it and
   use it, or drop all downstream claims.
2. **Finite-r RMT.** The MP/BBP/spike-map (`ρ = θ + σ²/θ`) arguments are asymptotic. A
   rigorous finite-r "optimal α" theorem is not done and probably needs a collaborator.
   Lower priority now that α is empirically flat — this would explain *why* it's flat.
3. **Rotation interval / `p_e` sweep at 2 epochs.** Everything in §1 uses
   `p_e=0.5, interval=3`, never tuned in the 2-epoch regime. Cheap, could add margin.
4. **AdaLoRA miscalibration analysis.** No GPU needed. Note peft's AdaLoRA is
   structurally incompatible with our functional/vmap per-sample-grad harness (it needs
   `.grad` on module params), so the baseline was realised as
   same-allocation-with-naive-score.
5. **Vision.** `examples/train_vision.py` does not exist. Scope creep unless a reviewer
   demands cross-modality.

---

## 7. Dead ends — do not re-run

- **Per-matrix rank *allocation*** (probe-spectra and W0-based, `--lora-xs-rank-alloc`):
  null in both DP and non-DP. Heterogeneity across the 196 matrices does not help.
- **Single-seed ε=1 anything.** ε=1 seed sd is 1.5e-3 to 9.9e-3 — one family
  (`fixed-re9`) spans **1.9e-2** across three seeds, larger than every effect we chase.
  All earlier ε=1 α orderings (incl. "α=∞ worst at ε=1") are unsupported.
- **1-epoch comparisons against LoRA.** LoRA scores 0.399 at 1 epoch vs 0.3456 at 2 —
  badly undertrained because DP noise ∝ √d and its d is 201× larger. A 1-epoch table
  flatters us and a reviewer will catch it. **All headline numbers use 2 epochs.**

---

## 8. Noise floors — pin these to the wall

| regime / family | same-config seed sd |
|---|---|
| non-DP, non-adaptive fixed `p_e` | 2.3e-5 |
| **non-DP, adaptive depth** | **5.0e-4** |
| DP ε=3 (1 epoch) | 3.7e-4 |
| DP ε=3, 2-epoch `cmp-` arms | 1.0e-4 – 3.2e-4 |
| DP ε=1 (1 epoch) | 1.5e-3 – 9.9e-3 |
| DP ε=1, 2-epoch `cmp-` arms | 7e-6 – 5.8e-4 |

Use the floor from the **same family** as the comparison. This is the single most
error-prone thing in the project (§3.5).

---

## 9. Key files

| path | what |
|---|---|
| `vendor/lora-privacy/.../peft_lora_xs/xse.py` | the optimizer; rotation + adaptive depth. `_ADAPTIVE_DEPTH*` env vars at L45–63; rotation at L401+; blockwise `R_new` at L524–541 |
| `vendor/lora-privacy/.../peft_lora_xs/allocation.py` | per-matrix rank allocation (dead end, kept) |
| `examples/train_causal_lm.py` | trainer; preset `qwen-coder-kstack-lora` at L1030 (sets ε=3, 2 epochs, r=16) |
| `deploy/zenml/run.py` | submit; `ARMS` at L56 |
| `docs/renyi-status-summary.md` | consolidated position for the lead |
| `docs/renyi-alpha-utility-verdict.md` | α investigation; **§11 and §12 are both superseded — see §3.5 above** |
| `docs/renyi-effective-rank-theory.md` | the proofs |
| `docs/renyi-concepts-primer.md` | beginner-friendly walkthrough |
| `campaign_logs/final_waiter.sh` | the in-flight watcher |

---

## 10. Proof status (one line each)

- **Monotonicity of `N_α`** — PROVEN, exact:
  `d/dα log N_α = −D_KL(p^(α)‖p)/(1−α)² ≤ 0`, escort `p^(α)_i = p_i^α/Σp_j^α`.
  Reproduced by three adversarial proof attempts.
- **ε-free adaptivity** — PROVEN, one line: the rotation is a function of the already
  privatised momentum, so post-processing immunity applies. This is load-bearing for §1.
- **Mediation / additive separability** — PROVEN: α enters only via the integer vector
  `(r_e,ℓ)_ℓ`, `r_e = r − ⌊N_α⌋ − m`. So `m` is a pure offset (verified at m=0: α=0.5
  and α=2 both realise depth 15.00, agreeing to 4e-5), and the **rounding staircase**
  explains why α ≳ 0.8 is behaviourally identical. Note it constrains the *vector*, not
  its mean — §3 tested the mean and found nothing.
- **T3, T4** — RETRACTED. T3 diverges Θ(r) (was claimed convergent); T4 is not
  universally MSE-optimal. Removed, not patched.
- **Finite-r RMT** — INCOMPLETE (see §6.2).
