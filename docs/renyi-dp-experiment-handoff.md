# Handoff / Resume Guide — Rényi effective-rank theory + DP-vs-non-DP experiment

**Written:** 2026-07-16. **Why:** Cadence/JetTrain GPUs went down mid-experiment
(0 workers dequeuing tenant-wide). This doc makes the work fully resumable on any
other environment (a coder box, a plain GPU host, or ZenML) with no dependence on
Cadence.

---

## TL;DR

We (a) wrote the ICLR-level theory + beginner primer for using **Rényi entropy as
a one-parameter (α) generalization of the Shannon effective rank** for
adaptive-depth/rotation under DP, and (b) instrumented the code to **test the
central mechanism**: *DP noise inflates the low-α effective rank of the core
spectrum, while the stable rank (α→∞) is not inflated.* The GPU experiment (two
identical LoRA-XSe runs, DP vs non-DP, only difference = noise) was queued on
Cadence but never ran because the cluster is down. **You can validate the theory
right now with no GPU** (see §3), and run the real experiment on any GPU with two
one-line commands (see §5).

---

## 1. What was produced this session (all committed)

**Submodule `vendor/lora-privacy` (branch `main`):**
- `docs/renyi-effective-rank-theory.md` — ICLR-level treatment: the Rényi rank
  family, the four classical ranks as α=0/1/2/∞ of one curve, the noise-bias
  theorem (bias ↓ monotonically with α; `N_∞ ≈ k + (r−k)/4c²`), the MSE result
  (stable rank is the optimal estimator), the "DP inverts the optimal Rényi
  order" thesis, and the connection to variable ranks + rotations.
- `docs/renyi-concepts-primer.md` — from-scratch explanation of every concept
  (singular values → energy spectrum → entropy → effective rank → the α family →
  bias/variance/MSE → Marchenko–Pastur/BBP → the DP inversion), each with an
  analogy and a worked example on one running spectrum σ=(10,1,1,1).
- `docs/renyi_synthetic_validation.py` — **no-GPU** script reproducing every
  numerical claim, including a DP-vs-non-DP demo mirroring the wandb metrics.
- `src/lora_privacy/peft_lora_xs/xse.py` — added `_renyi_rank_grid(p)` + the
  `_DIAG_ALPHAS` grid. Every rotation now emits, per layer, `r_eff_a0`,
  `r_eff_a0p5`, `r_eff_a1`, `r_eff_a2`, `r_eff_ainf` on the core R spectrum —
  independent of the configured adaptive-depth α, so one run yields the full
  α-curve.

**Parent repo `opaque` (branch `david-stan/manifold-experiments`):**
- `examples/train_causal_lm.py` — logs the layer-mean grid to wandb as
  `rotation/r_eff_a0 … r_eff_ainf` plus `rotation/renyi_gap_a0p5_ainf` (the
  headline: low-α minus stable rank).
- `.cadence/configs/renyi_dp_vs_nodp.yaml` — DP arm (ε=3).
- `.cadence/configs/renyi_nodp.yaml` — non-DP arm (`--noise-multiplier 0`).
- `docs/renyi-dp-experiment-handoff.md` — this file.

---

## 2. Resume on a fresh environment

```bash
# 1. clone parent + submodule
git clone git@github.com:JetBrains-Research/opaque.git
cd opaque
git checkout david-stan/manifold-experiments
git submodule update --init --recursive        # pulls vendor/lora-privacy at the right SHA

# 2. deps (uv — the project is a uv workspace; lora-privacy is a path dep)
uv sync --group examples --all-packages
#    (or: pip install -e . -e vendor/lora-privacy  + the examples extras)

# 3. secrets / tracking
export HF_TOKEN=...            # to download Qwen/Qwen2.5-Coder-7B + KStack
export WANDB_API_KEY=...
export WANDB_BASE_URL=https://jetbrains.wandb.io
export WANDB_ENTITY=federated-compute
export WANDB_PROJECT=opaque-lora-xs
```

Hardware: the real experiment uses Qwen2.5-Coder-7B → needs ~1×H100/H200 (≥40 GB;
80 GB comfortable). The theory validation (§3) needs nothing but numpy.

---

## 3. Validate the theory NOW (no GPU, no cluster)

This is the "does it make sense?" check and it does not need any of the above:

```bash
python vendor/lora-privacy/docs/renyi_synthetic_validation.py   # numpy only
```

Expected output (verified): identities match to 4 dp; bias falls monotonically
toward true k as α grows; MSE is minimized at α=∞; and the DP-vs-non-DP demo
shows the low-α inflation **gap ~2–3× larger under noise**. That last block is a
synthetic preview of exactly what the GPU runs measure.

---

## 4. The experiment, precisely

Two **identical** LoRA-XSe runs; the **only** difference is DP noise:

| Arm | command flag | mechanism |
|---|---|---|
| DP (ε=3) | *(default; noise calibrated)* | `dpsgd_acc.gaussian` |
| non-DP | `--noise-multiplier 0` | `acc.nonprivate()` (same clipping, no noise) |

Everything else identical: Qwen2.5-Coder-7B, KStack, r=16, p_e=0.333, 1 epoch,
adaptive-depth OFF. Rotation diagnostics log the full α-grid.

**Prediction:** DP run has `r_eff_a0p5 ≫ r_eff_ainf` (large
`rotation/renyi_gap_a0p5_ainf`); non-DP run's α-curve is much flatter (small
gap). The decisive signal is the *interaction*: gap_DP ≫ gap_non-DP. If the gaps
are similar, the noise-inflation mechanism is wrong and the theory needs revision.

---

## 5. Run it WITHOUT Cadence (any GPU box / coder instance)

The Cadence YAMLs just wrap these two commands. Run them directly:

```bash
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
metrics appear, without paying for the 7B): point at a tiny model and a few
steps, e.g. `--preset custom --model-name sshleifer/tiny-gpt2` (or any small
CausalLM) with `--num-train-samples 512 --num-epochs 1 --lora-method lora-xs
--lora-xse-p-e 0.333 --optimizer sgd --sgd-momentum 0.9`. Confirm the wandb run
shows `rotation/renyi_gap_a0p5_ainf`, then launch the real pair.

---

## 6. ZenML path (if moving off Cadence entirely)

The repo ships a ZenML integration (skill `jbr-fed-researcher:zenml-experiments`,
server `zenml.labs.jb.gg`, workspace `prod`). To port: wrap each of the two
commands in §5 as a ZenML step (a `@step` that shells out or calls the training
entrypoint), parameterized by `noise_multiplier` (None=DP, 0=non-DP), and a
`@pipeline` that runs both. Keep W&B logging as-is (it's independent of the
orchestrator). Confirm the ZenML stack has a GPU-backed orchestrator/step-operator
before submitting. Start from `list_stacks` / `list_pipelines` to see what's
available rather than assuming.

---

## 7. Reading the result

In wandb (`federated-compute/opaque-lora-xs`), overlay the two runs and plot:
- `rotation/r_eff_a0p5` and `rotation/r_eff_ainf` — DP's a0p5 sits well above its
  ainf; non-DP's two lines nearly coincide.
- `rotation/renyi_gap_a0p5_ainf` — the one-number headline; DP ≫ non-DP.

Note `rotation/r_eff_a0` ≡ 16 in both (Hartley counts all directions) — it's a
degenerate control, not a discriminator. The discriminating pair is **a0p5 vs
ainf**.

For the paper figure, average the gap over the late-training rotations for each
run and report gap_DP / gap_non-DP with a CI over ~3 seeds.

---

## 8. Cadence jobs left behind

Queued (never ran; cluster down). Cancel when convenient or let them auto-run if
the pool recovers:

- `55732` renyi-alpha-dp-eps3 (H100) — QUEUED
- `55733` renyi-alpha-nodp (H100) — QUEUED
- `55735` worker-liveness-probe (2 CPU) — QUEUED (proved the worker layer was down)

To relaunch on Cadence once it's back:
`start_execution_from_preset(".cadence/configs/renyi_dp_vs_nodp.yaml")` and
`... renyi_nodp.yaml` (project `JbrFed`).

---

## 9. Open next steps

1. Run the two-arm experiment (§5) on any available GPU; confirm gap_DP ≫ gap_non-DP.
2. If confirmed, produce the paper theory-figure: full α-curve, DP vs non-DP
   overlay, ~3 seeds, CIs.
3. Formalize the random-matrix proofs (Part VI of `renyi-effective-rank-theory.md`):
   deterministic-equivalent version of the bias theorem, BBP-consistency of the
   stable rank, the exploration-cost model closing the estimator↔task-optimum gap.
4. (Method) wire the stable-rank estimator into per-layer variable-rank allocation
   and an α-annealing schedule (see theory doc Part V).
