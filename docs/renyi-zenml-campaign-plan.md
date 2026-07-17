# Rényi Allocation — Full ZenML Campaign (execution-ready handoff)

**For:** an agent with GPU access + ability to build/push the ZenML training image.
**Goal:** run the extensive empirical campaign to decide whether Rényi/stable-rank
per-layer allocation pays off — **non-DP α-sweep first, then DP**, across as many
tasks as feasible (incl. vision). Priority order and rationale below.

This doc is self-contained: method spec, what's already implemented vs what you
must build, the exact run recipes, ZenML mechanics, and the analysis/decision
gates. Theory + regimes: `vendor/lora-privacy/docs/renyi-effective-rank-theory.md`
(+ `renyi-proof-status.md`). Hypotheses: `docs/renyi-utility-experiments-plan.md`.

---

## 0. Three findings that shape this plan

1. **Per-layer variable rank now works.** `LoraXSConfig.rank_pattern` + robust
   per-module matching in `model.py` are implemented; everything downstream
   already handled heterogeneous ranks. Budget `Σ_ℓ r_ℓ²` is held fixed to a
   uniform run, so allocation vs uniform is apples-to-apples (same params → same
   DP noise).
2. **peft-AdaLoRA is INCOMPATIBLE with this DP harness** (it needs `.grad` on
   module params + in-place mutation; this harness is functional/vmap with
   detached params — AdaLoRA's importance scorer crashes on `p.grad is None`).
   So the AdaLoRA *baseline* is realized as **the same allocation mechanism with
   a noise-naive score** (Shannon α=1, or `frob` energy = AdaLoRA-style
   magnitude importance). This is the *cleaner* comparison anyway: same code,
   only the scoring changes → isolates the Rényi-order insight. Report the
   peft-AdaLoRA incompatibility as a finding.
3. **ZenML campaign = shell loop over `python deploy/zenml/run.py`**, no ZenML
   code edits needed — BUT the trainer runs from the image baked at
   `/opt/opaque`, so **you must rebuild+push the image after pulling this
   branch** (`deploy/zenml/build_and_push.sh`), or set `OPAQUE_REPO_DIR`.

---

## 1. What is already implemented (this branch)

- **Method** — per-layer rank via `--lora-xs-rank-pattern-json <alloc.json>`
  (train_causal_lm.py). Loads `{module: r_ℓ}`; a post-build check prints the
  applied rank spread and WARNS on silent fallback.
- **Allocation math** — `lora_privacy.peft_lora_xs.allocation` (`renyi_rank`,
  `score_layer`, `allocate_ranks`, `allocate_from_spectra`); unit-tested offline
  (budget conserved, clamped, busy-layer→higher rank).
- **Probe dump** — `--dump-core-spectra <probe.json>` dumps each layer's core-R
  singular values after a short `--max-steps N` run (the DP-faithful source).
- **Allocator CLI** — `examples/compute_rank_allocation.py`: `--from-spectra`
  (probe, DP-faithful) or `--from-model` (W0, data-free); `--score-mode
  {renyi,frob,uniform}`; `--alpha {inf,2,1,0.5}`; fixed `--uniform-r` budget.
- **α-grid diagnostics** — every rotation logs `rotation/r_eff_a{0,0p5,1,2,inf}`
  + `rotation/renyi_gap_a0p5_ainf` (already merged).
- **Local batch** — `examples/run_renyi_campaign.sh` (method family × DP × r ×
  cadence × seed), tiered, DRY_RUN/FAST knobs.

## 2. What YOU must build first (blocking a full campaign)

- **Vision harness** — `examples/train_vision.py` (spec in §7). The DP core
  (`make_functional` + `clipped_grad` + gaussian noise + accounting + torchopt)
  is model-agnostic and reused unchanged; you write dataset/model/collate/loss.
  Verify with a tiny CPU smoke run before GPU.
- **(If wanted) real GLUE/vision presets** in `train_causal_lm.py` /
  `train_vision.py` mirroring `qwen-coder-kstack-lora`.
- **Rebuild the ZenML image** from this branch (see §6).

---

## 3. The roster (methods × the two "α"s — keep them distinct)

There are **two** independent α knobs; do not conflate:
- **allocation-α** — the Rényi order used to *score layers* for rank allocation
  (this doc's new method). `inf` = stable rank (ours), `1` = Shannon (naive),
  `frob` = AdaLoRA-style. Set via `compute_rank_allocation.py --alpha`.
- **adaptive-depth-α** — the existing within-layer re-exploration order
  (`XSE_ADAPTIVE_DEPTH_ALPHA`). Leave at default (1.0) / off unless sweeping it.

Methods to compare (all at matched budget `Σr_ℓ²`):

| id | method | how |
|----|--------|-----|
| U | uniform LoRA-XS | `--lora-method lora-xs` (no rank_pattern) |
| U+rot | uniform LoRA-XSe (rotation) | `+ --lora-xse-p-e 0.333` |
| A∞ | **alloc, stable rank (OURS)** | probe → `compute_rank_allocation --alpha inf` → `--lora-xs-rank-pattern-json` |
| A1 | alloc, Shannon (naive) | `--alpha 1` |
| A½ | alloc, α=0.5 (diversity, should be worst under DP) | `--alpha 0.5` |
| Afrob | alloc, energy importance (AdaLoRA-style) | `--score-mode frob` |
| L | standard LoRA (ref) | `--lora-method lora` |

The **thesis test** is A∞ vs {A1, Afrob, U}: same budget, only the score differs.

---

## 4. Task suite (cover as much as feasible)

| Tier | Task | Model | Status | Metric |
|------|------|-------|--------|--------|
| T1 | GLUE (SST-2, MNLI, QNLI, QQP) | RoBERTa-large | needs a preset (cheap, high-power, canonical DP-FT benchmark) | accuracy |
| T2 | KStack (code) | Qwen2.5-Coder-7B | **ready** (`qwen-coder-kstack-lora`) | HumanEval+/MBPP+ pass@1, eval/loss |
| T3 | commonsense-8 / GSM8K | Llama-2-7B or Qwen | needs a preset | accuracy / EM |
| T4 | CIFAR-100 (+ CIFAR-10) | ViT | **needs `train_vision.py`** (§7) | top-1 accuracy |

Start with **T2 (ready)** to validate the pipeline, then **T1** (cheapest, most
seeds → statistical power), then T3/T4 for breadth.

---

## 5. PHASING — non-DP α-sweep FIRST, then DP (priority order)

### Phase 1 — non-DP, extensive (establish the control) 🥇
Run the full roster (§3) **without** DP (`--noise-multiplier 0`) on T2 (+ T1 when
ready), all seeds. **Expectation (state it up front):** without noise the
allocation-α should barely matter (clean spectra → α-invariant), and A∞ ≈ A1 ≈
Afrob. That *flat* result is the **control**: it proves any α-effect seen in
Phase 2 is caused by DP noise, not by allocation per se. Also answers your core
question here: does variable allocation (any α) and LoRA-XSe beat **uniform
LoRA-XS**, and tie **LoRA**, with no privacy?

### Phase 2 — DP, the thesis 🥈
Repeat the roster **with** DP at ε ∈ {1, 3, 8} on T2 (+ T1). **Prediction:** A∞ ≥
A1 ≥ A½ and A∞ ≥ Afrob ≥ U, with the gap growing as ε ↓. This is where stable-
rank allocation should pay off.

### Phase 3 — regime map 🥉
A∞ vs U across (ε ∈ {1,3,8}) × (r ∈ {8,16,32}); heatmap of the advantage +
overlay the `P(N∞≥k)` contour. The theory says the win concentrates in the
high-noise / small-r corner.

### Phase 4 — breadth
T3 (reasoning) + T4 (vision), Phase-1/2 subset, to show generality.

Run Phase 1 fully before Phase 2 (per your priority). If Phase 1 shows variable
allocation never beats uniform even without noise, escalate before spending DP
budget.

---

## 6. Run recipes (exact)

### 6a. The allocation pipeline (3 steps)
```bash
# (1) short probe: uniform r, dump the (noised, if DP) core spectra
uv run python examples/train_causal_lm.py --preset qwen-coder-kstack-lora \
    --lora-method lora-xs --lora-xse-p-e 0.333 --target-epsilon 3 \
    --max-steps 60 --dump-core-spectra probe_eps3_r16.json --num-epochs 1

# (2) allocate at the SAME budget (uniform r=16), one JSON per scoring choice
for a in inf 1 0.5; do
  uv run python examples/compute_rank_allocation.py --from-spectra probe_eps3_r16.json \
      --score-mode renyi --alpha $a --uniform-r 16 --out alloc_eps3_a$a.json
done
uv run python examples/compute_rank_allocation.py --from-spectra probe_eps3_r16.json \
    --score-mode frob --uniform-r 16 --out alloc_eps3_frob.json

# (3) train each arm with its allocation (budget/noise identical to uniform r=16)
uv run python examples/train_causal_lm.py --preset qwen-coder-kstack-lora \
    --lora-method lora-xs --lora-xse-p-e 0.333 --target-epsilon 3 --num-epochs 1 \
    --lora-xs-rank-pattern-json alloc_eps3_ainf.json --eval-humaneval --eval-mbpp
```
Non-DP arms: swap `--target-epsilon 3` → `--noise-multiplier 0` (probe AND train).
Note: the probe consumes privacy; for strict accounting either fold the probe
steps into the budget or probe on a public split. For the experiment, keep the
probe short and identical across arms so it is a fair, controlled comparison.

### 6b. Launch on ZenML (campaign loop)
```bash
# one-time: rebuild image from THIS branch so /opt/opaque has the new code
bash deploy/zenml/build_and_push.sh            # sets a tag
export OPAQUE_DOCKER_TAG=<tag-from-build>
# optional GPU fallback if H100 saturated:
export OPAQUE_GPU_TYPE=A100_80GB OPAQUE_CPU_COUNT=12 OPAQUE_MEMORY_GB=160

# campaign: one ZenML run per arm × privacy × seed. --extra passes trainer flags.
for s in 42 43; do
 for dp in "nodp:--noise-multiplier 0" "eps3:--target-epsilon 3" "eps1:--target-epsilon 1"; do
   name=${dp%%:*}; flag=${dp#*:}
   for arm in "U:" "Ainf:--lora-xs-rank-pattern-json alloc_${name}_ainf.json" \
              "A1:--lora-xs-rank-pattern-json alloc_${name}_a1.json"; do
     an=${arm%%:*}; af=${arm#*:}
     python deploy/zenml/run.py dp --seed "$s" \
        --run-name "camp-$an-$name-s$s" \
        --extra --lora-method lora-xs --lora-xse-p-e 0.333 --num-epochs 1 $flag $af
   done
 done
done
```
(`run.py dp` is just the arm carrying the base flags; `--extra` appends/overrides.
Submits are async — the loop fans out onto the cluster. Monitor via the ZenML
dashboard / the wandb project.) The allocation JSONs must exist in the image or
be produced by a probe step first — simplest: run the probe locally/one ZenML
job, commit the JSONs (or stage on the shared PVC), rebuild, then launch.

### 6c. Local (single GPU box, no ZenML)
`bash examples/run_renyi_campaign.sh` (method family × DP × r × cadence) — this
already exists for the LoRA-XSe *family*; extend its roster with the alloc arms
by adding `--lora-xs-rank-pattern-json` lines.

---

## 7. `examples/train_vision.py` — spec (build this)

Reuse the DP core **unchanged**; only swap the text bits. Checklist:
1. Data: CIFAR-10/100 via `torchvision.datasets` (or HF `datasets`), standard
   train transforms; `collate` → `(pixel_values, labels)`.
2. Model: `ViTForImageClassification` (HF) or `torchvision.models.vit_b_16`,
   LoRA-XS/LoRA on the attention/MLP linears (`query,key,value,dense` etc.).
3. `per_example_loss_fn(params, pixel_values, labels)` → cross-entropy (mirror
   `train_causal_lm.py:~1381`).
4. Reuse verbatim: `make_functional(..., partition_trainable=True)`,
   `clipped_grad`/`adaptive_clipped_grad`, gaussian noise, the accountant loop,
   torchopt optimizer + `apply_updates`, and the XSe optimizer path. The patch
   router will skip ViT (no LLM family) → runs via generic vmap ops (correct,
   just no fused-kernel speedup).
5. Same LoRA-XS flags (`--lora-xs-rank-pattern-json`, `--dump-core-spectra`) so
   allocation works identically. Metric: top-1 accuracy on the test set.
6. Smoke-test on CPU with a tiny ViT + 2 batches before any GPU run.

---

## 8. Controls, metrics, analysis

- **Matched budget** across all arms (the alloc JSONs are built to the uniform-r
  budget; the trainer prints achieved rank spread — verify it's ~100% of budget
  and non-uniform).
- **Equal HP budget**, **≥5 seeds** on T1, **≥3** on 7B tiers, **paired tests**.
- Primary endpoint = **downstream metric** (pass@1 / accuracy / EM), not eval
  loss. Secondary: `eval/loss_min`, and the `rotation/r_eff_*` diagnostics.
- **Decision gates:** (Phase 1) does any allocation beat uniform without noise?
  (Phase 2) is A∞ ≥ A1/Afrob/U and does the gap grow as ε↓? If Phase 2 is flat
  even at ε=1, pivot to the analysis/critique paper (mechanism + AdaLoRA
  incompatibility + the confirmed inflation) — still publishable.

---

## 9. Handoff checklist for the running agent
- [ ] pull this branch; `uv sync --group examples --all-packages`
- [ ] `git submodule update --init` (per-layer rank lives in the submodule)
- [ ] rebuild+push the ZenML image (`deploy/zenml/build_and_push.sh`); set `OPAQUE_DOCKER_TAG`
- [ ] build `examples/train_vision.py` (§7) + CPU smoke test
- [ ] add GLUE / reasoning presets (T1/T3) if pursuing those tiers
- [ ] Phase 1 (non-DP α-sweep, T2 then T1) → Phase 2 (DP) → Phase 3 (regime map) → Phase 4 (breadth)
- [ ] analyze per §8; update `docs/renyi-utility-experiments-plan.md` figures with results
