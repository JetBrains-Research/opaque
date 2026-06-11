# DPO Validation Campaign — Mellum2 + Zeta NES

**Goal:** Confirm correctness of `opaque.transformers.trl.DPOTrainer` against
upstream `trl.DPOTrainer` (noDP parity) and characterise the DP path's
signal-vs-noise envelope on a real model + real preference data.

**Anchor decisions:**
- Model: `JetBrains/Mellum2-12B-A2.5B-Base` (12B MoE, 2.5B active). Needs
  `transformers≥5.10` — bumped in-yaml after `uv sync`.
- Dataset: `zed-industries/zeta`, `dpo` split (132 pairs). Schema
  `(events, input, output, rejected)` remapped to TRL-canonical
  `(prompt, chosen, rejected)` in the example scripts.
- Hyperparams: batch 16, microbatch 2, max-length 1024, LR 5e-5, β=0.1,
  LoRA r=16/α=32 on q/k/v/o/gate/up/down. 100 steps for Phase 2/3 (full
  runs); 15 steps for Phase 1 smoke.

## Decision tree (the "data-driven" part)

Each phase emits a verdict that drives the next:

```
PHASE 1 SMOKE (15 steps each, 4 parallel runs)
├─ P1A: train_dpo (mellum2_zeta) noDP        (exec 55415)
├─ P1B: train_dpo_trainer (mellum2_zeta) noDP (exec 55416)
├─ P1C: train_dpo_trl_baseline               (exec 55417)
└─ P1D: train_dpo_trainer (mellum2_zeta) DP ε=8 (exec 55418)

ALL 4 SUCCEED  → Phase 2 (parity)
P1C SUCCEEDS, opaque (A/B/D) FAILS → diagnose opaque ↔ transformers-v5
  compat (this is the realistic outcome — opaque was pinned to 4.57.x).
  Action: capture traceback, fall back to Mellum-4b-base for Phase 2/3
  (Llama arch, supported on transformers 4.57.x; loses Mellum2 size but
  unblocks the DPO validation).
P1C FAILS too → either transformers v5 is broken on Mellum2, the Zeta
  dataset has tokenization issues with Mellum2's tokenizer, or H200
  capacity. Diagnose via Cadence logs; do NOT proceed to Phase 2 until
  pinpointed.
P1A or P1B FAIL but P1D SUCCEEDS → unexpected; probably preset bug.
P1D OOMs → reduce microbatch to 1 in the DP yaml; retry just P1D.
```

```
PHASE 2 NODP PARITY (100 steps each, 3 runs, same seed 42)
├─ P2A: opaque-loop noDP (clip=1e9)   → from P1A preset, max-steps 100
├─ P2B: opaque-class noDP (clip=1e9)  → from P1B preset, max-steps 100
└─ P2C: TRL baseline                  → from P1C preset, NUM_STEPS=100

Compare train/loss curves on W&B:

CURVES OVERLAY (≤10% per-step gap, same monotone trend)
  → DPOTrainer at noDP matches TRL. Proceed to Phase 3.

CURVES DIVERGE (opaque higher than TRL by const offset)
  → Likely loss aggregation discrepancy (per-token mean vs sum).
     Check the `_NORM_LOSSES` set and `sigmoid` head normalization.

CURVES DIVERGE (opaque lower than TRL early, converges late)
  → Likely sampler difference (Poisson vs without-replacement). Compare
     against the `_NoEpochDPOTrainer` (WITH_REPLACEMENT=1) variant of the
     TRL baseline — relaunch P2C with WITH_REPLACEMENT=1 and re-overlay.

CURVES DIVERGE WILDLY
  → Tokenization mismatch: chat-template fallback may have fired on one
     trainer but not the other. Print first-batch token-id histograms for
     comparison.

CURVES OVERLAY at noDP but trainer-A vs trainer-B (loop vs class)
DIVERGE → bug in one of the two (likely the class-based one, since the
  loop-based one was used first historically). Pinpoint with a single
  micro-run that prints per-step grad norms on both.
```

```
PHASE 3 DP SIGNAL-VS-NOISE SWEEP (100 steps each, ≥6 runs)

Anchor: Phase 2's noDP curve = ground truth.

Sweep dimensions:
  clipping_norm ∈ {0.1, 1.0, 10.0}
  target_epsilon ∈ {3.0, 8.0, ∞ (noise=0)}

Cross product (excluding noise=0 with non-default clip — degenerate):
  P3-clip0.1-eps3      P3-clip0.1-eps8
  P3-clip1.0-eps3      P3-clip1.0-eps8     P3-clip1.0-eps∞ (== noDP at clip=1)
  P3-clip10-eps3       P3-clip10-eps8

For each, plot train/loss vs the noDP anchor:

LOSS DECREASES MEANINGFULLY (trend matches noDP, offset from noise)
  → DP DPO trains. Note the (clip, ε) at which signal first dominates.

LOSS STAYS FLAT at noDP starting value
  → Noise is overwhelming signal. Confirm by computing the SNR proxy:
     median(clipped_grad_norm) / noise_std. If < 1, expected behavior.
     Note as "noise floor" and try smaller ε (=lower noise) or larger
     batch.

LOSS DIVERGES / NaN
  → Clipping interacts badly with this loss head. Print per-example
     clip_rate; investigate.
```

## Run inventory

| Phase | Run name                                         | Exec ID | Status   | Notes |
|-------|--------------------------------------------------|---------|----------|-------|
| P1v1  | P1A-dpo-loop-mellum2-zeta-noDP-smoke              | 55415   | FAILED   | `KeyError: 'mellum'` — uv refused to upgrade transformers past project's `<5` pin |
| P1v1  | P1B-dpo-class-mellum2-zeta-noDP-smoke             | 55416   | FAILED   | same `KeyError: 'mellum'` |
| P1v1  | P1C-dpo-trl-mellum2-zeta-baseline-smoke           | 55417   | FAILED   | model loaded fine on trans v5; tripped on `DPOConfig overwrite_output_dir` removed in TRL 0.13+ |
| P1v1  | P1D-dpo-class-mellum2-zeta-DP-eps8-smoke          | 55418   | FAILED   | same `KeyError: 'mellum'` |
| P1v2  | P1Av2-dpo-loop-mellum4b-zeta-noDP-smoke           | 55419   | QUEUED   | fallback to Mellum-4b-base (Llama, transformers 4.57.x) |
| P1v2  | P1Bv2-dpo-class-mellum4b-zeta-noDP-smoke          | 55420   | QUEUED   | same |
| P1v2  | P1Cv2-dpo-trl-mellum2-zeta-baseline-smoke         | 55421   | QUEUED   | Mellum2 retained on TRL baseline; `overwrite_output_dir` dropped |
| P1v2  | P1Dv2-dpo-class-mellum4b-zeta-DP-eps8-smoke       | 55422   | FINISHED | DP ε=8 calibrated nm=0.501, clip=0.0625; loss flat 0.694 at noise floor (expected) |
| P2    | P2A-dpo-loop-mellum4b-zeta-noDP-100s              | 55423   | QUEUED   | num_epochs 50 so max_steps binds |
| P2    | P2B-dpo-class-mellum4b-zeta-noDP-100s             | 55424   | QUEUED   | parity-trio member |
| P2    | P2C-dpo-trl-mellum4b-zeta-baseline-100s           | 55425   | QUEUED   | parity anchor (Mellum-4b matches opaque path) |
| P2    | P2D-dpo-trl-mellum2-zeta-baseline-100s            | 55426   | QUEUED   | Mellum2 longer-horizon sanity |
| P3    | _pending Phase-2 noDP anchor_                     |         |          |          |

## Phase 3 staged commands (fire after Phase 2 lands)

Five DP-sweep runs on Mellum-4b + Zeta DPO, anchor = P2B opaque-class noDP curve.
All use `--clipping-mode fixed` to remove the auto-bound confound seen in Phase 1.

```
# Clip-only control (no noise) at three clip values — shows clipping bias alone
P3a clip=0.1 noise=0  EXTRA_ARGS="--max-steps 100 --noise-multiplier 0 --clipping-mode fixed --clipping-norm 0.1 --log-steps 2"
P3b clip=1.0 noise=0  EXTRA_ARGS="--max-steps 100 --noise-multiplier 0 --clipping-mode fixed --clipping-norm 1.0 --log-steps 2"
P3c clip=10  noise=0  EXTRA_ARGS="--max-steps 100 --noise-multiplier 0 --clipping-mode fixed --clipping-norm 10  --log-steps 2"

# Noise sweep at clip=1.0 — does signal exceed noise at eps=3 / eps=8?
P3d clip=1.0 eps=3    EXTRA_ARGS="--max-steps 100 --clipping-mode fixed --clipping-norm 1.0 --target-epsilon 3 --log-steps 2"
P3e clip=1.0 eps=8    EXTRA_ARGS="--max-steps 100 --clipping-mode fixed --clipping-norm 1.0 --target-epsilon 8 --log-steps 2"
```

If Phase 2's parity check FAILS (loop or class diverges from TRL beyond 10%
per-step), DO NOT proceed to Phase 3 — diagnose instead. The DP sweep is only
meaningful with a trusted noDP anchor.

## Phase 1 v2 results (2026-06-11)

| Run | Final train/loss | rewards/margins | rewards/acc | Steps | Notes |
|---|---|---|---|---|---|
| P1Av2 loop noDP (Mellum-4b) | 0.645 | n/a | n/a | 3 (truncated by num_epochs=1) | auto-clip 1e9 → 6.25e7 → 0.625 |
| P1Bv2 class noDP (Mellum-4b) | 0.660 | +0.088 | 0.214 | 15 (full) | clip rate 0; bound 6.25e7 too high |
| P1Cv2 TRL Mellum2 | 0.689 | +0.002 | 0.375 | 15 (full) | Mellum2 LOADS and TRAINS on transformers v5 |
| P1Dv2 class DP ε=8 (Mellum-4b) | 0.694 | -0.002 | 0 | 15 (full) | calibrated nm=0.501, clip=0.0625; flat as expected |

Memory peak: ~31 GB (Mellum-4b LoRA), ~58 GB (Mellum-4b + ref + microbatch 4 vmap), no model fit issues. Runtime per smoke: 45-200s (Mellum-4b fast, Mellum2 slower).

**Provisional read on parity:** opaque-class loss (0.660) is below TRL Mellum2 (0.689) at step 15 — but they're different models. Need the Mellum-4b TRL baseline (P2C) for a clean parity comparison.

**Auto-clipping behavior at noDP:** the adaptive bound shrank aggressively (1e9 → 6.25e7 → 0.625 across steps 0-2). Means our `--clipping-norm 1e9` arg was honored at init but the adaptive mode is dropping it fast. For Phase 3 DP sweep, switch to fixed clipping (`--clipping-mode fixed`) to remove that confound.

## Learnings from P1 v1 (logged for memory)

1. **uv pip install respects project constraints.** Running `uv pip install
   "transformers>=5.10,<6"` inside a uv-managed project where pyproject.toml
   pins `transformers>=4.57.0,<5` resolves as a no-op (uv prefers the
   project's tighter constraint). The `--quiet` flag swallowed the resolve
   output, making it look like the install succeeded.
   *Workaround that did work:* `uv run --with "transformers>=5.10"` —
   `--with` adds to the ephemeral environment and overrides the project
   constraint, but ONLY for scripts that don't import the project's own
   packages (so it can break the constraint without conflict).
   *Doesn't work for:* the opaque DPO scripts, which import
   `opaque.transformers.trl` and therefore need the project env, which is
   constraint-locked to `transformers<5`.
2. **TRL 0.13+ removed `overwrite_output_dir`** from `DPOConfig` (the kwarg
   lives on HF `TrainingArguments` but TRL no longer re-exports it). Any
   example script targeting recent TRL must drop it.
3. **Mellum2 support landed in transformers 5.x.** Until opaque's
   transformers-compat layer is migrated to v5 (a multi-week effort), the
   opaque DPO validation runs on **Mellum-4b-base** (Llama arch). The TRL
   baseline retains Mellum2 to confirm it loads & trains at all.

## Notes

- Eval is OFF on the opaque runs (no held-out slice — Zeta DPO has only 132
  pairs). Phase 2 parity is judged from train/loss curve overlay.
- TRL baseline run-tag inherited from W&B: `val/trl-trainers-r1b`.
- The opaque-trainer transformers-v5 compat issue is the highest-prior
  risk; if Phase 1 surfaces it, **fall back to Mellum-4b-base** (Llama,
  works on transformers 4.57.x) and re-run all of Phase 1.
