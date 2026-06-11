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

| Phase | Run name                                         | Exec ID | Status     | Notes |
|-------|--------------------------------------------------|---------|------------|-------|
| P1    | P1A-dpo-loop-mellum2-zeta-noDP-smoke              | 55415   | QUEUED     | clip=1e9 |
| P1    | P1B-dpo-class-mellum2-zeta-noDP-smoke             | 55416   | QUEUED     | clip=1e9 |
| P1    | P1C-dpo-trl-mellum2-zeta-baseline-smoke           | 55417   | QUEUED     | LoRA on |
| P1    | P1D-dpo-class-mellum2-zeta-DP-eps8-smoke          | 55418   | QUEUED     | preset DP defaults |
| P2    | _pending Phase-1 success_                         |         |            |          |
| P3    | _pending Phase-2 noDP anchor_                     |         |            |          |

## Notes

- Eval is OFF on the opaque runs (no held-out slice — Zeta DPO has only 132
  pairs). Phase 2 parity is judged from train/loss curve overlay.
- TRL baseline run-tag inherited from W&B: `val/trl-trainers-r1b`.
- The opaque-trainer transformers-v5 compat issue is the highest-prior
  risk; if Phase 1 surfaces it, **fall back to Mellum-4b-base** (Llama,
  works on transformers 4.57.x) and re-run all of Phase 1.
