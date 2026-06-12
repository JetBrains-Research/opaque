# HP Tuning Campaign Results — 2026-06-12 (in flight)

Branch `claude/modest-gates-WpC4d` @ `dead8577`. Sweeps A/B/C/D per [plan.md](plan.md); trial state ledger at [state.jsonl](state.jsonl).

Updated each loop cycle. Will be the morning-report deliverable.

## Status snapshot (as of cycle 1, 2026-06-12 ~20:00 UTC)

| Sweep | Submitted | Running | Queued | Finished | Best metric | Best HPs |
|---|---|---|---|---|---|---|
| A (Mellum-4b DP-DPO) | 1 | 1 (`55452`/`6itmyoxi`) | 0 | 0 | step 10: rewards/acc=0.66, loss=0.685 | — |
| B (Mellum-2.0 DP-DPO) | 1 | 0 | 1 (`55453`) | 0 | — | — |
| C (Mellum-2.0 SFT) | 1 | 0 | 1 (`55454`) | 0 | — | — |
| D (Mellum-4b SFT) | 1 | 0 | 1 (`55455`) | 0 | — | — |

## Known issues

- **Mellum-2.0 + Opaque RMSNorm:** Mellum-2.0's `q_norm` / `k_norm` use head_dim ~128 with many rows (`microbatch * seq_len * num_heads` > 32k), which falls outside Opaque RMSNorm's Triton row/block coverage. Triton kernel raises `Opaque RMSNorm: enable row_mode or use hidden dim / batch sizes that trigger the row kernel` at `packages/opaque-patches/.../kernels/rms_norm.py:225`. **Workaround:** pass `--no-performance-kernels` to both `train_dpo_trainer.py` and `train_sft_trainer.py` (DP-SGD path runs ~30% slower but is correct). Mellum-4b (Llama, no q_norm/k_norm) is unaffected.

## Loop event log

- **2026-06-12 20:00 UTC** — Cycle 1. A-T01 launched supervised (rate ~12s/step, nm=0.943 calibrated to ε=8; rewards/acc=0.66 at step 10 — pipeline healthy). B-T01 (55453), C-T01 (55454), D-T01 (55455) submitted to fill 4-slot budget.
- **2026-06-12 20:38 UTC** — Cycle 2. Mellum-4b (A-T01 55452, D-T01 55455) still RUNNING. Both Mellum-2.0 runs (B-T01 55453, C-T01 55454) FAILED after ~5 min on Opaque RMSNorm guard. Added `--no-performance-kernels` flag to `train_sft_trainer.py` (commit 1748af9f, mirrors DPO trainer's existing flag). Resubmitted B-T01 → 55456 and C-T01 → 55457 with `--no-performance-kernels` in EXTRA_ARGS. 8 planned probes remain queued. ScheduleWakeup 1800s.
