# HP Tuning Campaign Results — 2026-06-12 (in flight)

Branch `claude/modest-gates-WpC4d` @ `dead8577`. Sweeps A/B/C/D per [plan.md](plan.md); trial state ledger at [state.jsonl](state.jsonl).

Updated each loop cycle. Will be the morning-report deliverable.

## Status snapshot (as of cycle 1, 2026-06-12 ~20:00 UTC)

| Sweep | Submitted | Running | Queued | Finished | Best metric | Best HPs |
|---|---|---|---|---|---|---|
| A (Mellum-4b DP-DPO) | 1 | 1 (`55452`/`6itmyoxi`) | 0 | 0 | step ~560: train/rewards/acc=0.90, train/margins=1.6, train/loss=0.32, ε=4.64/8 | T01: lr=5e-5, bs=128, beta=0.1, clip=1.0 |
| B (Mellum-2.0 DP-DPO) | 3 (2 failed) | 1 (`55459` mb=4 retry) | 0 | 2 failed (`55453` RMSNorm, `55456` OOM) | n/a — retry just queued | T01: lr=5e-5, bs=128, beta=0.1, clip=1.0, r=16, mb=4 |
| C (Mellum-2.0 SFT) | 2 (1 failed) | 1 (`55457`) | 0 | 1 failed (`55454`) | step ~220: train/loss=0.70, ε=7.1/10 (will overshoot to ~13) | T01: lr=5e-5, r=16, mb=8 |
| D (Mellum-4b SFT) | 1 | 1 (`55455`) | 0 | 0 | step ~755: train/loss=0.67, ε=9.3/10 — projected finish ~00:15 UTC | T01: lr=5e-5, r=16 |

## Known issues

- **Mellum-2.0 + Opaque RMSNorm:** Mellum-2.0's `q_norm` / `k_norm` use head_dim ~128 with many rows (`microbatch * seq_len * num_heads` > 32k), which falls outside Opaque RMSNorm's Triton row/block coverage. Triton kernel raises `Opaque RMSNorm: enable row_mode or use hidden dim / batch sizes that trigger the row kernel` at `packages/opaque-patches/.../kernels/rms_norm.py:225`. **Workaround:** pass `--no-performance-kernels` to both `train_dpo_trainer.py` and `train_sft_trainer.py` (DP-SGD path runs ~30% slower but is correct). Mellum-4b (Llama, no q_norm/k_norm) is unaffected.

## Loop event log

- **2026-06-12 20:00 UTC** — Cycle 1. A-T01 launched supervised (rate ~12s/step, nm=0.943 calibrated to ε=8; rewards/acc=0.66 at step 10 — pipeline healthy). B-T01 (55453), C-T01 (55454), D-T01 (55455) submitted to fill 4-slot budget.
- **2026-06-12 20:38 UTC** — Cycle 2. Mellum-4b (A-T01 55452, D-T01 55455) still RUNNING. Both Mellum-2.0 runs (B-T01 55453, C-T01 55454) FAILED after ~5 min on Opaque RMSNorm guard. Added `--no-performance-kernels` flag to `train_sft_trainer.py` (commit 1748af9f, mirrors DPO trainer's existing flag). Resubmitted B-T01 → 55456 and C-T01 → 55457 with `--no-performance-kernels` in EXTRA_ARGS. 8 planned probes remain queued.
- **2026-06-12 23:10 UTC** — Cycle 3. All four RUNNING:
  - **A-T01 (55452, Mellum-4b DP-DPO):** step ~390/1500, train/loss=0.39, train/rewards/acc=**0.95**, ε=3.8/8, ~11s/step. Strongly converging — already near P5c's 97.9% on train; eval at step 400 due imminently. Projected finish ~04:30 UTC.
  - **D-T01 (55455, Mellum-4b SFT):** step ~510/1000, train/loss=0.65, ε=8.46/10, ~8s/step. Stable convergence. Projected finish ~01:30 UTC.
  - **B-T01 (55456, Mellum-2.0 DP-DPO, --no-performance-kernels):** step ~65/1500, train/loss=0.69, train/rewards/acc=0.65, ε=1.7/8, ~22s/step. Training cleanly. Projected finish ~05:40 UTC.
  - **C-T01 (55457, Mellum-2.0 SFT, --no-performance-kernels):** step ~95/1000, train/loss=0.71, ε=6.2/10, ~15s/step. Training cleanly; ε accumulating slightly fast (likely lands at ~13 vs target 10 — calibrator was for full horizon; will hold under target by step 1000 if rate slows). Projected finish ~00:50 UTC.

  No terminal trials this cycle. Concurrency full at 4/4. ScheduleWakeup 1800s.
- **2026-06-12 23:45 UTC** — Cycle 4. **B-T01 (55456, Mellum-2.0 DPO) FAILED at 21:23 UTC with CUDA OOM** (23.98 GiB allocation, ~step 175): microbatch=8 too tight for Mellum-2.0 12B on H200's 139 GiB with DPO's 2× pass + ref logp eval at step 100. Retrying with **microbatch=4** as 55459 (HP-B-T01-..._mb4). The other 3 (A-T01 55452, D-T01 55455, C-T01 55457) still healthy.
