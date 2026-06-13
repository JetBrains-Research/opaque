# HP Tuning Campaign Results — 2026-06-12 (in flight)

Branch `claude/modest-gates-WpC4d` @ `dead8577`. Sweeps A/B/C/D per [plan.md](plan.md); trial state ledger at [state.jsonl](state.jsonl).

Updated each loop cycle. Will be the morning-report deliverable.

## Status snapshot (as of cycle 1, 2026-06-12 ~20:00 UTC)

| Sweep | Submitted | Running | Queued | Finished | Best metric | Best HPs |
|---|---|---|---|---|---|---|
| A (Mellum-4b DP-DPO) | 3 | 2 (`55464` T02, `55465` T03) | 0 | **1 done** (`55452`/`6itmyoxi`) | **T01 final**: eval/loss=**0.274**, eval/rewards/acc=**0.898**, eval/margins=**4.58**, ε=**8.05** (vs P5c eval/loss=0.397) | T01 winning: lr=5e-5, bs=128, beta=0.1, clip=1.0 |
| B (Mellum-2.0 DP-DPO) | 4 (3 failed) | 1 (`55463` mb=4+evalb=4) | 0 | 3 failed (`55453` RMSNorm, `55456`+`55459` eval-time OOM) | n/a — retry still running | T01: lr=5e-5, bs=128, beta=0.1, clip=1.0, r=16, mb=4, eval_b=4 |
| C (Mellum-2.0 SFT) | 3 (1 failed) | 1 (`55466` T02) | 0 | **1 done** (`55457`/`k565ady9`), 1 failed | T01 final: train/loss=**0.707**, train/mean_token_acc=**0.820**, ε=10.06 | T01: lr=5e-5, r=16, mb=8 |
| D (Mellum-4b SFT) | 2 | 0 | 0 | **2 done** (`55455`/`xii5leqq`, `55461`/`zo9bwaa1`) | T01: loss=**0.647**, acc=**0.839**; T02 (lr=1e-4): loss=**0.644**, acc=**0.840** — tied | **sweep converged**: lr-insensitive between 5e-5 and 1e-4 |

## Fixed issues

- **Mellum-2.0 + Opaque RMSNorm:** Mellum-2.0's `q_norm` / `k_norm` (head_dim=128, n_rows>32k under vmap) hit a misclassified `RuntimeError` at `rms_norm.py:225`. The kernel's grid is `(n_rows,)` and the row kernel produces correct output at any shape — the guard was a launch-overhead warning incorrectly raised as an error. **Fixed in commit `5695b733`:** replaced with a one-shot `warnings.warn` bucketed by shape. Future Mellum-2.0 runs no longer need `--no-performance-kernels`. A Liger-style block kernel would amortize launch overhead in this regime; that's the follow-up.
- **Eval batch silently defaulted to HF stock 8:** `TrainingArguments.per_device_eval_batch_size` is now `Optional[int]` and `__post_init__` resolves `None` to `per_device_train_batch_size`. Callers who bump train batch for a big model get eval scaled to match unless they explicitly override (and they still can with `--per-device-eval-batch-size`).

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

- **2026-06-13 11:36 UTC** — Cycle 17. **TWO TERMINALS:**
  - **A-T03 (Mellum-4b DP-DPO, lr=1e-4 / β=0.3 / clip=2.0) FINAL — sweep A winner:** eval/loss=**0.088**, eval/rewards/acc=**0.964**, eval/rewards/margins=**12.20**, ε=8.05/8. Beats A-T01 baseline (loss 0.274, margins 4.58) by ~3× on margins; vs P5c's 500-step run (eval/loss 0.397) the verdict is decisive. Runtime 4.9h. W&B `80t6wrwi`.
  - **C-T02 (Mellum-2.0 SFT lr=1e-4 + lora_r=32, perf kernels ON via RMSNorm fix):** train/loss=**0.699**, mean_token_acc=**0.822**, ε=10.06. Essentially tied with C-T01 (loss 0.707 at lr=5e-5). **Sweep C verdict: Mellum-2.0 SFT lr-insensitive in [5e-5, 1e-4]**. RMSNorm fix verified in production — kernels-on ran cleanly with ~30% perf penalty from row kernel launches. Runtime 5.3h. W&B `2v0j351h`.

  Slots freed: A-T04 (β=0.5 probe) still in flight, B-T01 retry (Mellum-2.0 DPO) still in flight. No new submissions — sweep A's winner is clear, sweep C is converged.

- **2026-06-13 06:42 UTC** — Cycle 9. **A-T03 first eval at step 300 — stunning:**
  - eval/loss=**0.175**, eval/rewards/acc=**0.926**, eval/margins=**4.85** at step 300/1500
  - A-T03 at step 300 already beats A-T01's final at step 1500 (eval/loss 0.175 vs 0.274)
  - HPs: lr=1e-4, β=0.3, clip=2.0 — the more aggressive corner of the grid
  - Train (loss=0.17) and eval (loss=0.175) tracking tight; no overfit yet
  - Sweep A's clear lead candidate. A-T02 (lr=1e-5) still slow at step 300/0.91 rewards/acc.
  - B-T01 retry at step ~580/1500: train/rewards/acc=0.94 (Mellum-2.0 DPO works). C-T02 RMSNorm fix stable.

- **2026-06-13 04:05 UTC** — Cycle 7. **THREE TERMINALS:**
  - **A-T01** (Mellum-4b DP-DPO, 1500 steps): eval/loss=**0.274**, eval/rewards/acc=**0.898**, eval/margins=**4.58**, ε=**8.05/8** (calibrator nailed target). Compared to P5c's 500-step run (eval/loss=0.397, eval/rewards/acc=0.915 train but lower stability), the 1500-step horizon clearly improves loss and matches/exceeds reward gating. Runtime 4.9h.
  - **D-T01** (Mellum-4b SFT lr=5e-5) and **D-T02** (lr=1e-4) **tied at train/loss=0.647 vs 0.644** and mean_token_acc=0.839 vs 0.840 — sweep D is lr-insensitive in this range, **converged**.
  - **C-T01** (Mellum-2.0 SFT lr=5e-5): train/loss=**0.707**, mean_token_acc=**0.820**, ε=10.06. Mellum-2.0 lags Mellum-4b SFT (0.707 vs 0.647) at fixed steps, likely needs higher lr or longer horizon.

  Submitted 3 new probes: **A-T02** (55464, lr=1e-5/bs=64/beta=0.05), **A-T03** (55465, lr=1e-4/beta=0.3), **C-T02** (55466, Mellum-2.0 SFT lr=1e-4, lora_r=32, **dropped --no-performance-kernels** to test the RMSNorm fix in production). B-T01 retry (55463) still running. Concurrency 4/4.
- **2026-06-13 00:19 UTC** — Cycle 5. **D-T01 FINISHED ✓** (xii5leqq, 2h13m runtime): train/loss=**0.647**, train/mean_token_acc=**0.839**, privacy/epsilon=**10.06** (target 10 nailed), nm=0.418 calibrated. Submitted **D-T02** (55461, --learning-rate 1e-4) to fill the slot. B-T01 retry (55459) healthy at step 50 with mb=4 but slower (~30s/step → projected ~12.5h end-to-end; will partially complete by morning). A-T01 + C-T01 still on track.
- **2026-06-13 01:33 UTC** — Cycle 6. **B-T01 retry (55459, mb=4) FAILED again** with the same 23.98 GiB OOM at step ~120. Diagnosis: OOM was during **eval** (eval-batch defaulted to `--batch-size 128`, way too big for Mellum-2.0 12B). Microbatch reduction doesn't help eval. Resubmitted as **55463** with `--per-device-eval-batch-size 4 --num-eval-samples 100`. **A-T01 first eval at step 1300**: eval/loss=**0.277**, eval/rewards/accuracies=**0.896**, eval/margins=**4.4** — Mellum-4b DP-DPO is clearly working at these HPs. C-T01 step 790/1000 (loss=0.70, ε=9.4), D-T02 step 610/1000 (loss=0.65, ε=8.8).
