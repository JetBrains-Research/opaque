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

## Loop event log

- **2026-06-12 20:00 UTC** — Cycle 1. A-T01 launched supervised (rate ~12s/step, nm=0.943 calibrated to ε=8; rewards/acc=0.66 at step 10 — pipeline healthy). B-T01, C-T01, D-T01 submitted to fill 4-slot budget. 8 planned probes remain queued. ScheduleWakeup 1800s.
