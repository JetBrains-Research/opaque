# Overnight HP Tuning Campaign — 2026-06-12

Branch: `claude/modest-gates-WpC4d` @ `dead8577` (merge of post-#266 main +
trainer preset refresh).

W&B entity/project: `federated-compute/opaque`.
Cadence project: `JbrFed`.
Concurrency budget: **4 H200 runs in flight**.

## Sweeps

| ID | Pri | Preset | Model | Task | Dataset | Probe steps | Confirm steps |
|---|---|---|---|---|---|---|---|
| **A** | P0 | `train_dpo_trainer (mellum_codesec).yaml` (default) | Mellum-4b dense | DP-DPO ε=8 | CyberNative | 1500 | 5000 |
| **B** | P0 | `train_dpo_trainer (mellum_codesec).yaml` + EXTRA_ARGS `--model JetBrains/Mellum2-12B-A2.5B-Base --microbatch-size 8 --lora-modules q_proj k_proj v_proj o_proj` | Mellum-2.0 MoE | DP-DPO ε=8 | CyberNative | 1500 | 5000 |
| **C** | P1 | `train_sft_trainer (mellum_kstack).yaml` + EXTRA_ARGS `--model-name JetBrains/Mellum2-12B-A2.5B-Base --microbatch-size 8 --lora-modules q_proj k_proj v_proj o_proj` | Mellum-2.0 MoE | DP-SFT ε=10 | KStack | 1000 | 2000 |
| **D** | P2 | `train_sft_trainer (mellum_kstack).yaml` (default) | Mellum-4b dense | DP-SFT ε=10 | KStack | 1000 | 2000 |

## Initial probe grid (round 1)

### Sweep A (Mellum-4b DP-DPO)
| Trial | lr | batch | beta | clip | EXTRA_ARGS |
|---|---|---|---|---|---|
| A-T01 | 5e-5 | 128 | 0.1 | 1.0 | (defaults — supervised first launch) |
| A-T02 | 1e-5 | 64 | 0.05 | 0.5 | `--learning-rate 1e-5 --batch-size 64 --beta 0.05 --clipping-norm 0.5` |
| A-T03 | 1e-4 | 128 | 0.3 | 2.0 | `--learning-rate 1e-4 --beta 0.3 --clipping-norm 2.0` |

### Sweep B (Mellum-2.0 DP-DPO)
Base EXTRA_ARGS for the model flip:
`--model JetBrains/Mellum2-12B-A2.5B-Base --microbatch-size 8 --lora-modules q_proj k_proj v_proj o_proj`

| Trial | lr | batch | beta | clip | lora_r | Extra |
|---|---|---|---|---|---|---|
| B-T01 | 5e-5 | 128 | 0.1 | 1.0 | 16 | (base) |
| B-T02 | 1e-5 | 64 | 0.05 | 1.0 | 32 | `--learning-rate 1e-5 --batch-size 64 --beta 0.05 --lora-r 32 --lora-alpha 64` |
| B-T03 | 5e-5 | 128 | 0.3 | 2.0 | 16 | `--beta 0.3 --clipping-norm 2.0` |

### Sweep C (Mellum-2.0 DP-SFT)
Base EXTRA_ARGS:
`--model-name JetBrains/Mellum2-12B-A2.5B-Base --microbatch-size 8 --lora-modules q_proj k_proj v_proj o_proj`

| Trial | lr | lora_r | Extra |
|---|---|---|---|
| C-T01 | 5e-5 | 16 | (base) |
| C-T02 | 1e-4 | 32 | `--learning-rate 1e-4 --lora-r 32 --lora-alpha 64` |
| C-T03 | 1e-5 | 8 | `--learning-rate 1e-5 --lora-r 8 --lora-alpha 16` |

### Sweep D (Mellum-4b DP-SFT)
| Trial | lr | lora_r | EXTRA_ARGS |
|---|---|---|---|
| D-T01 | 5e-5 | 16 | (defaults) |
| D-T02 | 1e-4 | 16 | `--learning-rate 1e-4` |
| D-T03 | 1e-5 | 32 | `--learning-rate 1e-5 --lora-r 32 --lora-alpha 64` |

## Run naming

`HP-<sweep>-T<NN>-<key=val>_<key=val>` — e.g., `HP-A-T01-lr5e-5_bs128_beta0.1_clip1.0`.

## Selection policy (round 2+)

1. Take top-2 trials by `eval/rewards/accuracies` (DPO) or `eval/loss` (SFT) after round 1.
2. Refine: spawn ±1-axis neighbors of each winner.
3. Stop early if top-2 are within 1pp on eval rewards/acc AND eval/loss stagnates 3 trials in a row.
4. Hard cap: 10 trials per sweep.
5. Confirmation run at full horizon on the top-1 HP.

## Failure handling

- **OOM**: retry with microbatch halved (single retry).
- **CUDA / NaN / hung**: log + skip the HP point.
- **DP ε blow-out** (`privacy/epsilon > target × 1.2`): mark HP infeasible, reduce batch in next probe.
- **Cadence queue full / no provisioning**: pause submission; ScheduleWakeup unchanged.

## Loop cadence

`ScheduleWakeup(delaySeconds=1800, prompt=<<autonomous-loop-dynamic>>)` — 30 minute wake interval. Each cycle reconciles state, processes terminal trials, submits up to (4 − in_flight) new ones.
