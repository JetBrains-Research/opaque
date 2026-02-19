# Opaque Examples

Production-ready scripts demonstrating end-to-end differentially private training
with the Opaque library.

## Scripts

### `train_causal_lm.py` — Primary: DP-SGD LoRA for Causal LMs

Full end-to-end DP-SGD training script for any HuggingFace causal language model.
Exercises **all** library modules: clipping → noise → accounting → (optional) auditing.

**Features:**
- Adaptive or fixed per-example gradient clipping (`opaque.clipping`)
- Gaussian noise injection (`opaque.noise`)
- Privacy accounting calibrated from target (ε, δ) budget (`opaque.accounting`)
- Optional empirical privacy auditing (`opaque.auditing`)
- Optional [Weights & Biases](https://wandb.ai) experiment tracking (`--wandb`)
- Works with any HuggingFace causal LM (GPT-2, Qwen, LLaMA, Mistral, …)

**Quick start (smoke test, ~5 min on GPU):**

```bash
uv run --extra examples python examples/train_causal_lm.py --model_name gpt2
```

**With W&B tracking:**

```bash
pip install wandb
uv run --extra examples python examples/train_causal_lm.py \
    --model_name gpt2 \
    --wandb \
    --wandb_project my-dp-experiments
```

**With post-training privacy audit:**

```bash
uv run --extra examples python examples/train_causal_lm.py \
    --model_name gpt2 \
    --audit \
    --audit_canaries 200
```

**Production run (Qwen-7B, multi-hour):**

```bash
uv run --extra examples python examples/train_causal_lm.py \
    --model_name Qwen/Qwen2.5-7B \
    --num_train_samples 2000 \
    --batch_size 8 \
    --num_epochs 80
```

See [`README_train_causal_lm.md`](README_train_causal_lm.md) for full documentation
and configuration examples.

---

### `dp_sgd_simple.py` — Quickstart: Linear Model

Minimal DP-SGD demo on a synthetic linear regression task. Completes in < 1 minute.
Good starting point for understanding the core API without HuggingFace dependencies.

```bash
uv run python examples/dp_sgd_simple.py
```

---

### `adaptive_clipping.py` — Adaptive Clipping + Accounting

Demonstrates `adaptive_clipped_grad()` with explicit state-passing, and shows how to
use `acc.adaclip()` to accurately account for the privacy cost of adaptive clipping.

```bash
uv run --extra examples python examples/adaptive_clipping.py
```

---

### `accounting_complete.py` — Accounting API Showcase

Complete demonstration of the `opaque.accounting` API: mechanisms, composition,
calibration, multi-phase training, and all privacy metrics. No model training —
pure accounting showcase.

```bash
uv run python examples/accounting_complete.py
```

---

### `train_qwen_ddp.py` — Multi-GPU DDP Training

DDP-style distributed DP-SGD training. Requires multiple GPUs or single-GPU mode.

```bash
# Single GPU
uv run --extra examples python examples/train_qwen_ddp.py

# Multi-GPU (4 GPUs)
uv run --extra examples python -m torch.distributed.run \
    --nproc_per_node=4 examples/train_qwen_ddp.py
```

---

## Presets

The `presets/` directory contains YAML configuration files for common scenarios:

| Preset | File | Description |
|--------|------|-------------|
| Smoke test | `presets/smoke_test.yaml` | 5-min run on GPT-2, suitable for CI |

### Using a preset

Presets are self-documenting YAML files. Pass the args directly to the script:

```bash
# Smoke test
uv run --extra examples python examples/train_causal_lm.py \
    $(python -c "import yaml,sys; d=yaml.safe_load(open('examples/presets/smoke_test.yaml')); print(' '.join(f'--{k} {v}' for k,v in d['args'].items()))")
```

Or simply copy the args from the YAML and pass them manually.

---

## CLI Reference for `train_causal_lm.py`

```
usage: train_causal_lm.py [-h]
  [model]   --model_name NAME  --use_eager_attention
  [data]    --dataset NAME  --num_train_samples N  --max_seq_len N  --num_eval_samples N
  [train]   --batch_size N  --num_epochs N  --learning_rate LR  --optimizer {sgd,adam}
            --eval_steps N  --seed N
  [lora]    --lora_r N  --lora_alpha N  --lora_budget_modules [M ...]
  [dp]      --clip_norm C  --adaptive_clipping / --no_adaptive_clipping
            --target_clip_rate R  --clip_norm_max M  --quantile_noise_std S
            --microbatch_size N
  [privacy] --target_epsilon E  --target_delta D
            --calibration_min LO  --calibration_max HI  --calibration_tolerance TOL
  [audit]   --audit / --no_audit  --audit_canaries N  --audit_batch_size N
  [wandb]   --wandb  --wandb_project PROJECT  --wandb_entity ENTITY
```

### Key defaults (smoke-test friendly)

| Argument | Default | Notes |
|----------|---------|-------|
| `--model_name` | `gpt2` | Any HuggingFace causal LM |
| `--dataset` | `ag_news` | Any HuggingFace text dataset |
| `--num_train_samples` | `300` | Increase for production |
| `--batch_size` | `16` | Reduce for large models |
| `--num_epochs` | `3` | Increase for production |
| `--adaptive_clipping` | `True` | Disable with `--no_adaptive_clipping` |
| `--target_epsilon` | `3.0` | Standard privacy budget |
| `--target_delta` | `1e-5` | Standard DP delta |
| `--audit` | `False` | Enable with `--audit` |
| `--wandb` | `False` | Enable with `--wandb` |

---

## Privacy Output

Every run reports a calibrated privacy guarantee at the end:

```
Privacy:
  Target epsilon: 3.000
  Target delta: 1.0e-05
  Noise multiplier (calibrated): 0.8234
  Final epsilon: 3.001
```

The noise multiplier is automatically calibrated so the full training run achieves
exactly the target (ε, δ)-DP guarantee.

If `--audit` is set, the script also reports an **empirical** lower bound on ε from
membership inference (the empirical bound should be ≤ theoretical ε):

```
Theoretical ε=3.001 | Empirical ε(lower bound)=0.423
```
