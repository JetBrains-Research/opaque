# Optimizers (TorchOpt)

Opaque does **not** bundle optimizers. Use
[TorchOpt](https://torchopt.readthedocs.io/) functional optimizers for the
parameter-update step in your DP training loop.

## Why Functional Optimizers?

Opaque's gradient pipeline is **functional**: every function takes state in and
returns new state out.  TorchOpt follows the same pattern, making integration
seamless:

```python
import torchopt

optimizer = torchopt.adam(lr=1e-3)
opt_state = optimizer.init(params)

# Explicit state in → state out
updates, opt_state = optimizer.update(noisy_grads, opt_state, params=params)
params = torchopt.apply_updates(params, updates)
```

No hidden mutable state—every piece of the training loop is explicit.

## Complete Training Loop

```python
import torch
import torchopt
import opaque.accounting as acc
from opaque.clipping import clipped_grad
from opaque.noise import gaussian_noise

# Setup
clip_norm = 1.0
noise_multiplier = 1.1
num_steps = 1000

# 1. Gradient pipeline
grad_fn, clip_state = clipped_grad(
    loss_fn, l2_clip_norm=clip_norm, argnums=0, batch_argnums=1,
)
noise_fn, noise_state = gaussian_noise(stddev=noise_multiplier * clip_norm)

# 2. Optimizer
optimizer = torchopt.adam(lr=1e-3)
opt_state = optimizer.init(params)

# 3. Training
for step in range(num_steps):
    grads, clip_state = grad_fn(params, batch, state=clip_state)
    noisy_grads, noise_state = noise_fn(grads, noise_state)
    updates, opt_state = optimizer.update(noisy_grads, opt_state, params=params)
    params = torchopt.apply_updates(params, updates)
```

## Choosing an Optimizer

### SGD

```python
optimizer = torchopt.sgd(lr=0.01, momentum=0.9)
opt_state = optimizer.init(params)
```

Simple and predictable.  Good baseline for debugging.

### Adam (Recommended)

```python
optimizer = torchopt.adam(lr=1e-3)
opt_state = optimizer.init(params)
```

Adaptive learning rates help compensate for DP noise.  Usually the best choice
for DP training—converges faster and is more robust to hyperparameter choices.

### AdamW

```python
optimizer = torchopt.adamw(lr=1e-3, weight_decay=0.01)
opt_state = optimizer.init(params)
```

Adam with decoupled weight decay.  Useful for fine-tuning pre-trained models
(e.g., LoRA).

## Tips

**Start with Adam.**  Adam's per-parameter adaptive learning rates are
especially helpful when DP noise is added, since different parameters receive
different signal-to-noise ratios.

**Use the same LR schedule as non-DP.**  Standard learning rate schedules
(warmup, cosine decay) work well with DP training.  Apply them by updating
the optimizer's learning rate each step.

**Don't clip optimizer updates.**  Opaque clips *gradients* before the
optimizer.  The optimizer should see the full noisy gradient—clipping after
the optimizer would distort the adaptive state.

## DDP Compatibility

When using `torch.nn.parallel.DistributedDataParallel`, Opaque's functional
gradient pipeline runs *inside* each rank.  DDP handles the all-reduce of noisy
gradients across ranks.  No changes to the optimizer API are needed.

Use `PoissonSampler(distributed=True)` or set `RANK`/`WORLD_SIZE` environment
variables so that each rank samples from its shard of the dataset.

## Distributed Training

TorchOpt optimizers work seamlessly with DDP distributed training—**no changes needed**.

### Optimizer States in DDP

**TL;DR:** Optimizer states stay synchronized automatically across all GPUs when using TorchOpt's functional optimizers with DP-SGD.

#### How It Works

```python
import torchopt
from opaque.distributed import sum_gradients
from opaque.noise import gaussian_noise

# 1. Initialize optimizer (same on all devices)
opt = torchopt.adam(lr=1e-3)
opt_state = opt.init(params)

# 2. Training loop
for step, batch in enumerate(dataloader):
    # Get per-device clipped gradients
    grads, clip_state = grad_fn(params, batch, state=clip_state)
    
    # Aggregate across devices (all devices get same result)
    if distributed:
        grads = sum_gradients(grads)
    
    # Add noise (same seed → same noise on all devices)
    noisy_grads, noise_state = noise_fn(grads, noise_state)
    
    # Update (pure function → same result on all devices)
    updates, opt_state = opt.update(noisy_grads, opt_state, params=params)
    params = torchopt.apply_updates(params, updates)
    # ✅ opt_state is now identical on all devices (implicitly synchronized)
```

**Why it works:**

1. **Pure functions:** `opt.update()` is deterministic—same inputs always produce same outputs
2. **Identical gradients:** After `sum_gradients()` and noise (same seed), all devices have identical `noisy_grads`
3. **Identical updates:** All devices compute identical `opt_state` and `params` updates

**Result:** No explicit state synchronization needed—optimizer states evolve identically on all devices.

### Important Notes

**❌ Don't use `torchopt.distributed` for DDP:**

- `torchopt.distributed` is for **RPC-based parameter server parallelism** (different paradigm)
- For DDP, use `opaque.distributed` utilities (`sum_gradients`, etc.)
- See [Distributed Training](distributed.md#torchoptdistributed-vs-opaquedistributed) for details

**⚠️ Optimizer state drift:**

- Theoretical guarantee is sound, but **not empirically validated** in Opaque's tests
- For production, consider periodically checking state consistency (see [Optimizer State Validation](distributed.md#optimizer-state-synchronization-torchopt))

### Complete DDP Example

See [`examples/train_qwen_ddp.py`](https://github.com/evgri243/opaque/blob/main/examples/train_qwen_ddp.py) for a full working example:

```bash
# Launch DDP training on 4 GPUs
uv run python -m torch.distributed.run --nproc_per_node=4 examples/train_qwen_ddp.py
```

## See Also

- [Gradient Clipping](clipping.md) — clipping and adaptive clipping
- [Noise Addition](noise.md) — adding calibrated noise
- [Tutorial 04: DP Optimizers](../tutorials/04_dp_optimizers.ipynb)
- [API Reference](../api/optimizers.md)
- [TorchOpt Documentation](https://torchopt.readthedocs.io/)

---

**Next**: Learn about [Poisson Sampling & Microbatching](sampling.md) for privacy amplification
