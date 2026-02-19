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

## See Also

- [Gradient Clipping](clipping.md) — clipping and adaptive clipping
- [Noise Addition](noise.md) — adding calibrated noise
- [Tutorial 04: DP Optimizers](../tutorials/04_dp_optimizers.ipynb)
- [API Reference](../api/optimizers.md)
- [TorchOpt Documentation](https://torchopt.readthedocs.io/)

---

**Next**: Learn about [Poisson Sampling & Microbatching](sampling.md) for privacy amplification
