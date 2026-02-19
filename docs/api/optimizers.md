# Optimizers

Opaque does **not** bundle optimizers. Use [TorchOpt](https://torchopt.readthedocs.io/)
functional optimizers for the parameter-update step.

---

## Why TorchOpt?

TorchOpt provides **functional** optimizers: no hidden mutable state, explicit
`(updates, new_state) = optimizer.update(grads, state)` interface.  This matches
Opaque's functional gradient pipeline (`clipped_grad → noise_fn → optimizer`).

```python
import torchopt

optimizer = torchopt.adam(lr=1e-3)
opt_state = optimizer.init(params)

# Explicit state in, state out
updates, opt_state = optimizer.update(noisy_grads, opt_state, params=params)
params = torchopt.apply_updates(params, updates)
```

---

## Supported Optimizers

Any TorchOpt functional optimizer works. The most common choices for DP training:

### SGD

```python
optimizer = torchopt.sgd(lr=0.01, momentum=0.9)
```

### Adam

```python
optimizer = torchopt.adam(lr=1e-3)
```

### AdamW

```python
optimizer = torchopt.adamw(lr=1e-3, weight_decay=0.01)
```

---

## Complete Pattern

```python
import torchopt
from opaque.clipping import clipped_grad
from opaque.noise import gaussian_noise

# Gradient pipeline
grad_fn, clip_state = clipped_grad(loss_fn, l2_clip_norm=1.0, batch_argnums=1)
noise_fn, noise_state = gaussian_noise(stddev=noise_multiplier * 1.0)

# Optimizer
optimizer = torchopt.adam(lr=1e-3)
opt_state = optimizer.init(params)

for step in range(num_steps):
    grads, clip_state = grad_fn(params, batch, state=clip_state)
    noisy_grads, noise_state = noise_fn(grads, noise_state)
    updates, opt_state = optimizer.update(noisy_grads, opt_state, params=params)
    params = torchopt.apply_updates(params, updates)
```

---

## DDP Compatibility

When using `torch.nn.parallel.DistributedDataParallel`, Opaque's functional
gradient pipeline runs *inside* each rank.  DDP handles the all-reduce of noisy
gradients across ranks.  No changes to the clipping, noise, or optimizer API are
needed.

Use `PoissonSampler(distributed=True)` or set `RANK`/`WORLD_SIZE` environment
variables so that each rank samples from its shard of the dataset.

---

## See Also

- [Gradient Clipping API](core/clipping.md) — includes `adaptive_clipped_grad()`
- [Optimizers User Guide](../user-guide/optimizers.md)
- [Tutorial 04: DP Optimizers](../tutorials/04_dp_optimizers.ipynb)
- [TorchOpt Documentation](https://torchopt.readthedocs.io/)
