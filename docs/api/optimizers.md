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
from opaque.random import key

# Gradient pipeline
grad_fn, clip_state = clipped_grad(
    loss_fn, clipping_norm=1.0, batch_argnums=1,
    normalize_by=batch_size,
)
noise_fn, noise_state = gaussian_noise(stddev=noise_multiplier * clip_state.sensitivity, key=key(42))

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

Use `local_shard()` to partition the dataset across ranks and pass a
per-rank key via `fold_in(key, rank)` to each `PoissonSampler`.

---

## JME-AdamW (DP-FTRL with Adam)

Opaque provides `jme_adamw` for DP-FTRL training with Adam-style updates
and matrix-factorization correlated noise. See the
[Optimizers User Guide](../user-guide/optimizers.md#jme-adamw-adam-with-mf-correlated-noise)
for setup and usage.

::: opaque.optimizers
    options:
      show_source: true
      heading_level: 3

## See Also

- [Gradient Clipping API](clipping.md) — includes `adaptive_clipped_grad()` and `auto_clipped_grad()`
- [Optimizers User Guide](../user-guide/optimizers.md)
- [Fine-tuning an LLM Tutorial](../tutorials/llm_finetuning.ipynb)
- [TorchOpt Documentation](https://torchopt.readthedocs.io/)
