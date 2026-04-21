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
from opaque.core.clipping import clipped_grad
from opaque.dpsgd.noise import gaussian_noise
from opaque.core.random import key

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

## DP-Aware Optimizers

Standard Adam's second moment is biased upward in DP training because it
squares noised gradients ($\tilde{g}^2 = g^2 + 2gz + z^2$).  Opaque provides
two independent corrections — see the
[Optimizers User Guide](../user-guide/optimizers.md#the-second-moment-problem-in-dp-training)
for details:

- **`adamw_bc`** — subtracts the known noise variance from $\hat{v}_t$
  (Chooi et al., [arXiv:2511.07843](https://arxiv.org/abs/2511.07843)).
  Works with any Gaussian noise source.  With `noise_stddev=0` (default),
  identical to `torchopt.adamw`.
- **`adamw_jme`** — uses a separately privatized $g^2$ estimate from JME
  (Kalinin et al., [arXiv:2502.06597](https://arxiv.org/abs/2502.06597)).
  Requires MF correlated noise (`jme_noise`).

These address the same problem from different angles and **must not be
combined**.

::: opaque.dpsgd.optimizers.adamw_bc
    options:
      show_source: true
      heading_level: 3

::: opaque.dpftrl.optimizers.adamw_jme
    options:
      show_source: true
      heading_level: 3

## See Also

- [Gradient Clipping API](clipping.md) — includes `adaptive_clipped_grad()` and `auto_clipped_grad()`
- [Optimizers User Guide](../user-guide/optimizers.md)
- [Fine-tuning an LLM Tutorial](../tutorials/llm_finetuning.ipynb)
- [TorchOpt Documentation](https://torchopt.readthedocs.io/)
