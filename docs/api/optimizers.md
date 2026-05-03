# Optimizers

Opaque ships its own functional optimizer library at
[`opaque.optimizers`](#opaque.optimizers). The factories follow the
[TorchOpt](https://torchopt.readthedocs.io/) ``GradientTransformation``
protocol and accept optional DP-aware kwargs (``noise_stddev``,
``noisy_squared_grads``) at update time.

---

## API shape

Functional: no hidden mutable state, explicit
`(updates, new_state) = optimizer.update(grads, state)` interface.

```python
from opaque.optimizers import adamw

optimizer = adamw(lr=1e-3, weight_decay=0.01)
opt_state = optimizer.init(params)

# Explicit state in, state out
updates, opt_state = optimizer.update(noisy_grads, opt_state, params=params)
params = torchopt.apply_updates(params, updates)
```

---

## Supported Optimizers

The library ships:

- ``opaque.optimizers.adamw`` — universal Adam / AdamW; optional DP
  modes via ``noise_stddev`` (φ-EMA bias correction) or
  ``noisy_squared_grads`` (JME paired-stream second moment).  Knobs:
  ``decoupled_weight_decay``, ``update_rms_clip`` (StableAdamW).
- ``opaque.optimizers.lion`` — Lion (Tu et al., 2023); no DP-aware mode.
- ``opaque.optimizers.ademamix`` — AdEMAMix (Pagliardini et al., 2024);
  same DP-aware options as ``adamw``.
- ``opaque.optimizers.adafactor`` — Adafactor (factored second moment;
  vanilla + WD only in this release).
- ``opaque.optimizers.schedule_free`` — wrapper around any of the above
  (or ``torchopt.sgd``) implementing Defazio's schedule-free averaging.

For SGD use ``torchopt.sgd`` directly:

```python
import torchopt
optimizer = torchopt.sgd(lr=0.01, momentum=0.9)
```

---

## Complete Pattern

```python
import torchopt
from opaque.clipping import clipped_grad
from opaque.dpsgd.noise import gaussian_noise
from opaque.random import key

# Gradient pipeline
grad_fn, clip_state = clipped_grad(
    loss_fn, clipping_norm=1.0, batch_argnums=1,
    normalize_by=batch_size,
)
noise_fn, noise_state = gaussian_noise(stddev=noise_multiplier * clip_state.sensitivity, key=key(42))

# Optimizer
from opaque.optimizers import adamw

optimizer = adamw(lr=1e-3, weight_decay=0.01)
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
squares noised gradients ($\tilde{g}^2 = g^2 + 2gz + z^2$).  ``adamw``
(and ``ademamix``) accept two orthogonal corrections — see the
[Optimizers User Guide](../user-guide/optimizers.md#the-second-moment-problem-in-dp-training)
for details:

- **``noise_stddev`` (DP-AdamW-BC)** — subtracts the known noise variance
  from $\hat{v}_t$ via a β₂-EMA (Chooi et al.,
  [arXiv:2511.07843](https://arxiv.org/abs/2511.07843)).  Works with
  any Gaussian noise source.  With ``noise_stddev = 0`` (the default),
  the optimizer reduces to standard AdamW math.

  ```python
  optimizer = adamw(lr=1e-3, weight_decay=0.01, noise_stddev=initial_sigma)
  # Per-step override under adaptive clipping:
  updates, state = optimizer.update(grads, state, params=p, noise_stddev=current_sigma)
  ```

- **``noisy_squared_grads`` (JME)** — substitutes a separately privatized
  $g^2$ estimate from JME (Kalinin et al.,
  [arXiv:2502.06597](https://arxiv.org/abs/2502.06597)).  Requires MF
  correlated noise (``jme_noise``).

  ```python
  (noisy_grads, noisy_sq), state = jme_noise_fn(grads, state)
  updates, opt_state = optimizer.update(
      noisy_grads, opt_state, params=p, noisy_squared_grads=noisy_sq
  )
  ```

These address the same problem from different angles and **must not be
combined** — passing both kwargs at the same ``update()`` call raises
``ValueError``.

::: opaque.optimizers.adamw
    options:
      show_source: true
      heading_level: 3

## See Also

- [Gradient Clipping API](clipping.md) — includes `adaptive_clipped_grad()` and `auto_clipped_grad()`
- [Optimizers User Guide](../user-guide/optimizers.md)
- [Fine-tuning an LLM Tutorial](../tutorials/llm_finetuning.ipynb)
- [TorchOpt Documentation](https://torchopt.readthedocs.io/)
