# Optimizers

Opaque ships its own functional optimizer library at
[`opaque.optimizers`](#opaque.optimizers). The factories follow the
[TorchOpt](https://torchopt.readthedocs.io/) ``GradientTransformation``
protocol and accept optional DP-aware kwargs (``noise_stddev``,
``noisy_squared_grads``) at update time.

The library does **not** re-export torchopt primitives — Opaque ships
only what it adds value to.  For vanilla SGD / Adam / RMSprop / etc.
in a non-DP baseline or custom training loop, import them from
torchopt directly: ``from torchopt import sgd``.

---

## API shape

Functional: no hidden mutable state, explicit
`(updates, new_state) = optimizer.update(grads, state)` interface.

```python
import torchopt
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
- ``opaque.optimizers.lion`` — Lion (Chen et al., 2023); no DP-aware mode.
- ``opaque.optimizers.ademamix`` — AdEMAMix (Pagliardini et al., 2024);
  same DP-aware options as ``adamw``.
- ``opaque.optimizers.adafactor`` — Adafactor (factored second moment;
  vanilla + WD only in this release).
- ``opaque.optimizers.schedule_free`` — wrapper around any base
  ``GradientTransformation`` implementing Defazio's schedule-free
  averaging.  Composes with Opaque-built factories or torchopt
  primitives interchangeably.

For vanilla SGD / Adam / RMSprop / etc., import directly from torchopt:

```python
import torchopt
from opaque.optimizers import adamw, schedule_free

opt = torchopt.sgd(lr=0.01, momentum=0.9)        # canonical DP baseline
opt = adamw(lr=1e-3, noise_stddev=sigma)         # DP-AdamW-BC
opt = schedule_free(adamw(lr=1e-3))              # composition
opt = schedule_free(torchopt.sgd(lr=0.01))       # also fine
```

**A note on torchopt primitives under DP**: most are slow-but-functional
(`sgd`, `adam`, `rmsprop`, `radam`, `adadelta`).  Two are unsafe and
should be avoided for DP training: `torchopt.adagrad` (cumulative `∑ g²`
accumulates `t·σ²` with no decay; denominator runs away) and
`torchopt.adamax` (max-norm absorbs the half-normal noise mean
`σ·√(2/π)` permanently; per-coordinate LR is floored by noise).
Use ``adamw(..., noise_stddev=σ)`` for adaptive DP training instead.

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

---

## Serialisation

``state_dict`` and ``load_state_dict`` live in
``opaque.optimizers.serialization`` (a less-common building block,
not in the package's top-level ``__all__``).  ``state_dict`` flattens
any chain optimizer state into a ``dict[str, Any]`` of tensors and
Python primitives, ready for ``torch.save``; ``load_state_dict``
rebuilds the state from a freshly-initialised template:

```python
from opaque.optimizers import adamw
from opaque.optimizers.serialization import state_dict, load_state_dict

opt = adamw(lr=1e-3, weight_decay=0.01)
state = opt.init(params)
# ... train ...

# Save
torch.save(state_dict(state), "opt.pt")

# Load — template must have the same shape (init from same params).
template = opt.init(params)
state = load_state_dict(template, torch.load("opt.pt"))
```

Forward-compatible: paths missing from the saved dict keep the
template's value, so optimizers that gain new state fields between
releases load cleanly from older checkpoints.

## See Also

- [Gradient Clipping API](clipping.md) — includes `adaptive_clipped_grad()` and `auto_clipped_grad()`
- [Optimizers User Guide](../user-guide/optimizers.md)
- [Fine-tuning an LLM Tutorial](../tutorials/llm_finetuning.ipynb)
- [TorchOpt Documentation](https://torchopt.readthedocs.io/)
