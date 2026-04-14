# Optimizers

Opaque does not bundle optimizers. Use
[TorchOpt](https://torchopt.readthedocs.io/) functional optimizers for the
parameter-update step in your DP training loop.

## Why functional optimizers

Opaque's gradient pipeline is functional: every function takes state in and
returns new state out. TorchOpt follows the same pattern, making
integration seamless:

```python
import torchopt

optimizer = torchopt.adam(lr=1e-3)
opt_state = optimizer.init(params)

# Explicit state in -> state out
updates, opt_state = optimizer.update(noisy_grads, opt_state, params=params)
params = torchopt.apply_updates(params, updates)
```

No hidden mutable state. Every piece of the training loop is explicit.

## Complete training loop

```python
import torchopt
from opaque import clipped_grad, gaussian_noise
from opaque.random import key

# Gradient pipeline
grad_fn, clip_state = clipped_grad(
    loss_fn, clipping_norm=1.0, argnums=0, batch_argnums=1,
)
noise_fn, noise_state = gaussian_noise(
    stddev=noise_multiplier * clip_state.sensitivity, key=key(42),
)

# Optimizer
optimizer = torchopt.adam(lr=1e-3)
opt_state = optimizer.init(params)

# Training
for batch in dataloader:
    grads, clip_state = grad_fn(params, batch, state=clip_state)
    noisy_grads, noise_state = noise_fn(grads, noise_state)
    updates, opt_state = optimizer.update(noisy_grads, opt_state, params=params)
    params = torchopt.apply_updates(params, updates)
```

## Choosing an optimizer

**Adam** (`torchopt.adam`) is the recommended default. Its per-parameter
adaptive learning rates help compensate for DP noise, since different
parameters receive different signal-to-noise ratios. Adam typically
converges faster and is more robust to hyperparameter choices than SGD in
DP training.

```python
optimizer = torchopt.adam(lr=1e-3)
```

**AdamW** (`torchopt.adamw`) adds decoupled weight decay, useful for
fine-tuning pre-trained models:

```python
optimizer = torchopt.adamw(lr=1e-3, weight_decay=0.01)
```

**SGD** (`torchopt.sgd`) is simpler and useful as a debugging baseline:

```python
optimizer = torchopt.sgd(lr=0.01, momentum=0.9)
```

## JME-AdamW: Adam with MF correlated noise

When using matrix factorization (MF) noise for DP-FTRL, standard Adam
cannot be used directly — it computes the second moment by squaring the
noisy gradients, which breaks the MF noise correlation structure.

**JME** (Joint Moment Estimation, [arXiv:2502.06597](https://arxiv.org/abs/2502.06597))
solves this by privately estimating both moments via two independent MF noise
streams.  The second moment noise comes at ~22% additional privacy cost
under add/remove DP.

### Setup

```python
from opaque.noise.mf import jme_noise, band_mf_strategy
from opaque.optimizers import jme_adamw

# Strategy: momentum=beta1 (Adam's first moment workload)
strategy = band_mf_strategy(n_steps=1000, bands=8, momentum=0.9)

# Noise: jme_noise computes g², creates two MF streams, calibrates stddevs
noise_fn, noise_state = jme_noise(
    grad_template, strategy,
    noise_multiplier=sigma,
    key=key(42),
    zeta=clip_state.sensitivity,
    beta2=0.999,
)

# Optimizer: decoupled weight decay, callable LR schedule
optimizer = jme_adamw(lr=lr_schedule_fn, betas=(0.9, 0.999), weight_decay=0.01)
opt_state = optimizer.init(params)
```

### Training loop

```python
for batch in dataloader:
    grads, clip_state = grad_fn(params, batch, state=clip_state)
    (noisy_grads, noisy_sq), noise_state = noise_fn(grads, noise_state)
    updates, opt_state = optimizer.update(
        noisy_grads, opt_state,
        params=params, noisy_squared_grads=noisy_sq,
    )
    params = torchopt.apply_updates(params, updates)
```

### Accounting

Wrap the base mechanism with `acc.jme()` to account for both moment streams:

```python
mechanism = acc.band_mf(nm, sensitivity=S, num_groups=k)
if use_adam:
    mechanism = acc.jme(mechanism, zeta=clip_state.sensitivity,
                        max_column_norm=strategy._max_column_norm)
process = acc.cyclic_poisson(mechanism, sample_rate=q)
```

### CLI

```bash
python examples/train_dp_ftrl.py --preset smoke --optimizer adam --mechanism blt
```

Works with all MF mechanisms: `band_mf`, `blt`, `lambda_cgd`, `bisr`, `identity`.

## DP-specific optimizer considerations

### Why Adam works well with DP

In DP training, different parameters receive different signal-to-noise
ratios. Parameters with large true gradients (e.g., biases, early layers) get
a better ratio than parameters with small true gradients (e.g., deep
attention layers). Adam's per-parameter adaptive learning rate naturally
compensates for this, scaling up updates where the signal is strong relative
to noise and scaling down where it is weak. SGD applies the same learning
rate everywhere, which is suboptimal when noise levels vary across parameters.

### Weight decay and privacy

Weight decay (L2 regularization) does not consume privacy budget because it
is applied to the model parameters, not the data. AdamW's decoupled weight
decay is implemented as `params = params - wd * params` after the gradient
step — this is a deterministic function of the current parameters and does
not depend on the training data.

### Gradient accumulation

Opaque does not use gradient accumulation. Instead, `clipped_grad` processes
the entire batch (possibly in microbatches) and returns the sum of clipped
gradients in one call. This means you pass the full noisy gradient directly
to the optimizer — no manual accumulation is needed.

## Learning rate schedules

Standard LR schedules work with DP training. Linear warmup followed by cosine
decay is common for DP fine-tuning:

```python
import math

total_steps = 1000
warmup_steps = 50

for step in range(total_steps):
    if step < warmup_steps:
        lr = base_lr * step / warmup_steps
    else:
        progress = (step - warmup_steps) / (total_steps - warmup_steps)
        lr = base_lr * 0.5 * (1 + math.cos(math.pi * progress))

    optimizer = torchopt.adam(lr=lr)
    # ... or update the optimizer state's hyperparameters
```

A typical warmup is 5-10% of total training steps. The warmup helps
stabilize early updates when the model has not yet adapted to the noisy
gradient signal.

## Practical notes

**Do not clip optimizer updates.** Opaque clips *gradients* before the
optimizer sees them. Clipping after the optimizer would distort the
adaptive state.

**DDP compatibility.** In distributed training, optimizer states stay
synchronized automatically because `optimizer.update` is a pure function
and all ranks receive identical noisy gradients after `sum_gradients` and
noise addition (using the same key on all ranks). No explicit state
synchronization is needed.

## API reference

See [Optimizers API Reference](../api/optimizers.md) for TorchOpt
integration details.
