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
from opaque.core.clipping import clipped_grad
from opaque.dpsgd.noise import gaussian_noise
from opaque.core.random import key

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
DP training.  For an improved version that corrects the second-moment bias
introduced by DP noise, see
[the second-moment problem](#the-second-moment-problem-in-dp-training) below.

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

## The second-moment problem in DP training

Standard Adam computes its second moment as $v_t = \beta_2 v_{t-1} + (1-\beta_2) g_t^2$.
In DP training, the optimizer receives *noised* gradients $\tilde{g}_t = g_t + z_t$, so
the second moment becomes:

$$v_t = \beta_2 v_{t-1} + (1-\beta_2) \tilde{g}_t^2 = \beta_2 v_{t-1} + (1-\beta_2)(g_t^2 + 2g_t z_t + z_t^2)$$

The $z_t^2$ term inflates the denominator, shrinking effective learning rates.
The $2g_t z_t$ cross term averages out but adds variance.  This is a known
problem: Gaussian noise with variance $\sigma^2$ adds an expected bias of
$\sigma^2$ to every component of $\hat{v}_t$.

Opaque provides two independent solutions.  They address the *same* problem
from different angles and **must not be combined** — using both would
double-correct the second moment.

### AdamW-BC: bias correction by noise variance subtraction

**Idea:** The noise variance $\Phi_t = \sigma_t^2$ is *known* (we chose it),
so we can subtract it from the biased $\hat{v}_t$.

$$\phi_t = \beta_2 \phi_{t-1} + (1-\beta_2) \Phi_t, \qquad
\hat{v}^{\text{corrected}}_t = \max\!\bigl(\hat{v}_t - \hat{\phi}_t,\; \gamma\bigr)$$

This is Algorithm 2 from
[Chooi et al. (arXiv:2511.07843)](https://arxiv.org/abs/2511.07843).  It is
cheap (one extra scalar EMA), requires no changes to the noise mechanism,
and works with any i.i.d. Gaussian noise source.

When `noise_stddev=0` (default), `adamw_bc` is numerically identical to
`torchopt.adamw` — use it as a drop-in replacement even without BC.

```python
from opaque.dpsgd.optimizers import adamw_bc

# Without BC — identical to torchopt.adamw
optimizer = adamw_bc(lr=1e-3, weight_decay=0.01)

# With BC — pass sigma
noise_stddev = noise_multiplier * clip_state.sensitivity
optimizer = adamw_bc(lr=1e-3, weight_decay=0.01, noise_stddev=noise_stddev)
```

With adaptive clipping (where sensitivity changes each step), override
per step:

```python
updates, opt_state = optimizer.update(
    noisy_grads, opt_state, params=params,
    noise_stddev=noise_multiplier * clip_state.sensitivity,
)
```

### AdamW-JME: privately-estimated second moments

**Idea:** Instead of squaring the noisy gradient (which amplifies noise),
use a *separately privatized* estimate of $g_t^2$ from a second noise
stream.

**JME** (Joint Moment Estimation,
[Kalinin et al., arXiv:2502.06597](https://arxiv.org/abs/2502.06597))
maintains two independent matrix-factorization correlated noise streams —
one for $g_t$ (first moment) and one for $g_t^2$ (second moment).  The
optimizer receives both and uses each for its respective EMA:

$$\mu_t = \beta_1 \mu_{t-1} + (1-\beta_1) \tilde{g}_t, \qquad
v_t = \beta_2 v_{t-1} + (1-\beta_2) \widetilde{g^2}_t$$

The additional second-moment stream costs ~22% extra privacy budget under
add/remove DP.

JME requires a compatible MF noise mechanism (`jme_noise`), so it only
applies to DP-FTRL training — not to standard DP-SGD with i.i.d.
Gaussian noise.

### When to use which

| Scenario | Recommended | Why |
|---|---|---|
| DP-SGD with Gaussian noise | `adamw_bc` | No MF streams available for JME |
| DP-FTRL without Adam | `torchopt.sgd` | No second moment to correct |
| DP-FTRL with Adam | `adamw_jme` | Better v estimate; JME streams available |
| DP-FTRL with Adam, no extra budget | `adamw_bc` | BC is free; JME costs ~22% ε |

## AdamW-JME: setup and usage

```python
from opaque.dpftrl.noise import jme_noise, band_mf_strategy
from opaque.dpftrl.optimizers import adamw_jme

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
optimizer = adamw_jme(lr=lr_schedule_fn, betas=(0.9, 0.999), weight_decay=0.01)
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

Works with MF mechanisms supported by `jme_noise` auto-derivation (see [DP-FTRL](dp-ftrl.md)): `band_mf`, `blt`, `bisr`, `bsr`, `identity`. For `lambda_cgd`, pass `second_moment_strategy` explicitly.

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
