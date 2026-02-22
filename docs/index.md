# Opaque

**Functional DP-SGD for PyTorch**

Opaque provides composable primitives for differentially private model training
in PyTorch. Built on `torch.func`, every component uses explicit state -- no
hooks, no subclassing, no hidden mutation.

## Per-example gradient clipping

`clipped_grad` computes per-example gradients via `vmap` + `grad`, clips each
to a maximum L2 norm, and sums the result.

```python
from opaque import clipped_grad

grad_fn, clip_state = clipped_grad(
    loss_fn, l2_clip_norm=1.0, argnums=0, batch_argnums=1,
)
grads, clip_state = grad_fn(params, batch, state=clip_state)
```

## Noise injection

Add calibrated Gaussian noise scaled to the clipping sensitivity. All noise
functions return `(noise_fn, state)`. Pass the same key on all ranks for synchronized distributed noise.

```python
from opaque import gaussian_noise
from opaque.random import key

noise_fn, noise_state = gaussian_noise(
    stddev=noise_multiplier * clip_state.sensitivity(), key=key(42),
)
noisy_grads, noise_state = noise_fn(grads, noise_state)
```

## Privacy accounting

Composable `DpProcess` objects built on a Rust PLD engine. Mechanisms compose
with `*` (repeat) and `|` (heterogeneous composition). Query multiple privacy
metrics from the same object.

```python
import opaque.accounting as acc

step = acc.poisson(acc.gaussian(noise_multiplier), sample_rate=0.01)
training = step * 1000

eps = training.epsilon_at(delta=1e-5)
adv = training.advantage()
beta = training.beta_at(alpha=0.01)
```

## Noise calibration

Binary search for the noise multiplier (or any parameter) that satisfies a
target privacy budget.

```python
result = acc.calibrate(
    acc.epsilon_budget(3.0, delta=1e-5),
    lambda nm: acc.poisson(acc.gaussian(nm), sample_rate=0.01) * 1000,
    param_min=0.1, param_max=10.0,
)
noise_multiplier = result.param
```

## Poisson sampling

Privacy-amplifying batch sampling. Each example is included independently with
probability `sample_rate`, producing variable-size batches. Distributed mode is
detected and sharded automatically.

```python
from opaque.sampling import PoissonSampler
from opaque.random import key

sampler = PoissonSampler(dataset, sample_rate=0.01, num_epochs=10, key=key(0))
loader = DataLoader(dataset, batch_sampler=sampler)
```

## Distributed training

DDP-compatible: each device clips locally, gradients are aggregated via
AllReduce, and noise is added identically on every device (same key, same
noise).

```python
from opaque.distributed import sum_gradients

grads, clip_state = grad_fn(params, local_batch, state=clip_state)
grads = sum_gradients(grads)
noisy_grads, noise_state = noise_fn(grads, noise_state)
```

## Privacy auditing

Empirical privacy validation via membership inference attacks with
Clopper-Pearson confidence intervals.

```python
from opaque.auditing import audit

result = audit(member_scores, non_member_scores, confidence=0.95)
print(f"Empirical epsilon: {result.epsilon:.2f}")
```

## Next steps

**Getting started**: [Installation](getting-started/installation.md) and
[Quick Start](getting-started/quickstart.md).

**Understanding the API**: [User Guide](user-guide/index.md) covers each
module with detailed explanations, API patterns, and practical guidance.

**Hands-on practice**: [Tutorials](tutorials/README.md) are task-based Jupyter
notebooks that exercise the library on concrete problems.

**API details**: [API Reference](api/index.md) provides complete function
signatures and docstrings.
