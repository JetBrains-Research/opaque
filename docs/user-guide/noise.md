# Noise Addition

After clipping gradients, the next step in DP-SGD is adding calibrated noise
to the sum of clipped gradients. The noise obscures individual contributions,
providing the differential privacy guarantee. The amount of noise is
proportional to the sensitivity (from clipping) and the desired privacy level.

All noise functions in Opaque follow the same pattern: they return a
`(noise_fn, state)` tuple with immutable state.

For mathematical details, privacy analysis, and parameter guidance for
each mechanism, see the [Mechanisms](../mechanisms/index.md) reference.

For MF-specific assumptions (workload fidelity vs DP correctness, LR schedules,
JME, BSR scope), see [Correlated noise (DP-FTRL)](dp-ftrl.md).

## Gaussian noise

`gaussian_noise` is the standard noise mechanism for DP-SGD. It adds
independent Gaussian noise to each gradient tensor.

```python
from opaque.dpsgd.noise import gaussian_noise
from opaque.core.random import key

noise_fn, noise_state = gaussian_noise(
    stddev=noise_multiplier * clip_state.sensitivity,
    key=key(42),
)

noisy_grads, noise_state = noise_fn(grads, noise_state)
```

### Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `stddev` | `float` | Standard deviation of Gaussian noise. Typically `noise_multiplier * sensitivity`. |
| `key` | `RngKey` | Explicit RNG key for deterministic noise. Create with `key(seed)`. |

### Calibrating stddev

The noise standard deviation is `noise_multiplier * sensitivity`, where:

- `noise_multiplier` is determined by the target privacy budget (use
  `acc.calibrate()` to find it)
- `sensitivity` comes from `clip_state.sensitivity`

```python
import opaque.accounting as acc

result = acc.calibrate(
    acc.epsilon_budget(3.0, delta=1e-5),
    lambda nm: acc.poisson(acc.gaussian(nm), sample_rate) * num_steps,
    param_min=0.1, param_max=10.0,
)

noise_fn, noise_state = gaussian_noise(
    stddev=result.param * clip_state.sensitivity, key=key(42),
)
```

See [Privacy Accounting](accounting.md) for details on calibration.

### State flow

Each call to `noise_fn` returns a new state with an incremented step counter.
Always use the returned state for the next call.

```python
noise_fn, state = gaussian_noise(stddev=1.0, key=key(42))

for batch in dataloader:
    grads, clip_state = grad_fn(params, batch, state=clip_state)
    noisy_grads, state = noise_fn(grads, state)  # state advances
    updates, opt_state = optimizer.update(noisy_grads, opt_state)
    params = torchopt.apply_updates(params, updates)
```

Internally, noise at step t is generated from `fold_in(base_key, t)`, ensuring
deterministic per-step noise regardless of execution order.

### Zero stddev

When `stddev=0`, `gaussian_noise` returns a no-op function that passes
gradients through unchanged. This is useful for toggling DP on and off
without changing the training loop.

### Per-group noise

When using [per-group clipping](clipping.md#per-group-clipping), the
recommended approach is MSE-optimal allocation via `per_group_noise_stddev`,
which varies σ across groups — putting less noise on smaller-norm groups:

```python
from opaque.dpsgd.noise import per_group_noise_stddev

stddev = per_group_noise_stddev(clip_state, noise_multiplier)
noise_fn, noise_state = gaussian_noise(stddev=stddev, key=key(42))
```

This returns a `PerGroup` of per-group standard deviations with
$\sigma_i \propto \sqrt{C_i}$. Privacy accounting is identical to the
isotropic case — just `gaussian(nm)`.

The [training script](https://github.com/JetBrains-Research/opaque/blob/main/examples/train_causal_lm.py) uses this by default
when per-group clipping is active.

Alternatively, isotropic noise (same σ everywhere) also works:

```python
stddev = noise_multiplier * clip_state.sensitivity
noise_fn, noise_state = gaussian_noise(stddev=stddev, key=key(42))
```

`clip_state.sensitivity` returns a scalar $\lVert C \rVert_2 / n$ even with
per-group clipping norms, so no code changes are needed.

## Bounded Gaussian noise

Standard Gaussian noise has unbounded support, which means privatized outputs
can land arbitrarily far from the input. Bounded Gaussian noise restricts the
support to a finite interval, giving tighter privacy accounting because the
worst-case divergence is limited.

Opaque provides two variants that differ in how they handle the probability mass
that falls outside the bounds:

### Truncated (renormalized)

`truncated_gaussian_noise` draws from a Gaussian renormalized over the bounded
interval. No probability mass sits at the boundaries — the density is smooth
but slightly taller than the original Gaussian.

```python
from opaque.dpsgd.noise import truncated_gaussian_noise
from opaque.core.random import key

noise_fn, noise_state = truncated_gaussian_noise(
    stddev=1.0,
    radius=2.0,
    key=key(42),
)
noisy_grads, noise_state = noise_fn(grads, noise_state)
```

The truncation uses an inverse-CDF method: for each gradient element, noise is
sampled from a Gaussian centered on that element and truncated to the bounds.

For high-dimensional tasks like model training, the truncated Gaussian
converges to the standard Gaussian, so use `acc.gaussian(noise_multiplier)`
for accounting.

### Which variant to use

The truncated Gaussian provides bounded support, which can be useful when
gradient bounds are important for downstream optimization. For privacy
accounting, use `acc.gaussian()` regardless of the noise variant.

## Matrix-factorization noise (DP-FTRL)

Standard Gaussian noise is independent across training steps. Matrix-
factorization (MF) noise introduces correlations between steps so that noise
partially cancels when aggregated over the training run. This reduces the
effective noise on cumulative updates, improving accuracy for the same privacy
budget.

MF noise implements the DP-FTRL framework (Kairouz et al. 2021, Denisov et al.
2022). Instead of adding independent noise z_t at each step, the mechanism
adds correlated noise n_t = sum_i C_inv[t,i] * z_i, where C_inv is the
inverse of a strategy matrix chosen to minimize cumulative error.

The mechanism factors a **workload matrix** A (what the optimizer computes,
e.g., prefix sums) into:

- **C** (strategy matrix): Encodes the privacy mechanism
- **C^{-1}** (noising matrix): Generates correlated noise at each step
- **B = A C^{-1}** (decoder matrix): Relates noisy outputs back to workload queries

The sensitivity of C determines how much noise is needed, while the error of B
determines the effective noise on the output.

### When to use MF noise

Use MF noise when:

- Training runs many steps (hundreds to thousands)
- The privacy budget is tight and independent noise degrades accuracy too much
- You want better privacy-utility trade-offs at the same epsilon

MF noise has higher per-step overhead (maintaining correlation buffers) and
requires knowing the total number of steps in advance.

### Variants

Opaque provides five MF strategies, all used through the unified `mf_noise()` dispatcher:

| Strategy factory | Memory | Best for |
|----------|--------|----------|
| `band_mf_strategy()` | O(bands) | General use with cyclic Poisson amplification |
| `blt_strategy()` | O(buffers) | Long training runs (n > 5000), multi-epoch |
| `lambda_cgd_strategy()` | O(1) | Zero extra memory (PRNG replay) |
| `bisr_strategy()` | O(bandwidth) | Asymptotically optimal, arbitrary bandwidth |
| `identity_strategy()` | O(1) | Testing MF infrastructure with standard noise |

All strategies are created by factory functions and passed to `mf_noise()`:

```python
from opaque.dpftrl.noise import mf_noise, band_mf_strategy
from opaque.core.random import key

strategy = band_mf_strategy(n_steps=1000, bands=10)
noise_fn, noise_state = mf_noise(
    grad_template=params,
    strategy=strategy,
    stddev=noise_multiplier * clip_state.sensitivity,
    key=key(42),
)

for step in range(1000):
    grads, clip_state = grad_fn(params, batch, state=clip_state)
    noisy_grads, noise_state = noise_fn(grads, noise_state)
    params = params - lr * noisy_grads
```

The `grad_template` argument provides shape and dtype information for
pre-allocating noise buffers. Pass any pytree with the same structure as
the gradients (e.g., the model parameters).

In distributed training, pass the same `key(seed)` on all ranks to produce
identical noise. See [Distributed Training](distributed.md) and
[RNG Key](rng-key.md) for details.

### `band_mf_strategy`

Banded Toeplitz strategy. Optimizes banded Toeplitz coefficients for the
workload. Uses cyclic Poisson amplification for privacy accounting.

```python
from opaque.dpftrl.noise import mf_noise, band_mf_strategy
from opaque.core.random import key

strategy = band_mf_strategy(n_steps=1000, bands=10, momentum=0.95)
noise_fn, noise_state = mf_noise(
    params, strategy,
    stddev=noise_multiplier * clip_state.sensitivity,
    key=key(42),
)
```

### `blt_strategy`

Buffered Linear Toeplitz strategy. More memory-efficient than BandMF for
long training runs, using a parametric representation via exponential decay
buffers. Supports multi-epoch training via `min_sep` and `max_participations`.

```python
from opaque.dpftrl.noise import mf_noise, blt_strategy
from opaque.core.random import key

strategy = blt_strategy(
    n_steps=10000, min_sep=100, max_participations=5, max_buffers=10,
)
noise_fn, noise_state = mf_noise(
    params, strategy,
    stddev=noise_multiplier * clip_state.sensitivity,
    key=key(42),
)
```

### `lambda_cgd_strategy`

DP-λCGD strategy — uses PRNG seed replay instead of storing previous noise
vectors. Zero extra memory overhead compared to DP-SGD.

```python
from opaque.dpftrl.noise import mf_noise, lambda_cgd_strategy
from opaque.core.random import key

strategy = lambda_cgd_strategy(
    lambda_=0.9, n_steps=total_steps,
    min_sep=steps_per_epoch, max_participations=num_epochs,
)
noise_fn, noise_state = mf_noise(
    params, strategy,
    stddev=noise_multiplier * clip_state.sensitivity,
    key=key(42),
)
```

### `bisr_strategy`

BISR (Banded Inverse Square Root) strategy — generalises λCGD to arbitrary
bandwidth p ≥ 2. Asymptotically optimal.

```python
from opaque.dpftrl.noise import mf_noise, bisr_strategy
from opaque.core.random import key

strategy = bisr_strategy(
    n_steps=total_steps, bandwidth=4, momentum=0.95,
)
noise_fn, noise_state = mf_noise(
    params, strategy,
    stddev=noise_multiplier * clip_state.sensitivity,
    key=key(42),
)
```

### `identity_strategy`

Identity strategy — equivalent to standard DP-SGD (independent noise at each
step) but using the MF API. Useful for testing or as a baseline.

```python
from opaque.dpftrl.noise import mf_noise, identity_strategy
from opaque.core.random import key

strategy = identity_strategy()
noise_fn, noise_state = mf_noise(
    params, strategy,
    stddev=noise_multiplier * clip_state.sensitivity,
    key=key(42),
)
```

### Privacy accounting for MF noise

MF noise has different sensitivity than standard Gaussian noise because
the correlated strategy matrix amplifies or attenuates individual
contributions. The noise **strategy** computes `sensitivity` and
`gram_matrix` from the mechanism parameters; the accounting constructor
receives these values rather than recomputing them. This ensures that
noise generation and privacy accounting always agree on the mechanism.

```python
import opaque.accounting as acc
from opaque.dpftrl.noise import band_mf_strategy, lambda_cgd_strategy

# BandMF — strategy provides sensitivity and num_groups
strategy = band_mf_strategy(n_steps=1000, bands=10)
proc = acc.cyclic_poisson(
    acc.band_mf(1.0, sensitivity=strategy.sensitivity,
                num_groups=strategy.num_groups),
    sample_rate=0.01,
)
eps = proc.epsilon_at(1e-5)

# DP-λCGD / BISR / BLT — strategy provides sensitivity and gram_matrix
strategy = lambda_cgd_strategy(
    lambda_=0.9, n_steps=total_steps,
    min_sep=steps_per_epoch, max_participations=num_epochs,
)
proc = acc.balls_in_bins(
    acc.lambda_cgd(1.0, sensitivity=strategy.sensitivity,
                   gram_matrix=strategy.gram_matrix),
    num_bins=steps_per_epoch, num_epochs=num_epochs,
)
```

See [Privacy Accounting — Matrix factorization mechanisms](accounting.md#matrix-factorization-mechanisms)
for the full API.

### Multi-participation (multi-epoch)

When training for multiple epochs, each example participates multiple times.
Strategies that support multi-epoch patterns (`blt_strategy`, `lambda_cgd_strategy`,
`bisr_strategy`) accept `min_sep` and `max_participations` to compute tight
sensitivity bounds:

```python
from opaque.dpftrl.noise import mf_noise, blt_strategy
from opaque.core.random import key

strategy = blt_strategy(
    n_steps=5000,
    min_sep=100,            # minimum steps between participations
    max_participations=5,   # 5 epochs
)
noise_fn, state = mf_noise(
    grad_template, strategy,
    stddev=noise_multiplier * clipping_norm,
    key=key(42),
)
```

### Sensitivity

The sensitivity is computed internally by each strategy factory. You can
inspect it via the `sensitivity` attribute:

```python
strategy = band_mf_strategy(n_steps=1000, bands=10)
print(strategy.sensitivity)  # typically 1.0 for normalized strategies
```

### Comparison: DP-SGD vs MF strategies

For a linear regression with n=1000 steps, epsilon=1.0:

| Method | Strategy | Relative MSE | Memory |
|--------|-----------|-------------|--------|
| DP-SGD | `identity_strategy()` | Baseline | O(1) |
| BandMF | `band_mf_strategy(bands=4)` | ~0.7x | O(bands) |
| BLT | `blt_strategy(max_buffers=3)` | ~0.6x | O(buffers) |
| λCGD | `lambda_cgd_strategy(lambda_=0.9)` | ~0.7x | O(1) |
| BISR | `bisr_strategy(bandwidth=4)` | ~0.6x | O(bandwidth) |

Values are illustrative; actual results depend on problem specifics.

### MF noise with cyclic sampling

MF noise works best with `CyclicPoissonSampler`, which creates a predictable
sampling pattern that the noise strategy can exploit:

```python
from opaque.dpftrl.sampling import CyclicPoissonSampler
from opaque.core.random import key

sampler = CyclicPoissonSampler(
    dataset, sampling_prob=sample_rate, cycle_length=4,
    iterations=num_steps, key=key(0),
)
```

## Distributed noise synchronization

In distributed training, all devices must add the same noise to maintain model
consistency. Pass the **same key** on every rank:

```python
# Same key on all ranks → identical noise → models stay in sync
noise_fn, noise_state = gaussian_noise(
    stddev=noise_multiplier * sensitivity, key=key(42),
)
```

For independent per-rank noise (not typical for centralized DP-SGD), derive
a per-rank key via `fold_in`:

```python
from opaque.core.random import key, fold_in
import torch.distributed as dist

rank = dist.get_rank()
noise_fn, noise_state = gaussian_noise(
    stddev=noise_multiplier * sensitivity,
    key=fold_in(key(42), rank),
)
```

For validation, call `sync(noise_state)` to assert that the RNG key and
step counter match across ranks. The `sync()` dispatcher auto-detects the
noise state type. See [Distributed Training](distributed.md) for details.

## References

- [Choquette-Choo et al., 2023](https://arxiv.org/abs/2306.08153) -- BandMF
- [McMahan et al., 2024](https://arxiv.org/abs/2404.16706) -- BLT
- [Choquette-Choo et al., 2024](https://arxiv.org/abs/2408.08868) -- Multi-epoch BLT
- [Kalinin et al., 2026](https://arxiv.org/abs/2601.22334) -- DP-λCGD
- [Kalinin et al., 2026](https://arxiv.org/abs/2505.12128) -- BISR
- [McMahan et al., 2025](https://arxiv.org/abs/2504.21413) -- Inversion theorem
- [Kairouz et al., 2021](https://arxiv.org/abs/2103.00039) -- DP-FTRL

## API reference

See [Noise API Reference](../api/noise.md) for complete function signatures
and return types.
