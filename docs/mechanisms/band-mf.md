# BandMF Mechanism

BandMF adds **correlated noise** across training steps using a banded Toeplitz
strategy matrix. Unlike independent Gaussian noise (DP-SGD), the noise at each
step depends on noise from recent steps. This correlation is designed so that
noise partially cancels on cumulative updates, reducing effective noise for the
same privacy budget — the DP-FTRL framework.

## Idea

In standard DP-SGD, each step adds independent noise $z_t \sim \mathcal{N}(0, \sigma^2 I)$.
When computing cumulative sums (as optimizers like SGD with momentum or Adam
implicitly do), these independent noises accumulate: after $n$ steps, the
cumulative noise variance grows as $n \sigma^2$.

Matrix factorization flips this: instead of adding independent noise, the
mechanism adds **correlated** noise $n_t = \sum_i C^{-1}_{t,i} \cdot z_i$ where
$C^{-1}$ is the inverse of a carefully chosen strategy matrix $C$. The
correlations are designed so that:

1. Each individual step is still noisy enough for privacy
2. But cumulative sums see partially-cancelling noise, giving lower MSE

**BandMF** specifically uses a **banded lower-triangular Toeplitz** matrix for
$C$. "Banded" means each step's noise depends only on the last $b$ steps
(controlled by the `bands` parameter). "Toeplitz" means the coefficients
repeat — the correlation pattern is the same at every step.

**Key property**: The Toeplitz coefficients are optimized so that column norms
equal 1, giving single-participation sensitivity = 1.0.

## Mathematics

### Strategy matrix

The encoder $C$ is an $n \times n$ banded lower-triangular Toeplitz matrix:

$$C_{t,s} = \begin{cases} c_{t-s} & \text{if } 0 \leq t - s < b \\ 0 & \text{otherwise} \end{cases}$$

where $c_0, c_1, \ldots, c_{b-1}$ are the optimized Toeplitz coefficients and
$b$ is the number of bands. The coefficients satisfy $\sum_{j=0}^{b-1} c_j^2 = 1$
(column norm normalization).

### Noise generation

At step $t$, the mechanism:

1. Draws fresh noise $z_t \sim \mathcal{N}(0, \sigma^2 I)$
2. Computes correlated noise $n_t = \sum_{j=0}^{\min(t, b-1)} C^{-1}_{t, t-j} \cdot z_{t-j}$
3. Adds $n_t$ to the clipped gradient sum

Only the last $b$ noise vectors need to be stored — memory is $O(b \cdot d)$ where
$d$ is the parameter dimension.

### Sensitivity

Under single participation (each example appears at most once), the
sensitivity is the maximum column $\ell_2$ norm of $C$:

$$S = \max_j \|C_{\cdot, j}\|_2 = 1$$

This equals 1.0 by construction for the standard Toeplitz optimization.

### Privacy analysis

The entire training run reduces to a single Gaussian mechanism with
effective noise multiplier:

$$\sigma_{\text{eff}} = \frac{\sigma}{S} = \sigma$$

The PLD is computed once for this effective Gaussian (not per-step),
then optionally composed with subsampling amplification.

## Supported amplifications

### Cyclic Poisson subsampling (`cyclic_poisson`)

The primary amplification method for BandMF. The training run is decomposed
into $k = \lceil n / b \rceil$ independent **groups** of $b$ consecutive
steps. Within each group, cyclic Poisson subsampling means each example
is included independently with probability $q$.

**What this means**: instead of analyzing the full $n$-step run as one
mechanism, we analyze $k$ independent Poisson-subsampled Gaussian mechanisms
and compose them. Each group has:

- Effective noise multiplier: $\sigma_{\text{eff}} = \sigma / S$
- Sample rate: $q$

The total privacy is the $k$-fold self-composition of the per-group PLD:

$$\text{PLD}_{\text{total}} = \text{PLD}_{\text{group}}^{\otimes k}$$

This is computed efficiently with 2 FFTs (self-composition).

```python
proc = acc.cyclic_poisson(
    acc.band_mf(noise_multiplier=1.0, n_steps=1000, bands=10),
    sample_rate=0.01,
)
eps = proc.epsilon_at(delta=1e-5)
```

| Amplification | Supported | Notes |
|---------------|:---------:|-------|
| `poisson()` | No | Use `cyclic_poisson()` instead |
| `truncated_poisson()` | No | Not applicable to MF mechanisms |
| `cyclic_poisson()` | Yes | Decomposes into $\lceil n/b \rceil$ independent groups |

!!! note "Without amplification"
    You can also use BandMF without subsampling by omitting the
    `cyclic_poisson()` wrapper. This accounts for the full training run
    as a single Gaussian mechanism — useful for comparison or when
    subsampling is not applicable.

## Code examples

### Noise injection

```python
from opaque import band_mf_noise
from opaque.random import key

noise_fn, noise_state = band_mf_noise(
    grad_template=params,    # pytree with correct shapes/dtypes
    n_steps=1000,
    stddev=noise_multiplier * clip_state.sensitivity,
    key=key(42),
    bands=10,
)

for step in range(1000):
    grads, clip_state = grad_fn(params, batch, state=clip_state)
    noisy_grads, noise_state = noise_fn(grads, noise_state)
    params = params - lr * noisy_grads
```

### Privacy accounting

```python
import opaque.accounting as acc

# BandMF with cyclic Poisson amplification (recommended)
proc = acc.cyclic_poisson(
    acc.band_mf(noise_multiplier=1.0, n_steps=1000, bands=10),
    sample_rate=0.01,
)
eps = proc.epsilon_at(delta=1e-5)

# BandMF without amplification (for comparison)
proc_no_amp = acc.band_mf(noise_multiplier=1.0, n_steps=1000, bands=10)
eps_no_amp = proc_no_amp.epsilon_at(delta=1e-5)

print(f"With cyclic Poisson: ε={eps:.4f}")
print(f"Without amplification: ε={eps_no_amp:.4f}")
```

### Calibration

```python
result = acc.calibrate(
    acc.epsilon_budget(3.0, delta=1e-5),
    lambda nm: acc.cyclic_poisson(
        acc.band_mf(nm, n_steps=1000, bands=10),
        sample_rate=0.01,
    ),
    param_min=0.1,
    param_max=10.0,
)
noise_multiplier = result.param
```

### End-to-end with cyclic sampler

BandMF works best with `CyclicPoissonSampler`, which creates a predictable
sampling pattern that the noise strategy exploits:

```python
import torch
from opaque import band_mf_noise, clipped_grad
from opaque.sampling import CyclicPoissonSampler
from opaque.random import key, split
import opaque.accounting as acc

n_steps, bands = 1000, 10
sample_rate = 0.01

# Calibrate
result = acc.calibrate(
    acc.epsilon_budget(3.0, delta=1e-5),
    lambda nm: acc.cyclic_poisson(
        acc.band_mf(nm, n_steps, bands), sample_rate,
    ),
    param_min=0.1, param_max=10.0,
)

# Setup
key_samp, key_noise = split(key(42), num=2)
grad_fn, clip_state = clipped_grad(
    loss_fn, clipping_norm=1.0, batch_argnums=1,
    normalize_by=batch_size,
)
noise_fn, noise_state = band_mf_noise(
    params, n_steps, stddev=result.param * clip_state.sensitivity,
    key=key_noise, bands=bands,
)
sampler = CyclicPoissonSampler(
    dataset, sampling_prob=sample_rate, cycle_length=bands,
    iterations=n_steps, key=key_samp,
)
loader = torch.utils.data.DataLoader(dataset, batch_sampler=sampler)

# Train
for batch in loader:
    grads, clip_state = grad_fn(params, batch, state=clip_state)
    noisy_grads, noise_state = noise_fn(grads, noise_state)
    params = params - lr * noisy_grads
```

## Parameter guide

| Parameter | Range | Effect |
|-----------|-------|--------|
| `noise_multiplier` | 0.1 – 10.0 (calibrate) | Higher = more private. Use `acc.calibrate()`. |
| `n_steps` | Must be known in advance | Total training iterations. |
| `bands` | 1 – `n_steps` | Number of Toeplitz bands. More = better noise but more memory. |

**Tips**:

- **`bands` = 4–20** covers most workloads. Start with 10.
- **`bands` = 1** reduces to independent noise (like standard DP-SGD).
- **`bands` = `n_steps`** is the full (unbanded) Toeplitz — optimal noise
  reduction but $O(n)$ memory per parameter.
- The number of cyclic groups is $\lceil n / b \rceil$. Fewer groups means
  better per-group amplification but more composition steps.
- BandMF requires knowing `n_steps` before training starts. If the training
  length is uncertain, use standard Gaussian noise with early stopping.
- Pair with `CyclicPoissonSampler` for consistent sampling and accounting.

## References

- **Choquette-Choo et al. (2023)** — [Multi-Epoch Matrix Factorization Mechanisms for Private Machine Learning](https://arxiv.org/abs/2306.08153).
  BandMF mechanism with banded Toeplitz optimization and cyclic Poisson
  amplification.
- **Kairouz et al. (2021)** — [Practical and Private (Deep) Learning without Sampling or Shuffling](https://arxiv.org/abs/2103.00039).
  DP-FTRL framework.
- **Denisov et al. (2022)** — [Improved Differential Privacy for SGD via Optimal Private Linear Operators](https://arxiv.org/abs/2202.08312).
  Matrix factorization for DP-SGD.
