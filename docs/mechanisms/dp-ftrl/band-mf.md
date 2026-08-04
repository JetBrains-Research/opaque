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

### Poisson subsampling (`opaque.dpftrl.accounting.poisson`)

The primary amplification method for BandMF. The training run is decomposed
into $k = \lceil n / b \rceil$ independent **groups** of $b$ consecutive
steps. Within each group, **cyclic Poisson** participation means each example
in the active group is included independently with probability $q$ (this is
what :func:`opaque.dpftrl.accounting.poisson` composes over; there is no
separate ``cyclic_poisson`` factory).

**What this means**: instead of analyzing the full $n$-step run as one
mechanism, we analyze $k$ independent Poisson-subsampled Gaussian mechanisms
and compose them. Each group has:

- Effective noise multiplier: $\sigma_{\text{eff}} = \sigma / S$
- Sample rate: $q$

The total privacy is the $k$-fold self-composition of the per-group PLD:

$$\text{PLD}_{\text{total}} = \text{PLD}_{\text{group}}^{\otimes k}$$

This is computed efficiently with 2 FFTs (self-composition).

```python
import opaque.dpftrl.accounting as dpftrl_acc
from opaque.dpftrl.noise import band_mf_strategy

strategy = band_mf_strategy(bands=10)
proc = dpftrl_acc.poisson(
    dpftrl_acc.mf_gaussian(1.0, strategy),
    sample_rate=0.01,
    n_steps=1000,
)
eps = proc.epsilon_at(delta=1e-5)
assert eps > 0 and eps < float("inf"), f"epsilon out of range: {eps}"
print(f"Epsilon (δ=1e-5): {eps:.2f}")
```

| Amplification | Supported | Notes |
|---------------|:---------:|-------|
| `opaque.dpsgd.accounting.poisson` | No | DP-SGD per-step factory; different object |
| `opaque.dpsgd.accounting.poisson` (truncated) | No | DP-SGD only |
| `opaque.dpftrl.accounting.poisson` | Yes | Whole-process MF Poisson; $\lceil n/b \rceil$ groups for ``BandMf`` |

### b-min-sep subsampling (`b_min_sep`)

[Dong & Ganesh (2026)](https://arxiv.org/abs/2602.09338) introduce **warm-start
b-min-sep** subsampling: Poisson-style inclusion from the full dataset while
excluding examples that appeared in any of the previous $b-1$ batches. For
the same target expected batch size per iteration as cyclic Poisson (per-example
rate $p_0 = \mathbb{E}[|B|]/|D|$), the paper’s per-iteration inclusion probability
is $p = p_0 / (1 - p_0(b-1))$ when $b>1$.

Opaque pairs this with **Monte Carlo PLD** accounting (same family as BnB MC
for matrix mechanisms): pass the BandMF strategy’s first-column coefficients,
`n_steps`, and `p0` to `opaque.accounting.b_min_sep(...)`.
Training scripts can select it with `--band-mf-sampling b_min_sep` (see
`examples/train_dpftrl.py`).

!!! warning
    Monte Carlo PLDs are empirical point estimates. Their conservative grid
    bucketing does not replace the RC-4 confidence correction, so reported ε
    values are not upper confidence bounds.

| Amplification | Supported | Notes |
|---------------|:---------:|-------|
| `b_min_sep()` | Yes | MC PLD; default `num_mc_samples=100_000` |

For large `n_steps × num_mc_samples`, the implementation keeps **one copy** of
the MC random transcripts in **Rust** (compact `f64` arrays) and reuses them
for every noise-multiplier probe during calibration (no Python list blow-up).
Optional cap: set `OPAQUE_B_MIN_SEP_TRANSCRIPT_CACHE_MAX_BYTES` (default ~4 GiB);
use `0` to disable transcript reuse and fall back to one-shot MC per `pld()` call.

!!! note "Without amplification"
    You can also use BandMF without subsampling by omitting the
    :func:`opaque.dpftrl.accounting.poisson` wrapper (compose the Gaussian
    mechanism directly if your accounting path supports it). Useful for
    comparison when subsampling is not applicable.

!!! note
    The `dpftrl_acc.band_mf()` API takes pre-computed sensitivity and group count
    from the noise strategy. For end-to-end usage, `mf_gaussian_noise()` +
    `band_mf_strategy()` computes these automatically.

## Assumptions and limitations

- **DP correctness** is for the **implemented** banded Toeplitz matrix \(C\) and the sampler you pair with accounting. For Monte Carlo amplifications, RC-4 confidence correction is still required before a reported ε is an upper confidence bound.
- **`lr_schedule` (optional)**: workload coefficients are folded into a **lower-triangular Toeplitz** workload inside optimization. That matches the usual **constant** learning-rate momentum-SGD story. For a **varying** schedule \(\eta_t\), the implied Toeplitz workload can differ from the full map \(W_{t,s}=\eta_t\beta^{t-s}\): privacy is still valid for the constructed \(C\); **utility** alignment is approximate unless \(\eta\) is constant. See the [DP-FTRL user guide](../../user-guide/dp-ftrl.md).
- **Momentum \(\beta=0\)**: Opaque warns because the workload becomes essentially identity (little benefit over independent noise).

## Code examples

### Noise injection

```python
from opaque.dpftrl.noise import mf_gaussian_noise, band_mf_strategy
from opaque.random import key

strategy = band_mf_strategy(bands=10)
noise_fn, noise_state = mf_gaussian_noise(
    grad_template=params,
    strategy=strategy,
    n_steps=1000,
    noise_multiplier=noise_multiplier,
    key=key(42),
)

for step in range(1000):
    grads, clip_state = grad_fn(params, batch, state=clip_state)
    noisy_grads, noise_state = noise_fn(grads, noise_state)
    params = params - lr * noisy_grads.pytree
```

### Privacy accounting

The accounting constructor receives `sensitivity` and `num_groups` from
the same `band_mf_strategy` used for noise generation. This keeps both
components in sync:

```python
import opaque.dpftrl.accounting as dpftrl_acc
from opaque.dpftrl.noise import band_mf_strategy

strategy = band_mf_strategy(bands=10)

# BandMF with Poisson amplification (recommended)
proc = dpftrl_acc.poisson(
    dpftrl_acc.mf_gaussian(1.0, strategy),
    sample_rate=0.01,
    n_steps=1000,
)
eps = proc.epsilon_at(delta=1e-5)
assert eps > 0 and eps < float("inf"), f"epsilon out of range: {eps}"
```

!!! note
    Always use `strategy.sensitivity(n_steps=...)` and `strategy.num_groups` rather than
    hardcoded values. The strategy computes these from the optimized Toeplitz
    coefficients.

### End-to-end BandMF example

BandMF uses `opaque.dpftrl.sampling.CyclicPoissonSampler` with ``bands`` matching the
strategy so participation lines up with the noise.  The same class with
``bands=1`` gives plain Poisson on the full dataset each step for an identity MF
baseline (``identity_mf`` / ``identity_strategy``):

```python
from opaque.dpftrl.sampling import CyclicPoissonSampler
from opaque.random import key

sampler = CyclicPoissonSampler(
    dataset,
    sample_rate=0.01,
    bands=1,
    n_steps=1000,
    key=key(0),
)
```

BandMF training (``bands`` matches ``band_mf_strategy``):

```python
import torch
from opaque.dpftrl.clipping import clipped_grad
from opaque.dpftrl.noise import mf_gaussian_noise, band_mf_strategy
from opaque.dpftrl.sampling import CyclicPoissonSampler
from opaque.random import key, split

n_steps, bands = 1000, 10
sample_rate = 0.01

# Setup
key_samp, key_noise = split(key(42), num=2)
grad_fn, clip_state = clipped_grad(
    loss_fn, clipping_norm=1.0, batch_argnums=1,
    normalize_by=batch_size,
)
strategy = band_mf_strategy(bands=bands)
noise_fn, noise_state = mf_gaussian_noise(
    params, strategy,
    n_steps=n_steps,
    noise_multiplier=result.param,
    key=key_noise,
)
sampler = CyclicPoissonSampler(
    dataset,
    sample_rate=sample_rate,
    bands=bands,
    n_steps=n_steps,
    key=key_samp,
)
loader = torch.utils.data.DataLoader(dataset, batch_sampler=sampler)

# Train
for batch in loader:
    grads, clip_state = grad_fn(params, batch, state=clip_state)
    noisy_grads, noise_state = noise_fn(grads, noise_state)
    params = params - lr * noisy_grads.pytree
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
