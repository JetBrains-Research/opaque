# Matrix Factorization & DP-FTRL

Matrix factorization mechanisms replace the independent Gaussian noise of standard DP-SGD with **correlated noise** that achieves better utility at the same privacy budget. This is the technique behind DP-FTRL (Differentially Private Follow-The-Regularized-Leader).

## Why Correlated Noise?

In standard DP-SGD, each training step adds independent noise. When you compute running averages or sums (as optimizers do internally), the noise **accumulates** as O(sqrt(n)) over n steps.

Matrix factorization mechanisms instead add noise that is correlated across steps, designed so that the noise **partially cancels out** in the running sum. The result: same privacy guarantee, but lower effective noise on the model parameters.

**Typical improvement**: 10-50% utility gain at the same privacy budget.

## Key Concepts

### The Factorization A = B C

The mechanism factors a **workload matrix** A (what the optimizer computes, e.g., prefix sums) into:

- **C** (strategy matrix): Encodes the privacy mechanism
- **C^{-1}** (noising matrix): Used to generate correlated noise at each step
- **B = A C^{-1}** (decoder matrix): Relates noisy outputs back to workload queries

The **sensitivity** of C determines how much noise is needed, while the **error** of B determines the effective noise on the output.

### Strategy Types

| Strategy | Memory | Best For |
|----------|--------|----------|
| **BandMF** (Banded Toeplitz) | O(bands) | General purpose, good default |
| **BLT** (Buffered Linear Toeplitz) | O(num_buffers) | Large n, state-of-the-art |
| **Dense** | O(n^2) | Small n, exact solutions |

## Quick Start

All MF constructors follow the same `(noise_fn, state)` pattern as `gaussian_noise`:

```python
import torch
from opaque.noise import band_mf_noise

# A gradient template (zeros with the right shapes)
grad_template = {name: torch.zeros_like(p) for name, p in model.named_parameters()}

# Create noise function — grad_template is always the first argument
noise_fn, noise_state = band_mf_noise(
    grad_template,
    n=1000,
    bands=4,
    stddev=noise_multiplier * clip_norm,
    generator=42,
)

# Training loop
for batch in dataloader:
    clipped_grad = compute_clipped_grad(model, batch)

    # Add correlated noise
    noisy_grad, noise_state = noise_fn(clipped_grad, noise_state)

    # Assign noisy gradients and step
    for (name, p), g in zip(model.named_parameters(), noisy_grad.values()):
        p.grad = g.to(p.dtype)
    optimizer.step()
```

## BandMF: Banded Toeplitz Strategies

BandMF uses banded lower-triangular Toeplitz matrices. The `bands` parameter controls the trade-off between memory and utility:

```python
from opaque.noise import band_mf_noise

noise_fn, state = band_mf_noise(
    grad_template,
    n=1000,
    bands=4,
    stddev=noise_multiplier * clip_norm,
)
```

- `bands=1`: Equivalent to DP-SGD (no correlation)
- `bands=4`: Good default, significant improvement over DP-SGD
- `bands=n`: Optimal but uses O(n) memory

## BLT: Buffered Linear Toeplitz

BLT matrices provide state-of-the-art utility with O(num_buffers) memory:

```python
from opaque.noise import blt_mf_noise

noise_fn, state = blt_mf_noise(
    grad_template,
    n=10000,
    stddev=noise_multiplier * clip_norm,
    min_buffers=1,
    max_buffers=5,
)
```

## Dense Factorization

For small n, you can use a dense (optimal) factorization:

```python
from opaque.noise import dense_mf_noise

noise_fn, state = dense_mf_noise(
    grad_template,
    n=100,
    stddev=noise_multiplier * clip_norm,
)
```

## Identity (DP-SGD Equivalent)

For comparison, `identity_mf_noise` uses the identity matrix (equivalent to standard DP-SGD):

```python
from opaque.noise import identity_mf_noise

noise_fn, state = identity_mf_noise(
    grad_template,
    stddev=noise_multiplier * clip_norm,
)
```

## Custom Matrix

Bring your own noising matrix with `custom_mf_noise`:

```python
from opaque.noise import custom_mf_noise
from opaque.noise.matrix_factorization.toeplitz import (
    inverse_as_streaming_matrix,
    optimal_max_error_strategy_coefs,
)

coefs = optimal_max_error_strategy_coefs(1000)
noising = inverse_as_streaming_matrix(coefs)

noise_fn, state = custom_mf_noise(
    grad_template,
    noising,
    stddev=noise_multiplier * clip_norm,
    generator=42,
)
```

## Multi-Participation (Multi-Epoch)

When training for multiple epochs, each example participates multiple times. The sensitivity computation must account for this:

```python
from opaque.noise import blt_mf_noise

# min_sep = epoch_length / batch_size (minimum steps between participations)
noise_fn, state = blt_mf_noise(
    grad_template,
    n=5000,
    stddev=noise_multiplier * clip_norm,
    min_sep=100,
    max_participations=5,  # 5 epochs
)
```

## Sensitivity Computation

The sensitivity of the strategy matrix C determines the noise calibration:

```python
from opaque.noise.matrix_factorization.sensitivity import (
    single_participation_sensitivity,
    get_sensitivity_banded,
)

# Single-participation sensitivity
sens = single_participation_sensitivity(C_matrix)

# For banded matrices with min-sep
sens = get_sensitivity_banded(C_matrix, min_sep=100, max_participations=5)
```

## Comparison: DP-SGD vs BandMF vs BLT

For a linear regression with n=1000 steps, epsilon=1.0:

| Method | Mechanism | Final MSE | Memory |
|--------|-----------|-----------|--------|
| DP-SGD | Independent noise | ~2.5 | O(1) |
| BandMF (bands=4) | Banded Toeplitz | ~1.8 | O(4) |
| BLT (3 buffers) | Buffered Toeplitz | ~1.5 | O(3) |

*Values are illustrative; actual results depend on problem specifics.*

## References

- **BandMF**: [Choquette-Choo et al., 2023](https://arxiv.org/abs/2306.08153)
- **BLT**: [McMahan et al., 2024](https://arxiv.org/abs/2404.16706)
- **Multi-epoch BLT**: [Choquette-Choo et al., 2024](https://arxiv.org/abs/2408.08868)
- **Inversion theorem**: [McMahan et al., 2025](https://arxiv.org/abs/2504.21413)
- **DP-FTRL**: [Kairouz et al., 2021](https://arxiv.org/abs/2103.00039)

## See Also

- **[Noise Addition](noise.md)**: Standard Gaussian noise
- **[Gradient Clipping](clipping.md)**: Per-example clipping
- **[Sampling](sampling.md)**: Cyclic Poisson sampling for BandMF
