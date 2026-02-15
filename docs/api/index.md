# API Reference

Opaque provides a functional API for differential privacy in PyTorch. This reference documents all public functions and
classes.

## Module Organization

Opaque is organized into several modules, each focused on a specific aspect of DP training:

### Core Utilities

- **[PyTree Utilities](core/pytree_utils.md)**: Operations on PyTrees (nested structures of tensors)
  - `tree_map()`, `global_norm()`, `tree_leaves()`

- **[Functional Utilities](functional_utils.md)**: Utility functions for functional programming
  - `make_functional()` - Convert nn.Module to functional form

### DP-SGD Components

- **[Clipping](core/clipping.md)**: Per-sample gradient clipping ⭐
  - `clipped_grad()` - High-level gradient clipping (recommended)
  - `clipped_fun()` - Clip and sum function outputs
  - `clip_pytree()` - Low-level PyTree clipping

- **[Noise](noise.md)**: Noise injection for DP
  - `gaussian()` / `gaussian_stateful()` - Standard Gaussian noise
  - `bounded_gaussian()` / `bounded_gaussian_stateful()` - Bounded Gaussian noise (truncated normal)

- **[Accounting](accounting.md)**: Privacy budget tracking
  - `create()` - Initialize privacy state
  - `compose_poisson_gaussian()`, `compose_truncated_poisson_gaussian()` - Compose privacy
  - `get_epsilon()`, `get_beta()`, `get_advantage()` - Query privacy
  - `find_noise_multiplier_for_epsilon_delta()` - Calibrate noise

- **[Optimizers](optimizers.md)**: DP-aware optimizers
  - `adaptive_clipping()` - Adaptive clipping wrapper for TorchOpt

- **[Sampling](sampling.md)**: Privacy-amplifying sampling
  - `PoissonSampler` - Standard Poisson sampling
  - `TruncatedPoissonSampler` - Bounded Poisson sampling (recommended)

## Quick Reference

### Typical DP-SGD Workflow

```python
import torch
import opaque.accounting as acc
from opaque import clipped_grad, add_gaussian_noise

# 1. Calibrate noise
noise_multiplier = acc.find_noise_multiplier_for_epsilon_delta(
    epsilon=3.0, delta=1e-5, sample_rate=0.01, num_steps=1000
)

# 2. Create clipped gradient function
dp_grad_fn = clipped_grad(
    loss_fn, l2_clip_norm=1.0, argnums=0, batch_argnums=1
)

# 3. Training loop
privacy_state = acc.create()

for step in range(1000):
    grads = dp_grad_fn(params, batch)
    noisy_grads = add_gaussian_noise(grads, stddev=noise_multiplier)
    params = update(params, noisy_grads)
    privacy_state = acc.compose_poisson_gaussian(
        privacy_state, noise_multiplier, sample_rate=0.01, count=1
    )

# 4. Check final privacy
epsilon = acc.get_epsilon(privacy_state, delta=1e-5)
```

## Function Index

### Clipping

| Function         | Purpose                             | User Guide                                                        |
|------------------|-------------------------------------|-------------------------------------------------------------------|
| `clipped_grad()` | Differentiate loss with DP clipping | [Guide](../user-guide/clipping.md#clipped_grad-high-level-api)    |
| `clipped_fun()`  | Clip and sum function outputs       | [Guide](../user-guide/clipping.md#clipped_fun-primary-api)        |
| `clip_pytree()`  | Clip PyTree to max norm             | [Guide](../user-guide/clipping.md#clip_pytree-low-level-clipping) |

### Noise

| Function                       | Purpose                                      | User Guide                      |
|--------------------------------|----------------------------------------------|---------------------------------|
| `gaussian()`                   | Standard Gaussian noise (unbounded)          | [Guide](../user-guide/noise.md) |
| `gaussian_stateful()`          | Standard Gaussian with reproducible state    | [Guide](../user-guide/noise.md) |
| `bounded_gaussian()`           | Bounded Gaussian noise (truncated normal)    | [Guide](../user-guide/noise.md#bounded-gaussian-noise) |
| `bounded_gaussian_stateful()`  | Bounded Gaussian with reproducible state     | [Guide](../user-guide/noise.md#bounded-gaussian-noise) |

### Accounting (Composition)

| Function                               | Purpose                     | User Guide                                                              |
|----------------------------------------|-----------------------------|-------------------------------------------------------------------------|
| `create()`                             | Initialize privacy state    | [Guide](../user-guide/accounting.md#the-functional-accounting-api)      |
| `compose_poisson_gaussian()`           | Compose Poisson sampling    | [Guide](../user-guide/accounting.md#compose_poisson_gaussian)           |
| `compose_truncated_poisson_gaussian()` | Compose truncated Poisson   | [Guide](../user-guide/accounting.md#compose_truncated_poisson_gaussian) |
| `compose_sampled_gaussian()`           | Compose fixed-size sampling | [Guide](../user-guide/accounting.md#compose_sampled_gaussian)           |
| `compose_gaussian()`                   | Compose without sampling    | [Guide](../user-guide/accounting.md#compose_gaussian)                   |

### Accounting (Queries)

| Function          | Purpose                  | User Guide                                                  |
|-------------------|--------------------------|-------------------------------------------------------------|
| `get_epsilon()`   | Query (ε, δ)-DP          | [Guide](../user-guide/accounting.md#1-differential-privacy) |
| `get_beta()`      | Query (α, β) error rates | [Guide](../user-guide/accounting.md#3-error-rates)          |
| `get_advantage()` | Query f-DP advantage     | [Guide](../user-guide/accounting.md#2-f-dp-advantage)       |

### Accounting (Calibration)

| Function                                    | Purpose                  | User Guide                                                     |
|---------------------------------------------|--------------------------|----------------------------------------------------------------|
| `find_noise_multiplier_for_epsilon_delta()` | Find noise for (ε, δ)    | [Guide](../user-guide/accounting.md#calibrate-for)             |
| `find_noise_multiplier_for_advantage()`     | Find noise for advantage | [Guide](../user-guide/accounting.md#calibrate-for-advantage)   |
| `find_noise_multiplier_for_err_rates()`     | Find noise for (α, β)    | [Guide](../user-guide/accounting.md#calibrate-for-error-rates) |

### Optimizers

| Function              | Purpose                               | User Guide                           |
|-----------------------|---------------------------------------|--------------------------------------|
| `adaptive_clipping()` | Wrap optimizer with adaptive clipping | [Guide](../user-guide/optimizers.md) |

### Sampling

| Class                     | Purpose                    | User Guide                                                    |
|---------------------------|----------------------------|---------------------------------------------------------------|
| `PoissonSampler`          | Standard Poisson sampling  | [Guide](../user-guide/sampling.md#standard-poisson-sampling)  |
| `TruncatedPoissonSampler` | Truncated Poisson sampling | [Guide](../user-guide/sampling.md#truncated-poisson-sampling) |

## Type Hints

Opaque uses type hints throughout. Key types:

```python
# PyTree: Nested structure of tensors
PyTree = dict[str, torch.Tensor] | tuple[torch.Tensor, ...]

# Privacy state (immutable)
PrivacyState = dp_accounting.pld.PrivacyLoss

# Generator for reproducible noise
Generator = torch.Generator | None
```

## Design Philosophy

Opaque's API follows these principles:

1. **Functional-first**: Immutable state, pure functions
2. **Composable**: Small, focused functions that combine naturally
3. **Type-safe**: Comprehensive type hints
4. **PyTorch-native**: Built on `torch.func`, works with standard PyTorch
5. **JAX-inspired**: API closely mirrors JAX-Privacy for familiarity

## See Also

- **[User Guides](../user-guide/index.md)**: Conceptual explanations and examples
- **[Tutorials](../tutorials/README.md)**: Interactive Jupyter notebooks
- **[Quick Start](../getting-started/quickstart.md)**: Get started in 5 minutes

---

**Browse by module**: Use the navigation on the left to explore detailed API documentation
