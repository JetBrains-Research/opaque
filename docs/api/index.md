# API Reference

Opaque provides a functional API for differential privacy in PyTorch. This reference documents all public functions and
classes.

## Module Organization

Opaque is organized into several modules, each focused on a specific aspect of DP training:

### Core Utilities

- **[Random Number Generation](rng.md)**: Immutable RNG keys for deterministic DP
  - `RngKey` - Immutable key type
  - `key()`, `random_key()` - Create keys
  - `split()`, `fold_in()` - Manipulate keys
  - `set_reproducible_pytorch_seed()` - PyTorch/CUDNN reproducibility
  - `generator_from_key()` - PyTorch generator bridge

- **[Utilities](utilities.md)**: Functional and PyTree utilities
  - `make_functional()` - Convert nn.Module to functional form
  - `tree_map()`, `tree_map_with_path()`, `partition()`, `merge()`, `global_norm()`, `tree_leaves()`

### DP-SGD Components

- **[Clipping](clipping.md)**: Per-sample gradient clipping
  - `clipped_grad()` - High-level gradient clipping
  - `clipped_fun()` - Clip and sum function outputs
  - `clip_pytree()` - Low-level PyTree clipping
  - `adaptive_clipped_grad()` - Clipped gradients with auto-tuned clip norm

- **[Noise](noise.md)**: Noise injection for DP
  - `gaussian_noise()` - Standard Gaussian noise
  - `bounded_gaussian_noise()` - Bounded Gaussian noise (truncated normal)
  - `band_mf_noise()`, `blt_mf_noise()`, `dense_mf_noise()` - Correlated noise (DP-FTRL)
  - `identity_mf_noise()`, `custom_mf_noise()` - MF API utilities

- **[Accounting](accounting.md)**: Privacy budget tracking
  - `gaussian()`, `poisson()`, `truncated_poisson()` - Mechanism constructors → typed subclasses
  - `DpProcess` operators: `*` (repeat), `|` (compose)
  - `.epsilon_at()`, `.delta_at()`, `.advantage()`, `.beta_at()` - Privacy metrics
  - `calibrate()` - Binary-search noise multiplier for target privacy

- **[Sampling](sampling.md)**: Privacy-amplifying sampling
  - `PoissonSampler` - Standard Poisson sampling
  - `TruncatedPoissonSampler` - Bounded Poisson sampling
  - `CyclicPoissonSampler` - Cyclic Poisson sampling (BandMF)

### Validation & Debugging

- **[Auditing](auditing.md)**: Empirical privacy validation
  - `epsilon_clopper_pearson()`, `epsilon_one_run()` - Estimate epsilon from attacks
  - `audit()` - Comprehensive privacy audit
  - `auc()`, `beta_at()` - Attack utility metrics

- **[Distributed](distributed.md)**: Multi-GPU training with DDP
  - `sum_gradients()` - Sum clipped gradients across GPUs (for DP training)
  - `reduce_pytree()` - Generic PyTree reduction
  - `sync()` - Auto-dispatch sync for any state/aux type
  - `sync_object()` - Synchronize scalar fields of a dataclass
  - `is_distributed()`, `get_rank()`, `get_world_size()` - Distributed utilities

## Quick Reference

### Typical DP-SGD Workflow

```python
import opaque.accounting as acc
from opaque import clipped_grad, gaussian_noise
from opaque.random import key

# Calibrate noise multiplier
result = acc.calibrate(
    acc.epsilon_budget(3.0, delta=1e-5),
    lambda nm: acc.poisson(acc.gaussian(nm), sample_rate=0.01) * 1000,
    param_min=0.1, param_max=5.0,
)

# Set up DP components
grad_fn, clip_state = clipped_grad(loss_fn, l2_clip_norm=1.0, batch_argnums=1)
noise_fn, noise_state = gaussian_noise(
    stddev=result.param * clip_state.sensitivity(), key=key(42),
)

# Training loop
for batch in dataloader:
    grads, clip_state = grad_fn(params, batch, state=clip_state)
    noisy_grads, noise_state = noise_fn(grads, noise_state)
    params = update(params, noisy_grads)
```

See [Quick Start](../getting-started/quickstart.md) for a complete working example.

## Function Index

### Clipping

| Function         | Purpose                             | User Guide                                                        |
|------------------|-------------------------------------|-------------------------------------------------------------------|
| `clipped_grad()` | Differentiate loss with DP clipping | [Guide](../user-guide/clipping.md#clipped_grad-recommended-api) |
| `clipped_fun()`  | Clip and sum function outputs       | [Guide](../user-guide/clipping.md#clipped_fun-general-purpose-clipping) |
| `clip_pytree()`  | Clip PyTree to max norm             | [Guide](../user-guide/clipping.md#clip_pytree-low-level-clipping) |

### Noise

| Function                       | Purpose                                      | User Guide                      |
|--------------------------------|----------------------------------------------|---------------------------------|
| `gaussian_noise()`                   | Standard Gaussian noise (unbounded)          | [Guide](../user-guide/noise.md) |
| `bounded_gaussian_noise()`           | Bounded Gaussian noise (truncated normal)    | [Guide](../user-guide/noise.md#bounded-gaussian-noise) |
| `band_mf_noise()`                   | BandMF correlated noise (DP-FTRL)           | [Guide](../user-guide/noise.md) |
| `blt_mf_noise()`                    | BLT correlated noise (DP-FTRL)              | [Guide](../user-guide/noise.md) |
| `dense_mf_noise()`                  | Dense optimal correlated noise               | [Guide](../user-guide/noise.md) |
| `identity_mf_noise()`               | Identity noise via MF API                    | [Guide](../user-guide/noise.md) |
| `custom_mf_noise()`                 | Bring-your-own noising matrix                | [Guide](../user-guide/noise.md) |

### Accounting (Mechanisms)

| Function                  | Purpose                           | User Guide                                                              |
|---------------------------|-----------------------------------|-------------------------------------------------------------------------|
| `gaussian()`              | Gaussian mechanism                | [Guide](../user-guide/accounting.md#mechanisms)                         |
| `poisson()`               | Poisson-subsampled Gaussian       | [Guide](../user-guide/accounting.md#mechanisms)                         |
| `truncated_poisson()`     | Truncated Poisson Gaussian        | [Guide](../user-guide/accounting.md#mechanisms)                         |
| `parallel_poisson()`      | Parallel Poisson subsampling      | [Guide](../user-guide/accounting.md#mechanisms)                         |
| `adaclip()`               | Adaptive clipping mechanism       | [Guide](../user-guide/accounting.md#mechanisms)                         |
| `eps_delta()`             | Fixed (ε, δ) guarantee            | [Guide](../user-guide/accounting.md#mechanisms)                         |
| `identity()`              | Zero privacy loss                 | [Guide](../user-guide/accounting.md#mechanisms)                         |

### Accounting (Composition & Metrics)

| Method / Operator         | Purpose                           | User Guide                                                              |
|---------------------------|-----------------------------------|-------------------------------------------------------------------------|
| `process * k`             | Repeat k times                    | [Guide](../user-guide/accounting.md#core-concepts)                      |
| `a \| b`                  | Heterogeneous composition         | [Guide](../user-guide/accounting.md#core-concepts)                      |
| `.epsilon_at(delta)`      | Query (ε, δ)-DP                   | [Guide](../user-guide/accounting.md#privacy-metrics)                    |
| `.delta_at(epsilon)`      | Query δ for given ε               | [Guide](../user-guide/accounting.md#privacy-metrics)                    |
| `.advantage()`            | Query f-DP advantage              | [Guide](../user-guide/accounting.md#privacy-metrics)                    |
| `.beta_at(alpha)`         | Query (α, β) error rates          | [Guide](../user-guide/accounting.md#privacy-metrics)                    |

### Calibration

| Function                  | Purpose                           | User Guide                                                              |
|---------------------------|-----------------------------------|-------------------------------------------------------------------------|
| `calibrate()`             | Find noise for target privacy     | [Guide](../user-guide/accounting.md#calibration)                        |
| `epsilon_budget()`        | Budget for (ε, δ) calibration     | [Guide](../user-guide/accounting.md#calibration)                        |
| `delta_budget()`          | Budget for δ calibration           | [Guide](../user-guide/accounting.md#calibration)                        |
| `advantage_budget()`      | Budget for advantage calibration  | [Guide](../user-guide/accounting.md#calibration)                        |
| `beta_budget()`           | Budget for (α, β) calibration     | [Guide](../user-guide/accounting.md#calibration)                        |
| `risk_budget()`           | Budget for Bayes risk calibration | [Guide](../user-guide/accounting.md#calibration)                        |

### Sampling

| Class                     | Purpose                    | User Guide                                                    |
|---------------------------|----------------------------|---------------------------------------------------------------|
| `PoissonSampler`          | Standard Poisson sampling  | [Guide](../user-guide/sampling.md#poisson-sampling) |
| `TruncatedPoissonSampler` | Truncated Poisson sampling | [Guide](../user-guide/sampling.md#poisson-sampling) |
| `CyclicPoissonSampler`    | Cyclic Poisson sampling (BandMF) | [Guide](../user-guide/sampling.md#poisson-sampling) |

### Privacy Auditing

| Function / Method                    | Purpose                         | User Guide                                     |
|--------------------------------------|---------------------------------|------------------------------------------------|
| `auditing.setup()`                   | Prepare canary experiment       | [Guide](../user-guide/auditing.md)             |
| `auditing.evaluate()`               | Score canaries and compute audit | [Guide](../user-guide/auditing.md)             |
| `AuditResult.epsilon_at()`          | Epsilon bound (auto-selects method) | [Guide](../user-guide/auditing.md)          |
| `AuditResult.epsilon_clopper_pearson()` | Conservative epsilon bound  | [Guide](../user-guide/auditing.md)             |
| `AuditResult.epsilon_one_run()`     | Tighter bound (Nasr et al.)     | [Guide](../user-guide/auditing.md)             |
| `AuditResult.auc()`                 | Membership inference AUC        | [Guide](../user-guide/auditing.md)             |
| `AuditResult.beta_at()`             | Type-II error at given alpha    | [Guide](../user-guide/auditing.md)             |

### Distributed

| Function               | Purpose                     | User Guide                                 |
|------------------------|-----------------------------|--------------------------------------------|
| `sum_gradients()`      | Sum clipped gradients (DP-specific) | [Guide](../user-guide/distributed.md) |
| `reduce_pytree()`      | Generic PyTree reduction    | [Guide](../user-guide/distributed.md)      |
| `reduce_scalar()`      | Reduce scalar across devices | [Guide](../user-guide/distributed.md)      |
| `gather_tensors()`     | Gather tensors from all ranks | [Guide](../user-guide/distributed.md)      |
| `sync()`               | Auto-dispatch sync for any state/aux | [Guide](../user-guide/distributed.md) |
| `sync_object()`        | Sync scalar fields of a dataclass | [Guide](../user-guide/distributed.md) |
| `is_distributed()`     | Check if DDP is active      | [Guide](../user-guide/distributed.md)      |
| `get_rank()`           | Get current GPU index       | [Guide](../user-guide/distributed.md)      |
| `get_world_size()`     | Get total number of GPUs    | [Guide](../user-guide/distributed.md)      |

## Type Hints

Opaque uses type hints throughout. Key types:

```python
import opaque.accounting as acc

# PyTree: Nested structure of tensors
PyTree = dict[str, torch.Tensor] | tuple[torch.Tensor, ...]

# DpProcess: Composable privacy process (from Rust PLD engine)
process: DpProcess = acc.poisson(acc.gaussian(1.1), 0.01) * 1000

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
- **[Quick Start](../getting-started/quickstart.md)**: End-to-end DP-SGD example

---

**Browse by module**: Use the navigation on the left to explore detailed API documentation
