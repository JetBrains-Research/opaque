# opaque-accounting

PLD computation engine for differential privacy accounting, implemented in Rust with Python bindings via PyO3.

## Overview

`opaque_accounting` is the numerical backend for `opaque.accounting`. It provides
flat functions that take scalar parameters and return opaque `Pld` handles.
Python owns composition, repetition, caching, and calibration.

The engine uses the Privacy Loss Distribution (PLD) framework with
Connect-the-Dots discretization ([Doroshenko et al., 2022](https://arxiv.org/abs/2207.04380)).
It computes (epsilon, delta)-DP, f-DP advantage, (alpha, beta) error rates,
and Bayes risk — all from a single PLD computation.

### Why PLD?

| Method | Bound Quality | Composition | Speed |
|--------|--------------|-------------|-------|
| Basic composition | Loose | O(k) additive | Fast |
| **PLD (this crate)** | **Tight** | **FFT convolution** | **O(n log n)** |

PLD gives the tightest known bounds for composed mechanisms via
FFT-based convolution in O(n log n) time.

## Building

### As part of Opaque (recommended)

```bash
uv run maturin develop --release -m crates/dp-accounting/Cargo.toml
```

### Standalone Rust

```toml
[dependencies]
opaque-accounting = { path = "crates/dp-accounting" }
```

## Quick Start

### Python (via `opaque.accounting`)

Users interact through the Python wrapper, not the native module directly:

```python
import opaque.accounting as acc

# Single Gaussian mechanism
step = acc.gaussian(0.5)
print(step.epsilon_at(1e-5))

# DP-SGD: Poisson-subsampled Gaussian, 1000 steps
training = acc.poisson(acc.gaussian(0.5), sample_rate=0.01) * 1000
eps = training.epsilon_at(delta=1e-5)
print(f"DP-SGD: epsilon={eps:.2f}")

# Heterogeneous composition (warmup + training)
warmup = acc.poisson(acc.gaussian(0.15), 0.001) * 100
training = acc.poisson(acc.gaussian(0.25), 0.001) * 400
total = warmup | training
print(total.epsilon_at(1e-5))
```

### Rust

```rust
use opaque_accounting::mechanisms::gaussian_pld;
use opaque_accounting::amplification::poisson_gaussian_pld;
use opaque_accounting::DiscretizationConfig;

let config = DiscretizationConfig::default();
let pld = gaussian_pld(0.5, &config)?;
let eps = pld.epsilon_at(1e-5);
println!("epsilon = {eps:.4}");

// Poisson-subsampled, self-composed 1000 times
let step = poisson_gaussian_pld(0.5, 0.01, &config)?;
let training = step.self_compose(1000);
println!("epsilon = {:.4}", training.epsilon_at(1e-5));
```

## Architecture

Rust is a PLD computation engine: flat functions that take scalar
parameters and return opaque `Pld` handles.

```
opaque_accounting (Rust crate)
├── mechanisms/       # gaussian_pld, eps_delta_pld, identity_pld
├── amplification/    # poisson_gaussian_pld, truncated_poisson_gaussian_pld, parallel_poisson_gaussian_pld
├── transformations/  # adaclip_sensitivity (adaptive clipping)
├── pld/              # PrivacyLossDistribution: metrics, compose, self_compose
├── discretization/   # Connect-the-Dots algorithm, DiscretizationConfig
├── numerics/         # FFT, log-space arithmetic, special functions
├── adjacency.rs      # Add/remove adjacency relation
├── error.rs          # Error types
└── python/           # PyO3 bindings (flat module, no sub-modules)
```

### Pld type

All functions return `PrivacyLossDistribution` (exposed as `Pld` in Python):

| Method | Returns | Description |
|--------|---------|-------------|
| `epsilon_at(delta)` | float | Smallest epsilon for (epsilon, delta)-DP |
| `delta_at(epsilon)` | float | Smallest delta for (epsilon, delta)-DP |
| `advantage()` | float | f-DP advantage (= delta_at(0)) |
| `beta_at(alpha)` | float | Type-II error at given Type-I error |
| `risk_at(prior)` | float | Bayes risk under optimal adversary |
| `compose(other)` | Pld | Heterogeneous composition (FFT convolution) |
| `self_compose(count)` | Pld | Homogeneous k-fold composition |

Operators: `pld * k` (self_compose), `a \| b` (compose).

### DiscretizationConfig

Fine-tune numerical precision when defaults are insufficient:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `discretization` | 1e-4 | Grid spacing for PLD discretization |
| `log_mass_truncation_bound` | -50.0 | Log tail truncation threshold |
| `pessimistic_estimate` | True | Upper-bound rounding (safe for privacy) |
| `max_grid_size` | 10,000,000 | Auto-coarsen if grid exceeds this |

## Validation

```bash
# Rust unit tests
cargo test

# Python cross-validation (requires dp-accounting)
pytest tests/ -v
```

## References

- Doroshenko et al., ["Connect the Dots: Tighter Discrete Approximations of Privacy Loss Distributions"](https://arxiv.org/abs/2207.04380), PoPETs 2022
- Abadi et al., ["Deep Learning with Differential Privacy"](https://arxiv.org/abs/1607.00133), CCS 2016
- Koskela et al., ["Tight Differential Privacy for Discrete-Valued Mechanisms"](https://arxiv.org/abs/2106.08567), AISTATS 2021
- Andrew et al., ["Differentially Private Learning with Adaptive Clipping"](https://arxiv.org/abs/2106.07136), NeurIPS 2021
- Dong et al., ["Gaussian Differential Privacy"](https://arxiv.org/abs/1905.02383), JRSS-B 2022

## License

Apache-2.0
