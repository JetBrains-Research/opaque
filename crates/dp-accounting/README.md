# opaque-dp-accounting

High-performance differential privacy accounting using Privacy Loss Distributions (PLD), implemented in Rust with Python bindings via PyO3.

## Overview

`opaque_dp_accounting` provides tight privacy accounting for DP-SGD and related mechanisms using the PLD framework with Connect-the-Dots discretization ([Doroshenko et al., 2022](https://arxiv.org/abs/2207.04380)). It computes (epsilon, delta)-DP, f-DP advantage, (alpha, beta) error rates, and Bayes risk — all from a single PLD computation.

### Why PLD?

| Method | Bound Quality | Composition | Speed |
|--------|--------------|-------------|-------|
| Basic composition | Loose | O(k) additive | Fast |
| Renyi DP (RDP) | Good | Moment-based | Fast |
| **PLD (this crate)** | **Tight** | **FFT convolution** | **O(n log n)** |

PLD gives the **tightest known bounds** for composed mechanisms via FFT-based convolution in O(n log n) time.

## Installation

### Python (via maturin)

```bash
cd crates/dp-accounting
python -m venv .venv && source .venv/bin/activate
pip install maturin dp-accounting  # dp-accounting for cross-validation
maturin develop --release
```

### Rust

```toml
[dependencies]
opaque-dp-accounting = { path = "crates/dp-accounting" }
```

## Quick Start

### Python

```python
import opaque_dp_accounting as dp

# --- Single mechanism ---
proc = dp.gaussian(1.1)
print(proc.epsilon_at(1e-5))   # epsilon for this noise level
print(proc.delta_at(1.0))      # delta at epsilon=1.0

# --- Standard DP-SGD (Poisson-subsampled Gaussian, 1000 steps) ---
step = dp.poisson(noise_multiplier=1.1, sample_rate=0.01)
training = step * 1000                     # repeat 1000 times
eps = training.epsilon_at(delta=1e-5)      # query epsilon
print(f"DP-SGD: epsilon={eps:.2f}")

# --- One-liner ---
eps = dp.compute_epsilon(1.1, 0.01, 1000, delta=1e-5)

# --- Calibrate noise for target privacy ---
nm = dp.calibrate_noise(
    target_epsilon=8.0,
    target_delta=1e-5,
    sample_rate=0.01,
    num_steps=1000,
)
print(f"Use noise_multiplier={nm:.4f}")
```

### Rust

```rust
use opaque_dp_accounting::functional::*;

// Poisson-subsampled Gaussian, 1000 steps
let step = poisson(gaussian(1.1)?, 0.01);
let training = repeat(step, 1000)?;
let epsilon = training.epsilon_at(1e-5)?;
println!("epsilon = {epsilon:.4}");
```

## Python API Reference

### Mechanisms

| Function | Description |
|----------|-------------|
| `gaussian(noise_multiplier, config=None)` | Gaussian mechanism (sensitivity=1) |
| `poisson(noise_multiplier, sample_rate, config=None)` | Poisson-subsampled Gaussian |
| `truncated_poisson(noise_multiplier, sample_rate, batch_size_cap, dataset_size, config=None)` | Truncated Poisson (capped batch size) |
| `accumulate(noise_multiplier, sample_rate, microbatches, config=None)` | Gradient accumulation (mixture of Gaussians) |
| `eps_delta(epsilon, delta=0, config=None)` | Fixed (epsilon, delta)-DP mechanism |
| `identity(config=None)` | Zero privacy loss |
| `adaclip(noise_multiplier, quantile_noise_std, config=None)` | Adaptive clipping ([Andrew et al. 2021](https://arxiv.org/abs/2106.07136)) |
| `poisson_adaclip(noise_multiplier, quantile_noise_std, sample_rate, config=None)` | Poisson + AdaClip combined |

### Composition

| Function / Operator | Description |
|---------------------|-------------|
| `process * k` | Repeat a process k times (homogeneous composition) |
| `a \| b` | Compose two processes (heterogeneous composition) |
| `repeat(process, count)` | Functional form of `process * count` |
| `compose(left, right)` | Functional form of `left \| right` |

### Privacy Metrics

All metrics are computed from the same PLD — no redundant computation.

| Method | Returns | Description |
|--------|---------|-------------|
| `proc.epsilon_at(delta)` | float | Smallest epsilon for (epsilon, delta)-DP |
| `proc.delta_at(epsilon)` | float | Smallest delta for (epsilon, delta)-DP |
| `proc.advantage()` | float | f-DP advantage (= delta_at(0)) |
| `proc.beta_at(alpha)` | float | Type-II error at given Type-I error |
| `proc.risk_at(prior)` | float | Bayes risk under optimal adversary |

### Convenience

| Function | Description |
|----------|-------------|
| `compute_epsilon(noise_multiplier, sample_rate, num_steps, delta)` | One-liner for DP-SGD epsilon |
| `calibrate_noise(target_epsilon, target_delta, sample_rate, num_steps, ...)` | Find noise for target privacy |

### Debugging & Introspection

```python
proc = dp.poisson(1.1, 0.01) * 1000

# Quick display
print(proc)
# Poisson(noise_multiplier=1.1, sample_rate=0.01) | eps(delta=1e-5)=3.73

# Constructor parameters
proc.describe()
# {'type': 'Repeat(...)', 'inner': 'Poisson(...)', 'count': 1000}

# PLD grid diagnostics
info = proc.pld_info()
# {'grid_size': 84001, 'discretization': 0.0001, 'infinity_mass': 1.2e-10,
#  'total_mass': 1.0, 'elapsed_ms': 42.3, ...}

# Full privacy summary
print(proc.summary(delta=1e-5))
# --- Repeat(Poisson(...), k=1000) ---
# epsilon(delta=1e-5)  = 3.73
# delta(epsilon=1)     = 2.1e-02
# advantage            = 4.5e-01
# beta(alpha=0.05)     = 0.12
# risk(prior=0.5)      = 0.38
# ---
# PLD grid: 84001 bins, disc=0.0001, inf_mass=1.2e-10
# PLD computed in 42.3 ms
```

### DiscretizationConfig

Fine-tune numerical precision when defaults are insufficient:

```python
cfg = dp.DiscretizationConfig(
    discretization=1e-3,        # coarser grid (faster, less precise)
    log_mass_truncation_bound=-50.0,  # wider tails
    pessimistic_estimate=False,  # optimistic (lower-bound) rounding
    max_grid_size=1_000_000,    # limit memory
)

proc = dp.gaussian(1.1, config=cfg)
```

| Parameter | Default | Description |
|-----------|---------|-------------|
| `discretization` | 1e-4 | Grid spacing for PLD discretization |
| `log_mass_truncation_bound` | -32.0 | log2 tail truncation threshold |
| `pessimistic_estimate` | True | Upper-bound rounding (safe for privacy) |
| `max_grid_size` | 10,000,000 | Auto-coarsen if grid exceeds this |

## Architecture

```
opaque_dp_accounting
├── functional/           # Core functional API
│   ├── mechanisms/       # Gaussian, EpsDelta, Identity
│   ├── amplification/    # Poisson, TruncatedPoisson, Accumulated
│   ├── transforms/       # AdaClip
│   ├── composition.rs    # compose(), repeat()
│   ├── calibrate.rs      # Binary search calibration
│   ├── discretization/   # Connect-the-Dots algorithm
│   ├── pld/              # Privacy Loss Distribution + PMF
│   └── process.rs        # Process trait (core abstraction)
├── math_helpers/         # FFT, log-space arithmetic, special functions
├── python/               # PyO3 bindings
└── error.rs              # Error types
```

### The `Process` Trait

Every privacy mechanism implements the `Process` trait:

```rust
pub trait Process {
    fn pld(&self) -> Result<PrivacyLossDistribution>;
}
```

This single method produces a PLD from which all privacy metrics are derived. Composition works by convolving PLDs via FFT — no approximation needed.

## Validation

The crate is validated at two levels:

- **371 Rust unit tests** covering mechanisms, composition, calibration, amplification, and edge cases
- **40 Python cross-validation tests** comparing against Google's `dp-accounting` library with conservative tolerances derived from Connect-the-Dots discretization error

Run the tests:

```bash
# Rust
cargo test

# Python (requires dp-accounting in venv)
source .venv/bin/activate
pytest tests/test_against_dp_accounting.py -v
```

## References

- Doroshenko et al., ["Connect the Dots: Tighter Discrete Approximations of Privacy Loss Distributions"](https://arxiv.org/abs/2207.04380), PoPETs 2022
- Abadi et al., ["Deep Learning with Differential Privacy"](https://arxiv.org/abs/1607.00133), CCS 2016
- Koskela et al., ["Tight Differential Privacy for Discrete-Valued Mechanisms"](https://arxiv.org/abs/2106.08567), AISTATS 2021
- Andrew et al., ["Differentially Private Learning with Adaptive Clipping"](https://arxiv.org/abs/2106.07136), NeurIPS 2021
- Dong et al., ["Gaussian Differential Privacy"](https://arxiv.org/abs/1905.02383), JRSS-B 2022

## License

Apache-2.0
