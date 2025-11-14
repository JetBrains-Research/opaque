# Opaque

**Functional Differential Privacy for PyTorch LoRA Fine-tuning**

Opaque is a PyTorch port of Google's [JAX-Privacy](https://github.com/google-deepmind/jax_privacy), adapted specifically for differentially private (DP) fine-tuning of Large Language Models (LLMs) using LoRA (Low-Rank Adaptation).

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/pytorch-2.0+-red.svg)](https://pytorch.org/)
[![CI](https://github.com/evgri243/opaque/actions/workflows/ci.yml/badge.svg)](https://github.com/evgri243/opaque/actions/workflows/ci.yml)
[![Docs](https://github.com/evgri243/opaque/actions/workflows/docs.yml/badge.svg)](https://github.com/evgri243/opaque/actions/workflows/docs.yml)
[![Code style: ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff)
[![Tests](https://img.shields.io/badge/tests-111%20passing-brightgreen.svg)](https://github.com/evgri243/opaque)
[![Stage](https://img.shields.io/badge/stage-2%20complete-success.svg)](https://github.com/evgri243/opaque)

---

## Project Vision

Bring production-quality differential privacy to PyTorch's LLM fine-tuning ecosystem with:

- **Functional API**: Composable DP primitives inspired by JAX-Privacy's modern design
- **LoRA-First**: Optimized for parameter-efficient fine-tuning
- **PyTorch Native**: Built on `torch.func` (functional transformations)
- **Zero Surprises**: Fail-fast error handling for security-critical DP training

**Status**: 🎉 Stage 1 & 2 Complete — DP-SGD ready with clipping, noise injection, and privacy accounting!

---

## Quick Start

### Installation

```bash
# Basic installation
pip install opaque-dp  # Not yet published

# From source (development)
git clone https://github.com/evgri243/opaque.git
cd opaque
uv sync
```

### Minimal Example: DP-SGD Training

```python
import torch
import torch.nn as nn
import opaque.accounting as acc
from opaque import (
    make_functional,
    clipped_grad,
    add_gaussian_noise,
)

# 1. Define model and convert to functional
model = nn.Linear(10, 1)
fmodel, params = make_functional(model)

# 2. Define per-example loss
def loss_fn(params, example):
    x, y = example
    pred = fmodel(params, x)
    return ((pred - y) ** 2).mean()

# 3. Calibrate noise for target privacy
sample_rate = 0.01  # batch_size / dataset_size
num_steps = 1000

noise_multiplier = acc.find_noise_multiplier_for_epsilon_delta(
  epsilon=3.0,
  delta=1e-5,
  sample_rate=sample_rate,
  num_steps=num_steps,
)

# 4. Create clipped gradient function
clip_norm = 1.0
clipped_grad_fn = clipped_grad(
    loss_fn,
    argnums=0,
    batch_argnums=1,
    l2_clip_norm=clip_norm,
)

# 5. Training loop
privacy_state = acc.create()

for step in range(num_steps):
    # Compute clipped gradients
    grads = clipped_grad_fn(params, (X_batch, y_batch))

    # Add calibrated noise
    noisy_grads = add_gaussian_noise(
        grads, stddev=noise_multiplier * clip_norm
    )

    # Update parameters
    params = tuple(p - lr * g for p, g in zip(params, noisy_grads))

    # Track privacy (compose one step)
    privacy_state = acc.compose_poisson_gaussian(
      privacy_state,
      noise_multiplier=noise_multiplier,
      sample_rate=sample_rate,
      count=1,
    )

# Get final privacy
epsilon = acc.get_epsilon(privacy_state, delta=1e-5)
print(f"Privacy: (ε={epsilon:.2f}, δ=1e-5)")  # Should be ≈ 3.0
```

**See**: [Tutorial 02](docs/tutorials/02_differential_privacy_noise_and_accounting.ipynb) for complete walkthrough

---

## Core Concepts

### Differential Privacy in a Nutshell

Differential privacy (DP) provides mathematical guarantees that a model doesn't memorize individual training examples. This is critical when training on sensitive data (medical records, private messages, etc.).

**DP-SGD Algorithm** ([Abadi et al. 2016](https://arxiv.org/abs/1607.00133)):
1. Compute per-example gradients (not batch average)
2. **Clip** each example's gradient to maximum L2 norm
3. Add **Gaussian noise** to the sum of clipped gradients
4. Update model parameters

**Privacy Budget**: Controlled by (ε, δ) parameters
- Smaller ε = stronger privacy (more noise) and may lower accuracy
- Typical ranges: ε ∈ [1, 10], δ ∈ [1e-6, 1e-5]

### Why LoRA?

[LoRA](https://arxiv.org/abs/2106.09685) (Low-Rank Adaptation) fine-tunes only a small fraction of model parameters:

- **Efficiency**: DP-SGD overhead is ~2x (instead of ~10x for full fine-tuning)
- **Memory**: Per-example gradients only for adapter weights (rank r << d)
- **Quality**: Often matches full fine-tuning performance

---

## Project Status & Roadmap

### Stage 0: Planning (Completed)
- [x] Analyze JAX-Privacy architecture
- [x] Design PyTorch port strategy
- [x] Define TDD-inspired workflow

### ✅ Stage 1: Core Clipping (Complete!)

**Timeline**: Completed 2025-11-11 | [Detailed Plan](docs/development/stage1-plan.md)

- [x] `opaque.utils.pytree` - PyTree operations with optree
- [x] `opaque.clipping` - Full clipping API:
  - `clip_pytree()` - Low-level PyTree clipping
  - `clipped_fun()` - Clip and sum function outputs (primary API)
  - `clipped_grad()` - High-level gradient clipping
  - `BoundedSensitivityCallable` - Wrapper with sensitivity tracking
- [x] Numerical validation against JAX-Privacy main (all tests pass within 1e-5)
- [x] Full API parity with JAX-Privacy main branch (single-device features)

**Deliverable**: ✅ Complete functional API matching JAX-Privacy

### ✅ Stage 2: Noise Injection & Privacy Accounting (Complete!)

**Timeline**: Completed 2025-11-14 | [Detailed Plan](docs/development/stage2-plan.md)

- [x] `opaque.noise` - Gaussian noise generation
  - `add_gaussian_noise()` - Stateless functional API
  - Reproducibility with `torch.Generator`
  - PyTree support
- [x] `opaque.accounting` - Functional privacy accounting
  - Immutable state API: `create()`, `compose_*()`, `get_*()`
  - Composition: `compose_poisson_gaussian()`, `compose_truncated_poisson_gaussian()`, etc.
  - Privacy queries: `get_epsilon()`, `get_beta()`, `get_advantage()`
  - Three privacy metrics: (ε, δ)-DP, f-DP advantage, (α, β) error rates
  - Calibration using riskcal: `find_noise_multiplier_for_epsilon_delta()`, etc.
- [x] `opaque.sampling` - Poisson sampling mechanisms
  - `PoissonSampler` - Standard Poisson sampling
  - `TruncatedPoissonSampler` - Bounded batch sizes
- [x] `opaque.optimizers` - DP optimizer wrappers
  - `adaptive_clipping()` - Adaptive clipping wrapper for TorchOpt optimizers
- [x] 111 tests passing (55 accounting + 56 optimizer tests)
- [x] Numerical equivalence with JAX-Privacy confirmed

**Deliverable**: ✅ Complete DP-SGD implementation with functional privacy accounting

### Stage 3: Integration & End-to-End (Next)

- [ ] End-to-end DP-SGD training examples
- [ ] Integration with PyTorch optimizers
- [ ] Memory-efficient microbatching
- [ ] Performance optimization

### Stage 4: High-Level API (Future)
- [ ] `opaque.api.make_private()` - One-line DP wrapper
- [ ] `DPConfig` - Configuration dataclass
- [ ] Integration with Hugging Face `peft` library
- [ ] Automatic LoRA detection

---

## Documentation

### Tutorials

- [Tutorial 01: Gradient Clipping from Basics](docs/tutorials/01_gradient_clipping_from_basics.ipynb) — Learn gradient
  clipping with `clipped_grad()`
- [Tutorial 02: Differential Privacy - Noise and Accounting](docs/tutorials/02_differential_privacy_noise_and_accounting.ipynb) —
  Complete DP-SGD with noise injection and privacy accounting

### Development Documentation

- [Roadmap](docs/development/roadmap.md) — Project timeline and milestones
- [Stage 1 Plan](docs/development/stage1-plan.md) — Gradient clipping implementation
- [Stage 2 Plan](docs/development/stage2-plan.md) — Noise injection and accounting implementation
- [Design Decisions](docs/development/design-decisions.md) — Technical choices and rationale
- [Contributing Guide](CONTRIBUTING.md) — Development workflow

---

## Development Workflow (TDD-Inspired)

We follow a rigorous test-driven approach:

1. **Discover**: Study JAX-Privacy implementation
2. **JAX Test**: Create reference test against JAX-Privacy (optional, requires `jax-validation` group)
3. **Failing Test**: Write Opaque test that fails
4. **Implement**: Make the test pass
5. **Document**: Add docstrings and API reference
6. **Example**: Create usage example (if warranted)

```bash
# Run standard tests (no JAX needed)
uv run pytest

# Run JAX validation tests (optional, requires JAX)
uv run --group jax-validation pytest -m jax_validation

# Examples (coming soon)
```

See [CONTRIBUTING.md](./CONTRIBUTING.md) for detailed guidelines.

---

## Contributing

We welcome contributions! Please see [CONTRIBUTING.md](./CONTRIBUTING.md) for:

- Setting up development environment
- Testing requirements
- Code style guidelines
- Pull request process

**Good First Issues**: Check the [issue tracker](https://github.com/evgri243/opaque/issues) for beginner-friendly tasks.

---

## Comparison with Alternatives

| Feature | Opaque | [Opacus](https://opacus.ai/) | [JAX-Privacy](https://github.com/google-deepmind/jax_privacy) |
|---------|--------|--------|-------------|
| **Framework** | PyTorch | PyTorch | JAX |
| **API Style** | Functional | Object-oriented | Functional (new) + OOP (old) |
| **LoRA Focus** |  Yes | � Possible |  Yes |
| **torch.func** |  Yes | L Custom hooks | N/A |
| **Maturity** | =� Alpha |  Production |  Production |
| **LLM Examples** | Coming soon | Limited | Gemma, Llama |

**Why Opaque?**
- **Modern PyTorch**: Built on `torch.func` (functional transformations)
- **Composability**: Separate clipping, noise, accounting modules
- **LoRA-Optimized**: Exploits low-rank structure for efficiency
- **JAX-Inspired**: Proven functional API design

**Why Not Opaque? (Yet)**
- **Maturity**: Opacus is battle-tested, Opaque is experimental
- **Coverage**: Opacus supports conv layers, batchnorm, etc.
- **If you need production DP today, use Opacus!**

---

## References

### Papers
- [Deep Learning with Differential Privacy](https://arxiv.org/abs/1607.00133) (Abadi et al. 2016) - DP-SGD algorithm
- [LoRA: Low-Rank Adaptation](https://arxiv.org/abs/2106.09685) (Hu et al. 2021) - Parameter-efficient fine-tuning
- [Unlocking High-Accuracy DP Image Classification](https://arxiv.org/abs/2204.13650) (De et al. 2022) - Practical DP training

### Libraries
- [JAX-Privacy](https://github.com/google-deepmind/jax_privacy) - Original inspiration
- [Opacus](https://opacus.ai/) - PyTorch DP library (different design)
- [dp-accounting](https://github.com/google/differential-privacy/tree/main/python/dp_accounting) - Privacy budget tracking

### Tutorials
- [DP-SGD Algorithm Explained](https://medium.com/pytorch/differential-privacy-series-part-1-dp-sgd-algorithm-explained-12512c3959a3)
- [PyTorch torch.func Tutorial](https://pytorch.org/tutorials/intermediate/functorch_tutorial.html)

---

## License

Apache 2.0 - See [LICENSE](./LICENSE) for details.

---

## Acknowledgments

- **JAX-Privacy Team** (Google DeepMind) for the excellent reference implementation
- **Opacus Team** (Meta) for pioneering PyTorch DP training
- **PyTorch Team** for `torch.func` making functional DP possible

---

**Questions?** Open an [issue](https://github.com/evgri243/opaque/issues) or start a [discussion](https://github.com/evgri243/opaque/discussions)!
