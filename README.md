# Opaque

**Functional Differential Privacy for PyTorch LoRA Fine-tuning**

Opaque is a PyTorch port of Google's [JAX-Privacy](https://github.com/google-deepmind/jax_privacy), adapted specifically for differentially private (DP) fine-tuning of Large Language Models (LLMs) using LoRA (Low-Rank Adaptation).

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/pytorch-2.0+-red.svg)](https://pytorch.org/)
[![CI](https://github.com/JetBrains-Research/opaque/actions/workflows/ci.yml/badge.svg)](https://github.com/JetBrains-Research/opaque/actions/workflows/ci.yml)
[![Docs](https://github.com/JetBrains-Research/opaque/actions/workflows/docs.yml/badge.svg)](https://github.com/JetBrains-Research/opaque/actions/workflows/docs.yml)
[![Code style: ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff)
[![Tests](https://img.shields.io/badge/tests-111%20passing-brightgreen.svg)](https://github.com/JetBrains-Research/opaque)
[![Stage](https://img.shields.io/badge/stage-2%20complete-success.svg)](https://github.com/JetBrains-Research/opaque)

---

## Project Vision

Bring production-quality differential privacy to PyTorch's LLM fine-tuning ecosystem with:

- **Functional API**: Composable DP primitives inspired by JAX-Privacy's modern design
- **LoRA-First**: Optimized for parameter-efficient fine-tuning
- **PyTorch Native**: Built on `torch.func` (functional transformations)
- **Zero Surprises**: Fail-fast error handling for security-critical DP training

**Status**: 🎉 Stage 1 & 2 Complete — DP-SGD ready with functional clipping and noise injection!

---

## Quick Start

### Installation

```bash
# Basic installation
pip install opaque-dp  # Not yet published

# From source (development)
git clone https://github.com/JetBrains-Research/opaque.git
cd opaque
uv sync
```

### Minimal Example: DP-SGD Training

```python
import torch
from opaque import clipped_grad, gaussian

# 1. Define your loss function
def loss_fn(params, x, y):
    predictions = x @ params
    return ((predictions - y) ** 2).sum()

# 2. Configure DP-SGD components (once, outside loop)
grad_fn = clipped_grad(loss_fn, l2_clip_norm=1.0, batch_argnums=1)
noise_fn = gaussian(stddev=1.1 * grad_fn.clip_norm)

# 3. Training loop - clean functional composition!
params = torch.randn(10, requires_grad=False)

for batch_x, batch_y in dataloader:
    # Compute clipped gradients
    grads = grad_fn(params, batch_x, batch_y)

    # Add noise - natural composition!
    noisy_grads = noise_fn(grads)

    # Update parameters
    params = params - learning_rate * noisy_grads

# Privacy accounting handled externally (dp_accounting or jbr-fed-accounting)
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
  - Plain functions with `.clip_norm` attribute (no wrapper classes)
- [x] Numerical validation against JAX-Privacy main (all tests pass within 1e-5)
- [x] Full API parity with JAX-Privacy main branch (single-device features)

**Deliverable**: ✅ Complete functional API matching JAX-Privacy

### ✅ Stage 2: Noise Injection & Optimizers (Complete!)

**Timeline**: Completed 2025-11-14 | [Detailed Plan](docs/development/stage2-plan.md)

- [x] `opaque.noise` - Higher-order noise functions (NEW API!)
  - `gaussian(stddev)` - Stateless noise function
  - `gaussian_stateful(stddev, seed)` - Reproducible noise with explicit state
  - Clean functional composition: `noise_fn(grad_fn(...))`
- [x] `opaque.sampling` - Poisson sampling mechanisms
  - `PoissonSampler` - Standard Poisson sampling
  - `TruncatedPoissonSampler` - Bounded batch sizes
- [x] `opaque.optimizers` - DP optimizer wrappers
  - `adaptive_clipping()` - Adaptive clipping wrapper for TorchOpt optimizers
- [x] 218 tests passing (all core functionality)
- [x] Numerical equivalence with JAX-Privacy confirmed

**Note**: Privacy accounting is now external (use `dp_accounting` or `jbr-fed-accounting`)

**Deliverable**: ✅ Complete DP-SGD training implementation with clean functional API

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

1. **Test First**: Write failing test that defines the API
2. **Implement**: Make the test pass (minimal code)
3. **Document**: Add docstrings and API reference
4. **Refactor**: Improve code quality
5. **Verify**: Run full test suite

```bash
# Run all tests
uv run pytest

# Run with coverage
uv run pytest --cov=opaque --cov-report=html
```

See [CONTRIBUTING.md](./CONTRIBUTING.md) for detailed guidelines.

---

## Contributing

We welcome contributions! Please see [CONTRIBUTING.md](./CONTRIBUTING.md) for:

- Setting up development environment
- Testing requirements
- Code style guidelines
- Pull request process

**Good First Issues**: Check the [issue tracker](https://github.com/JetBrains-Research/opaque/issues) for beginner-friendly tasks.

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

**Questions?** Open an [issue](https://github.com/JetBrains-Research/opaque/issues) or start a [discussion](https://github.com/JetBrains-Research/opaque/discussions)!
