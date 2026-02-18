# Opaque

**Functional Differential Privacy for PyTorch LoRA Fine-tuning**

Opaque is a PyTorch port of Google's [JAX-Privacy](https://github.com/google-deepmind/jax_privacy), adapted specifically for differentially private (DP) fine-tuning of Large Language Models (LLMs) using LoRA (Low-Rank Adaptation).

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/pytorch-2.0+-red.svg)](https://pytorch.org/)
[![CI](https://github.com/JetBrains-Research/opaque/actions/workflows/ci.yml/badge.svg)](https://github.com/JetBrains-Research/opaque/actions/workflows/ci.yml)
[![Docs](https://github.com/JetBrains-Research/opaque/actions/workflows/docs.yml/badge.svg)](https://github.com/JetBrains-Research/opaque/actions/workflows/docs.yml)
[![Code style: ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff)
[![Tests](https://img.shields.io/badge/tests-458%20passing-brightgreen.svg)](https://github.com/JetBrains-Research/opaque)

---

## Project Vision

Bring production-quality differential privacy to PyTorch's LLM fine-tuning ecosystem with:

- **Functional API**: Composable DP primitives inspired by JAX-Privacy's modern design
- **LoRA-First**: Optimized for parameter-efficient fine-tuning
- **PyTorch Native**: Built on `torch.func` (functional transformations)
- **Zero Surprises**: Fail-fast error handling for security-critical DP training

---

## Quick Start

### Installation

```bash
# Core library (only torch + optree)
pip install opaque-dp  # Not yet published

# Then add your ML stack
pip install transformers peft  # For LLMs
# or
pip install torchvision         # For CV
# or whatever you need!

# Development
git clone https://github.com/JetBrains-Research/opaque.git
cd opaque
uv sync --group dev  # Core tests
uv sync --group dev --group compat  # + HF tests
```

**Philosophy**: Opaque provides DP-SGD primitives. You bring the models.

See [Installation Guide](#installation-guide) for detailed instructions.

### Minimal Example: DP-SGD Training

```python
import torch
import opaque.accounting as acc
from opaque import clipped_grad, gaussian_noise

# 1. Define your loss function
def loss_fn(params, x, y):
    predictions = x @ params
    return ((predictions - y) ** 2).sum()

# 2. Configure DP-SGD components (once, outside loop)
grad_fn, clip_state = clipped_grad(loss_fn, l2_clip_norm=1.0, batch_argnums=(1, 2))
noise_fn, noise_state = gaussian_noise(stddev=1.1 * clip_state.sensitivity())

# 3. Training loop - clean functional composition!
params = torch.randn(10, requires_grad=False)

for batch_x, batch_y in dataloader:
    # Compute clipped gradients (state flows through)
    grads, clip_state = grad_fn(params, batch_x, batch_y, state=clip_state)

    # Add noise
    noisy_grads, noise_state = noise_fn(grads, noise_state)

    # Update parameters
    params = params - learning_rate * noisy_grads

# 4. Privacy accounting
step = acc.poisson(acc.gaussian(1.1), sample_rate=0.01)
training = step * num_steps
epsilon = training.epsilon_at(1e-5)
print(f"Privacy: (ε={epsilon:.2f}, δ=1e-5)")
```

**See**: [Quick Start Guide](docs/getting-started/quickstart.md) for complete walkthrough

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

## Project Status

Core DP-SGD primitives are implemented and validated:

- **Clipping**: `clip_pytree()`, `clipped_fun()`, `clipped_grad()`, `adaptive_clipped_grad()` — Per-example gradient clipping with functional state
- **Noise**: `gaussian_noise()`, `bounded_gaussian_noise()`, plus matrix factorization noise (`band_mf_noise`, `blt_mf_noise`, `dense_mf_noise`, `custom_mf_noise`, `identity_mf_noise`)
- **Sampling**: `PoissonSampler`, `TruncatedPoissonSampler`, `CyclicPoissonSampling` — Privacy-amplified batch selection
- **Accounting**: Rust-based PLD engine with tight composition bounds, calibration, and the `Accountant` training-loop helper
- **Auditing**: Empirical privacy auditing via membership inference
- **Profiling**: Memory tracking and microbatch size auto-tuning
- **HuggingFace Compatibility**: Auto-patching for vmap-compatible forward passes (LLaMA, Mistral, Qwen2, Phi, OLMo, Gemma2)
- **Validation**: 458 tests passing, numerical equivalence with JAX-Privacy (atol=1e-5)

---

## Documentation

### Tutorials

- [Tutorial 01: Gradient Clipping from Basics](docs/tutorials/01_gradient_clipping_from_basics.ipynb) — Learn per-example gradient clipping
- [Tutorial 02: Noise and Accounting](docs/tutorials/02_differential_privacy_noise_and_accounting.ipynb) — Noise injection and privacy accounting
- [Tutorial 03: Complete DP-SGD Training](docs/tutorials/03_complete_dp_sgd_training.ipynb) — End-to-end training loop
- [Tutorial 04: DP Optimizers](docs/tutorials/04_dp_optimizers.ipynb) — TorchOpt integration and adaptive clipping
- [Tutorial 05: Sampling and Microbatching](docs/tutorials/05_sampling_and_microbatching.ipynb) — Poisson sampling strategies
- [Tutorial 06: LoRA HuggingFace Training](docs/tutorials/06_lora_huggingface_dp_training.ipynb) — Fine-tune LLMs with DP
- [Tutorial 07: Privacy Auditing](docs/tutorials/07_privacy_auditing.ipynb) — Empirical privacy validation

### Guides

- [Quick Start](docs/getting-started/quickstart.md) — Train your first DP model in 5 minutes
- [User Guide](docs/user-guide/index.md) — Concepts and best practices
- [API Reference](docs/api/index.md) — Full API documentation

### Development

- [Contributing Guide](CONTRIBUTING.md) — Development workflow

---

## Development

```bash
# Run all tests
uv run pytest

# Run with coverage
uv run pytest --cov=opaque --cov-report=html

# Format & lint
uv run ruff format src/ tests/
uv run ruff check src/ tests/
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
| **Maturity** | Alpha |  Production |  Production |
| **LLM Examples** | LoRA tutorials | Limited | Gemma, Llama |

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

## Installation Guide

### For End Users

#### Install Opaque
```bash
pip install opaque-dp
```
**Includes**: `torch>=2.10`, `optree>=0.17`

#### Then Add Your ML Stack
```bash
# For LLM fine-tuning
pip install transformers peft datasets

# For computer vision
pip install torchvision

# For optimizers
pip install torchopt

# Whatever you need!
```

**Philosophy**: Opaque is DP infrastructure. You choose your models and libraries.

### For Developers

#### Quick Start (Development)
```bash
git clone https://github.com/JetBrains-Research/opaque.git
cd opaque
uv sync --group dev
```
**Includes**: Core tests, linting, formatting

#### Full Development (All Features)
```bash
uv sync --all-groups
```
**Includes**: Everything (tests, docs, examples, benchmarks)

#### Selective Installation
```bash
# Just compatibility testing
uv sync --group compat

# Run examples and tutorials
uv sync --group examples

# Build documentation
uv sync --group docs

# Benchmark against Opacus
uv sync --group benchmark

# Multiple groups
uv sync --group dev --group compat --group examples
```

### Common Workflows

**Researcher**: Use Opaque for custom DP experiments
```bash
pip install opaque-dp
# Then install whatever models/datasets you need
```

**ML Engineer**: Fine-tune LLMs with DP
```bash
pip install opaque-dp
pip install transformers peft datasets
```

**Contributor**: Develop and test code
```bash
git clone https://github.com/JetBrains-Research/opaque.git
cd opaque
uv sync --group dev
pytest tests/  # Core tests only

# Optional: Test HuggingFace compatibility
uv sync --group compat
pytest tests/compat/
```

**Documentation Writer**: Build docs locally
```bash
uv sync --group docs
mkdocs serve
```

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
