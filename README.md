# Opaque

**Functional Differential Privacy for PyTorch LoRA Fine-tuning**

Opaque is a PyTorch port of Google's [JAX-Privacy](https://github.com/google-deepmind/jax_privacy), adapted specifically for differentially private (DP) fine-tuning of Large Language Models (LLMs) using LoRA (Low-Rank Adaptation).

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/pytorch-2.0+-red.svg)](https://pytorch.org/)

---

## Project Vision

Bring production-quality differential privacy to PyTorch's LLM fine-tuning ecosystem with:

- **Functional API**: Composable DP primitives inspired by JAX-Privacy's modern design
- **LoRA-First**: Optimized for parameter-efficient fine-tuning
- **PyTorch Native**: Built on `torch.func` (functional transformations)
- **Zero Surprises**: Fail-fast error handling for security-critical DP training

**Status**: Early Development — Core clipping module in progress (Stage 1)

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

### Minimal Example

Note: This snippet shows the planned API and will work after Stage 1 is implemented. Do not run yet.

```python
import torch
from opaque.core.clipping import clipped_grad

# Define loss for a single example
def loss_fn(params, data):
    return 0.5 * ((data - params) ** 2).mean()

# Create clipped gradient function
clipped_grad_fn = clipped_grad(
    loss_fn,
    l2_clip_norm=1.0,      # Clip each example's gradient to max norm 1.0
    normalize_by=3.0,       # Divide by batch size
)

# Compute clipped gradients on batch
params = torch.tensor(3.0, requires_grad=True)
data = torch.tensor([0.0, 7.0, -2.0])  # Batch of 3 examples

clipped_grads = clipped_grad_fn(params, data)
print(clipped_grads)  # Gradients clipped per-example, then summed
```

**Next**: Add Gaussian noise and privacy accounting (coming in Stage 2+)

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

### 📋 Stage 1: Core Clipping (Ready to Start)
**Timeline**: 3 weeks | [Detailed Plan](docs/development/stage1-plan.md)

- [ ] `opaque.core.pytree_utils` - PyTree operations
- [ ] `opaque.core.clipping` - Per-example gradient clipping
- [ ] Numerical validation against JAX-Privacy
- [ ] Examples: Linear regression, basic usage

**Deliverable**: `clipped_grad()` function matching JAX-Privacy behavior

### Stage 2: Noise Injection (Weeks 4-5)
- [ ] `opaque.core.noise` - Gaussian noise generation
- [ ] Reproducibility with `torch.Generator`
- [ ] Integration with clipped gradients

### Stage 3: Privacy Accounting (Weeks 6-7)
- [ ] Wrap Google's `dp-accounting` library
- [ ] Noise calibration for target (�, �)
- [ ] Privacy budget tracking

### Stage 4: High-Level API (Weeks 8-9)
- [ ] `opaque.api.make_private()` - One-line DP wrapper
- [ ] `DPConfig` - Configuration dataclass
- [ ] Integration with Hugging Face `peft` library

---

## Documentation

- [Stage 1 Plan](docs/development/stage1-plan.md) — Current implementation plan
- [Design Decisions](docs/development/design-decisions.md) — Technical choices and rationale
- [Architecture Overview](docs/development/architecture.md)
- [Roadmap](docs/development/roadmap.md)
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
