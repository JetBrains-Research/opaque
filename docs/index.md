# Opaque

**Functional Differential Privacy for PyTorch LoRA Fine-tuning**

Opaque is a PyTorch port of Google's [JAX-Privacy](https://github.com/google-deepmind/jax_privacy), adapted specifically for differentially private (DP) fine-tuning of Large Language Models (LLMs) using LoRA (Low-Rank Adaptation).

## Features

- **Functional API**: Composable DP primitives inspired by JAX-Privacy
- **LoRA-First**: Optimized for parameter-efficient fine-tuning
- **PyTorch Native**: Built on `torch.func` functional transformations
- **Test-Driven**: Validated against JAX-Privacy reference implementation

## Status

!!! success "Stage 1 Complete"
    ✅ Core clipping module is complete and tested! 79 tests passing with numerical validation against JAX-Privacy.

    🔜 Next: Stage 2 (Noise Injection) to enable full DP-SGD.

## Quick Example

```python
import torch
from opaque.clipping import clipped_grad


# Define loss for a single example
def loss_fn(params, data):
    return 0.5 * ((data - params) ** 2).mean()


# Create clipped gradient function
clipped_grad_fn = clipped_grad(
    loss_fn,
    l2_clip_norm=1.0,  # Clip each example's gradient to max norm 1.0
    normalize_by=3.0,  # Divide by batch size
)

# Compute clipped gradients on batch
params = torch.tensor(3.0, requires_grad=True)
data = torch.tensor([0.0, 7.0, -2.0])  # Batch of 3 examples

clipped_grads = clipped_grad_fn(params, data)
```

## Installation

!!! note
    Opaque is not yet published to PyPI. Install from source:

```bash
git clone https://github.com/evgri243/opaque.git
cd opaque
uv sync
```

## Next Steps

### For Learners
- 📚 [Tutorial: Gradient Clipping from Basics](tutorials/01_gradient_clipping_from_basics.ipynb) - Start here!
- 📖 [Quick Start Guide](getting-started/quickstart.md)
- 📖 [DP Basics](user-guide/dp-basics.md)

### For Developers
- 🔧 [API Reference](api/core/clipping.md)
- 🔧 [Contributing Guide](development/contributing.md)
- 🔧 [Design Decisions](development/design-decisions.md)
