# opaque-torch

PyTorch provider for Opaque's backend-neutral engine primitives.

## Install

```bash
pip install opaque-torch
```

Or install the default bundle:

```bash
pip install opaque
```

## Usage

Torch tensors and modules select this provider automatically. Explicit selection
is available through `opaque.torch.torch_backend()`.

Activation registers the portable core, distributed and observability profiles,
native `Tensor` and `Parameter` serialization, allocator controls, and trace
annotations. Torch-only helpers, including module functionalization and RNG
bridges, live under `opaque.torch`.
