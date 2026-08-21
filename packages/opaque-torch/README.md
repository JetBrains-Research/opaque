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

Activation registers the portable core, distributed and observability
profiles, the optional `ExecutionProfile` transforms (`compile`,
`checkpoint`, `optimize_saved_activations`), native `Tensor` and `Parameter`
serialization, allocator controls, and trace annotations. Torch-only helpers,
including module functionalization and RNG bridges, live under `opaque.torch`.

The Torch execution provider implements:

- `compile(fn)` → `torch.compile(fn)`
- `checkpoint(fn)` → `torch.utils.checkpoint.checkpoint(fn, ..., use_reentrant=False)`
- `optimize_saved_activations(fn)` → enters `torch.autograd.graph.save_on_cpu(pin_memory=True)` per call

Checkpoint/functorch compatibility is the provider's own concern:
`opaque.torch.apply_runtime_patches()` applies it, `opaque.torch.checkpoint`
exposes the individual installers and the capability probes, and none of it
needs `opaque-patches` installed. Higher layers call this one first — fixing
Hugging Face requires fixing torch — so their users still make a single call.
