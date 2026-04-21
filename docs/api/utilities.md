# Utilities

## Functional Utilities

The `opaque.functional` module provides the `torch.func` bridges that turn
standard `nn.Module` models into the pure-function form DP-SGD needs.
`opaque.clipping.clipped_grad` and friends assume their `loss_fn`
argument is functional in this sense.

**Key function**: `make_functional()` — convert a PyTorch module to
functional form.

```python
import torch.nn as nn
from opaque.functional import make_functional

model = nn.Linear(10, 1)
fmodel, params = make_functional(model)
output = fmodel(params, input_tensor)
```

Functional models allow `torch.func.vmap` to compute per-example gradients
efficiently, which is essential for DP-SGD.

**See also**: [Quick Start Guide](../getting-started/quickstart.md) for
functional API usage.

::: opaque.functional
    options:
      show_source: true
      heading_level: 3

---

## PyTree Utilities

The `opaque.core.pytree` module provides helpers for working with PyTrees —
nested structures of tensors used throughout Opaque. For convenience the
most common entry points (`tree_map`, `tree_leaves`, `global_norm`,
`partition`, `merge`) are also re-exported from `opaque.core`.

**PyTrees** are nested dictionaries or tuples of tensors, commonly used to
represent model parameters:

```python
params = {
    "encoder": {"weight": tensor1, "bias": tensor2},
    "decoder": {"weight": tensor3, "bias": tensor4},
}
```

This module provides:

- **`tree_map()`** — apply a function to all leaves
- **`tree_map_with_path()`** — apply a function with key paths to all leaves
- **`tree_leaves()`** — extract all leaf tensors
- **`partition()`** — split a tree into two by predicate
- **`merge()`** — recombine partitioned trees
- **`global_norm()`** — compute L2 norm across the entire tree

**See also**: [Gradient Clipping Guide](../user-guide/clipping.md) for PyTree
usage in DP-SGD.

::: opaque.core.pytree
    options:
      show_source: true
      heading_level: 3
