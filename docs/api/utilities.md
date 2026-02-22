# Utilities

## Functional Utilities

The `opaque.utils.functional` module provides utilities for functional
programming with PyTorch models.

Opaque uses PyTorch's functional API (`torch.func`) for computing per-example
gradients. This requires converting standard `nn.Module` models to functional
form.

**Key function**: `make_functional()` -- Convert a PyTorch module to functional
form.

```python
import torch.nn as nn
from opaque import make_functional

model = nn.Linear(10, 1)
fmodel, params = make_functional(model)
output = fmodel(params, input_tensor)
```

Functional models allow `torch.func.vmap` to compute per-example gradients
efficiently, which is essential for DP-SGD.

**See also**: [Quick Start Guide](../getting-started/quickstart.md) for
functional API usage

::: opaque.utils.functional
    options:
      show_source: true
      heading_level: 3

---

## PyTree Utilities

The `opaque.utils.pytree` module provides utilities for working with PyTrees --
nested structures of tensors used throughout Opaque.

**PyTrees** are nested dictionaries or tuples of tensors, commonly used to
represent model parameters:

```python
params = {
    "encoder": {"weight": tensor1, "bias": tensor2},
    "decoder": {"weight": tensor3, "bias": tensor4},
}
```

This module provides:

- **`tree_map()`** -- Apply a function to all leaves
- **`tree_map_with_path()`** -- Apply a function with key paths to all leaves
- **`tree_leaves()`** -- Extract all leaf tensors
- **`partition()`** -- Split a tree into two by predicate
- **`merge()`** -- Recombine partitioned trees
- **`global_norm()`** -- Compute L2 norm across entire tree

**See also**: [Gradient Clipping Guide](../user-guide/clipping.md) for PyTree
usage in DP-SGD

::: opaque.utils.pytree
    options:
      show_source: true
      heading_level: 3
