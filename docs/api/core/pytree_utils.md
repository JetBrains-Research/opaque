# PyTree Utilities

The `opaque.utils.pytree` module provides utilities for working with PyTrees - nested structures of tensors used
throughout Opaque.

## Overview

**PyTrees** are nested dictionaries or tuples of tensors, commonly used to represent model parameters:

```python
# Example PyTree
params = {
    "encoder": {"weight": tensor1, "bias": tensor2},
    "decoder": {"weight": tensor3, "bias": tensor4},
}
```

This module provides operations on PyTrees:

- **`tree_map()`** - Apply function to all leaves
- **`global_norm()`** - Compute L2 norm across entire tree
- **`tree_leaves()`** - Extract all leaf tensors

These utilities are used internally by clipping functions and are also useful for custom DP implementations.

**See also**: [Gradient Clipping Guide](../../user-guide/clipping.md) for PyTree usage in DP-SGD

## API Documentation

::: opaque.utils.pytree
    options:
      show_source: true
      heading_level: 2
