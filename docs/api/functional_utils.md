# Functional Utilities

The `opaque.utils.functional` module provides utilities for functional programming with PyTorch models.

## Overview

Opaque uses PyTorch's functional API (`torch.func`) for computing per-example gradients. This requires converting
standard `nn.Module` models to functional form.

**Key function**: `make_functional()` - Convert a PyTorch module to functional form

### Example

```python
import torch.nn as nn
from opaque import make_functional

# Standard PyTorch model
model = nn.Linear(10, 1)

# Convert to functional
fmodel, params = make_functional(model)

# Use with explicit parameters
output = fmodel(params, input_tensor)
```

**Why functional?** Functional models allow `torch.func.vmap` to compute per-example gradients efficiently, which is
essential for DP-SGD.

**See also**: [Quick Start Guide](../getting-started/quickstart.md) for functional API usage

## API Documentation

::: opaque.utils.functional
    options:
      show_source: true
      heading_level: 2
