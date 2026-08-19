# opaque.ops

Portable, backend-dispatched array operations. Every exported callable is a
canonical `Primitive`: it accepts and returns native arrays and dtype objects
of the active backend, and Opaque deliberately wraps neither.

```python
from opaque import ops

norm_sq = ops.sum(ops.square(grad))
host = ops.to_host(scores)      # detached numpy copy
```

These are the building blocks the mechanisms are written in; user code
usually stays a level above them (clipping, noise, optimizers). They are
public because provider authors implement them and advanced callers compose
them.

## Module

::: opaque.ops
    options:
        show_source: false
        heading_level: 3
        members: true
        filters: ["!^_"]
