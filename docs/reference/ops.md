# opaque.ops

Portable, backend-dispatched array operations. Every exported callable is a
canonical `Primitive`: it accepts and returns native arrays and dtype objects
of the active backend, and Opaque deliberately wraps neither.

```python
from opaque import ops

norm_sq = ops.sum(ops.square(grad))
host = ops.to_host(scores)  # detached numpy copy
```

These are the building blocks the mechanisms are written in; training code
stays a level above them, at clipping, noise, and optimizers.

## Overview

- **Inspection** — `is_array()`, `shape()`, `dtype()`, `is_floating()`,
  `is_complex()`, `is_low_precision()`, `isfinite()`
- **Creation and placement** — `zeros()`, `zeros_like()`, `ones_like()`,
  `scalar()`, `clone()`, `detach()`, `transfer()`, `to_host()`
- **Arithmetic** — `add()`, `subtract()`, `multiply()`, `divide()`, `pow()`,
  `reciprocal()`, `abs()`, `square()`, `sqrt()`, `rsqrt()`, `exp()`, `erf()`,
  `erfinv()`
- **Reductions** — `sum()`, `mean()`, `all()`, `amin()`, `amax()`,
  `scalar_item()`
- **Comparison and selection** — `greater()`, `maximum()`, `minimum()`,
  `clamp()`, `where()`, `nan_to_num()`
- **Structure** — `expand_dims()`, `squeeze()`, `slice_array()`,
  `concatenate()`
- **Dtype** — `astype()`, `promote_dtype()`, `real_dtype()`,
  `accumulator_dtype()`, `float32()`, `finfo_eps()`,
  `finfo_smallest_normal()`

`transfer()` is the one deliberately provider-shaped operation here; the rest
are portable across every backend implementing the core profile.

## Module

::: opaque.ops
    options:
        show_source: false
        heading_level: 3
        members: true
        filters: ["!^_"]
