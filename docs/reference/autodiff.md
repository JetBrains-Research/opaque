# opaque.autodiff

Backend-dispatched functional autodiff transforms. These mirror the
`torch.func` shapes on every provider: transforms are constructed once and
bind to the active backend at call time.

```python
from opaque.autodiff import grad_and_value, vmap

grad_fn = grad_and_value(loss_fn)          # -> (grads, value)
per_example = vmap(loss_fn, in_axes=(None, 0))
```

`vmap` carries an explicit `randomness` contract (default `"same"`), so
stochastic modules such as dropout behave deterministically under
per-example vectorization unless a caller opts out.

## Transforms

::: opaque.autodiff.grad_and_value
    options:
        show_source: true
        heading_level: 3

::: opaque.autodiff.vmap
    options:
        show_source: true
        heading_level: 3
