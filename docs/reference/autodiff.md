# opaque.autodiff

Backend-dispatched functional autodiff transforms. These mirror the
`torch.func` shapes on every provider: transforms are constructed once and
bind to the active backend at call time.

```python
from opaque.autodiff import grad_and_value, vmap

grad_fn = grad_and_value(loss_fn)          # -> (grads, value)
per_example = vmap(loss_fn, in_axes=(None, 0))
```

`vmap` carries an explicit `randomness` contract (default `"same"`): one draw
from the framework's ambient generator is reused for every element, so a
stochastic module such as dropout applies the identical mask to each example,
and that mask still changes from run to run unless the generator is seeded.
Pass `"different"` for independent draws, or `"error"` to reject ambient
randomness. Opaque's own randomness is keyed and unaffected — `normal(key, ...)`
inside a mapped function ignores the setting; derive a key per element for
per-example noise, as in [Domain separation](rng.md#domain-separation).

## Transforms

::: opaque.autodiff.grad_and_value
    options:
        show_source: true
        heading_level: 3

::: opaque.autodiff.vmap
    options:
        show_source: true
        heading_level: 3
