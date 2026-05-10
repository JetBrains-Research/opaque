# Clipping `fun` helpers

`opaque.dpsgd.clipping.clipped_grad` (and its `dpftrl` mirror) is the
standard one-shot path: build the per-example loss, clip, sum,
optionally normalize. Under the hood it's built on a smaller
power-user surface in
`opaque.api.engine.clipping.fun` that you can call directly when you
need fine-grained control.

## When to use it

- You're clipping something that **isn't** a per-example loss
  gradient — e.g. a pre-computed activation tensor, an arbitrary
  PyTree return value, a privacy-budget-bounded statistic.
- You're building a custom mechanism that needs to clip mid-pipeline.
- You're implementing a new sensitivity-bound clipping rule and need
  to plug into the rest of the DP pipeline (noise + accounting).

## The four primitives

```python
from opaque.api.engine.clipping.fun import (
    clipped_fun,        # clip + sum any per-example function output
    auto_clipped_fun,   # AUTO-S variant of clipped_fun
    clip_pytree,        # clip an already-batched pytree of per-example values
    auto_scale_pytree,  # AUTO-S variant of clip_pytree
)
```

### `clipped_fun`

```python
clipped_fun(
    fun: Callable,
    *,
    has_aux: bool = False,
    batch_argnums: int | tuple[int, ...] = 0,
    clipping_norm: float | PerGroup = 1.0,
    normalize_by: float = 1.0,
    return_aux: bool = False,
    second_moment: bool = False,
    microbatch_size: int | None = None,
) -> tuple[Callable, FixedClipState]
```

Wraps a function `fun(args...) -> value` (where `value` is a PyTree)
in `vmap(grad(...))` semantics, clips each per-example output to a
fixed norm, and sums. Returns `(clipped_fn, state)`.

Use this when you have a per-example function that returns *something
the clipping is meaningful for* (an embedding, a statistic) but you
don't want to differentiate through a loss.

### `auto_clipped_fun`

The AUTO-S variant — automatic per-example scaling with a smooth-min
function (Bu et al. NeurIPS 2023). Same signature as `clipped_fun`
plus a `gamma: float` smoothness parameter. Constant per-record
sensitivity (`R`), so it composes with both Gaussian and
matrix-factorization mechanisms.

### `clip_pytree`

```python
clip_pytree(
    pytree: TensorPytree,
    *,
    max_norm: float | PerGroup,
    normalize_by: float = 1.0,
) -> ClippedPytree
```

Lower level still — given a pytree where leaves are tensors with a
leading batch dimension, clip each per-example slice's L2 norm and
sum. No `vmap`, no `grad`. Use when you've already produced
per-example tensors via some other route (autograd, manual
differencing, gradient checkpointing).

### `auto_scale_pytree`

AUTO-S variant of `clip_pytree`. Same signature plus the smoothness
parameter; same constant-sensitivity property.

## Worked example: clipping per-example activations for a custom mechanism

```python
from opaque.api.engine.clipping.fun import clipped_fun
from opaque.dpsgd.noise import gaussian_noise
from opaque.random import key

def per_example_activation(params, batch):
    # ... compute a per-example summary statistic, return a PyTree ...
    return {"summary": ...}

clipped_fn, clip_state = clipped_fun(
    per_example_activation,
    batch_argnums=1,
    clipping_norm=1.0,
)
clipped, _ = clipped_fn(params, batch, state=clip_state)

noise_fn, noise_state = gaussian_noise(noise_multiplier=1.0, key=key(0))
noised, _ = noise_fn(clipped, noise_state)
# noised is a NoisedPytree carrying max_norm + noise_stddev metadata
```

The `NoisedPytree` flows downstream into optimisers and aggregators
exactly as the standard DP-SGD path does.

## Aux outputs

Pass `return_aux=True` to get a `ClippedFunAux` (for `clipped_fun`)
or `AutoClippedFunAux` (for `auto_clipped_fun`) alongside the
clipped output. Aux carries per-example pre-clip / post-clip norms,
the clipping rate, and the per-example values; use them for
diagnostics or to drive an adaptive threshold.

## Microbatching

`microbatch_size: int | None` accumulates clipped sums over micro-
batches without ever materialising the full per-example tensor — the
standard pattern when memory is tight. The accumulator preserves the
clipping invariant exactly (no double-clipping).

## See also

- [`opaque.dpsgd.clipping`](../reference/clipping.md) — the high-level
  `clipped_grad` surface.
- [`opaque.types`](../reference/utilities.md) — `ClippedPytree`,
  `NoisedPytree`, `PerGroup`.
