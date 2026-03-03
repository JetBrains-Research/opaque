# Per-Example Gradient Clipping

Per-example gradient clipping bounds the influence of each training example on
the model update. This is the core operation that makes DP-SGD possible:
clipping establishes a known sensitivity, which determines how much noise is
needed for a given privacy guarantee.

Opaque provides three clipping APIs at different levels of abstraction.

## `clipped_grad` -- recommended API

`clipped_grad` wraps a per-example loss function. It computes per-example
gradients, clips each to a maximum L2 norm, and sums the result. This is the
primary API for DP-SGD training.

```python
from opaque import clipped_grad

def loss_fn(params, x, y):
    return ((x @ params - y) ** 2).sum()

grad_fn, clip_state = clipped_grad(
    loss_fn,
    argnums=0,             # differentiate w.r.t. first argument (params)
    batch_argnums=(1, 2),  # second and third arguments are batched
    l2_clip_norm=1.0,
)

grads, clip_state = grad_fn(params, batch_x, batch_y, state=clip_state)
```

### How it works

1. `torch.func.grad_and_value` computes the gradient of `loss_fn` with respect
   to the argument at position `argnums`.
2. `torch.func.vmap` vectorizes this over the batch dimension of the arguments
   at positions `batch_argnums`, producing one gradient per example.
3. Each per-example gradient is clipped to L2 norm at most `l2_clip_norm`.
4. The clipped gradients are summed across the batch.

The returned `clip_state` is a `FixedClipState` containing the clip norm and
a `sensitivity()` method used to calibrate noise.

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `loss_fn` | `Callable` | required | Per-example loss function. Must return a scalar (or `(scalar, aux)` if `has_aux=True`). |
| `argnums` | `int \| tuple[int, ...]` | `0` | Which arguments to differentiate. |
| `has_aux` | `bool` | `False` | If True, `loss_fn` returns `(loss, aux)`. The aux data is returned per-example. |
| `l2_clip_norm` | `float` | required | Maximum L2 norm for per-example gradients. |
| `batch_argnums` | `int \| tuple[int, ...]` | `1` | Which arguments have a batch dimension. |
| `microbatch_size` | `int \| None` | `None` | Process batch in chunks to reduce memory. |
| `normalize_by` | `float` | `1.0` | Divide the clipped output and sensitivity by this value. Useful for averaging (set to batch size). |
| `pre_clipping_transform` | `Callable` | identity | Transform applied to each per-example gradient before clipping. |
| `dtype` | `torch.dtype \| None` | `None` | Accumulation dtype (e.g., float32 for float16 inputs). |
| `return_aux` | `bool` | `False` | Return per-example diagnostics. |

### State flow

`clipped_grad` returns `(grad_fn, clip_state)`. The state must be threaded
through each call:

```python
grad_fn, clip_state = clipped_grad(loss_fn, l2_clip_norm=1.0, batch_argnums=1)

for batch in dataloader:
    grads, clip_state = grad_fn(params, batch, state=clip_state)
    # clip_state is immutable; the returned value is the same object
```

With fixed clipping, the state never changes -- `FixedClipState` is immutable
and the same instance is returned on every call. The state-passing convention
exists for API consistency with `adaptive_clipped_grad`, where the state does
change.

### Sensitivity

The sensitivity is the maximum change in the clipped gradient sum when one
example is added, removed, or replaced. Noise is calibrated to this value.

```python
sensitivity = clip_state.sensitivity()
# With l2_clip_norm=1.0: sensitivity = 1.0

noise_fn, noise_state = gaussian_noise(
    stddev=noise_multiplier * sensitivity, key=key(42),
)
```

### Diagnostics

Set `return_aux=True` to get per-example gradient norms and loss values:

```python
grad_fn, clip_state = clipped_grad(
    loss_fn, l2_clip_norm=1.0, batch_argnums=1, return_aux=True,
)

(grads, aux), clip_state = grad_fn(params, batch, state=clip_state)
# aux.grad_norms: per-example L2 norms before clipping
# aux.clipped_grad_norms: per-example L2 norms after clipping
# aux.loss_values: per-example loss values
# aux.clipping_norm: the L2 clip norm used
```

`adaptive_clipped_grad` returns `AdaptiveClippedGradAux` instead, which has
a `clipping_rate` field (fraction of gradients clipped) instead of
`clipping_norm`.

## `clipped_fun` -- general-purpose clipping

`clipped_fun` clips and sums the outputs of any function, not just gradients.
`clipped_grad` is built on top of `clipped_fun`.

```python
from opaque import clipped_fun

def per_example_fn(params, example):
    return compute_something(params, example)

clipped_fn, clip_state = clipped_fun(
    per_example_fn,
    batch_argnums=1,
    l2_clip_norm=1.0,
)

summed_result, clip_state = clipped_fn(params, batch, state=clip_state)
```

Use `clipped_fun` when you already have per-example outputs (not necessarily
gradients) and want to clip-and-sum them.

## `clip_pytree` -- low-level clipping

`clip_pytree` clips a single PyTree of tensors to a maximum L2 norm. It does
not handle batching or summation.

```python
from opaque import clip_pytree

grads = {"weight": torch.tensor([3.0, 4.0]), "bias": torch.tensor([1.0])}
clipped_grads, aux = clip_pytree(grads, clip_norm=1.0)
# aux.norm = 5.099 (original L2 norm)
# clipped_grads: scaled so global L2 norm <= 1.0
```

Use `clip_pytree` when you have pre-computed per-example outputs and want
fine-grained control over clipping. Most users should use `clipped_grad`
instead.

## Microbatching

Per-example gradient computation via `vmap` requires memory proportional to
`batch_size * model_parameters`. For large models, this may exceed GPU memory.

Microbatching processes the batch in smaller chunks, accumulating clipped
gradients incrementally. The result is mathematically identical to processing
the full batch.

```python
grad_fn, clip_state = clipped_grad(
    loss_fn,
    l2_clip_norm=1.0,
    batch_argnums=1,
    microbatch_size=16,  # process 16 examples at a time
)
```

Each microbatch of 16 examples is vmapped, clipped, and summed. The partial
sums are accumulated in-place, so peak memory is proportional to
`microbatch_size * model_parameters` instead of `batch_size * model_parameters`.

### Choosing microbatch size

Use `TrainingProfiler` from `opaque.profiling` to run a short sweep and
select the largest stable microbatch that does not OOM:

```python
from opaque.profiling import TrainingProfiler, reset_peak_memory

profiler = TrainingProfiler(device)
for candidate_mb in [64, 32, 16, 8, 4, 2, 1]:
    grad_fn, state = clipped_grad(
        loss_fn,
        l2_clip_norm=1.0,
        batch_argnums=(1, 2),
        microbatch_size=candidate_mb,
    )

    reset_peak_memory(device)
    with profiler.step(batch_size=batch_size):
        grads, aux = grad_fn(params, batch_x, batch_y, state=state)

    print(candidate_mb, profiler.current_metrics()["memory_peak_gb"])
```

See [Memory Profiling](memory-profiling.md) for details.

## Adaptive clipping

`adaptive_clipped_grad` automatically adjusts the clip norm during training
using the geometric adaptation rule from
[Andrew et al. 2021](https://arxiv.org/abs/1905.03871). Instead of manually
tuning the clip norm, you specify a target fraction of gradients that should
be clipped (the *target quantile*).

```python
from opaque import adaptive_clipped_grad
from opaque.random import key

grad_fn, clip_state = adaptive_clipped_grad(
    loss_fn,
    batch_argnums=1,
    initial_clip_norm=1.0,
    target_quantile=0.5,   # aim for 50% of gradients clipped
    key=key(7),            # required for quantile noise
)

grads, clip_state = grad_fn(params, batch, state=clip_state)
# clip_state.clip_norm has been updated
```

### How adaptive clipping works

After each step:

1. Compute the fraction of per-example gradients whose norm exceeded the
   current clip norm.
2. Add calibrated Gaussian noise to this fraction (for privacy).
3. Apply a geometric update: if the noisy clipping rate exceeds the target
   quantile, increase the clip norm; otherwise, decrease it.

The update rule is:

    C_{t+1} = C_t * exp(eta * (noisy_rate_t - target_quantile))

where eta is the `learning_rate` parameter (default 0.2).

### State changes

Unlike `clipped_grad`, the state from `adaptive_clipped_grad` **does change**
on each call. The returned `AdaptiveClipState` contains the updated clip norm,
step counter, and clipping statistics. Always use the returned state for the
next call.

```python
for batch in dataloader:
    grads, clip_state = grad_fn(params, batch, state=clip_state)
    # clip_state.clip_norm may have changed
    # clip_state.clipping_rate shows fraction clipped this step
```

### Privacy accounting for adaptive clipping

Adaptive clipping introduces an additional privacy cost (the noisy clipping
rate query). Account for it using `acc.adaclip()`:

```python
import opaque.accounting as acc

step = acc.poisson(
    acc.adaclip(acc.gaussian(noise_multiplier),
                quantile_noise_multiplier=0.05,
                batch_size=batch_size),
    sample_rate,
)
training = step * num_steps
eps = training.epsilon_at(1e-5)
```

### Distributed adaptive clipping

In distributed training, the clip norm must be consistent across devices.
After each step, synchronize the state:

```python
from opaque.distributed import sync

grads, clip_state = grad_fn(params, batch, state=clip_state)
clip_state = sync(clip_state)  # aggregate counts across ranks
grads = dist_utils.sum_gradients(grads)
```

`sync()` dispatches to `sync_adaptive_clip_state` internally, which aggregates
`num_clipped` and `total` across ranks, recomputes the global clipping rate,
and updates `clip_norm` to be identical on every device.

## Loss function requirements

The loss function passed to `clipped_grad` must:

1. **Return a scalar** for each example. Opaque differentiates this scalar to
   produce per-example gradients.
2. **Accept batched arguments** at the positions specified by `batch_argnums`.
   These arguments have a batch dimension that `vmap` maps over.
3. **Be compatible with `torch.func`**. Operations using in-place mutation,
   data-dependent control flow, or non-functional layers may fail under `vmap`.
   Gradient checkpointing (`torch.utils.checkpoint`) is incompatible; use
   microbatching instead (see [Known Limitations](../limitations.md)).

## Common patterns

### Functional model conversion

PyTorch models store parameters internally. To use them with `clipped_grad`,
convert to functional form:

```python
from opaque import make_functional, clipped_grad

fmodel, params = make_functional(model)

def loss_fn(params, x, y):
    pred = fmodel(params, x.unsqueeze(0)).squeeze()
    return (pred - y) ** 2

grad_fn, clip_state = clipped_grad(loss_fn, argnums=0, batch_argnums=(1, 2),
                                   l2_clip_norm=1.0)
```

### Separating trainable and frozen parameters

For parameter-efficient methods (LoRA, adapters), separate trainable from
frozen parameters:

```python
fmodel, trainable, frozen = make_functional(model, partition_trainable=True)

def loss_fn(trainable_params, input_ids, labels):
    out = fmodel(trainable_params, frozen, input_ids=input_ids.unsqueeze(0),
                 labels=labels.unsqueeze(0))
    return out.loss

grad_fn, clip_state = clipped_grad(loss_fn, argnums=0, batch_argnums=(1, 2),
                                   l2_clip_norm=1.0)
```

Only the trainable parameters receive per-example gradients. Frozen parameters
are treated as constants by `vmap`.

## API reference

See [Clipping API Reference](../api/clipping.md) for complete function
signatures, all parameters, and return types.
