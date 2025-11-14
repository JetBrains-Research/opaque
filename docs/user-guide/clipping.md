# Per-Sample Gradient Clipping

Per-sample gradient clipping is the **foundation of differential privacy** in deep learning. It bounds the influence any
single training example can have on the model, enabling formal privacy guarantees.

## Why Clipping Matters

In standard neural network training, gradients from different examples can have vastly different magnitudes. A single "
outlier" example could have disproportionate influence on the model update.

**For differential privacy**, this is a problem: If one person's data causes huge gradients, removing that person would
noticeably change the model. **Clipping solves this by ensuring all examples contribute equally bounded gradients.**

### The Core Idea

```python
# Standard SGD (vulnerable)
batch_gradients = [grad(example) for example in batch]
update = mean(batch_gradients)  # Some gradients can dominate!

# DP-SGD with clipping (private)
batch_gradients = [grad(example) for example in batch]
clipped_gradients = [clip_to_norm(g, C) for g in batch_gradients]  # Bounded!
update = mean(clipped_gradients)  # + noise (added separately)
```

Each gradient is clipped to maximum L2 norm **C** (the "clip norm"). This bounds sensitivity: changing one example
changes the sum of gradients by at most **2C**.

## The Three Clipping APIs

Opaque provides three functions for gradient clipping, from low-level to high-level:

### 1. `clip_pytree()` - Low-Level Clipping

Clips a PyTree of tensors to maximum L2 norm:

```python
from opaque import clip_pytree

# Clip a dictionary of gradients
grads = {"weight": torch.tensor([3.0, 4.0]), "bias": torch.tensor([1.0])}
clipped_grads, norm = clip_pytree(grads, clip_norm=1.0)
# norm = 5.099 (sqrt(3^2 + 4^2 + 1^2))
# clipped_grads = {"weight": tensor([0.588, 0.784]), "bias": tensor([0.196])}
```

**Use when**: You already have gradients and just need to clip them.

**See**: [API Reference](../api/core/clipping.md#clip_pytree)

### 2. `clipped_fun()` - Primary API

Wraps any function to clip and sum its outputs across a batch:

```python
from opaque import clipped_fun

def per_example_fn(params, example):
    """Any function that returns PyTree (e.g., gradients)."""
    return compute_gradient(params, example)

# Create clipped version
clipped_fn = clipped_fun(
    per_example_fn,
    batch_argnums=1,  # Argument 1 (example) is batched
    l2_clip_norm=1.0,
)

# Automatically clips per-example outputs and sums
summed_output = clipped_fn(params, batch_of_examples)
```

**Use when**: You have a function that processes single examples and want automatic batching + clipping.

**See**: [API Reference](../api/core/clipping.md#clipped_fun)

### 3. `clipped_grad()` - High-Level API ⭐

The **recommended way** to compute DP gradients:

```python
from opaque import clipped_grad

def loss_fn(params, example):
    """Loss for a single example."""
    x, y = example
    pred = model(params, x)
    return (pred - y) ** 2

# Create DP gradient function
dp_grad_fn = clipped_grad(
    loss_fn,
    argnums=0,  # Differentiate w.r.t. first argument (params)
    batch_argnums=1,  # Second argument (example) is batched
    l2_clip_norm=1.0,
)

# Use like torch.func.grad, but with clipping!
grads = dp_grad_fn(params, (X_batch, y_batch))
```

**Use when**: You want to differentiate a loss function with DP guarantees (most common case).

**See**: [API Reference](../api/core/clipping.md#clipped_grad)

## Understanding Clip Norms

The **clip norm C** is a critical hyperparameter that controls the privacy-utility tradeoff.

### How Clipping Works

For each example's gradient **g**:

1. Compute L2 norm: `||g|| = sqrt(sum(g_i^2))`
2. If `||g|| > C`, scale down: `g_clipped = g * (C / ||g||)`
3. Otherwise, keep unchanged: `g_clipped = g`

### Choosing the Clip Norm

!!! tip "Start with C=1.0"
This is a reasonable default for normalized data. Adjust based on results.

**Effects of clip norm**:

- **C too small**: Most gradients get clipped → slow convergence, poor accuracy
- **C too large**: Little clipping → need more noise → slower convergence OR weaker privacy
- **C just right**: Modest clipping, moderate noise, good accuracy

**Rule of thumb**:

- C=1.0 for normalized data (images, embeddings)
- C=0.1 for very small models
- C=5.0-10.0 for large language models

### Adaptive Clipping (Recommended)

Instead of fixed `C`, use **adaptive clipping** to automatically adjust based on gradient statistics:

```python
from opaque.optimizers import adaptive_clipping
import torchopt

base_optimizer = torchopt.sgd(lr=0.01)
optimizer = adaptive_clipping(
    base_optimizer,
    initial_clip_norm=1.0,
    target_quantile=0.5,  # Clip at median gradient norm
)
```

**See**: [Optimizers Guide](optimizers.md) for details

## Per-Example Gradients with `vmap`

Opaque uses PyTorch's `torch.func.vmap` to compute per-example gradients efficiently:

```python
import torch.func as F

def loss_fn(params, example):
    return compute_loss(params, example)

# Manual per-example gradients (slow!)
grads = [F.grad(loss_fn)(params, ex) for ex in examples]

# Vectorized per-example gradients (fast!)
grad_fn = F.vmap(F.grad(loss_fn), in_dims=(None, 0))
grads = grad_fn(params, examples)  # Batch dimension is 0 for examples
```

`clipped_grad()` handles this automatically via the `batch_argnums` parameter.

## Common Patterns

### Pattern 1: Simple Classification

```python
import torch.nn.functional as F
from opaque import clipped_grad

def loss_fn(params, example):
    x, y = example
    logits = model(params, x.unsqueeze(0)).squeeze(0)
    return F.cross_entropy(logits.unsqueeze(0), y.unsqueeze(0))

dp_grad_fn = clipped_grad(loss_fn, l2_clip_norm=1.0, argnums=0, batch_argnums=1)

# Training loop
for X_batch, y_batch in dataloader:
    grads = dp_grad_fn(params, (X_batch, y_batch))
    # ... add noise and update params
```

### Pattern 2: Regression with Auxiliary Outputs

```python
def loss_fn(params, example):
    x, y = example
    pred = model(params, x)
    loss = (pred - y) ** 2
    aux_info = {"prediction": pred}
    return loss, aux_info

dp_grad_fn = clipped_grad(
    loss_fn,
    l2_clip_norm=1.0,
    argnums=0,
    batch_argnums=1,
    has_aux=True,  # Return auxiliary outputs
)

grads, aux = dp_grad_fn(params, (X_batch, y_batch))
# aux.values contains {"prediction": ...}
```

### Pattern 3: Multiple Arguments

```python
def loss_fn(model_params, feature_params, example):
    """Loss that depends on two parameter sets."""
    x, y = example
    features = extract_features(feature_params, x)
    pred = model(model_params, features)
    return (pred - y) ** 2

dp_grad_fn = clipped_grad(
    loss_fn,
    l2_clip_norm=1.0,
    argnums=(0, 1),  # Clip gradients for both parameter sets
    batch_argnums=2,  # Third argument is batched
)

model_grads, feature_grads = dp_grad_fn(model_params, feature_params, (X_batch, y_batch))
```

## Advanced Features

### Return Gradient Norms

Useful for debugging and adaptive clipping:

```python
dp_grad_fn = clipped_grad(loss_fn, l2_clip_norm=1.0, return_grad_norms=True, ...)

grads, aux = dp_grad_fn(params, batch)
print(f"Gradient norms: {aux.grad_norms}")  # See which examples were clipped
```

### NaN-Safe Clipping

By default, Opaque uses NaN-safe clipping (replaces NaN/inf gradients with zeros):

```python
dp_grad_fn = clipped_grad(loss_fn, nan_safe=True, ...)  # Default
```

Disable for debugging:

```python
dp_grad_fn = clipped_grad(loss_fn, nan_safe=False, ...)  # Will error on NaN
```

### Pre-Clipping Transformation

Apply transformations before clipping (e.g., gradient preprocessing):

```python
def normalize_grads(grads):
    """Normalize each layer's gradients independently."""
    return {k: v / (v.norm() + 1e-8) for k, v in grads.items()}

dp_grad_fn = clipped_grad(
    loss_fn,
    l2_clip_norm=1.0,
    pre_clipping_transform=normalize_grads,
)
```

## Microbatching for Memory Efficiency

Computing per-example gradients requires more memory than standard batching. Use microbatching to reduce memory usage:

```python
# Instead of processing entire batch at once:
# grads = dp_grad_fn(params, large_batch)  # OOM!

# Process in smaller microbatches:
microbatch_size = 8
total_grads = None

for i in range(0, len(batch), microbatch_size):
    microbatch = batch[i:i+microbatch_size]
    grads = dp_grad_fn(params, microbatch)

    if total_grads is None:
        total_grads = grads
    else:
        total_grads = {k: total_grads[k] + grads[k] for k in grads}

# total_grads now contains sum of all clipped gradients
```

**Note**: Microbatching doesn't change privacy guarantees! Each example is still clipped individually.

## Comparison with Standard Gradients

| Feature       | Standard `torch.func.grad`     | Opaque `clipped_grad`                       |
|---------------|--------------------------------|---------------------------------------------|
| **API**       | `grad(loss_fn)(params, batch)` | `clipped_grad(loss_fn, ...)(params, batch)` |
| **Gradients** | Batch-average                  | Per-example, clipped, summed                |
| **Memory**    | Low (batch-level)              | High (per-example)                          |
| **Privacy**   | ❌ None                         | ✅ Bounded sensitivity                       |
| **Use case**  | Standard training              | DP training                                 |

## See Also

- **[Tutorial 01](../tutorials/01_gradient_clipping_from_basics.ipynb)**: Interactive gradient clipping tutorial
- **[Quick Start](../getting-started/quickstart.md)**: Complete DP-SGD example
- **[Noise Addition](noise.md)**: Next step after clipping
- **[API Reference](../api/core/clipping.md)**: Detailed API documentation

---

**Next**: Learn about [Noise Addition](noise.md) to add privacy guarantees
