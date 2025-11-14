# JAX-Privacy Comparison

This document explains the relationship between Opaque and JAX-Privacy, including which API we're porting and why.

---

## JAX-Privacy Architecture

JAX-Privacy provides three distinct APIs for differential privacy. Understanding these helps clarify Opaque's design
decisions.

### 1. Old API (`dp_sgd/` module)

**Characteristics**:

- Object-oriented design (~3,550 LOC)
- Explicit state management
- High flexibility, high complexity
- Used by Keras integration

**Key Classes**:

- `DpsgdGradientComputer`: Orchestrates clipping + noise
- `DpsgdTrainingAccountant`: Privacy budget tracking

**Pros**:

- Maximum control over training loop
- Well-tested in production

**Cons**:

- Verbose API
- Steep learning curve
- Requires manual state management

### 2. New API (`experimental/clipping.py`) ✅ **We're porting this!**

**Characteristics**:

- Functional design (~550 LOC)
- Composable primitives
- Clean separation of concerns
- Built for the future

**Key Function**: `clipped_grad(fun, l2_clip_norm=...)`

**How it works**:

```python
# Computes per-example gradients, clips each, then sums
# Does NOT add noise - that's a separate step!
clipped_grad_fn = clipped_grad(loss_fn, l2_clip_norm=1.0)
grads = clipped_grad_fn(params, data)
```

**Pros**:

- **Composable**: Clipping and noise are separate modules
- **Stateless**: Functions in, values out
- **Easier testing**: No hidden state
- **Microbatching built-in**: Memory efficiency
- **Future-proof**: JAX-Privacy is moving this direction

**Cons**:

- Less mature than old API
- Fewer examples (for now)

### 3. Keras API (High-Level Wrapper)

**Characteristics**:

- One-line integration: `make_private(model, config)`
- Wraps old API internally
- Easiest for users, least flexible

**Example**:

```python
from jax_privacy import make_private, DPKerasConfig

config = DPKerasConfig(
    l2_norm_clip=1.0,
    noise_multiplier=1.1,
    num_microbatches=256
)

model = make_private(model, config)
model.fit(train_data, epochs=10)
```

**Pros**:

- Minimal code changes
- Good for quick prototyping

**Cons**:

- Keras-specific
- Less control over training loop

---

## Why Port the New API?

Opaque is porting JAX-Privacy's **New API** (functional approach). Here's why:

### 1. **Modular Design**

Clipping and noise are separate concerns:

```python
# Opaque approach (following new JAX-Privacy API)
clipped_grads = clipped_grad(loss_fn, l2_clip_norm=1.0)(params, data)
noisy_grads = add_noise(clipped_grads, noise_scale=σ)
```

This allows:

- Testing clipping independently
- Swapping noise mechanisms
- Composing with other DP primitives

### 2. **Functional Paradigm Matches PyTorch**

PyTorch's `torch.func` module is designed for functional transformations:

| JAX                 | PyTorch             |
|---------------------|---------------------|
| `jax.vmap`          | `torch.func.vmap`   |
| `jax.grad`          | `torch.func.grad`   |
| Stateless functions | Stateless functions |

Functional design is a natural fit.

### 3. **LoRA-Friendly**

The new API's features map well to LoRA:

- **`pre_clipping_transform`**: Select LoRA adapters only
- **Microbatching**: Critical for LLM memory constraints
- **`keep_batch_dim=False`**: Useful for per-user clipping (federated learning)

### 4. **Future-Proof**

JAX-Privacy developers indicate the new API is the future direction. By porting this, Opaque stays aligned with upstream
development.

### 5. **Cleaner Testing**

Stateless functions are easier to test:

```python
# No setup/teardown, just call the function
result = clipped_grad(loss_fn, l2_clip_norm=1.0)(params, data)
assert compute_norm(result) <= expected_norm
```

---

## PyTorch Functional Equivalents

Here's how JAX constructs map to PyTorch:

| JAX Function                           | Purpose                        | PyTorch Equivalent                       | Notes                   |
|----------------------------------------|--------------------------------|------------------------------------------|-------------------------|
| `jax.vmap(fn, in_dims=...)`            | Vectorize over batch dimension | `torch.func.vmap(fn, in_dims=...)`       | ✅ Direct equivalent     |
| `jax.grad(fn, argnums=...)`            | Compute gradients              | `torch.func.grad(fn, argnums=...)`       | ✅ Direct equivalent     |
| `jax.value_and_grad(fn, has_aux=True)` | Get value AND gradient         | **No direct equivalent**                 | ⚠️ See workaround below |
| `jax.lax.scan(fn, init, xs)`           | Sequential accumulation        | Custom loop or checkpointing             | Manual implementation   |
| `jax.tree_util.tree_map(fn, tree)`     | Apply to nested structures     | `torch.utils._pytree.tree_map(fn, tree)` | ✅ Direct equivalent     |
| `jax.random.PRNGKey(seed)`             | Reproducible randomness        | `torch.Generator().manual_seed(seed)`    | Different API           |
| `optax.GradientTransformation`         | Optimizer interface            | `torch.optim.Optimizer`                  | Different API           |

### ⚠️ Critical API Difference: `value_and_grad`

One of the most important differences discovered during implementation:

**JAX:**

```python
# jax.value_and_grad returns BOTH value and gradient
grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
(value, aux), grad = grad_fn(params, data)  # Returns ((value, aux), grad)
```

**PyTorch:**

```python
# torch.func.grad returns ONLY gradient (no value!)
grad_fn = torch.func.grad(loss_fn, has_aux=True)
grad, aux = grad_fn(params, data)  # Returns (grad, aux) - NO VALUE!
```

**Workaround in Opaque:**

We created a `_value_and_grad()` helper to provide JAX-compatible behavior:

```python
def _value_and_grad(fun, argnums=0, has_aux=False):
    """Mimic jax.value_and_grad behavior in PyTorch."""
    grad_fn = torch.func.grad(fun, argnums=argnums, has_aux=has_aux)

    if has_aux:
        def wrapper(*args, **kwargs):
            value, aux = fun(*args, **kwargs)  # Call original function
            gradient, _ = grad_fn(*args, **kwargs)  # Get gradient
            return (value, aux), gradient  # JAX format
    else:
        def wrapper(*args, **kwargs):
            value = fun(*args, **kwargs)
            gradient = grad_fn(*args, **kwargs)
            return value, gradient

    return wrapper
```

This is critical for `clipped_grad` which needs both the value (for `return_values=True`) and the gradient.

### ⚠️ PyTorch vmap Cannot Handle None in Namedtuples

Another important difference:

**JAX:**

```python
AuxOutput = namedtuple("AuxOutput", ["values", "grad_norms", "aux"])
aux = AuxOutput(values=tensor, grad_norms=None, aux=None)  # None OK!
jax.vmap(fn_returning_aux, out_dims=0)  # Works fine
```

**PyTorch:**

```python
AuxOutput = namedtuple("AuxOutput", ["values", "grad_norms", "aux"])
aux = AuxOutput(values=tensor, grad_norms=None, aux=None)
torch.func.vmap(fn_returning_aux, out_dims=0)  # ERROR!
# ValueError: vmap(...): must only return Tensors, got type <class 'NoneType'>
```

**Workaround in Opaque:**

Use dicts instead of namedtuples inside vmapped functions, then convert back to namedtuples after vmap:

```python
# Inside grad_fn (before vmap)
aux_dict = {}
if return_values:
    aux_dict["values"] = value
if return_grad_norms:
    aux_dict["grad_norms"] = norm
return result, aux_dict  # Dict OK for vmap!

# After clipped_fun returns (after vmap)
aux_output = AuxiliaryOutput(
    values=aux_dict.get("values"),
    grad_norms=aux_dict.get("grad_norms"),
    aux=aux_dict.get("aux"),
)
```

---

## High-Level API (Stage 4)

While we're porting the functional API first, Opaque will **also** provide a high-level API similar to JAX-Privacy's
Keras API:

```python
from opaque import make_private, DPConfig

config = DPConfig(
    l2_norm_clip=1.0,
    noise_multiplier=1.1,
    target_epsilon=3.0,
    target_delta=1e-5,
)

# Automatic LoRA detection
model = make_private(model, config)

# Standard PyTorch training loop
optimizer.zero_grad()
loss.backward()
optimizer.step()  # DP-SGD happens automatically
```

This combines:

- **Functional core** (composable, testable) from new API
- **User-friendly wrapper** inspired by Keras API

---

## References

- [JAX-Privacy GitHub](https://github.com/google-deepmind/jax_privacy)
- [JAX-Privacy Docs](https://jax-privacy.readthedocs.io/)
- [Experimental Clipping Module](https://github.com/google-deepmind/jax_privacy/blob/main/jax_privacy/src/experimental/clipping.py)
- [PyTorch torch.func Tutorial](https://pytorch.org/tutorials/intermediate/functorch_tutorial.html)
