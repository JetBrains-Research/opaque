# Design Comparison: Code Examples

**Context**: This document provides concrete code examples comparing three design approaches for Opaque.

---

## Three Approaches Compared

1. **Current Opaque (Stage 1-2)**: Flat functional API
2. **Class-Based (RFC_OPAQUE_ARCHITECTURE.md)**: Abstract base classes
3. **Higher-Order Functional (Proposed)**: Functions returning functions

---

## Example 1: Basic DP-SGD Training Loop

### Current Opaque API

```python
import opaque
from opaque import clipped_grad, add_gaussian_noise

# Define components inline
loss_fn = lambda params, x, y: F.cross_entropy(model(x, params), y)

# Training loop
for batch in dataloader:
    # Compute clipped gradients (function returns result directly)
    grads = clipped_grad(
        loss_fn,
        l2_clip_norm=1.0,
        batch_argnums=(1, 2),
    )(params, batch['x'], batch['y'])

    # Add noise (function returns result directly)
    noisy_grads = add_gaussian_noise(
        grads,
        noise_multiplier=1.1,
        clip_norm=1.0,
    )

    # Update
    params = optimizer_step(params, noisy_grads)
```

**Characteristics**:
- ✅ Simple, direct function calls
- ✅ No state management
- ❌ Repetitive parameter passing
- ❌ No composition
- ❌ Hard to configure once, use many times

---

### Class-Based Approach

```python
import opaque
from opaque.mechanisms.clipping import L2Clipper
from opaque.mechanisms.noise import Gaussian

# Configure mechanisms (instantiate strategy objects)
clipper = L2Clipper(clip_norm=1.0)
noise = Gaussian(noise_multiplier=1.1)

loss_fn = lambda params, x, y: F.cross_entropy(model(x, params), y)

# Training loop
for batch in dataloader:
    # Compute per-example gradients (manual vmap)
    per_example_grads = compute_per_example_grads(
        loss_fn,
        params,
        batch['x'],
        batch['y'],
    )

    # Apply clipping strategy
    clipped_grads = clipper.clip(per_example_grads)

    # Apply noise distribution (need sensitivity from clipper)
    noisy_grads = noise.add_noise_pytree(
        clipped_grads,
        sensitivity=clipper.sensitivity(),
    )

    # Update
    params = optimizer_step(params, noisy_grads)
```

**Characteristics**:
- ✅ Configure once, use many times
- ✅ Clear separation of concerns
- ✅ Easy to swap implementations
- ❌ Manual sensitivity passing
- ❌ More verbose (explicit .clip(), .add_noise_pytree() calls)
- ❌ Requires understanding class hierarchy

---

### Higher-Order Functional Approach (Proposed)

```python
import opaque

# Configure mechanisms (higher-order functions return functions)
grad_fn = opaque.clipped_grad(
    loss_fn,
    l2_clip_norm=1.0,
    batch_argnums=(1, 2),
)

# Get sensitivity automatically from wrapped function
sensitivity = grad_fn.sensitivity("replace_one")

# Create noise function
noise_fn = opaque.gaussian(
    noise_multiplier=1.1,
    sensitivity=sensitivity,
)

# Training loop
for batch in dataloader:
    # Use configured functions
    grads = grad_fn(params, batch['x'], batch['y'])
    noisy_grads = noise_fn(grads)

    # Update
    params = optimizer_step(params, noisy_grads)
```

**Characteristics**:
- ✅ Configure once, use many times
- ✅ Sensitivity tracked automatically
- ✅ Natural composition: `noise_fn(grad_fn(...))`
- ✅ Less verbose than classes
- ✅ Mirrors jbr-fed-accounting API
- ✅ Familiar to functional programming users

---

## Example 2: Adaptive Clipping

### Class-Based Approach

```python
from opaque.mechanisms.clipping import AdaptiveClipper

# Instantiate with initial config
clipper = AdaptiveClipper(
    initial_norm=1.0,
    target_quantile=0.5,
    lr=0.01,
)

for batch in dataloader:
    per_example_grads = compute_per_example_grads(...)

    # Clip (mutates internal state)
    clipped_grads, metrics = clipper.clip_and_adapt(per_example_grads)

    # Access current state
    print(f"Current clip norm: {clipper.current_norm}")

    noisy_grads = noise.add_noise_pytree(clipped_grads, clipper.sensitivity())
    params = optimizer_step(params, noisy_grads)
```

**State Management**: Mutable object state (`self.current_norm` updated on each call)

---

### Higher-Order Functional Approach

**Option A: Closure-based state (simpler)**

```python
from opaque import adaptive_clipper

# Returns function with internal state in closure
clip_fn = adaptive_clipper(
    initial_norm=1.0,
    target_quantile=0.5,
    lr=0.01,
)

for batch in dataloader:
    per_example_grads = compute_per_example_grads(...)

    # Clip (updates closure state)
    clipped_grads, metrics = clip_fn(per_example_grads)

    # Access metrics returned
    print(f"Current clip norm: {metrics['clip_norm']}")

    noisy_grads = noise_fn(clipped_grads)
    params = optimizer_step(params, noisy_grads)
```

**Option B: Explicit state passing (more functional)**

```python
from opaque import adaptive_clipper

# Returns (update_fn, initial_state)
clip_fn, clip_state = adaptive_clipper(
    initial_norm=1.0,
    target_quantile=0.5,
    lr=0.01,
)

for batch in dataloader:
    per_example_grads = compute_per_example_grads(...)

    # Clip (returns new state)
    clipped_grads, clip_state = clip_fn(per_example_grads, clip_state)

    # Access state explicitly
    print(f"Current clip norm: {clip_state.clip_norm}")

    noisy_grads = noise_fn(clipped_grads)
    params = optimizer_step(params, noisy_grads)
```

**State Management**:
- Option A: Closure (mutable but encapsulated)
- Option B: Explicit state passing (fully immutable)

**Trade-off**: Option A is more convenient, Option B is more testable/reproducible.

---

## Example 3: Composition

### Class-Based Approach

**Problem**: How to compose clipping + noise?

**Option 1: Manual composition**

```python
clipper = L2Clipper(clip_norm=1.0)
noise = Gaussian(noise_multiplier=1.1)

def process_grads(grads):
    clipped = clipper.clip(grads)
    noisy = noise.add_noise_pytree(clipped, clipper.sensitivity())
    return noisy

# Use in loop
for batch in dataloader:
    grads = compute_per_example_grads(...)
    noisy_grads = process_grads(grads)
    ...
```

**Option 2: Composite class**

```python
class CompositeProcessor:
    def __init__(self, clipper, noise_dist):
        self.clipper = clipper
        self.noise_dist = noise_dist

    def process(self, grads):
        clipped = self.clipper.clip(grads)
        noisy = self.noise_dist.add_noise_pytree(clipped, self.clipper.sensitivity())
        return noisy

processor = CompositeProcessor(clipper, noise)

for batch in dataloader:
    grads = compute_per_example_grads(...)
    noisy_grads = processor.process(grads)
    ...
```

---

### Higher-Order Functional Approach

**Natural composition** via function chaining:

```python
# Option 1: Direct composition
grad_fn = clipped_grad(loss_fn, l2_clip_norm=1.0)
noise_fn = gaussian(noise_multiplier=1.1, sensitivity=grad_fn.sensitivity())

process = lambda params, batch: noise_fn(grad_fn(params, batch))

# Use in loop
for batch in dataloader:
    noisy_grads = process(params, batch)
    ...
```

**Option 2: Helper function**

```python
from opaque import compose

# Compose multiple functions
process = compose(
    noise_fn,   # Applied last
    grad_fn,    # Applied first
)

for batch in dataloader:
    noisy_grads = process(params, batch)
    ...
```

**Option 3: Pipeline builder**

```python
# Fluent API (if we want to go further)
pipeline = (
    opaque.gradient(loss_fn)
    .clip(l2_norm=1.0)
    .noise(gaussian(noise_multiplier=1.1))
    .build()
)

for batch in dataloader:
    noisy_grads = pipeline(params, batch)
    ...
```

---

## Example 4: Per-Layer Clipping

### Class-Based Approach

```python
from opaque.mechanisms.clipping import PerLayerClipper

clipper = PerLayerClipper({
    'transformer.layers': 1.0,
    'transformer.head': 0.5,
})

for batch in dataloader:
    per_example_grads = compute_per_example_grads(...)
    clipped = clipper.clip(per_example_grads)
    ...
```

---

### Higher-Order Functional Approach

```python
from opaque import per_layer_clipper

clip_fn = per_layer_clipper({
    'transformer.layers': 1.0,
    'transformer.head': 0.5,
})

for batch in dataloader:
    per_example_grads = compute_per_example_grads(...)
    clipped = clip_fn(per_example_grads)
    ...
```

**Adding a custom clipping strategy**:

**Class-Based**:

```python
from opaque.mechanisms.clipping import ClippingStrategy

class MyCustomClipper(ClippingStrategy):
    def __init__(self, alpha: float):
        self.alpha = alpha

    def clip(self, grads: PyTree) -> PyTree:
        # Custom logic
        ...

    def sensitivity(self) -> float:
        return self.alpha

# Use
clipper = MyCustomClipper(alpha=2.0)
```

**Functional**:

```python
def my_custom_clipper(alpha: float):
    """Custom clipping function factory."""
    def clip(grads: PyTree) -> PyTree:
        # Custom logic
        ...
    return clip

# Use
clip_fn = my_custom_clipper(alpha=2.0)
```

**Winner**: Functional is less boilerplate (no need to inherit from ABC).

---

## Example 5: Integration with Accounting (Future)

**Scenario**: Emit accounting events during training for later analysis.

### Class-Based Approach

```python
from opaque.accounting import AccountingCollector

# Instantiate mechanisms with accounting callback
collector = AccountingCollector()

clipper = L2Clipper(clip_norm=1.0, accounting_hook=collector.record)
noise = Gaussian(noise_multiplier=1.1, accounting_hook=collector.record)

for batch in dataloader:
    grads = compute_per_example_grads(...)

    # Mechanisms emit events via hook
    clipped = clipper.clip(grads)  # Emits: ClippingEvent(norm=1.0)
    noisy = noise.add_noise_pytree(clipped, clipper.sensitivity())  # Emits: NoiseEvent(...)

    params = optimizer_step(params, noisy)

# Query accounting
epsilon = collector.get_epsilon(delta=1e-5)
```

---

### Higher-Order Functional Approach

**Option 1: Decorators**

```python
from opaque.accounting import with_accounting

collector = AccountingCollector()

# Wrap functions to emit events
grad_fn = with_accounting(
    clipped_grad(loss_fn, l2_clip_norm=1.0),
    collector=collector,
    event_type="clipping",
)

noise_fn = with_accounting(
    gaussian(noise_multiplier=1.1),
    collector=collector,
    event_type="noise",
)

for batch in dataloader:
    grads = grad_fn(params, batch)  # Emits event
    noisy = noise_fn(grads)          # Emits event
    ...

epsilon = collector.get_epsilon(delta=1e-5)
```

**Option 2: Context manager**

```python
from opaque.accounting import AccountingContext

with AccountingContext() as acc:
    grad_fn = clipped_grad(loss_fn, l2_clip_norm=1.0)
    noise_fn = gaussian(noise_multiplier=1.1)

    for batch in dataloader:
        grads = acc.track(grad_fn)(params, batch)
        noisy = acc.track(noise_fn)(grads)
        ...

    epsilon = acc.get_epsilon(delta=1e-5)
```

---

## Summary Table

| Feature | Current API | Class-Based | Functional (Proposed) |
|---------|-------------|-------------|-----------------------|
| **Composition** | Manual | Manual/Helper Class | Natural (function calls) |
| **Configuration** | Per-call | Once (instantiation) | Once (HOF call) |
| **State Management** | Explicit passing | Mutable object | Closure or explicit |
| **Extensibility** | New function | Subclass ABC | New function |
| **Boilerplate** | Low | High (classes) | Low (functions) |
| **Type Safety** | Function hints | ABC/Protocol | Function hints/Protocol |
| **Alignment with jbr-fed-accounting** | Moderate | Low | High |
| **Alignment with JAX-Privacy** | Moderate | Low | High |
| **PyTorch idioms** | Good (torch.func) | Good (nn.Module-like) | Excellent (torch.func) |

---

## Recommendation

**Adopt Higher-Order Functional Approach** for these reasons:

1. **Natural composition**: `noise_fn(grad_fn(...))` reads like math
2. **Less boilerplate**: No class hierarchies to understand
3. **Proven pattern**: JAX-Privacy uses this successfully
4. **Alignment**: Mirrors jbr-fed-accounting's compositional API
5. **Flexibility**: Easy to add new mechanisms without inheritance

**Migration path**: Keep current API as deprecated wrappers for 1-2 releases.
