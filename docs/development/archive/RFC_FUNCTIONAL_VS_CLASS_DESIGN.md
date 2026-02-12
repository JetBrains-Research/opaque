# RFC: Functional vs Class-Based Design for Opaque

**Status:** Design Exploration
**Author:** Claude (based on analysis of JAX-Privacy and jbr-fed-accounting)
**Date:** 2026-02-12

---

## Executive Summary

This RFC compares two architectural approaches for Opaque's DP training library:

1. **Class-Based Approach**: Abstract base classes with concrete implementations (ClippingStrategy, NoiseDistribution, etc.)
2. **Functional Approach**: Higher-order functions that return functions (following JAX-Privacy and jbr-fed-accounting patterns)

**Key Finding**: The functional approach aligns better with:
- PyTorch's `torch.func` paradigm (vmap, grad, functional_call)
- jbr-fed-accounting's compositional design
- JAX-Privacy's proven patterns (BoundedSensitivityCallable)

---

## Background

### Current State

Opaque currently has a **flat functional design**:

```python
# Direct function calls (no composition yet)
clipped_grads = clipped_grad(loss_fn, l2_clip_norm=1.0)(params, batch)
noisy_grads = add_gaussian_noise(clipped_grads, noise_multiplier=1.1, clip_norm=1.0)
```

### User's Challenge

The user proposed exploring a **compositional functional design** similar to jbr-fed-accounting:

```python
# Higher-order functions returning functions
sample = truncated(poisson(sample_rate=0.01), max_batch_size=64, dataset_size=len(data_loader))
clip = adaclip(...)
add_noise = gaussian(noise_multiplier=0.3)

batch = sample(...)
grad = grad_sample(loss, params, batch)
grad = avg(clip(grad))
```

### Alternative Considered

In previous RFC (RFC_OPAQUE_ARCHITECTURE.md), we proposed a **class-based modular design**:

```python
# Abstract base classes with concrete implementations
clipper = L2Clipper(clip_norm=1.0)
noise = Gaussian(noise_multiplier=1.1)

grads = compute_per_example_grads(params, batch)
clipped = clipper.clip(grads)
noisy = noise.add_noise_pytree(clipped, clipper.sensitivity())
```

---

## Analysis: JAX-Privacy's Functional Design

### Core Pattern: BoundedSensitivityCallable

JAX-Privacy uses a **lightweight wrapper dataclass** to track sensitivity:

```python
@dataclasses.dataclass(frozen=True)
class BoundedSensitivityCallable:
    fun: Callable[..., Any]
    l2_norm_bound: float
    has_aux: bool

    def sensitivity(self, neighboring_relation: dp_accounting.NeighboringRelation):
        # Returns 1.0 for ADD/REMOVE, 2.0 for REPLACE_ONE (if rescale_to_unit_norm)
        # Returns l2_norm_bound or 2*l2_norm_bound otherwise
        ...

    def __call__(self, *args, **kwargs):
        return self.fun(*args, **kwargs)
```

### Clipping API

**Higher-order function** that returns the wrapper:

```python
def clipped_grad(
    fun: Callable,
    *,
    l2_clip_norm: float,
    batch_argnums: int | Sequence[int] = 1,
    ...
) -> BoundedSensitivityCallable:
    """Returns a function that computes clipped gradients."""

    # Internal implementation details
    def grad_fn(*args, **kwargs):
        # ... compute, clip, sum ...
        return clipped_value, aux

    # Return wrapped function with sensitivity tracking
    norm_bound = 1.0 if rescale_to_unit_norm else l2_clip_norm
    return BoundedSensitivityCallable(grad_fn, norm_bound, has_aux)
```

**Usage Example**:

```python
# Create the clipped gradient function
grad_fn = jax_privacy.clipped_grad(
    loss_fn,
    l2_clip_norm=1.0,
    batch_argnums=(1, 2),
    return_values=True,
)

# Use it in training loop
grads, aux = grad_fn(params, batch_x, batch_y)

# Access sensitivity for noise calibration
sensitivity = grad_fn.sensitivity(dp_accounting.NeighboringRelation.REPLACE_ONE)
```

### Noise Addition API

JAX-Privacy uses **optax.GradientTransformation** (stateful for PRNG):

```python
# Factory function returning GradientTransformation
privatizer = noise_addition.gaussian_privatizer(
    stddev=noise_multiplier * sensitivity,
    prng_key=jax.random.key(42)
)

# Initialize state
noise_state = privatizer.init(model_params)

# Apply noise in training loop
noisy_grads, noise_state = privatizer.update(grads, noise_state)
```

**Key Insight**: Noise is the **only stateful component** (for PRNG key). Everything else is pure functions.

---

## Comparison: Functional vs Class-Based

### 1. Composition & Chaining

**Functional Approach** (natural composition):

```python
# Each function returns a function
clip_fn = l2_clipper(clip_norm=1.0)
noise_fn = gaussian(noise_multiplier=1.1, clip_norm=1.0)

# Compose directly
process = lambda grads: noise_fn(clip_fn(grads))

# Or in training loop
grads = compute_grads(params, batch)
grads = clip_fn(grads)
grads = noise_fn(grads)
```

**Class-Based Approach** (manual chaining):

```python
# Instantiate strategy objects
clipper = L2Clipper(clip_norm=1.0)
noise = Gaussian(noise_multiplier=1.1)

# Manual method calls
grads = compute_grads(params, batch)
grads = clipper.clip(grads)
grads = noise.add_noise_pytree(grads, clipper.sensitivity())
```

**Winner**: Functional. Composition is built into the language (function calls), not a custom API.

### 2. Type Safety & IDE Support

**Functional Approach**:

```python
def l2_clipper(clip_norm: float) -> Callable[[PyTree], tuple[PyTree, float]]:
    """Returns a function that clips PyTree to L2 norm."""
    def clip(grads: PyTree) -> tuple[PyTree, float]:
        ...
    return clip

# Type hints on returned function are clear
clip_fn: Callable[[PyTree], tuple[PyTree, float]] = l2_clipper(1.0)
```

**Class-Based Approach**:

```python
class ClippingStrategy(ABC):
    @abstractmethod
    def clip(self, grads: PyTree) -> PyTree:
        ...

    @abstractmethod
    def sensitivity(self) -> float:
        ...

# Abstract base class requires runtime checks
clipper: ClippingStrategy = L2Clipper(1.0)
```

**Winner**: Tie. Both support type hints. Classes provide Protocol/ABC guarantees, functions are simpler.

### 3. Alignment with jbr-fed-accounting

**jbr-fed-accounting Rust API** (from previous analysis):

```rust
// Higher-order functions returning Process trait objects
let mechanism = gaussian(noise_multiplier)
    .poisson(sample_rate)
    .truncated(max_batch_size, dataset_size)
    .repeat(num_steps);

let epsilon = mechanism.epsilon_at(delta)?;
```

**Functional PyTorch Equivalent**:

```python
# Mirror the Rust API structure
mechanism = repeat(
    truncated(
        poisson(
            gaussian(noise_multiplier=1.1),
            sample_rate=0.01
        ),
        max_batch_size=64,
        dataset_size=10000
    ),
    count=1000
)

# Could apply to training step
step_fn = mechanism.apply(compute_and_clip_grads)
```

**Class-Based Equivalent**:

```python
# Mechanism composition via wrapper classes
mechanism = RepeatedMechanism(
    TruncatedPoissonMechanism(
        PoissonMechanism(
            GaussianMechanism(noise_multiplier=1.1),
            sample_rate=0.01
        ),
        max_batch_size=64,
        dataset_size=10000
    ),
    count=1000
)

# Apply requires explicit method
step_fn = mechanism.apply_to(compute_and_clip_grads)
```

**Winner**: Functional. Much closer to jbr-fed-accounting's compositional API.

### 4. Extensibility

**Functional Approach**:

```python
# Add new clipping strategy: just write a function
def per_layer_clipper(layer_norms: dict[str, float]) -> Callable[[PyTree], PyTree]:
    def clip(grads: PyTree) -> PyTree:
        return {k: clip_pytree(v, layer_norms[k]) for k, v in grads.items()}
    return clip

# Immediately usable
clip_fn = per_layer_clipper({'transformer': 1.0, 'head': 0.5})
```

**Class-Based Approach**:

```python
# Add new clipping strategy: subclass ABC
class PerLayerClipper(ClippingStrategy):
    def __init__(self, layer_norms: dict[str, float]):
        self.layer_norms = layer_norms

    def clip(self, grads: PyTree) -> PyTree:
        ...

    def sensitivity(self) -> float:
        ...

# Requires more boilerplate
clipper = PerLayerClipper({'transformer': 1.0, 'head': 0.5})
```

**Winner**: Functional. Less boilerplate, faster iteration.

### 5. Testing & Mocking

**Functional Approach**:

```python
# Easy to mock: just replace with another function
def mock_clipper(grads):
    return grads, 1.0  # No clipping for test

# Use in test
with unittest.mock.patch('opaque.clipping.l2_clipper', return_value=mock_clipper):
    ...
```

**Class-Based Approach**:

```python
# Mock via inheritance or composition
class MockClipper(ClippingStrategy):
    def clip(self, grads):
        return grads

    def sensitivity(self):
        return 1.0

# Or use unittest.mock.Mock with spec
clipper = unittest.mock.Mock(spec=ClippingStrategy)
```

**Winner**: Tie. Both approaches support testing well.

### 6. Memory & Performance

**Functional Approach**:

```python
# Closures capture only what they need
def l2_clipper(clip_norm: float):
    def clip(grads):
        # Only clip_norm captured in closure
        return clip_pytree(grads, clip_norm)
    return clip

# Lightweight: just function + captured variables
clip_fn = l2_clipper(1.0)
```

**Class-Based Approach**:

```python
# Class instances have overhead
class L2Clipper:
    def __init__(self, clip_norm: float):
        self.clip_norm = clip_norm
        # Potentially other state

    def clip(self, grads):
        return clip_pytree(grads, self.clip_norm)

# Instance overhead: dict, vtable, etc.
clipper = L2Clipper(1.0)
```

**Winner**: Functional. Closures are more memory-efficient than class instances.

---

## Proposed Functional API Design for Opaque

### Clipping Module

```python
# opaque/clipping/__init__.py

from dataclasses import dataclass
from typing import Callable, Any

@dataclass(frozen=True)
class BoundedSensitivityCallable:
    """Wrapper tracking sensitivity of a function."""
    fun: Callable[..., Any]
    l2_norm_bound: float
    has_aux: bool = False

    def sensitivity(self, neighboring: str = "add_remove") -> float:
        """Return sensitivity based on neighboring relation."""
        multiplier = 2.0 if neighboring == "replace_one" else 1.0
        return self.l2_norm_bound * multiplier

    def __call__(self, *args, **kwargs):
        return self.fun(*args, **kwargs)


def clipped_grad(
    loss_fn: Callable,
    *,
    l2_clip_norm: float,
    batch_argnums: int | tuple[int, ...] = 1,
    rescale_to_unit_norm: bool = False,
    has_aux: bool = False,
    return_values: bool = False,
    return_grad_norms: bool = False,
    **kwargs,
) -> BoundedSensitivityCallable:
    """Create a function that computes clipped per-example gradients.

    Returns:
        BoundedSensitivityCallable: Wrapped function with sensitivity tracking
    """
    # Implementation using torch.func.vmap and torch.func.grad
    ...

    norm_bound = 1.0 if rescale_to_unit_norm else l2_clip_norm
    return BoundedSensitivityCallable(grad_fn, norm_bound, has_aux)


def l2_clipper(
    clip_norm: float,
    rescale_to_unit_norm: bool = False,
) -> Callable[[PyTree], tuple[PyTree, float]]:
    """Create a function that clips PyTree to L2 norm.

    Returns:
        Function that takes PyTree and returns (clipped, norm)
    """
    def clip(grads: PyTree) -> tuple[PyTree, float]:
        return clip_pytree(grads, clip_norm, rescale_to_unit_norm)
    return clip


def per_layer_clipper(
    layer_norms: dict[str, float],
) -> Callable[[PyTree], PyTree]:
    """Create a function that clips each layer to different norms."""
    def clip(grads: PyTree) -> PyTree:
        return {
            k: clip_pytree(v, layer_norms[k])[0]
            for k, v in grads.items()
        }
    return clip


def adaptive_clipper(
    initial_norm: float,
    target_quantile: float = 0.5,
    lr: float = 0.01,
) -> Callable[[PyTree], tuple[PyTree, dict]]:
    """Create a stateful adaptive clipping function.

    Returns:
        Function that maintains internal state for clip norm adaptation
    """
    # State stored in closure
    clip_norm = initial_norm

    def clip(grads: PyTree) -> tuple[PyTree, dict]:
        nonlocal clip_norm

        # Compute per-example norms
        norms = torch.stack([global_norm(g) for g in grads])

        # Update clip norm based on quantile
        target = torch.quantile(norms, target_quantile)
        clip_norm = clip_norm + lr * (target - clip_norm)

        # Clip gradients
        clipped = clip_pytree(grads, clip_norm)[0]

        return clipped, {'clip_norm': clip_norm, 'mean_norm': norms.mean()}

    return clip
```

### Noise Module

```python
# opaque/noise/__init__.py

from typing import Callable, Any
import torch

# Stateless noise functions (regenerate PRNG internally)
def gaussian(
    noise_multiplier: float,
    sensitivity: float = 1.0,
) -> Callable[[PyTree], PyTree]:
    """Create a function that adds Gaussian noise.

    Note: This is stateless. For reproducibility, use gaussian_stateful().

    Returns:
        Function that adds noise: grads -> noisy_grads
    """
    def add_noise(grads: PyTree) -> PyTree:
        return add_gaussian_noise(
            grads,
            noise_multiplier=noise_multiplier,
            sensitivity=sensitivity,
        )
    return add_noise


# Stateful noise functions (for reproducible PRNG sequences)
@dataclass
class PrivatizerState:
    """State for stateful noise addition (tracks PRNG)."""
    generator: torch.Generator
    step: int = 0


def gaussian_stateful(
    noise_multiplier: float,
    sensitivity: float = 1.0,
    seed: int = 0,
) -> tuple[Callable[[PyTree, PrivatizerState], tuple[PyTree, PrivatizerState]], PrivatizerState]:
    """Create a stateful Gaussian noise privatizer.

    Returns:
        (update_fn, initial_state): Tuple of update function and initial state
    """
    def update(grads: PyTree, state: PrivatizerState) -> tuple[PyTree, PrivatizerState]:
        noisy = add_gaussian_noise(
            grads,
            noise_multiplier=noise_multiplier,
            sensitivity=sensitivity,
            generator=state.generator,
        )
        return noisy, PrivatizerState(state.generator, state.step + 1)

    initial_state = PrivatizerState(torch.Generator().manual_seed(seed))
    return update, initial_state


def laplace(
    epsilon: float,
    sensitivity: float = 1.0,
) -> Callable[[PyTree], PyTree]:
    """Create a function that adds Laplace noise (pure DP)."""
    def add_noise(grads: PyTree) -> PyTree:
        scale = sensitivity / epsilon
        return tree_map(
            lambda x: x + torch.distributions.Laplace(0, scale).sample(x.shape),
            grads,
        )
    return add_noise
```

### Sampling Module

```python
# opaque/sampling/__init__.py

def poisson(
    sample_rate: float,
) -> Callable[[DataLoader], Iterator]:
    """Create a Poisson subsampling function."""
    def sample(dataloader: DataLoader) -> Iterator:
        sampler = PoissonSampler(len(dataloader.dataset), sample_rate)
        return DataLoader(
            dataloader.dataset,
            batch_sampler=sampler,
            collate_fn=dataloader.collate_fn,
        )
    return sample


def truncated_poisson(
    sample_rate: float,
    max_batch_size: int,
    dataset_size: int,
) -> Callable[[DataLoader], Iterator]:
    """Create a truncated Poisson subsampling function."""
    def sample(dataloader: DataLoader) -> Iterator:
        sampler = TruncatedPoissonSampler(
            dataset_size,
            sample_rate,
            max_batch_size,
        )
        return DataLoader(
            dataloader.dataset,
            batch_sampler=sampler,
            collate_fn=dataloader.collate_fn,
        )
    return sample
```

### End-to-End Usage

```python
import opaque
import torch
from torch.utils.data import DataLoader

# Setup
model = TinyLLaMA()
params = dict(model.named_parameters())
dataloader = DataLoader(dataset, batch_size=32)

# Define DP mechanisms (higher-order functions)
sample_fn = opaque.sampling.truncated_poisson(
    sample_rate=0.01,
    max_batch_size=64,
    dataset_size=len(dataset),
)

grad_fn = opaque.clipping.clipped_grad(
    loss_fn,
    l2_clip_norm=1.0,
    batch_argnums=(1, 2),
    return_values=True,
)

# Get sensitivity for noise calibration
sensitivity = grad_fn.sensitivity("replace_one")

noise_fn, noise_state = opaque.noise.gaussian_stateful(
    noise_multiplier=1.1,
    sensitivity=sensitivity,
    seed=42,
)

# Training loop
for epoch in range(num_epochs):
    sampled_loader = sample_fn(dataloader)

    for batch in sampled_loader:
        # Compute clipped gradients
        grads, aux = grad_fn(params, batch['input_ids'], batch['labels'])

        # Add noise
        noisy_grads, noise_state = noise_fn(grads, noise_state)

        # Update parameters
        params = optimizer_step(params, noisy_grads)
```

---

## Migration Path from Current API

### Current API (Stage 1 & 2)

```python
# Current: Direct function calls
grads = clipped_grad(loss_fn, l2_clip_norm=1.0)(params, batch)
noisy = add_gaussian_noise(grads, noise_multiplier=1.1, clip_norm=1.0)
```

### Proposed Functional API

```python
# Proposed: Higher-order functions
grad_fn = clipped_grad(loss_fn, l2_clip_norm=1.0)  # Returns BoundedSensitivityCallable
grads = grad_fn(params, batch)

noise_fn = gaussian(noise_multiplier=1.1, sensitivity=grad_fn.sensitivity())
noisy = noise_fn(grads)
```

### Backward Compatibility

Keep existing API as convenience wrappers:

```python
# opaque/__init__.py

# Old API (deprecated but functional)
def compute_clipped_grads(loss_fn, params, batch, l2_clip_norm):
    """Convenience wrapper for backward compatibility."""
    warnings.warn("Use clipped_grad() instead", DeprecationWarning)
    grad_fn = clipping.clipped_grad(loss_fn, l2_clip_norm=l2_clip_norm)
    return grad_fn(params, batch)
```

---

## Recommendations

### ✅ Adopt Functional Approach

**Reasons**:

1. **Alignment**: Matches JAX-Privacy (proven pattern) and jbr-fed-accounting (composability)
2. **PyTorch Idioms**: Natural fit with `torch.func` (vmap, grad, functional_call)
3. **Composition**: Built-in function composition, no custom API needed
4. **Extensibility**: Easy to add new mechanisms without inheritance hierarchy
5. **Memory Efficiency**: Closures lighter than class instances
6. **Research Flexibility**: Quick iteration on new mechanisms

**Trade-offs Accepted**:

- Lose static interface guarantees from ABC (but Protocol can provide this)
- Stateful components (noise PRNG) require explicit state passing (already doing this with functional optimizers)

### 📋 Implementation Plan

**Stage 3: Functional Refactoring** (2-3 weeks)

1. **Week 1**: Introduce `BoundedSensitivityCallable` wrapper
   - Update `clipped_grad()` to return wrapper
   - Add `sensitivity()` method
   - Tests: Verify backward compatibility

2. **Week 2**: Add higher-order noise functions
   - `gaussian()`, `laplace()` returning functions
   - `gaussian_stateful()` with explicit state
   - Tests: Equivalence with current `add_gaussian_noise()`

3. **Week 3**: Add compositional sampling
   - `poisson()`, `truncated_poisson()` returning functions
   - Integration tests with clipping + noise
   - Update tutorial notebook

4. **Deprecation**: Mark old API as deprecated, keep for 2 releases

---

## Open Questions

1. **State Management**: Should we use closure (like adaptive_clipper example) or always explicit state passing?
   - **Proposal**: Explicit state for reproducibility-critical components (noise), closures for convenience (adaptive clipping)

2. **Type Hints**: How to annotate returned functions for best IDE support?
   - **Proposal**: Use `Callable[[PyTree], PyTree]` and `Protocol` for complex cases

3. **Composition Operators**: Should we add explicit composition helpers?
   ```python
   # Option A: Manual composition
   process = lambda grads: noise_fn(clip_fn(grads))

   # Option B: Explicit helper
   process = compose(noise_fn, clip_fn)
   ```
   - **Proposal**: Start with manual, add helpers if patterns emerge

4. **Integration with TorchOpt**: How does functional design interact with TorchOpt's functional optimizers?
   - **Current**: Works well, both use explicit state passing
   - **No changes needed**

---

## References

- **JAX-Privacy**: `/Users/evgri243/Workspaces/external/jax_privacy`
  - `clipping.py`: BoundedSensitivityCallable pattern
  - `noise_addition.py`: Stateful privatizers using optax.GradientTransformation
  - `examples/jax_api_example.py`: End-to-end usage

- **jbr-fed-accounting**: `../../federated-compute/federated-research/packages/dp-accounting/crates/jbr-fed-accounting`
  - Compositional Rust API: `gaussian(nm).poisson(rate).repeat(count)`
  - Process trait for evaluation

- **Opaque Current State**:
  - Stage 1 & 2 complete: Functional clipping + noise
  - Stage 3 goal: TorchOpt integration (already uses functional paradigm)

---

## Conclusion

The **functional approach** is the clear winner for Opaque. It:

- Aligns with both reference libraries (JAX-Privacy, jbr-fed-accounting)
- Fits PyTorch's `torch.func` paradigm naturally
- Enables compositional experimentation (the user's original goal)
- Reduces boilerplate compared to class hierarchies

**Next Step**: User approval, then implement Stage 3 (Functional Refactoring) as outlined above.
