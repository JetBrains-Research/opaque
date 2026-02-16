# TorchOpt Integration Pattern: Understanding GradientTransformation

## Summary

**TorchOpt/Optax Vision**: Function composition via `chain()`, NOT custom optimizers or wrappers.

**Pattern**: All gradient operations (clipping, noise, adaptive mechanisms) are `GradientTransformation` objects that can be composed using `torchopt.chain()`.

**Recommendation for Adaptive Clipping**: Create a `GradientTransformation` in `src/opaque/transform/` that can be chained with any optimizer.

---

## How TorchOpt Works: Complete Example

### 1. Basic Usage - AdamW

```python
import torch
import torchopt

# Model parameters (PyTree)
params = {
    'weight': torch.randn(10, 5, requires_grad=True),
    'bias': torch.randn(10, requires_grad=True),
}

# 1. CREATE OPTIMIZER - Returns GradientTransformation
adamw = torchopt.adamw(lr=0.01, weight_decay=0.01)
# Type: GradientTransformation(init=<function>, update=<function>)

# 2. INITIALIZE STATE
opt_state = adamw.init(params)
# Returns: ScaleByAdamState(
#     mu={'weight': tensor(...), 'bias': tensor(...)},  # 1st moment
#     nu={'weight': tensor(...), 'bias': tensor(...)},  # 2nd moment
#     count={'weight': tensor(0), 'bias': tensor(0)}    # step count
# )
# Actually a tuple: (ScaleByAdamState, EmptyState, EmptyState)

# Training loop
for step in range(100):
    # Compute loss and gradients (standard PyTorch)
    loss = compute_loss(params, batch)
    grads = torch.autograd.grad(loss, params.values())
    grads = dict(zip(params.keys(), grads))

    # 3. APPLY UPDATE - Returns updates + NEW state
    updates, opt_state = adamw.update(grads, opt_state, params=params)
    # IMPORTANT: opt_state is IMMUTABLE - new object returned

    # 4. APPLY UPDATES TO PARAMETERS
    params = torchopt.apply_updates(params, updates)
```

### 2. State Structure

State is an **immutable tuple** where each element corresponds to a transformation:

```python
# Single transformation (scale_by_adam)
state = ScaleByAdamState(mu=..., nu=..., count=...)

# Multiple transformations (via chain)
adamw_state = (
    EmptyState(),           # from flip_sign_and_add_weight_decay
    ScaleByAdamState(...),  # from scale_by_adam
    EmptyState(),           # from add_decayed_weights
    EmptyState(),           # from scale_by_neg_lr
)
```

### 3. Composition via `chain()`

AdamW is **composed of 4 transformations**:

```python
def adamw(...) -> GradientTransformation:
    return chain(
        flip_sign_and_add_weight_decay(weight_decay=0.0, maximize=False),
        scale_by_adam(b1=0.9, b2=0.999, eps=1e-8),
        add_decayed_weights(weight_decay=0.01, mask=None),
        scale_by_neg_lr(lr=0.01),
    )
```

**How `chain()` works**:
- `init_fn`: Calls each transformation's `init()` and returns tuple of states
- `update_fn`: Passes output of transformation N to input of N+1, threads state through

---

## GradientTransformation Pattern

### Anatomy of a Transformation

Every transformation is a pure function factory returning `(init_fn, update_fn)`:

```python
def my_transformation(hyperparams) -> GradientTransformation:
    """Factory function - captures hyperparameters in closure."""

    def init_fn(params: Params) -> OptState:
        """Initialize state from parameters.

        Args:
            params: PyTree of parameters (dict of tensors)

        Returns:
            Initial state (any Python object, usually NamedTuple)
        """
        # Example: initialize buffers matching param structure
        state = tree_map(lambda p: torch.zeros_like(p), params)
        return MyState(buffer=state, count=0)

    def update_fn(
        updates: Updates,     # Gradient updates (PyTree)
        state: OptState,      # Current state (from previous step)
        *,
        params: Params | None = None,  # Optional current params
        inplace: bool = True,          # Whether to modify tensors in-place
    ) -> tuple[Updates, OptState]:
        """Transform gradients and return new state.

        Returns:
            (transformed_updates, new_state)
        """
        # Transform gradients
        new_updates = transform(updates, state, params)

        # Compute new state (IMMUTABLE - return new object)
        new_state = MyState(
            buffer=update_buffer(state.buffer, updates),
            count=state.count + 1,
        )

        return new_updates, new_state

    return GradientTransformation(init_fn, update_fn)
```

### Example: `clip_grad_norm` (TorchOpt's built-in clipping)

```python
def clip_grad_norm(max_norm: float, norm_type: float = 2.0) -> GradientTransformation:
    """Clip gradient norm (NOT per-example)."""

    def init_fn(params: Params) -> OptState:
        return EmptyState()  # No state needed

    def update_fn(
        updates: Updates,
        state: OptState,
        *,
        params: Params | None = None,
        inplace: bool = True,
    ) -> tuple[Updates, OptState]:
        # Compute global norm
        total_norm = torch.norm(
            torch.stack([torch.norm(g, norm_type) for g in tree_leaves(updates)]),
            norm_type,
        )

        # Clip coefficient
        clip_coef = min(max_norm / (total_norm + 1e-6), 1.0)

        # Apply clipping
        new_updates = tree_map(lambda g: g * clip_coef, updates)

        return new_updates, state  # State unchanged

    return GradientTransformation(init_fn, update_fn)
```

---

## Optax/TorchOpt Vision: Function Composition

### Philosophy

**From Optax docs** (TorchOpt follows same design):

> Optax is built around the concept of **composable gradient transformations**. Each transformation is a pure function that:
> 1. Has no side effects
> 2. Explicitly passes state
> 3. Can be chained with other transformations
>
> This enables building complex optimizers from simple, reusable components.

### Three Ways to Extend (Ranked by Preference)

#### 1. ✅ **Function Composition (PREFERRED)**

Create a new `GradientTransformation` and compose with existing optimizers:

```python
# Define adaptive clipping transformation
adaptive_clip = torchopt.adaptive_clip_grad_norm(
    target_quantile=0.5,
    learning_rate=0.01,
)

# Compose with ANY optimizer
optimizer = torchopt.chain(
    adaptive_clip,           # Custom transformation
    torchopt.adamw(lr=0.01), # Base optimizer
)

# Or compose with other transformations
optimizer = torchopt.chain(
    torchopt.zero_nan_grads(),     # Handle NaNs
    adaptive_clip,                  # Adaptive clipping
    torchopt.clip_grad_norm(10.0), # Additional hard clip
    torchopt.sgd(lr=0.01),         # Base optimizer
)
```

**Why this is preferred**:
- Works with ANY optimizer (SGD, Adam, AdamW, RMSprop, custom)
- Pure functional composition
- No optimizer-specific code
- Matches Optax/TorchOpt philosophy

#### 2. ⚠️ **Wrapper Function (DISCOURAGED)**

Create a factory that returns a chained transformation:

```python
def sgd_with_adaptive_clip(lr, target_quantile, ...):
    """Helper function - just sugar for chain()."""
    return torchopt.chain(
        torchopt.adaptive_clip_grad_norm(target_quantile, ...),
        torchopt.sgd(lr),
    )
```

**Problems**:
- Need separate wrapper for each optimizer (sgd_with_adaptive_clip, adam_with_adaptive_clip, ...)
- Users lose ability to compose with other transformations
- Not flexible

#### 3. ❌ **Custom Optimizer (ANTI-PATTERN)**

Create optimizer-specific classes like Opacus does:

```python
class AdaptiveClipAdamW(GradientTransformation):
    """Don't do this - defeats composability."""
    ...
```

**Problems**:
- Need separate class for each optimizer
- Can't compose with other transformations
- Rigid, non-functional design
- Not the Optax/TorchOpt way

---

## Recommended Design for Adaptive Clipping

### Create `src/opaque/transform/adaptive_clip.py`

```python
"""Adaptive gradient clipping transformation (Andrew et al. 2021)."""

from typing import NamedTuple
import torch
from torchopt.base import GradientTransformation
from torchopt.typing import OptState, Params, Updates
from opaque.utils.pytree import global_norm

class AdaptiveClipState(NamedTuple):
    """State for adaptive clipping."""
    clip_norm: torch.Tensor      # Current clipping bound C_t
    step: int                     # Step count
    unclipped_count: int          # Number of unclipped examples in batch
    sample_count: int             # Batch size

def adaptive_clip_grad_norm(
    target_unclipped_quantile: float,
    clipbound_learning_rate: float,
    initial_clip_norm: float = 1.0,
    max_clip_norm: float = 10.0,
    min_clip_norm: float = 0.1,
) -> GradientTransformation:
    """Adaptive per-example gradient clipping (Andrew et al. 2021).

    This transformation adapts the clipping bound C_t over time using
    geometric multiplicative updates:

        C_{t+1} = C_t * exp(η * (ρ_t - γ))

    where:
        - ρ_t = fraction of examples NOT clipped in batch t
        - γ = target_unclipped_quantile (target for ρ_t)
        - η = clipbound_learning_rate

    Args:
        target_unclipped_quantile: Target fraction of unclipped gradients (γ)
        clipbound_learning_rate: Learning rate for clip bound adaptation (η)
        initial_clip_norm: Initial clipping bound C_0
        max_clip_norm: Maximum allowed clipping bound
        min_clip_norm: Minimum allowed clipping bound

    Returns:
        GradientTransformation that can be chained with any optimizer

    Example:
        >>> import torchopt
        >>> from opaque.transform import adaptive_clip_grad_norm
        >>>
        >>> # Compose with any optimizer
        >>> optimizer = torchopt.chain(
        ...     adaptive_clip_grad_norm(target_unclipped_quantile=0.5,
        ...                             clipbound_learning_rate=0.2),
        ...     torchopt.adamw(lr=0.001),
        ... )
        >>>
        >>> # Or use standalone
        >>> clipper = adaptive_clip_grad_norm(...)
        >>> opt_state = clipper.init(params)
        >>> updates, opt_state = clipper.update(grads, opt_state)

    References:
        - Andrew et al., 2021: https://arxiv.org/abs/1905.03871
    """

    def init_fn(params: Params) -> OptState:
        return AdaptiveClipState(
            clip_norm=torch.tensor(initial_clip_norm),
            step=0,
            unclipped_count=0,
            sample_count=0,
        )

    def update_fn(
        updates: Updates,
        state: OptState,
        *,
        params: Params | None = None,
        inplace: bool = True,
    ) -> tuple[Updates, OptState]:
        # Compute per-example norms (assumes updates are per-example)
        # NOTE: This assumes updates are in shape [batch_size, ...] for each param
        per_example_norms = []
        for update_tensor in updates.values():
            if update_tensor.ndim >= 1:
                # Flatten all dims except first (batch dim)
                flat = update_tensor.view(update_tensor.shape[0], -1)
                norms = torch.norm(flat, p=2, dim=1)
                per_example_norms.append(norms)

        # Global per-example norm
        batch_size = per_example_norms[0].shape[0]
        per_example_norm = torch.stack(per_example_norms, dim=1).norm(2, dim=1)

        # Compute clip factors
        clip_factors = (state.clip_norm / (per_example_norm + 1e-6)).clamp(max=1.0)

        # Count unclipped
        unclipped_count = (clip_factors == 1.0).sum().item()

        # Apply clipping to each parameter
        clipped_updates = {}
        for name, update_tensor in updates.items():
            # Reshape clip_factors to broadcast correctly
            clip_shape = [batch_size] + [1] * (update_tensor.ndim - 1)
            clip_factors_reshaped = clip_factors.view(*clip_shape)

            if inplace:
                clipped_updates[name] = update_tensor.mul_(clip_factors_reshaped)
            else:
                clipped_updates[name] = update_tensor * clip_factors_reshaped

        # Update clipping bound using geometric update
        unclipped_frac = unclipped_count / batch_size
        clip_multiplier = torch.exp(
            clipbound_learning_rate * (unclipped_frac - target_unclipped_quantile)
        )
        new_clip_norm = (state.clip_norm * clip_multiplier).clamp(
            min=min_clip_norm,
            max=max_clip_norm,
        )

        # Return new state (IMMUTABLE)
        new_state = AdaptiveClipState(
            clip_norm=new_clip_norm,
            step=state.step + 1,
            unclipped_count=unclipped_count,
            sample_count=batch_size,
        )

        return clipped_updates, new_state

    return GradientTransformation(init_fn, update_fn)
```

### Usage Examples

```python
# 1. With SGD
optimizer = torchopt.chain(
    adaptive_clip_grad_norm(target_unclipped_quantile=0.5, clipbound_learning_rate=0.2),
    torchopt.sgd(lr=0.01),
)

# 2. With Adam
optimizer = torchopt.chain(
    adaptive_clip_grad_norm(target_unclipped_quantile=0.5, clipbound_learning_rate=0.2),
    torchopt.adam(lr=0.001),
)

# 3. With AdamW
optimizer = torchopt.chain(
    adaptive_clip_grad_norm(target_unclipped_quantile=0.5, clipbound_learning_rate=0.2),
    torchopt.adamw(lr=0.001, weight_decay=0.01),
)

# 4. Complex composition
optimizer = torchopt.chain(
    torchopt.zero_nan_grads(),          # Handle NaNs first
    adaptive_clip_grad_norm(...),        # Adaptive clipping
    torchopt.clip_grad_norm(10.0),      # Hard upper bound
    torchopt.adamw(lr=0.001),           # Base optimizer
)

# Initialize and use
state = optimizer.init(params)
for step in range(1000):
    grads = compute_gradients(params, batch)
    updates, state = optimizer.update(grads, state, params=params)
    params = torchopt.apply_updates(params, updates)
```

---

## Integration with Noise Injection

Noise injection can ALSO be a `GradientTransformation`:

```python
# src/opaque/transform/gaussian_noise.py

def add_gaussian_noise(
    noise_multiplier: float,
    l2_sensitivity: float,
) -> GradientTransformation:
    """Add Gaussian noise to gradients for DP-SGD."""

    def init_fn(params: Params) -> OptState:
        return EmptyState()

    def update_fn(
        updates: Updates,
        state: OptState,
        *,
        params: Params | None = None,
        inplace: bool = True,
    ) -> tuple[Updates, OptState]:
        stddev = noise_multiplier * l2_sensitivity

        def add_noise(t):
            noise = torch.randn_like(t) * stddev
            return t.add_(noise) if inplace else t + noise

        noisy_updates = tree_map(add_noise, updates)
        return noisy_updates, state

    return GradientTransformation(init_fn, update_fn)
```

**Full DP-SGD pipeline**:

```python
dp_optimizer = torchopt.chain(
    adaptive_clip_grad_norm(...),           # Adaptive clipping (C_t varies)
    add_gaussian_noise(noise_multiplier=1.1),  # Noise injection
    torchopt.sgd(lr=0.01),                  # Base optimizer
)
```

---

## Comparison with Current Design

### Current Design (Stateful Callable)

```python
# src/opaque/clipping/adaptive.py

clipped_grad_fn = adaptive_clipped_grad(
    loss_fn,
    target_unclipped_quantile=0.5,
    clipbound_learning_rate=0.2,
)

# Returns AdaptiveSensitivityCallable with mutable state
for step in range(1000):
    grads = clipped_grad_fn(params, batch_x, batch_y)  # State mutated internally
    # Apply optimizer separately
    optimizer.step()
```

**Problems**:
1. Mutable state (not functional)
2. Separate from optimizer (not composable)
3. Hard to combine with other transformations
4. Doesn't match TorchOpt philosophy

### Proposed Design (GradientTransformation)

```python
# src/opaque/transform/adaptive_clip.py

optimizer = torchopt.chain(
    adaptive_clip_grad_norm(target_unclipped_quantile=0.5, clipbound_learning_rate=0.2),
    torchopt.adamw(lr=0.001),
)

state = optimizer.init(params)
for step in range(1000):
    grads = compute_grads(params, batch)
    updates, state = optimizer.update(grads, state, params=params)  # State explicit
    params = torchopt.apply_updates(params, updates)
```

**Benefits**:
1. Pure functional (immutable state)
2. Composable with any optimizer
3. Matches TorchOpt/Optax design
4. Easy to add other transformations

---

## Next Steps

1. **Remove `src/opaque/clipping/adaptive.py`** (stateful design)
2. **Create `src/opaque/transform/`** package
3. **Implement `adaptive_clip_grad_norm()`** as `GradientTransformation`
4. **Optionally**: Create `add_gaussian_noise()` as `GradientTransformation`
5. **Update examples** to use `torchopt.chain()`

This matches the Optax/TorchOpt vision: **composable, pure functional transformations**.
