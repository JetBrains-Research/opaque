# Adaptive Clipping Architecture: Where Should It Live?

**Date**: 2026-02-12
**Question**: Should adaptive clipping be in the **clipping module** or the **optimizer module**?

## Current Implementation (Opaque)

```
src/opaque/
├── clipping/
│   ├── adaptive.py          # ✅ adaptive_clipped_grad() HERE
│   ├── clipped_grad.py
│   └── clipped_fun.py
└── optimizers/              # Empty (deleted wrong implementation)
```

**Our current approach**: Adaptive clipping is a **clipping mechanism**.

## JAX-Privacy Architecture

After examining JAX-Privacy source code, here's how they structure it:

```
jax_privacy/
├── clipping.py              # ✅ clipped_grad(), clipped_fun() - NO ADAPTIVE
├── dp_sgd/
│   ├── gradients.py         # ✅ GradientComputer (handles clipping + noise)
│   └── optim.py             # Noise addition utilities
└── training/
    ├── dp_updater.py        # ✅ Updater (orchestrates gradient computation + optimizer update)
    └── optimizer_config.py  # ✅ OptimizerConfig (wraps Optax optimizers)
```

### Key Insights from JAX-Privacy

1. **Clipping module (`clipping.py`)**:
   - Contains `clipped_grad()`, `clipped_fun()`
   - **No adaptive clipping** - only fixed-threshold clipping
   - Pure functional transformations

2. **GradientComputer (`dp_sgd/gradients.py`)**:
   - Computes clipped gradients AND adds noise
   - Passed `clipping_norm` as parameter (not adaptive)
   - `DpsgdGradientComputer` is stateless

3. **Optimizer (`training/optimizer_config.py`)**:
   - Uses standard Optax optimizers (Adam, SGD, etc.)
   - `AgcOptimizerConfig`: Implements AGC (Adaptive Gradient Clipping by Brock et al.)
   - AGC is **chained with optimizer** using `optax.chain()`

4. **Adaptive Gradient Clipping (AGC)**:
   ```python
   # From optimizer_config.py
   agc_optimizer = optax.chain(
       optax.adaptive_grad_clip(clipping=0.01, eps=1e-3),  # ← AGC HERE
       base_optimizer,  # ← Then optimizer
   )
   ```
   - AGC is from **Brock et al. 2021** (NOT Andrew et al. 2021!)
   - Clips relative to parameter norm: `clip = clipping * ||params||`
   - **Lives in optimizer configuration**, not clipping module

### JAX-Privacy Does NOT Implement Andrew et al. 2021

**Important discovery**: JAX-Privacy does **NOT** implement the Andrew et al. 2021 adaptive clipping algorithm (quantile-based geometric updates).

They only have:
- Fixed-threshold clipping (in `clipping.py`)
- AGC - Adaptive Gradient Clipping (in optimizer, via Optax)

## Opacus Architecture

From our earlier investigation:

```
opacus/
├── optimizers/
│   ├── optimizer.py         # ✅ DPOptimizer (clipping + noise + optimizer)
│   └── adaclipoptimizer.py  # ✅ AdaClipDPOptimizer (Andrew et al. 2021)
```

**Opacus approach**: Adaptive clipping is in the **optimizer**.

### How Opacus Does It

```python
class AdaClipDPOptimizer(DPOptimizer):
    def clip_and_accumulate(self):
        # Standard clipping (inherited from DPOptimizer)
        per_sample_clip_factor = (self.max_grad_norm / per_sample_norms).clamp(max=1.0)
        grad = torch.einsum("i,i...", clip_factor_on_device, grad_sample)

        # Track unclipped count
        self.unclipped_num += (per_sample_clip_factor < 1).sum()

    def update_max_grad_norm(self):
        # Andrew et al. 2021: Geometric update
        unclipped_frac = self.unclipped_num / self.sample_size
        self.max_grad_norm *= torch.exp(
            -self.clipbound_learning_rate * (unclipped_frac - self.target_unclipped_quantile)
        )
```

**Opacus rationale**: Adaptive clipping requires tracking statistics across batches (unclipped count), which is **stateful**. This state naturally lives in the optimizer.

## TorchOpt Architecture

From our investigation:

```
torchopt/
├── base.py                  # GradientTransformation (init_fn, update_fn)
├── clip.py                  # ✅ clip_grad_norm() - NO ADAPTIVE
└── transform/               # Various gradient transformations
```

**TorchOpt approach**:
- Clipping is a **GradientTransformation** (stateless)
- No adaptive clipping implementation
- Follows Optax pattern: `(init_fn, update_fn)` with explicit state-passing

## Analysis: Where Should Adaptive Clipping Live?

### Option 1: In Clipping Module (Current)

```python
# src/opaque/clipping/adaptive.py
grad_fn = adaptive_clipped_grad(loss_fn, initial_clip_norm=1.0)
grads = grad_fn(params, x, y)  # Stateful callable
```

**Pros**:
- Conceptually: "adaptive clipping is a kind of clipping"
- Modular: Can use any optimizer with adaptive clipping
- Separation of concerns: Clipping ≠ Optimization

**Cons**:
- Requires stateful wrapper (breaks functional purity)
- Doesn't match TorchOpt/Optax patterns
- JAX-Privacy doesn't do this

### Option 2: As GradientTransformation (TorchOpt-style)

```python
# src/opaque/transform/adaptive_clip.py
init_fn, update_fn = adaptive_clip(initial_clip_norm=1.0)
state = init_fn()

for x, y in dataloader:
    grads = compute_grads(loss_fn, params, x, y)
    clipped_grads, state = update_fn(grads, state)
```

**Pros**:
- ✅ Matches TorchOpt/Optax patterns perfectly
- ✅ Functional state-passing (pure functions)
- ✅ Composable with `torchopt.chain()`
- ✅ Thread-safe

**Cons**:
- More verbose for users
- State management burden on user
- Not what Opacus does

### Option 3: In Optimizer Module (Opacus-style)

```python
# src/opaque/optimizers/adaptive_clip_optimizer.py
class AdaptiveClipOptimizer:
    def __init__(self, base_optimizer, initial_clip_norm=1.0):
        self.base_opt = base_optimizer
        self.clip_norm = initial_clip_norm
        # ... state

    def step(self, grads):
        # Clip with current threshold
        clipped = clip(grads, self.clip_norm)

        # Update threshold
        self.clip_norm *= exp(...)

        # Apply optimizer
        return self.base_opt.step(clipped)
```

**Pros**:
- Matches Opacus (de facto PyTorch standard for DP)
- State management natural for optimizer
- Less user burden

**Cons**:
- Mixes concerns: clipping + optimization
- Less modular
- Doesn't match TorchOpt patterns
- Against our functional philosophy

## Recommendation

**Move adaptive clipping to a GradientTransformation** (Option 2) because:

1. ✅ **Consistency**: Matches TorchOpt/Optax ecosystem
2. ✅ **Functional purity**: No hidden mutable state
3. ✅ **Composability**: Can chain with other transformations
4. ✅ **Separation of concerns**: Clipping is separate from optimization
5. ✅ **Future-proof**: If TorchOpt adds adaptive clipping, it'll use this pattern

### Proposed API

```python
from opaque.transform import adaptive_clip
from opaque import gaussian
import torchopt

# Create adaptive clipping transformation
clip_transform = adaptive_clip(
    initial_clip_norm=1.0,
    target_quantile=0.5,
    learning_rate=0.2,
)

# Create optimizer
opt = torchopt.adam(lr=1e-3)

# Initialize states
clip_state = clip_transform.init(params)
opt_state = opt.init(params)

# Training loop
for x, y in dataloader:
    # Compute raw gradients
    grads = compute_grads(loss_fn, params, x, y)

    # Apply adaptive clipping (returns new state)
    clipped_grads, clip_state = clip_transform.update(grads, clip_state)

    # Get sensitivity for noise
    sensitivity = clip_state.sensitivity

    # Add DP noise
    noise_fn = gaussian(stddev=1.1 * sensitivity)
    noisy_grads = noise_fn(clipped_grads)

    # Apply optimizer
    updates, opt_state = opt.update(noisy_grads, opt_state, params=params)
    params = torchopt.apply_updates(params, updates)
```

### Implementation Structure

```
src/opaque/
├── clipping/               # Fixed-threshold clipping only
│   ├── clipped_grad.py
│   └── clipped_fun.py
├── transform/              # ✅ NEW: Gradient transformations
│   ├── __init__.py
│   └── adaptive_clip.py    # ✅ adaptive_clip() returns GradientTransformation
└── noise/
    └── gaussian.py
```

## Alternative: Hybrid Approach

If we want to support both patterns:

1. **GradientTransformation** (recommended, functional)
2. **Convenience wrapper** (stateful, for users who prefer Opacus-style)

```python
# Functional API (recommended)
clip_transform = adaptive_clip(initial_clip_norm=1.0)
state = clip_transform.init(params)
clipped, state = clip_transform.update(grads, state)

# Convenience wrapper (stateful, Opacus-style)
from opaque.clipping import adaptive_clipped_grad  # Wrapper around transform
grad_fn = adaptive_clipped_grad(loss_fn, initial_clip_norm=1.0)
grads = grad_fn(params, x, y)  # State hidden internally
```

## Action Items

- [ ] Create `src/opaque/transform/` module
- [ ] Implement `adaptive_clip()` as `GradientTransformation`
- [ ] Define `AdaptiveClipState` with `.sensitivity` property
- [ ] Optionally: Keep `adaptive_clipped_grad()` as convenience wrapper
- [ ] Update documentation with both patterns
- [ ] Add examples showing composition with TorchOpt

## Conclusion

**Adaptive clipping should be a GradientTransformation**, not live in clipping or optimizer modules. This matches the TorchOpt/Optax functional philosophy and provides the best composability.

The key insight: **Andrew et al. 2021 adaptive clipping is NOT standard in JAX-Privacy**. It's a specialized technique that Opacus implements. We should implement it in a way that's consistent with our functional API philosophy (TorchOpt), not the OOP approach (Opacus).
