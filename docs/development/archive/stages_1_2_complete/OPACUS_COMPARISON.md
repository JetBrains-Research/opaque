# Comparison: Opaque vs Opacus Clipping and Normalization

**Date**: 2026-02-12
**Key Finding**: ✅ **Opacus does NOT normalize to unit norm by default**

## Opacus Approach

### Standard Clipping (DPOptimizer)

```python
# From opacus/optimizers/optimizer.py lines 452-456
per_sample_norms = torch.stack(per_param_norms, dim=1).norm(2, dim=1)
per_sample_clip_factor = (
    self.max_grad_norm / (per_sample_norms + 1e-6)
).clamp(max=1.0)

# Line 465: Apply clipping
grad = torch.einsum("i,i...", clip_factor_on_device, grad_sample)

# Line 486: Add noise scaled to max_grad_norm
noise = _generate_noise(
    std=self.noise_multiplier * self.max_grad_norm,  # σ = z · C
    reference=p.summed_grad,
    ...
)
```

**Analysis**:
- `clip_factor = min(1.0, C / ||g||)` - Standard clipping, no normalization
- Clipped gradient norm: `||g_clipped|| ≤ C`
- **Sensitivity: C** (max_grad_norm)
- Noise: `σ = noise_multiplier · C` ✓ **Correctly scaled!**

### Adaptive Clipping (AdaClipDPOptimizer)

```python
# From opacus/optimizers/adaclipoptimizer.py lines 106-108
per_sample_clip_factor = (self.max_grad_norm / (per_sample_norms + 1e-6)).clamp(
    max=1.0
)

# Line 122: Apply clipping (same as DPOptimizer)
grad = torch.einsum("i,i...", clip_factor_on_device, grad_sample)

# Line 132: Calls super().add_noise() which uses:
# noise = _generate_noise(std=self.noise_multiplier * self.max_grad_norm)

# Lines 148-151: Update clipping bound
self.max_grad_norm *= torch.exp(
    -self.clipbound_learning_rate
    * (unclipped_frac - self.target_unclipped_quantile)
)
```

**Analysis**:
- Same clipping as DPOptimizer: `clip_factor = min(1.0, C / ||g||)`
- **No normalization to unit norm!**
- Sensitivity varies: `S_t = C_t` (adaptive max_grad_norm)
- Noise **varies with C_t**: `σ_t = noise_multiplier · C_t` ✓
- Privacy accounting must handle **time-varying sensitivity**

### Key Difference from JAX-Privacy

| Aspect | Opacus | JAX-Privacy (rescale=True) |
|--------|--------|----------------------------|
| **Clipping** | `g * min(1, C/||g||)` | `g * min(1, C/||g||) / C` |
| **Max norm** | `||g_clipped|| ≤ C` | `||g_clipped|| ≤ 1.0` |
| **Sensitivity** | `C` (varies if adaptive) | `1.0` (fixed) |
| **Noise scale** | `σ = z · C` (varies) | `σ = z · 1.0` (fixed) |
| **Adaptive clipping** | Time-varying sensitivity | Fixed sensitivity |

## Implications for Our Implementation

### Current State

We have both options available:
```python
# Option 1: Opacus-style (no normalization)
grad_fn = clipped_grad(loss_fn, l2_clip_norm=C, rescale_to_unit_norm=False)
# Sensitivity = C
# Noise: gaussian(stddev=noise_multiplier * C)

# Option 2: JAX-Privacy-style (with normalization)
grad_fn = clipped_grad(loss_fn, l2_clip_norm=C, rescale_to_unit_norm=True)
# Sensitivity = 1.0
# Noise: gaussian(stddev=noise_multiplier * 1.0)
```

### Pros and Cons

**Opacus Approach (rescale=False)**:
- ✅ More intuitive: clipping norm = max gradient norm
- ✅ Direct control over gradient magnitude
- ✅ Learning rate doesn't need adjustment when changing C
- ❌ For adaptive clipping: time-varying sensitivity complicates accounting
- ❌ Noise scale varies with C (must adjust σ when C changes)

**JAX-Privacy Approach (rescale=True)**:
- ✅ Fixed sensitivity = 1.0 (simpler accounting)
- ✅ For adaptive clipping: no need to adjust noise scale
- ✅ Clipping norm doesn't affect gradient scale (absorbed into LR)
- ❌ Less intuitive: clipping norm ≠ max gradient norm
- ❌ Must adjust learning rate when changing C

## Adaptive Clipping: Which Approach?

### Andrew et al. 2021 Paper

The paper "[Differentially Private Learning with Adaptive Clipping](https://arxiv.org/abs/1905.03871)" (which Opacus implements) doesn't explicitly say whether to normalize.

However, examining the math:
- Algorithm adapts clipping bound C_t
- Privacy analysis assumes **time-varying sensitivity S_t = C_t**
- Noise must scale: `σ_t = z · C_t`

**This is exactly what Opacus does!**

### Why JAX-Privacy Uses Normalization

JAX-Privacy likely uses `rescale_to_unit_norm=True` for:
1. **Simplicity**: Fixed sensitivity = 1.0 simplifies accounting
2. **Flexibility**: Users can change C without adjusting noise
3. **Modularity**: Clipping and noise are independent

But this is a **design choice**, not a requirement.

## Recommendations for Opaque

### Recommendation 1: Default to Opacus Approach (No Normalization)

```python
def adaptive_clipped_grad(
    fun,
    initial_clip_norm=0.1,
    # rescale_to_unit_norm NOT set (defaults to False in clipped_grad)
    **kwargs
):
    """Adaptive gradient clipping (Andrew et al. 2021).

    Returns gradients with L2 sensitivity equal to current clip_norm.
    You must scale noise accordingly:

        noise_fn = gaussian(stddev=noise_multiplier * grad_fn.clip_norm)

    Example:
        >>> grad_fn = adaptive_clipped_grad(loss_fn, initial_clip_norm=0.1)
        >>> for x, y in dataloader:
        ...     grads = grad_fn(params, x, y)
        ...     # Scale noise to current clipping threshold
        ...     noise_fn = gaussian(stddev=1.1 * grad_fn.clip_norm)
        ...     noisy_grads = noise_fn(grads)
        ...     # Update params...
    """
```

**Pros**:
- Matches Opacus (de facto PyTorch standard)
- More intuitive for users familiar with DP-SGD
- Clear: sensitivity = clip_norm (exposed via attribute)

**Cons**:
- User must remember to scale noise with `clip_norm`
- More complex accounting (time-varying sensitivity)

### Recommendation 2: Support Both Modes with Clear Documentation

```python
def adaptive_clipped_grad(
    fun,
    initial_clip_norm=0.1,
    rescale_to_unit_norm=False,  # Explicit default
    **kwargs
):
    """Adaptive gradient clipping (Andrew et al. 2021).

    Args:
        rescale_to_unit_norm: If False (default, Opacus-style), sensitivity
            varies with clip_norm. If True (JAX-Privacy-style), sensitivity
            is fixed at 1.0.

    Returns function with attributes:
        .clip_norm: Current clipping threshold
        .sensitivity: Current L2 sensitivity (clip_norm if rescale=False, 1.0 if True)

    Example (Opacus-style):
        >>> grad_fn = adaptive_clipped_grad(loss_fn, rescale_to_unit_norm=False)
        >>> grads = grad_fn(params, x, y)
        >>> noise_fn = gaussian(stddev=1.1 * grad_fn.clip_norm)  # Varies!

    Example (JAX-Privacy-style):
        >>> grad_fn = adaptive_clipped_grad(loss_fn, rescale_to_unit_norm=True)
        >>> grads = grad_fn(params, x, y)
        >>> noise_fn = gaussian(stddev=1.1 * 1.0)  # Fixed!
    """
```

Add `.sensitivity` property:
```python
# In adaptive_clipped_grad implementation
grad_fn.sensitivity = 1.0 if rescale_to_unit_norm else state["clip_norm"]
```

**Pros**:
- Flexibility: supports both workflows
- Explicit: sensitivity is exposed as attribute
- Clear documentation prevents misuse

**Cons**:
- More API surface to maintain
- Users must understand the difference

### Recommendation 3: Wait for Accounting Integration

Since we're waiting for `jbr-fed-accounting`:
1. **Document current state clearly** in all clipping functions
2. **Add `.sensitivity` property** to make it explicit
3. **Add examples** showing correct noise scaling for both modes
4. **When accounting arrives**: Integrate automatic sensitivity tracking

## Action Items

- [ ] Add `.sensitivity` property to `adaptive_clipped_grad()`
- [ ] Update all clipping function docstrings to document sensitivity
- [ ] Add examples showing noise scaling for `rescale_to_unit_norm=False` and `True`
- [ ] Create user guide: "Understanding Sensitivity in DP-SGD"
- [ ] Update `RESCALE_TO_UNIT_NORM_ANALYSIS.md` with Opacus comparison
- [ ] When accounting available: Add automatic sensitivity tracking

## Conclusion

**Opacus does NOT normalize by default**, which means:
1. ✅ Our default (`rescale_to_unit_norm=False`) matches Opacus
2. ✅ We're providing more flexibility than Opacus (optional normalization)
3. ⚠️ We must document sensitivity clearly to prevent misuse
4. 📋 We should add `.sensitivity` property for explicit tracking

The key insight: **Both approaches are valid**, but require different noise scaling:
- **Opacus-style**: `σ = z · C` (varies)
- **JAX-Privacy-style**: `σ = z · 1.0` (fixed)

Users must know which they're using!
