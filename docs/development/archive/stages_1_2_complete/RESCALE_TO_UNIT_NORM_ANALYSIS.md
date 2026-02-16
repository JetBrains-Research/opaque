# Analysis: `rescale_to_unit_norm` and Privacy Guarantees

**Date**: 2026-02-12
**Status**: 🔴 **CRITICAL PRIVACY ISSUE IDENTIFIED**

## The Problem

The `rescale_to_unit_norm` parameter in our clipping functions **directly affects the L2 sensitivity** of the DP mechanism, which determines how much noise is required for a given privacy guarantee. **Currently, we do not account for this in our privacy accounting.**

## Technical Details

### L2 Sensitivity

From JAX-Privacy documentation:

> The L2 sensitivity of the returned function with respect to the batch arguments under add/remove or zero-out differential privacy definitions is guaranteed to be:
> - **1.0** if `rescale_to_unit_norm` is `True`
> - **`l2_clip_norm`** if `rescale_to_unit_norm` is `False`
>
> Under replace-one DP, the sensitivity is doubled (2.0 or 2 * `l2_clip_norm`).

### What `rescale_to_unit_norm` Does

When `rescale_to_unit_norm=True`:
1. Clip gradient to norm C (if needed)
2. **Additionally divide by C**, so final norm ≤ 1.0

```python
# rescale_to_unit_norm=False
clipped_grad = grad * min(1, C / ||grad||)  # norm ≤ C, sensitivity = C

# rescale_to_unit_norm=True
clipped_grad = grad * min(1, C / ||grad||) / C  # norm ≤ 1.0, sensitivity = 1.0
```

### Privacy Implications

For the Gaussian mechanism with noise standard deviation σ:

**Privacy guarantee depends on the noise-to-sensitivity ratio: σ / sensitivity**

- `rescale_to_unit_norm=False`: Need σ = C · z for privacy parameter ε,δ
  - Effective ratio: (C · z) / C = z ✓

- `rescale_to_unit_norm=True`: Need σ = 1.0 · z for same ε,δ
  - Effective ratio: (C · z) / 1.0 = C · z ✗ (too much noise!)
  - **OR** use σ = z (less noise) with ratio z / 1.0 = z ✓

## Current State in Opaque

### Where `rescale_to_unit_norm` Appears

1. **`clip_pytree()`** - Low-level clipping primitive
   ```python
   def clip_pytree(pytree, clip_norm, rescale_to_unit_norm=False, ...):
       # ...
       if rescale_to_unit_norm:
           scale = scale / clip_norm_tensor  # Divide by C
   ```

2. **`clipped_fun()`** - Function transformation
   ```python
   def clipped_fun(fun, l2_clip_norm=1.0, rescale_to_unit_norm=False, ...):
       # Passes rescale_to_unit_norm to clip_pytree
   ```

3. **`clipped_grad()`** - Gradient clipping
   ```python
   def clipped_grad(fun, l2_clip_norm=1.0, rescale_to_unit_norm=False, ...):
       # Passes rescale_to_unit_norm to clipped_fun
   ```

4. **`adaptive_clipped_grad()`** - Adaptive clipping
   ```python
   def adaptive_clipped_grad(fun, **clipped_grad_kwargs):
       # Allows rescale_to_unit_norm via kwargs
   ```

### What We DON'T Have

❌ **No accounting module that adjusts for sensitivity**
- We removed `opaque.accounting` (will be external via `jbr-fed-accounting`)
- Current noise injection in `gaussian()` doesn't know about sensitivity
- Users can pass `rescale_to_unit_norm=True` without realizing they need less noise

## The Danger

**Scenario**: User does this:

```python
# Clips to norm 1.0 AND rescales (sensitivity = 1.0, not 10.0!)
grad_fn = clipped_grad(loss_fn, l2_clip_norm=10.0, rescale_to_unit_norm=True)

# User thinks: "C=10, so I need noise σ = 10 * noise_multiplier"
noise_fn = gaussian(stddev=10.0 * 1.1)  # 😱 WAY TOO MUCH NOISE!

# What they should use: σ = 1.0 * noise_multiplier (sensitivity is 1.0)
noise_fn = gaussian(stddev=1.0 * 1.1)  # ✓ Correct
```

**Result**:
- If they use `stddev=C * noise_multiplier`: **Wasted utility** (100x more noise than needed!)
- If they guess wrong: **Privacy violation** or **terrible accuracy**

## Why `rescale_to_unit_norm` Exists

From JAX-Privacy comments and DP-SGD literature:

### Purpose 1: Simplify Hyperparameter Tuning

When `rescale_to_unit_norm=True`:
- Clipping threshold C doesn't affect the **scale** of updates
- Learning rate and noise multiplier can be tuned independently of C
- Common practice: absorb C into learning rate (re-parameterization)

### Purpose 2: Enable Adaptive Clipping

With adaptive clipping (Andrew et al. 2021):
- C adapts over time: C₁ = 0.1, C₂ = 0.15, C₃ = 0.12, ...
- If `rescale_to_unit_norm=False`: Need to adjust noise σ at each step (σₜ = Cₜ · z)
- If `rescale_to_unit_norm=True`: Noise stays constant (σ = 1.0 · z), **sensitivity is always 1.0**

**This is why adaptive clipping papers use rescaling!**

## Relationship to Adaptive Clipping

### Andrew et al. 2021 Approach

The paper "Differentially Private Learning with Adaptive Clipping" implicitly assumes:

1. Gradients are clipped to adaptive threshold Cₜ
2. **Then rescaled to unit norm** (sensitivity = 1.0 always)
3. Noise with fixed σ = 1.0 · z is added
4. Privacy accounting uses fixed sensitivity = 1.0

Without rescaling:
- Sensitivity changes: S₁ = 0.1, S₂ = 0.15, S₃ = 0.12, ...
- Would need complex accounting for time-varying sensitivity
- Privacy analysis becomes much harder

### What This Means for Us

Our `adaptive_clipped_grad()` implementation:
- ✅ **Does**: Adapt clipping threshold Cₜ correctly
- ❌ **Doesn't**: Rescale to unit norm by default
- ❌ **Doesn't**: Document the sensitivity implications

## Solutions

### Option 1: Make `rescale_to_unit_norm=True` Default for Adaptive Clipping

```python
def adaptive_clipped_grad(
    fun,
    initial_clip_norm=0.1,
    rescale_to_unit_norm=True,  # ← Force this for adaptive clipping!
    **kwargs
):
    # Ensure sensitivity is always 1.0
    # User must use σ = 1.0 * noise_multiplier
```

**Pros**:
- Matches Andrew et al. 2021 approach
- Fixed sensitivity = 1.0 simplifies accounting
- Aligns with JAX-Privacy best practices

**Cons**:
- Breaking change (if users expect sensitivity = Cₜ)
- Need to document noise calibration clearly

### Option 2: Return Sensitivity from Clipping Functions

```python
grad_fn = adaptive_clipped_grad(loss_fn, initial_clip_norm=0.1)
grads = grad_fn(params, x, y)

# Expose current sensitivity
sensitivity = grad_fn.clip_norm if not rescale else 1.0
noise_fn = gaussian(stddev=sensitivity * noise_multiplier)
```

**Pros**:
- Explicit about sensitivity
- Allows both rescaled and non-rescaled modes

**Cons**:
- More complex API
- User can still get it wrong

### Option 3: Integrated Noise Injection

```python
# Clipping + noise in one function
grad_fn = dp_clipped_grad(
    loss_fn,
    initial_clip_norm=0.1,
    noise_multiplier=1.1,
    rescale_to_unit_norm=True,
)
# Returns DP gradients directly
```

**Pros**:
- Impossible to misuse
- Sensitivity handled internally

**Cons**:
- Less composable
- Against our functional design philosophy

### Option 4: Deprecate `rescale_to_unit_norm` Until We Have Accounting

```python
def clip_pytree(..., rescale_to_unit_norm=False):
    if rescale_to_unit_norm:
        raise NotImplementedError(
            "rescale_to_unit_norm requires privacy accounting to be used safely. "
            "This feature is currently disabled until accounting module is available."
        )
```

**Pros**:
- Prevents misuse now
- Can re-enable when `jbr-fed-accounting` is integrated

**Cons**:
- Breaks adaptive clipping (needs rescaling)
- Removes a useful feature

## Recommendation

**Hybrid Approach**:

1. **For `adaptive_clipped_grad()`**: Force `rescale_to_unit_norm=True` and document clearly
   - Sensitivity is always 1.0
   - User must use `noise_stddev = 1.0 * noise_multiplier`
   - Add clear example in docstring

2. **For `clipped_grad()`/`clipped_fun()`**: Add strong warning when `rescale_to_unit_norm=True`
   ```python
   if rescale_to_unit_norm:
       warnings.warn(
           "rescale_to_unit_norm=True changes L2 sensitivity to 1.0. "
           "You must scale noise accordingly: stddev = 1.0 * noise_multiplier, "
           "not l2_clip_norm * noise_multiplier. See documentation for details.",
           PrivacyWarning
       )
   ```

3. **Documentation**: Add comprehensive guide on sensitivity and noise calibration

4. **Wait for accounting**: When `jbr-fed-accounting` is integrated, add automatic sensitivity tracking

## Action Items

- [ ] Update `adaptive_clipped_grad()` to force `rescale_to_unit_norm=True`
- [ ] Add `PrivacyWarning` when `rescale_to_unit_norm=True` in other functions
- [ ] Document sensitivity implications in all clipping function docstrings
- [ ] Add example showing correct noise scaling for both modes
- [ ] Create user guide: "Understanding Sensitivity and Noise in DP-SGD"
- [ ] File issue: "Integrate sensitivity tracking when jbr-fed-accounting available"

## References

1. JAX-Privacy clipping.py formal guarantees
2. Andrew et al. "Differentially Private Learning with Adaptive Clipping" (NeurIPS 2021)
3. Abadi et al. "Deep Learning with Differential Privacy" (CCS 2016)
4. DP-SGD sensitivity calibration best practices
