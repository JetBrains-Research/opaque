# Stage 2: Noise Injection & Privacy Accounting

**Goal**: Add Gaussian noise to aggregated clipped gradients and track privacy budget

**Timeline**: 3 weeks

**Status**: 📋 Ready to start

---

## Overview

Stage 2 implements two critical components for DP-SGD:

1. **Noise addition**: Add calibrated Gaussian noise to clipped gradients
2. **Privacy accounting**: Track privacy budget (ε, δ) and calibrate noise multipliers

After Stage 1 (per-example gradient clipping), we add calibrated noise to the **sum** of clipped gradients and track the
cumulative privacy cost to achieve formal differential privacy guarantees.

### Key Principle

Noise is added to the **aggregated** (summed) gradients, not per-example:

```python
# 1. Per-example clipped gradients (Stage 1)
per_example_grads = clipped_grad(loss_fn, ...)(params, batch_data)

# 2. Aggregate across batch
sum_grads = tree_map(lambda g: g.sum(dim=0), per_example_grads)

# 3. Add noise to sum (Stage 2 - Noise)
sensitivity = clipped_grad.sensitivity()
stddev = noise_multiplier * sensitivity
noisy_sum = add_gaussian_noise(sum_grads, stddev, generator)

# 4. Track privacy (Stage 2 - Accounting)
accountant.step(noise_multiplier, batch_size, dataset_size)
epsilon, delta = accountant.get_privacy_spent()

# 5. Update model
params = params - learning_rate * noisy_sum
```

### Deliverables

1. **`opaque.noise`** (~80 LOC)
  - `add_gaussian_noise()` - Functional API (stateless)
  - PyTree support via `tree_map()`

2. **`opaque.accounting`** (~150 LOC)
  - `PrivacyAccountant` - Wrapper around Google's dp-accounting
  - `calibrate_noise()` - Find noise_multiplier for target (ε, δ)
  - RDP and PLD accounting support

3. **Tests** (~300 LOC)
  - Noise: Unit tests, statistical validation, JAX-Privacy validation
  - Accounting: Budget tracking, calibration, composition tests

---

## Week 1: Noise Implementation

### Days 1-2: Functional Noise API

**File**: `src/opaque/noise.py`

**Key Design Decision**: Use **stateless functional API** instead of JAX's stateful `GradientTransformation`.

**Rationale**:

- JAX-Privacy: Stateful (maintains PRNG key in `optax.GradientTransformation`)
- PyTorch: Can be stateless (pass seed/generator directly)
- Simpler for users: no state management needed

**Implementation**:

```python
"""Gaussian noise addition for differential privacy."""

from typing import Optional
import torch
from opaque.pytree_utils import tree_map


def add_gaussian_noise(
    grads,
    stddev: float,
    generator: Optional[torch.Generator] = None,
):
    """Add Gaussian noise to gradients for differential privacy.

    Adds i.i.d. Gaussian noise N(0, stddev²) to each element of the gradient
    PyTree. The noise standard deviation should be calibrated as:
        stddev = noise_multiplier × sensitivity
    where sensitivity comes from the clipping bound.

    Args:
        grads: Gradient PyTree (dict/tuple of tensors)
        stddev: Standard deviation of Gaussian noise
        generator: Optional torch.Generator for reproducibility

    Returns:
        Noisy gradients with same structure and dtypes as input

    Example:
        >>> # After clipping
        >>> clipped_fn = clipped_grad(loss_fn, l2_clip_norm=1.0)
        >>> per_ex_grads = clipped_fn(params, batch_data)
        >>>
        >>> # Aggregate
        >>> sum_grads = tree_map(lambda g: g.sum(dim=0), per_ex_grads)
        >>>
        >>> # Add noise
        >>> sensitivity = clipped_fn.sensitivity()
        >>> stddev = noise_multiplier * sensitivity
        >>> generator = torch.Generator().manual_seed(42)
        >>> noisy = add_gaussian_noise(sum_grads, stddev, generator)
    """
    if stddev < 0:
        raise ValueError(f"stddev must be non-negative, got {stddev}")

    if stddev == 0:
        # No noise (for testing/debugging)
        return grads

    def add_noise_to_tensor(tensor: torch.Tensor) -> torch.Tensor:
        """Add noise to a single tensor, preserving dtype and device."""
        noise = torch.randn(
            tensor.shape,
            dtype=tensor.dtype,
            device=tensor.device,
            generator=generator,
        ) * stddev
        return tensor + noise

    return tree_map(add_noise_to_tensor, grads)
```

**Why stateless?**

- JAX requires state because `jax.random.split()` needs to track key
- PyTorch `Generator` can be passed directly, no tracking needed
- User controls when to create new generator
- Simpler API: `add_gaussian_noise(grads, stddev, generator)` vs managing state tuple

**Tests to write** (`tests/core/test_noise.py`):

```python
def test_add_noise_single_tensor():
    """Basic noise addition to single tensor."""
    grad = torch.randn(10, 5)
    stddev = 1.0
    generator = torch.Generator().manual_seed(42)

    noisy = add_gaussian_noise(grad, stddev, generator)

    assert noisy.shape == grad.shape
    assert noisy.dtype == grad.dtype
    assert not torch.allclose(noisy, grad)


def test_add_noise_pytree():
    """Noise addition to PyTree (dict of tensors)."""
    grads = {
        'weight': torch.randn(10, 5),
        'bias': torch.randn(10),
    }
    noisy = add_gaussian_noise(grads, stddev=1.0)

    assert set(noisy.keys()) == set(grads.keys())
    assert noisy['weight'].shape == grads['weight'].shape
    assert noisy['bias'].shape == grads['bias'].shape


def test_zero_stddev():
    """stddev=0 should return original gradients."""
    grad = torch.randn(5, 3)
    noisy = add_gaussian_noise(grad, stddev=0.0)
    assert torch.equal(noisy, grad)


def test_negative_stddev_raises():
    """Negative stddev should raise ValueError."""
    grad = torch.randn(5, 3)
    with pytest.raises(ValueError):
        add_gaussian_noise(grad, stddev=-1.0)


def test_dtype_preservation():
    """Noise should preserve input dtype."""
    for dtype in [torch.float32, torch.float64]:
        grad = torch.randn(10, 5, dtype=dtype)
        noisy = add_gaussian_noise(grad, stddev=1.0)
        assert noisy.dtype == dtype
```

### Days 3-4: Statistical Validation

**Goal**: Verify noise has correct statistical properties

**Tests** (`tests/core/test_noise.py` continued):

```python
import scipy.stats
import numpy as np


def test_noise_normality():
    """Verify noise follows N(0, stddev²) using K-S test."""
    stddev = 1.5
    generator = torch.Generator().manual_seed(42)

    # Generate many samples
    n_samples = 10000
    grad = torch.zeros(n_samples)
    noisy = add_gaussian_noise(grad, stddev, generator)
    noise = (noisy - grad).numpy()

    # Kolmogorov-Smirnov test
    # H0: noise ~ N(0, stddev²)
    _, p_value = scipy.stats.kstest(
        noise,
        scipy.stats.norm(0, stddev).cdf
    )

    # Accept at 5% significance level
    assert p_value > 0.025


def test_noise_stddev():
    """Empirical stddev should match specified stddev."""
    stddev = 2.0
    generator = torch.Generator().manual_seed(123)

    # Many samples
    n_samples = 50000
    grad = torch.zeros(n_samples)
    noisy = add_gaussian_noise(grad, stddev, generator)
    noise = noisy - grad

    empirical_std = noise.std().item()

    # Within 5% (statistical variation)
    assert abs(empirical_std - stddev) / stddev < 0.05


def test_reproducibility():
    """Same seed produces same noise."""
    grad = torch.randn(100, 50)
    stddev = 1.0

    gen1 = torch.Generator().manual_seed(42)
    noisy1 = add_gaussian_noise(grad, stddev, gen1)

    gen2 = torch.Generator().manual_seed(42)
    noisy2 = add_gaussian_noise(grad, stddev, gen2)

    assert torch.equal(noisy1, noisy2)


def test_uniqueness():
    """Different calls produce different noise."""
    grad = torch.randn(100, 50)
    generator = torch.Generator().manual_seed(42)

    noisy1 = add_gaussian_noise(grad, 1.0, generator)
    noisy2 = add_gaussian_noise(grad, 1.0, generator)

    assert not torch.allclose(noisy1, noisy2, atol=1e-6)
```

### Day 5: Noise Documentation

**Tasks**:

1. Complete docstrings (Google style)
2. Add examples to docstrings
3. Update module `__init__.py`

---

## Week 2: Privacy Accounting

### Days 1-3: Privacy Accountant Implementation

**File**: `src/opaque/accounting.py`

**Goal**: Wrap Google's `dp-accounting` library for privacy budget tracking

**Implementation**:

```python
"""Privacy accounting for differential privacy."""

from typing import Optional
from dp_accounting import dp_event, rdp


class PrivacyAccountant:
  """Track privacy budget across training steps using RDP accounting.

  Wraps Google's dp-accounting library to provide a simple interface
  for tracking cumulative privacy loss during DP-SGD training.

  Example:
      >>> # Initialize accountant
      >>> accountant = PrivacyAccountant(
      ...     noise_multiplier=1.1,
      ...     sample_rate=0.01,  # batch_size / dataset_size
      ... )
      >>>
      >>> # Training loop
      >>> for epoch in range(10):
      ...     for batch in dataloader:
      ...         # ... DP-SGD step ...
      ...         accountant.step()
      ...
      ...     eps = accountant.get_epsilon(delta=1e-5)
      ...     print(f"Epoch {epoch}: ε={eps:.2f}")
  """

  def __init__(
    self,
    noise_multiplier: float,
    sample_rate: float,
    delta: float = 1e-5,
  ):
    """Initialize privacy accountant.

    Args:
        noise_multiplier: Ratio of noise stddev to sensitivity
        sample_rate: Batch size / dataset size (for subsampling)
        delta: Target delta for (ε, δ)-DP
    """
    self.noise_multiplier = noise_multiplier
    self.sample_rate = sample_rate
    self.delta = delta
    self.steps = 0

    # RDP accountant from dp-accounting
    self._accountant = rdp.RdpAccountant()

  def step(self, num_steps: int = 1) -> None:
    """Record privacy cost of training step(s).

    Args:
        num_steps: Number of steps to account for (default 1)
    """
    event = dp_event.PoissonSampledDpEvent(
      self.sample_rate,
      dp_event.GaussianDpEvent(self.noise_multiplier)
    )

    self._accountant.compose(event, num_steps)
    self.steps += num_steps

  def get_epsilon(self, delta: Optional[float] = None) -> float:
    """Get current privacy budget ε for given δ.

    Args:
        delta: Target delta (uses constructor delta if None)

    Returns:
        Current epsilon value
    """
    if delta is None:
      delta = self.delta

    return self._accountant.get_epsilon(delta)

  def get_privacy_spent(
    self,
    delta: Optional[float] = None
  ) -> tuple[float, float]:
    """Get current privacy budget (ε, δ).

    Args:
        delta: Target delta (uses constructor delta if None)

    Returns:
        Tuple of (epsilon, delta)
    """
    if delta is None:
      delta = self.delta

    epsilon = self.get_epsilon(delta)
    return epsilon, delta


def calibrate_noise(
  target_epsilon: float,
  target_delta: float,
  sample_rate: float,
  num_steps: int,
  epsilon_tolerance: float = 0.01,
) -> float:
  """Find noise_multiplier that achieves target (ε, δ) privacy.

  Uses binary search to find the smallest noise_multiplier that
  satisfies the privacy budget after num_steps of DP-SGD.

  Args:
      target_epsilon: Target epsilon
      target_delta: Target delta
      sample_rate: Batch size / dataset size
      num_steps: Total number of training steps
      epsilon_tolerance: Acceptable error in epsilon

  Returns:
      Calibrated noise_multiplier

  Example:
      >>> # Find noise for ε=3, δ=1e-5 over 1000 steps
      >>> noise_mult = calibrate_noise(
      ...     target_epsilon=3.0,
      ...     target_delta=1e-5,
      ...     sample_rate=0.01,
      ...     num_steps=1000,
      ... )
      >>> print(f"Use noise_multiplier={noise_mult:.2f}")
  """
  # Binary search
  low, high = 0.1, 100.0

  while high - low > 0.01:
    mid = (low + high) / 2

    accountant = PrivacyAccountant(mid, sample_rate, target_delta)
    accountant.step(num_steps)
    epsilon = accountant.get_epsilon(target_delta)

    if epsilon < target_epsilon - epsilon_tolerance:
      # Too much noise, reduce it
      high = mid
    elif epsilon > target_epsilon + epsilon_tolerance:
      # Not enough noise, increase it
      low = mid
    else:
      return mid

  return (low + high) / 2
```

**Tests** (`tests/core/test_accounting.py`):

```python
def test_accountant_initialization():
  """Test basic accountant creation."""
  acc = PrivacyAccountant(
    noise_multiplier=1.0,
    sample_rate=0.01,
    delta=1e-5,
  )
  assert acc.steps == 0


def test_accountant_step():
  """Test privacy budget increases with steps."""
  acc = PrivacyAccountant(1.0, 0.01, 1e-5)

  eps0 = acc.get_epsilon()
  assert eps0 == 0.0  # No steps yet

  acc.step()
  eps1 = acc.get_epsilon()
  assert eps1 > 0.0

  acc.step(10)
  eps2 = acc.get_epsilon()
  assert eps2 > eps1
  assert acc.steps == 11


def test_more_noise_less_privacy_cost():
  """More noise should result in lower epsilon."""
  sample_rate = 0.01
  num_steps = 100

  acc1 = PrivacyAccountant(0.5, sample_rate, 1e-5)
  acc1.step(num_steps)

  acc2 = PrivacyAccountant(1.0, sample_rate, 1e-5)
  acc2.step(num_steps)

  acc3 = PrivacyAccountant(2.0, sample_rate, 1e-5)
  acc3.step(num_steps)

  eps1 = acc1.get_epsilon()
  eps2 = acc2.get_epsilon()
  eps3 = acc3.get_epsilon()

  # More noise = lower epsilon
  assert eps1 > eps2 > eps3


def test_calibrate_noise():
  """Test noise calibration for target privacy."""
  noise_mult = calibrate_noise(
    target_epsilon=3.0,
    target_delta=1e-5,
    sample_rate=0.01,
    num_steps=1000,
  )

  # Verify it achieves target
  acc = PrivacyAccountant(noise_mult, 0.01, 1e-5)
  acc.step(1000)
  eps = acc.get_epsilon()

  assert abs(eps - 3.0) < 0.1  # Within tolerance
```

### Days 4-5: Integration Example

**File**: `examples/03_dp_sgd_with_accounting.py`

**Goal**: Complete DP-SGD example with privacy tracking

```python
"""DP-SGD with Privacy Accounting."""
import torch
from opaque.clipping import clipped_grad
from opaque.noise import add_gaussian_noise
from opaque.accounting import PrivacyAccountant, calibrate_noise
from opaque.pytree_utils import tree_map


def main():
  # Data setup
  n_samples, n_features = 1000, 20
  X = torch.randn(n_samples, n_features)
  y = torch.randn(n_samples)

  # DP-SGD config
  target_epsilon = 3.0
  target_delta = 1e-5
  batch_size = 32
  n_epochs = 10
  lr = 0.01
  l2_clip_norm = 1.0

  # Calculate training steps
  n_steps = (n_samples // batch_size) * n_epochs
  sample_rate = batch_size / n_samples

  # Calibrate noise multiplier
  noise_multiplier = calibrate_noise(
    target_epsilon=target_epsilon,
    target_delta=target_delta,
    sample_rate=sample_rate,
    num_steps=n_steps,
  )

  print(f"Target: ε={target_epsilon}, δ={target_delta}")
  print(f"Calibrated noise_multiplier: {noise_multiplier:.3f}\n")

  # Initialize privacy accountant
  accountant = PrivacyAccountant(
    noise_multiplier=noise_multiplier,
    sample_rate=sample_rate,
    delta=target_delta,
  )

  # Model and clipping
  w = torch.randn(n_features, requires_grad=True)

  def loss_fn(w, x, y):
    return 0.5 * ((x @ w - y) ** 2).mean()

  clipped_fn = clipped_grad(
    loss_fn,
    argnums=0,
    l2_clip_norm=l2_clip_norm,
    batch_argnums=(1, 2),
  )

  sensitivity = clipped_fn.sensitivity()
  stddev = noise_multiplier * sensitivity
  generator = torch.Generator().manual_seed(42)

  # Training loop
  for epoch in range(n_epochs):
    perm = torch.randperm(n_samples)
    X_shuffled = X[perm]
    y_shuffled = y[perm]

    for i in range(n_samples // batch_size):
      # Batch
      start = i * batch_size
      end = start + batch_size
      X_batch = X_shuffled[start:end]
      y_batch = y_shuffled[start:end]

      # 1. Clipped gradients
      per_ex_grads = clipped_fn(w, X_batch, y_batch)

      # 2. Aggregate
      sum_grads = per_ex_grads.sum(dim=0)

      # 3. Add noise
      noisy_grads = add_gaussian_noise(sum_grads, stddev, generator)

      # 4. Update
      w = w - lr * noisy_grads

      # 5. Track privacy
      accountant.step()

    # Report privacy spent
    epsilon, delta = accountant.get_privacy_spent()
    with torch.no_grad():
      train_loss = loss_fn(w, X, y)

    print(f"Epoch {epoch + 1}: loss={train_loss:.4f}, "
          f"ε={epsilon:.2f}, δ={delta:.2e}")

  print(f"\nFinal privacy: ε={epsilon:.2f}, δ={delta:.2e}")
  print(f"Target was: ε={target_epsilon}, δ={target_delta}")


if __name__ == "__main__":
  main()
```

---

## Week 3: JAX Validation + Polish

### Days 1-2: JAX-Privacy Numerical Validation

**File**: `tests/jax_validation/test_noise.py`

**Challenge**: JAX and PyTorch use different PRNG algorithms, so exact match is impossible. Instead, validate *
*statistical equivalence**.

```python
import pytest
import jax
import jax.numpy as jnp
from jax_privacy.noise_addition import gaussian_privatizer
import torch
from opaque.noise import add_gaussian_noise


@pytest.mark.jax_validation
def test_noise_statistical_equivalence():
    """Verify PyTorch noise has same statistics as JAX-Privacy."""
    stddev = 1.5

    # JAX privatizer (stateful)
    jax_privatizer = gaussian_privatizer(
        stddev=stddev,
        prng_key=jax.random.PRNGKey(42)
    )

    # Generate JAX samples
    n_samples = 5000
    shape = (100, 50)
    jax_grads = jnp.zeros(shape)
    jax_state = jax_privatizer.init(jax_grads)

    jax_noise_samples = []
    for _ in range(n_samples):
        noisy, jax_state = jax_privatizer.update(jax_grads, jax_state)
        jax_noise_samples.append(np.array(noisy).flatten())
    jax_noise = np.concatenate(jax_noise_samples)

    # Generate PyTorch samples
    torch_generator = torch.Generator().manual_seed(42)
    torch_grads = torch.zeros(shape)

    torch_noise_samples = []
    for _ in range(n_samples):
        noisy = add_gaussian_noise(torch_grads, stddev, torch_generator)
        torch_noise_samples.append(noisy.numpy().flatten())
    torch_noise = np.concatenate(torch_noise_samples)

    # Compare statistics (not values)
    # 1. Mean ~0
    assert abs(jax_noise.mean()) < 0.05
    assert abs(torch_noise.mean()) < 0.05

    # 2. Std ~stddev
    assert abs(jax_noise.std() - stddev) / stddev < 0.05
    assert abs(torch_noise.std() - stddev) / stddev < 0.05

    # 3. Both pass normality test
    _, jax_p = scipy.stats.kstest(jax_noise, scipy.stats.norm(0, stddev).cdf)
    _, torch_p = scipy.stats.kstest(torch_noise, scipy.stats.norm(0, stddev).cdf)

    assert jax_p > 0.025
    assert torch_p > 0.025
```

### Days 3-4: Accounting Documentation & Additional Tests

**Tasks**:

1. Add comprehensive docstrings to accounting module
2. Test edge cases (very small/large epsilon, different deltas)
3. Test composition over many steps
4. Validate against known DP-SGD privacy bounds

**Additional Tests** (`tests/core/test_accounting.py`):

```python
def test_privacy_amplification_by_subsampling():
  """Smaller batch size should give better privacy."""
  noise_mult = 1.0
  num_steps = 100

  # Large batch (q=0.1)
  acc1 = PrivacyAccountant(noise_mult, 0.1, 1e-5)
  acc1.step(num_steps)

  # Small batch (q=0.01)
  acc2 = PrivacyAccountant(noise_mult, 0.01, 1e-5)
  acc2.step(num_steps)

  eps1 = acc1.get_epsilon()
  eps2 = acc2.get_epsilon()

  # Smaller batch = smaller epsilon (better privacy)
  assert eps2 < eps1


def test_composition_property():
  """Privacy degrades linearly in naive composition."""
  noise_mult = 1.0
  sample_rate = 0.01

  acc = PrivacyAccountant(noise_mult, sample_rate, 1e-5)

  # Measure after 100, 200, 300 steps
  acc.step(100)
  eps_100 = acc.get_epsilon()

  acc.step(100)
  eps_200 = acc.get_epsilon()

  acc.step(100)
  eps_300 = acc.get_epsilon()

  # Should grow (not necessarily linearly due to RDP)
  assert eps_100 < eps_200 < eps_300
```

### Day 5: Complete DP-SGD Example (moved from Week 2)

```python
"""DP-SGD for Linear Regression."""
import torch
from opaque.clipping import clipped_grad
from opaque.noise import add_gaussian_noise
from opaque.pytree_utils import tree_map


def main():
    torch.manual_seed(42)

    # Synthetic data
    n_samples, n_features = 1000, 20
    X = torch.randn(n_samples, n_features)
    true_w = torch.randn(n_features)
    y = X @ true_w + 0.1 * torch.randn(n_samples)

    # Model
    w = torch.randn(n_features)

    # DP-SGD config
    l2_clip_norm = 1.0
    noise_multiplier = 1.1
    batch_size = 32
    n_epochs = 10
    lr = 0.01

    # Clipping
    def loss_fn(w, x, y):
        return 0.5 * ((x @ w - y) ** 2).mean()

    clipped_fn = clipped_grad(
        loss_fn,
        argnums=0,
        l2_clip_norm=l2_clip_norm,
        batch_argnums=(1, 2),
    )

    # Noise
    sensitivity = clipped_fn.sensitivity()
    stddev = noise_multiplier * sensitivity
    generator = torch.Generator().manual_seed(123)

    print(f"Sensitivity: {sensitivity:.4f}")
    print(f"Noise stddev: {stddev:.4f}\n")

    # Training loop
    n_batches = n_samples // batch_size
    for epoch in range(n_epochs):
        perm = torch.randperm(n_samples)
        X_shuffled = X[perm]
        y_shuffled = y[perm]

        epoch_loss = 0.0
        for i in range(n_batches):
            # Batch
            start = i * batch_size
            end = start + batch_size
            X_batch = X_shuffled[start:end]
            y_batch = y_shuffled[start:end]

            # 1. Per-example clipped grads
            per_ex_grads = clipped_fn(w, X_batch, y_batch)

            # 2. Aggregate
            sum_grads = per_ex_grads.sum(dim=0)

            # 3. Add noise
            noisy_grads = add_gaussian_noise(sum_grads, stddev, generator)

            # 4. Update
            w = w - lr * noisy_grads

            # Track loss
            with torch.no_grad():
                epoch_loss += loss_fn(w, X_batch, y_batch).item()

        print(f"Epoch {epoch+1}: loss={epoch_loss/n_batches:.4f}")

    # Final evaluation
    with torch.no_grad():
        final_loss = loss_fn(w, X, y)
        error = torch.norm(w - true_w).item()
        print(f"\nFinal loss: {final_loss:.4f}")
        print(f"Weight error: {error:.4f}")


if __name__ == "__main__":
    main()
```

### Day 5: Final Documentation & Polish

**Tasks**:

1. Update main README with noise + accounting example
2. Ensure all docstrings are complete
3. Update CLAUDE.md with Stage 2 status
4. Update roadmap.md

**Module exports** (`src/opaque/__init__.py`):

```python
from opaque.noise import add_gaussian_noise
from opaque.accounting import PrivacyAccountant, calibrate_noise

__all__ = [
  # ... existing exports
  'add_gaussian_noise',
  'PrivacyAccountant',
  'calibrate_noise',
]
```

---

## Success Criteria

### Functional Requirements - Noise

1. ✅ `add_gaussian_noise()` adds N(0, stddev²) noise to PyTrees
2. ✅ Preserves tensor dtypes and shapes
3. ✅ Reproducible with same Generator seed
4. ✅ Works with nested PyTrees

### Functional Requirements - Accounting

1. ✅ `PrivacyAccountant` tracks ε over training steps
2. ✅ `calibrate_noise()` finds noise_multiplier for target (ε, δ)
3. ✅ Supports RDP accounting via dp-accounting
4. ✅ Handles composition over many steps
5. ✅ Privacy amplification by subsampling works correctly

### Statistical Requirements

1. ✅ Normality test passes (K-S test, p > 0.025)
2. ✅ Empirical stddev matches specified stddev (within 5%)
3. ✅ Same seed produces identical noise
4. ✅ Different calls produce different noise

### Privacy Requirements

1. ✅ Privacy budget increases monotonically with steps
2. ✅ More noise results in smaller epsilon
3. ✅ Smaller batch size (lower sample_rate) gives better privacy
4. ✅ Calibrated noise achieves target (ε, δ) within tolerance

### Code Quality

1. ✅ >90% code coverage
2. ✅ Google-style docstrings with examples
3. ✅ Passes Ruff formatting/linting
4. ✅ Works on CPU and GPU

---

## Design Decisions

### 1. Stateless vs. Stateful

**Decision**: Use **stateless functional API**

**Rationale**:

- JAX-Privacy is stateful because `jax.random.split()` requires tracking PRNG key
- PyTorch `Generator` can be passed directly, no state management needed
- Simpler for users: just pass `generator` argument
- Matches PyTorch conventions (e.g., `torch.randn(..., generator=g)`)

**Comparison**:

```python
# JAX-Privacy (stateful)
privatizer = gaussian_privatizer(stddev=0.5, prng_key=key)
state = privatizer.init(params)
noisy, new_state = privatizer.update(grads, state)  # Returns new state

# Opaque (stateless)
generator = torch.Generator().manual_seed(42)
noisy = add_gaussian_noise(grads, stddev=0.5, generator=generator)  # No state
```

### 2. Integration with Clipping

**Decision**: Keep separate (don't integrate into `clipped_grad()`)

**Rationale**:

- Separation of concerns: clipping and noise are independent
- User may want to inspect clipped gradients before adding noise
- Privacy accounting happens between clipping and noise
- Easier to test independently

**Future**: Could add optional `noise_stddev` parameter in Stage 3+

### 3. Privacy Accounting Integration

**Decision**: Integrate Google's `dp-accounting` library directly

**Rationale**:

- Google's `dp-accounting` is the standard for privacy accounting
- Provides both RDP and PLD (Privacy Loss Distribution) accounting
- Well-tested and maintained by DP experts
- Used by JAX-Privacy and other DP libraries

**Why not implement from scratch?**

- Privacy accounting is complex and error-prone
- Requires deep DP theory knowledge
- Google's library is battle-tested
- Focus our effort on PyTorch integration, not theory

**User workflow**:

```python
# 1. Calibrate noise for target privacy
noise_multiplier = calibrate_noise(
  target_epsilon=3.0,
  target_delta=1e-5,
  sample_rate=batch_size / dataset_size,
  num_steps=total_steps,
)

# 2. Create accountant
accountant = PrivacyAccountant(
  noise_multiplier=noise_multiplier,
  sample_rate=batch_size / dataset_size,
)

# 3. Training loop
for batch in dataloader:
  # ... DP-SGD step ...
  accountant.step()

  # Check privacy budget
  epsilon, delta = accountant.get_privacy_spent()
```

---

## Reference: JAX-Privacy Implementation

**Files examined**:

- `jax_privacy/noise_addition/additive_privatizers.py` (248 LOC)
  - `gaussian_privatizer()` - returns `optax.GradientTransformation`
  - `_iid_normal_noise()` - core noise generation (lines 214-220)
  - Pattern: `init(params) → state`, `update(grads, state) → (noisy_grads, new_state)`

**Key observations**:

1. JAX uses `optax.tree.random_like()` for PyTree noise generation
2. State contains PRNG key that gets split on each call
3. Simple Gaussian is identity matrix factorization
4. Noise added to **sum of clipped grads**, not per-example

**PyTorch simplifications**:

- No state management needed (Generator is explicit)
- Use `tree_map()` instead of `optax.tree.random_like()`
- ~80 LOC vs JAX's 248 LOC (deferred matrix factorization, distributed)

**Deferred for future (beyond Stage 5)**:

- Correlated noise (matrix factorization privatizer)
- StreamingMatrix for memory-efficient temporal correlation
- Distributed noise generation with sharding

---

## What Comes Next

After Stage 2 is complete:

**Stage 3: Privacy Accounting** (2 weeks)

- Wrap Google's `dp-accounting` library
- Implement `calibrate_noise()` for target (ε, δ)
- Privacy budget tracking

**Stage 4: High-Level API** (2 weeks)

- `make_private()` one-line wrapper
- Integration with Hugging Face PEFT
- Automatic LoRA detection

**Stage 5: Polish** (2-3 weeks)

- Tutorial notebooks
- Performance optimization
- PyPI publication

See [Project Roadmap](roadmap.md) for details.
