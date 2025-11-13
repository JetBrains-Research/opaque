# Stage 3: Functional Optimizers & DP-Adam-AC

**Goal**: Implement complete DP-SGD/DP-Adam training with TorchOpt, including state-of-the-art DP-Adam-AC (Adaptive
Clipping)

**Timeline**: 4-5 weeks

**Status**: 📋 Planning Complete → Ready to Implement

**Key Innovation**: Include **DP-Adam-AC** from [arxiv:2510.05288](https://arxiv.org/abs/2510.05288) - state-of-the-art
adaptive clipping

---

## Overview

Stage 3 builds on clipping (Stage 1) and noise+accounting (Stage 2) to create production-ready DP training workflows.
We'll use **TorchOpt** as our functional optimizer library, following the **Optax pattern** (same as JAX-Privacy).

### Why TorchOpt?

TorchOpt provides JAX-like functional optimizers for PyTorch:

- ✅ **Functional API**: Stateless `GradientTransformation` pattern matching Optax
- ✅ **Composable**: Chain transformations like `chain(scale_by_adam(), scale_by_lr())`
- ✅ **Efficient**: Fused operations and PyTree support
- ✅ **Standard Optimizers**: SGD, Adam, AdamW, RMSprop with momentum
- ✅ **Meta-Learning Ready**: Differentiable optimization support

**Pattern**:
```python
# TorchOpt follows Optax's GradientTransformation pattern
optimizer = torchopt.adam(lr=0.001, betas=(0.9, 0.999))  # Returns GradientTransformation
opt_state = optimizer.init(params)                        # Initialize state
updates, opt_state = optimizer.update(grads, opt_state)  # Get parameter updates
params = torchopt.apply_updates(params, updates)          # Apply updates
```

### What is DP-Adam-AC?

**DP-Adam-AC** (Adaptive Clipping) from [Zuo et al., 2024](https://arxiv.org/abs/2510.05288) improves upon standard
DP-Adam by:

1. **Adaptive Clipping**: Dynamically adjusts gradient clipping threshold based on percentiles of recent gradient norms
2. **Clip-Rate-Based LR Scaling**: Modulates learning rate when clipping is too frequent or rare
3. **EMA Parameter Smoothing**: Uses exponential moving average for better privacy-utility tradeoff
4. **No Weight Decay**: DP noise acts as implicit regularization

**Key Benefits**:

- 📈 Better accuracy than fixed clipping (1-3% improvement on benchmarks)
- 🎯 Self-tuning: adapts to gradient scale changes during training
- 🔒 Same privacy guarantees as standard DP-Adam
- 🚀 Practical for LLM fine-tuning

---

## Deliverables

### 1. `opaque.optimizers` Package (~400 LOC)

**File Structure**:
```
src/opaque/optimizers/
├── __init__.py          # Public API exports
├── base.py              # Base DP optimizer interface (~80 LOC)
├── dp_sgd.py            # DP-SGD implementation (~60 LOC)
├── dp_adam.py           # Basic DP-Adam (~60 LOC)
└── dp_adam_ac.py        # DP-Adam-AC with adaptive clipping (~200 LOC)
```

**Key Components**:

#### `base.py` - Base DP Optimizer Interface
```python
class DPOptimizerState(NamedTuple):
    """State for DP optimizer."""
    opt_state: Any  # TorchOpt optimizer state
    accountant: Any  # Privacy accountant (RDP/PLD)
    noise_gen: torch.Generator  # For reproducible noise
    step: int  # Training step counter

def make_dp_optimizer(
    base_optimizer: GradientTransformation,
    *,
    l2_clip_norm: float,
    noise_multiplier: float,
    sample_rate: float,
    target_delta: float,
    accountant_type: str = "rdp",
) -> tuple[Callable, Callable]:
    """Wrap TorchOpt optimizer with DP functionality.

    Returns:
        (init_fn, step_fn) where:
          - init_fn(params) -> DPOptimizerState
          - step_fn(params, grads, state) -> (new_params, new_state, metrics)
    """
```

#### `dp_sgd.py` - Basic DP-SGD

```python
def dp_sgd(
    learning_rate: float,
    momentum: float = 0.0,
    *,
    l2_clip_norm: float,
    noise_multiplier: float,
    sample_rate: float,
    target_delta: float,
    accountant_type: str = "rdp",
) -> tuple[Callable, Callable]:
    """Create DP-SGD optimizer.

    Example:
        >>> init_fn, step_fn = dp_sgd(
        ...     learning_rate=0.1,
        ...     l2_clip_norm=1.0,
        ...     noise_multiplier=1.1,
        ...     sample_rate=0.01,
        ...     target_delta=1e-5,
        ... )
        >>> state = init_fn(params)
        >>> for batch in dataloader:
        ...     grads = compute_grads(batch)
        ...     params, state, metrics = step_fn(params, grads, state)
        ...     print(f"ε = {metrics['epsilon']:.2f}")
    """
    base_opt = torchopt.sgd(lr=learning_rate, momentum=momentum)
    return make_dp_optimizer(
        base_opt,
        l2_clip_norm=l2_clip_norm,
        noise_multiplier=noise_multiplier,
        sample_rate=sample_rate,
        target_delta=target_delta,
        accountant_type=accountant_type,
    )
```

#### `dp_adam.py` - Basic DP-Adam
```python
def dp_adam(
    learning_rate: float = 1e-3,
    betas: tuple[float, float] = (0.9, 0.999),
    eps: float = 1e-8,
    *,
    l2_clip_norm: float,
    noise_multiplier: float,
    sample_rate: float,
    target_delta: float,
    accountant_type: str = "rdp",
) -> tuple[Callable, Callable]:
    """Create DP-Adam optimizer (standard, fixed clipping)."""
    base_opt = torchopt.adam(lr=learning_rate, betas=betas, eps=eps, weight_decay=0.0)
    return make_dp_optimizer(
        base_opt,
        l2_clip_norm=l2_clip_norm,
        noise_multiplier=noise_multiplier,
        sample_rate=sample_rate,
        target_delta=target_delta,
        accountant_type=accountant_type,
    )
```

#### `dp_adam_ac.py` - DP-Adam-AC (Adaptive Clipping) 🆕

**Core Algorithm** (from paper):

```
Algorithm: DP-Adam-AC
1. Compute per-example gradients (via vmap)
2. Clip each gradient to C (adaptive threshold)
3. Add Gaussian noise N(0, (σ·C)²I)
4. Update with Adam: θ ← θ - γ·η·m̂/(√v̂ + ε)
5. Update EMA params: θ̂ ← d·θ̂ + (1-d)·θ
6. Track clip rate: ρ = fraction of clipped gradients
7. Adjust C ← Percentile(recent_norms, q=100·(1-ρ*))
8. Adjust γ based on ρ (increase if too low, decrease if too high)
```

**Implementation**:
```python
class AdaptiveClipState(NamedTuple):
    """State for adaptive clipping."""
    unit_norm_buffer: list[float]  # Recent gradient unit norms
    current_clip_norm: float  # Current adaptive C
    lr_multiplier: float  # Current γ multiplier
    ema_params: PyTree  # EMA-smoothed parameters

def dp_adam_ac(
    learning_rate: float = 3e-4,
    betas: tuple[float, float] = (0.9, 0.999),
    eps: float = 1e-8,
    *,
    initial_clip_norm: float = 3.0,
    noise_multiplier: float,
    sample_rate: float,
    target_delta: float,
    # Adaptive clipping params
    target_clip_rate: float = 0.20,
    history_size: int = 1000,
    clip_norm_range: tuple[float, float] = (0.1, 10.0),
    # Dynamic LR params
    lr_multiplier_range: tuple[float, float] = (0.1, 2.0),
    lr_adjust_factors: tuple[float, float] = (1.01, 0.995),
    clip_rate_thresholds: tuple[float, float] = (0.10, 0.30),
    # EMA params
    ema_decay: float = 0.999,
    accountant_type: str = "rdp",
) -> tuple[Callable, Callable]:
    """Create DP-Adam-AC optimizer with adaptive clipping.

    Based on: Zuo et al., "DP-Adam-AC: Privacy-preserving Fine-Tuning of
    Localizable Language Models Using Adam Optimization with Adaptive Clipping"
    https://arxiv.org/abs/2510.05288

    Args:
        learning_rate: Base learning rate (η_base in paper)
        betas: Adam momentum decay rates (β₁, β₂)
        eps: Adam numerical stability constant (ε)
        initial_clip_norm: Initial clipping threshold C
        noise_multiplier: DP noise scale (σ)
        sample_rate: Batch sampling rate
        target_delta: Target δ for (ε, δ)-DP
        target_clip_rate: Target fraction of clipped gradients (ρ*)
        history_size: Number of recent norms to track (H)
        clip_norm_range: (C_min, C_max) bounds for adaptive C
        lr_multiplier_range: (γ_min, γ_max) bounds for LR scaling
        lr_adjust_factors: (↑, ↓) multiplicative adjustments
        clip_rate_thresholds: (ρ_low, ρ_high) for LR adjustment
        ema_decay: EMA smoothing factor (d)
        accountant_type: "rdp" or "pld"

    Returns:
        (init_fn, step_fn) for DP-Adam-AC optimization

    Example:
        >>> # Setup
        >>> init_fn, step_fn = dp_adam_ac(
        ...     learning_rate=3e-4,
        ...     initial_clip_norm=3.0,
        ...     noise_multiplier=1.1,
        ...     sample_rate=0.01,
        ...     target_delta=1e-5,
        ...     target_clip_rate=0.20,  # Aim for 20% clipping
        ... )
        >>>
        >>> # Training loop
        >>> state = init_fn(params)
        >>> for epoch in range(10):
        ...     for batch in dataloader:
        ...         grads = compute_grads(batch)
        ...         params, state, metrics = step_fn(params, grads, state)
        ...
        ...         # Monitor adaptive behavior
        ...         print(f"ε={metrics['epsilon']:.2f}, "
        ...               f"C={metrics['clip_norm']:.2f}, "
        ...               f"ρ={metrics['clip_rate']:.2%}, "
        ...               f"γ={metrics['lr_multiplier']:.2f}")
    """
```

### 2. `opaque.adaptive` Package (~150 LOC) 🆕

Helper functions for adaptive clipping:

```
src/opaque/adaptive/
├── __init__.py
├── clip_buffer.py      # Gradient norm history tracking (~60 LOC)
└── lr_scheduler.py     # Clip-rate-based LR scaling (~90 LOC)
```

**`clip_buffer.py`**:
```python
class ClipNormBuffer:
    """Efficient buffer for tracking gradient norms."""

    def __init__(self, capacity: int = 1000):
        self.capacity = capacity
        self.buffer: deque[float] = deque(maxlen=capacity)

    def update(self, norms: torch.Tensor, batch_sizes: torch.Tensor):
        """Add unit-normalized gradient norms."""
        unit_norms = norms / torch.maximum(batch_sizes, torch.ones_like(batch_sizes))
        self.buffer.extend(unit_norms.tolist())

    def get_percentile(self, q: float) -> float:
        """Compute q-th percentile of buffer."""
        if not self.buffer:
            return 1.0
        return float(torch.quantile(torch.tensor(list(self.buffer)), q))

    def get_clip_rate(self, threshold: float) -> float:
        """Fraction of norms exceeding threshold."""
        if not self.buffer:
            return 0.0
        return sum(1 for n in self.buffer if n > threshold) / len(self.buffer)
```

### 3. Tutorial 03 - Updated (Jupyter notebook)

**Current**: Basic DP-SGD training (already exists)

**Add Section 7**: DP-Adam-AC Comparison

- Train with DP-SGD, DP-Adam, DP-Adam-AC
- Visualize adaptive clip norm evolution
- Show clip rate vs training progress
- Compare final accuracies

### 4. Tests (~400 LOC)

```
tests/optimizers/
├── __init__.py
├── test_dp_sgd.py           # Basic DP-SGD tests (~80 LOC)
├── test_dp_adam.py          # Basic DP-Adam tests (~80 LOC)
├── test_dp_adam_ac.py       # DP-Adam-AC tests (~120 LOC)
└── test_adaptive_clip.py    # Adaptive clipping logic (~120 LOC)
```

**Key Tests**:

1. **Correctness**: DP-Adam-AC produces valid parameter updates
2. **Adaptive Behavior**: Clip norm adjusts based on gradient statistics
3. **LR Scaling**: Learning rate responds to clip rate
4. **EMA Smoothing**: EMA parameters track training params
5. **Privacy**: Accountant correctly tracks privacy budget
6. **Numerical**: Compare against JAX-Privacy (if applicable)

---

## Implementation Timeline

### Week 1: TorchOpt Integration & Base DP Optimizer

**Days 1-2**: Setup & Exploration

- Add TorchOpt dependency to `pyproject.toml`
- Explore TorchOpt functional API patterns
- Understand `GradientTransformation` composition
- Write integration tests with simple PyTree examples

**Days 3-4**: Base DP Optimizer Wrapper

- Implement `opaque/optimizers/base.py`
- Create `DPOptimizerState` dataclass
- Implement `make_dp_optimizer()` wrapper
- Integrate with existing `clipped_grad`, `add_gaussian_noise`, `RDPAccountant`
- Write unit tests for state management

**Day 5**: Basic DP-SGD

- Implement `opaque/optimizers/dp_sgd.py`
- Create `dp_sgd()` function using TorchOpt SGD
- Test against manual DP-SGD implementation
- Validate privacy accounting

### Week 2: Basic DP-Adam & Testing

**Days 1-2**: Basic DP-Adam

- Implement `opaque/optimizers/dp_adam.py`
- Create `dp_adam()` function using TorchOpt Adam
- Test Adam-specific features (momentum, bias correction)
- Compare with PyTorch's Adam on non-DP problem

**Days 3-5**: Comprehensive Testing

- Write `tests/optimizers/test_dp_sgd.py`
- Write `tests/optimizers/test_dp_adam.py`
- End-to-end training tests (logistic regression)
- Privacy budget validation
- Numerical equivalence checks

### Week 3: Adaptive Clipping Infrastructure

**Days 1-2**: Clip Norm Buffer

- Implement `opaque/adaptive/clip_buffer.py`
- `ClipNormBuffer` class with efficient percentile computation
- Unit tests for buffer operations
- Performance benchmarks (ensure <1ms overhead)

**Days 3-4**: LR Scheduler

- Implement `opaque/adaptive/lr_scheduler.py`
- Clip-rate-based LR adjustment logic
- Tests for LR scaling behavior
- Edge case handling (empty buffer, extreme clip rates)

**Day 5**: Integration Testing

- Test adaptive components with mock optimizer
- Validate percentile computation accuracy
- Stress test with large buffers

### Week 4: DP-Adam-AC Implementation

**Days 1-3**: Core DP-Adam-AC

- Implement `opaque/optimizers/dp_adam_ac.py`
- `AdaptiveClipState` dataclass
- `dp_adam_ac()` main function
- Integrate all adaptive components
- EMA parameter smoothing

**Days 4-5**: DP-Adam-AC Testing

- Write `tests/optimizers/test_dp_adam_ac.py`
- Unit tests for each adaptive mechanism
- Integration tests with training loop
- Compare fixed vs adaptive clipping
- Validate clip rate convergence to target

### Week 5: Documentation & Tutorial

**Days 1-2**: Tutorial Update

- Add Section 7 to Tutorial 03
- DP-Adam-AC training example
- Visualization of adaptive behavior
- Performance comparison table

**Days 3-4**: Documentation

- API reference for all optimizers
- Adaptive clipping design doc
- Performance benchmarks
- Migration guide from Opacus

**Day 5**: Polish & Review

- Code review and refactoring
- Coverage gaps filled
- Final JAX-Privacy validation
- Performance profiling

---

## Success Criteria

### Must Have ✅

1. ✅ DP-SGD functional optimizer working
2. ✅ DP-Adam functional optimizer working
3. ✅ DP-Adam-AC with all adaptive features
4. ✅ 90%+ test coverage for optimizers
5. ✅ Tutorial 03 updated with DP-Adam-AC example
6. ✅ Privacy accounting integrated

### Nice to Have 🎯

1. 🎯 JAX-Privacy numerical validation (if comparable API exists)
2. 🎯 Performance benchmarks vs Opacus
3. 🎯 Learning rate schedulers (cosine, linear)
4. 🎯 AdamW variant with decoupled weight decay

### Future Work 🔮

1. 🔮 DP-AdaGrad, DP-RMSprop
2. 🔮 Differentiable privacy budgets (meta-learning)
3. 🔮 Per-layer adaptive clipping
4. 🔮 Automatic hyperparameter tuning

---

## Key Design Decisions

### 1. TorchOpt over Custom Implementation

**Decision**: Use TorchOpt as optimizer backend

**Rationale**:

- ✅ Follows JAX-Privacy's Optax pattern
- ✅ Well-tested, production-ready
- ✅ Functional design matches our architecture
- ✅ Composable transformations
- ✅ Meta-learning support (bonus)

**Alternative**: Custom implementation

- ❌ Reinventing wheel
- ❌ More testing burden
- ❌ Harder to maintain

### 2. Separate Adaptive Package

**Decision**: Create `opaque.adaptive` for reusable components

**Rationale**:

- ✅ Adaptive clipping can be used with any optimizer
- ✅ Clear separation of concerns
- ✅ Easier testing
- ✅ Future: per-layer clipping, other adaptive strategies

### 3. EMA as Post-Processing

**Decision**: EMA smoothing is optional post-processing, not part of optimizer state

**Rationale**:

- ✅ Zero privacy cost (deterministic post-processing)
- ✅ Can be applied to any optimizer
- ✅ Clean separation from optimization logic
- ✅ Matches paper's approach

### 4. State Management

**Decision**: Use NamedTuples for optimizer state (following TorchOpt)

**Rationale**:

- ✅ Immutable by default
- ✅ Compatible with TorchOpt patterns
- ✅ Type-safe
- ✅ Easy to serialize

---

## References

1. **DP-Adam-AC Paper**: Zuo et al., "DP-Adam-AC: Privacy-preserving Fine-Tuning of Localizable Language Models Using
   Adam Optimization with Adaptive Clipping", arXiv:2510.05288, October 2024
2. **TorchOpt**: https://github.com/metaopt/torchopt
3. **Optax**: https://github.com/deepmind/optax (JAX-Privacy uses this)
4. **JAX-Privacy**: https://github.com/google-deepmind/jax_privacy
5. **Original DP-SGD**: Abadi et al., "Deep Learning with Differential Privacy", CCS 2016

---

## Next Stage Preview

**Stage 4: Privacy Amplification & Sampling** will focus on:

- Poisson sampling for better privacy amplification
- Secure shuffling with fixed privacy cost
- Microbatching for memory efficiency
- Truncated Poisson (already have accounting for this!)

DP-Adam-AC provides the optimizer foundation for efficient, practical DP training!
