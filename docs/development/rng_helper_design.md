# RNG Helper API: Before and After

## Summary

We've introduced **convenience helpers** that make the required-key API ergonomic while preserving our core design principles:

1. **All randomness explicit** - No hidden global state
2. **Deterministic by default** - Keys make reproducibility trivial  
3. **Composable primitives** - Build complex patterns from simple functions

## New API

```python
from opaque.random import random_key, training_key

# For prototyping (non-deterministic)
key = random_key()

# For training loops (deterministic, follows step → rank → worker derivation)
key = training_key(base_seed=42, step=step)

# For distributed training with synchronized noise
key = training_key(base_seed=42, step=step, rank=local_rank, synchronized=True)

# For distributed training with per-rank noise
key = training_key(base_seed=42, step=step, rank=local_rank, synchronized=False)

# Auto mode: synchronized if no rank, unsynchronized otherwise
key = training_key(base_seed=42, step=step, rank=local_rank, synchronized="auto")
```

## Before (Hypothetical Optional Key)

```python
# If we allowed key=None with random default:
noise_fn, state = gaussian_noise(stddev=1.0)  # Where does randomness come from?
noise_fn2, state2 = gaussian_noise(stddev=1.0)  # Same as above? Different?

# Problems:
# - Hidden non-determinism
# - Accidental non-reproducible experiments  
# - Unclear distributed semantics
# - Global state creeps back in
```

## After (With Helpers)

```python
# Prototyping: explicitly non-deterministic
key = random_key()
noise_fn, state = gaussian_noise(stddev=1.0, key=key)

# Training: explicitly deterministic
for step in range(num_steps):
    key = training_key(base_seed=42, step=step)
    noise_fn, state = gaussian_noise(stddev=1.0, key=key)
    # ... train ...

# Benefits:
# - Intent is clear at call site
# - Reproducibility by construction
# - Explicit control over distributed semantics
# - No global state
```

## Real-World Example

See `examples/dp_sgd_simple.py` for a complete training loop:

```python
from opaque.random import training_key

step = 0
for epoch in range(epochs):
    for batch_x, batch_y in dataloader:
        # Create deterministic key for this step
        key = training_key(base_seed=42, step=step)
        
        # Configure noise with explicit key
        noise_fn, noise_state = gaussian_noise(
            stddev=noise_multiplier * clip_norm,
            key=key,
        )
        
        # Compute clipped + noisy gradients
        grads, clip_state = grad_fn(params, batch_x, batch_y, state=clip_state)
        noisy_grads, noise_state = noise_fn(grads, noise_state)
        
        # Update
        params = params - lr * noisy_grads
        step += 1
```

## Design Rationale

### Why Not Allow None?

1. **What would None mean?**
   - `key(0)` → All runs identical (bad default)
   - Random seed → Non-reproducible (defeats JAX philosophy)
   - Global counter → Mutable state (defeats functional API)

2. **Security/Auditing:**
   - DP guarantees depend on *all* randomness being controlled
   - Optional keys create accidental non-reproducible audits
   - Explicit keys force thinking: "Where does this randomness come from?"

3. **JAX Precedent:**
   - JAX has no global RNG state - keys are always explicit
   - This prevents subtle non-determinism bugs
   - Forces reproducibility by construction

### Why Helpers Are Better

- **Intentional friction at the right level**: Core API requires keys, helpers make common patterns easy
- **Visible decisions**: `random_key()` vs `training_key()` documents intent
- **No hidden state**: Everything remains functional and composable
- **Escape hatch preserved**: Power users can still use `key()`, `split()`, `fold_in()` directly

## Distributed Training Patterns

### Centralized DP-SGD (synchronized noise)

```python
# Same noise on all ranks for model convergence
key = training_key(base_seed=42, step=step, rank=rank, synchronized=True)
noise_fn, state = gaussian_noise(stddev=1.0, key=key)
```

### Per-Rank DP (unsynchronized noise)

```python
# Different noise per rank
key = training_key(base_seed=42, step=step, rank=rank, synchronized=False)
noise_fn, state = gaussian_noise(stddev=1.0, key=key)
```

### Auto Mode (default behavior)

```python
# Automatically synchronized if no rank, unsynchronized if rank provided
key = training_key(base_seed=42, step=step, rank=rank, synchronized="auto")
```

## Complete Derivation Chain

```python
# Full chain: step → rank → worker
key = training_key(
    base_seed=42,
    step=current_step,
    rank=local_rank,           # For DDP/FSDP
    worker_id=worker_info.id,  # For DataLoader workers
    synchronized=False,
)
```

This follows our canonical derivation order and ensures:
- Different noise per step (privacy amplification)
- Different noise per rank (when unsynchronized)
- Different noise per DataLoader worker (data loading parallelism)

## Tests

See `tests/utils/test_rng_helpers.py` for 18 comprehensive tests covering:
- Non-deterministic key generation
- Deterministic training keys with step derivation
- Synchronized vs unsynchronized distributed modes
- Auto mode behavior
- Worker ID folding
- Full derivation chain validation
- Integration with noise functions
