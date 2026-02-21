# RNG Helper API: Design Rationale

## Summary

Opaque uses **explicit random keys** with a minimal API surface:

1. **All randomness explicit** - No hidden global state
2. **Deterministic by default** - Keys make reproducibility trivial
3. **Composable primitives** - Build complex patterns from `key()`, `fold_in()`, `split()`

## API

```python
from opaque.random import key, fold_in, split, random_key

# Deterministic key
k = key(42)

# Per-step derivation
step_key = fold_in(k, step)

# Multi-value derivation (variadic fold_in)
step_rank_key = fold_in(k, step, rank)

# Independent child keys
noise_key, sample_key = split(k, num=2)

# Non-deterministic (prototyping only)
k = random_key()
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

## After (With Primitives)

```python
# Prototyping: explicitly non-deterministic
k = random_key()
noise_fn, state = gaussian_noise(stddev=1.0, key=k)

# Training: explicitly deterministic with fold_in
base = key(42)
for step in range(num_steps):
    step_key = fold_in(base, step)
    noise_fn, state = gaussian_noise(stddev=1.0, key=step_key)
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
from opaque.random import key, fold_in

base = key(42)
step = 0
for epoch in range(epochs):
    for batch_x, batch_y in dataloader:
        # Create deterministic key for this step
        step_key = fold_in(base, step)

        # Configure noise with explicit key
        noise_fn, noise_state = gaussian_noise(
            stddev=noise_multiplier * clip_norm,
            key=step_key,
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
   - `key(0)` -> All runs identical (bad default)
   - Random seed -> Non-reproducible (defeats JAX philosophy)
   - Global counter -> Mutable state (defeats functional API)

2. **Security/Auditing:**
   - DP guarantees depend on *all* randomness being controlled
   - Optional keys create accidental non-reproducible audits
   - Explicit keys force thinking: "Where does this randomness come from?"

3. **JAX Precedent:**
   - JAX has no global RNG state - keys are always explicit
   - This prevents subtle non-determinism bugs
   - Forces reproducibility by construction

### Why Primitives Are Better Than Helpers

- **Fewer concepts**: `fold_in(key(42), step, rank)` vs `training_key(base_seed=42, step=step, rank=rank)`
- **Composable**: Variadic `fold_in` handles any derivation chain
- **Visible decisions**: `random_key()` vs `key(42)` documents intent
- **No hidden state**: Everything remains functional and composable

## Distributed Training Patterns

### Centralized DP-SGD (synchronized noise)

```python
# Same noise on all ranks — same key, no rank folded in
base = key(42)
noise_fn, state = gaussian_noise(stddev=1.0, key=fold_in(base, step))
```

### Per-Rank DP (independent noise)

```python
# Different noise per rank — fold in rank
base = key(42)
noise_fn, state = gaussian_noise(stddev=1.0, key=fold_in(base, step, rank))
```

## Complete Derivation Chain

```python
# Full chain: step -> rank -> worker
step_key = fold_in(key(42), current_step, local_rank, worker_info.id)
```

This ensures:
- Different noise per step (privacy amplification)
- Different noise per rank (when rank is provided)
- Different noise per DataLoader worker (data loading parallelism)

## Tests

See `tests/utils/test_rng_helpers.py` for comprehensive tests covering:
- Non-deterministic key generation
- Variadic fold_in derivation
- Integration with noise functions
