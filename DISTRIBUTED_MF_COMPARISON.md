# Distributed Training: Technical Comparison Matrix

This document compares different approaches to distributed training with differential privacy.

---

## Comparison Table: Distributed DP Training Approaches

| Aspect | Standard DP-SGD | MF Single-Device | MF Distributed (Naive) | MF Distributed (Synchronized) |
|--------|----------------|------------------|------------------------|-------------------------------|
| **Noise Type** | Independent | Correlated | Correlated | Correlated |
| **Noise Accumulation** | O(√n) | Partial cancellation | Broken | Partial cancellation |
| **Utility vs DP-SGD** | Baseline | +10-50% | ❌ Broken | +10-50% |
| **State Management** | Stateless | Sequential state | Per-device state | Synchronized seeds |
| **Communication** | AllReduce only | N/A (single device) | AllReduce + State sync | AllReduce only ✅ |
| **Privacy Accounting** | Simple | Same as DP-SGD | ❌ Breaks | Same as DP-SGD ✅ |
| **Implementation** | ✅ Done | ✅ Done | ❌ Wrong | ⏳ Phase 5 |
| **Code Changes** | None (auto) | N/A | Significant | None (auto) ✅ |

**Legend**: ✅ = Good/Complete, ⏳ = In Progress, ❌ = Broken/Invalid, N/A = Not Applicable

---

## Detailed Comparison

### 1. Standard DP-SGD (Current: ✅ Working)

```python
# Each device independently
clipped_grads = clip(grads)
noise = torch.randn_like(grads) * stddev  # Independent noise per device
noisy_grads = clipped_grads + noise

# Aggregate across devices
dist.all_reduce(noisy_grads, op=ReduceOp.SUM)
```

**Pros**:
- ✅ Simple implementation
- ✅ No coordination needed
- ✅ Privacy accounting straightforward
- ✅ Works with any parallelism strategy (DDP, FSDP, etc.)

**Cons**:
- ❌ Suboptimal utility (baseline)
- ❌ Noise accumulates as O(√n) in running sums

**Use case**: Baseline, when simplicity matters more than utility

---

### 2. Matrix Factorization - Single Device (Current: ✅ Working)

```python
# State tracks position in noise correlation matrix
noise_fn, state = band_mf_noise(grad_template, n=1000, bands=4, stddev=...)

for batch in dataloader:
    clipped_grads = clip(grads)
    noisy_grads, state = noise_fn(clipped_grads, state)  # Correlated noise
    optimizer.step()
```

**Pros**:
- ✅ 10-50% utility improvement over DP-SGD
- ✅ Same privacy guarantee as DP-SGD
- ✅ Memory efficient (BandMF: O(bands), BLT: O(buffers))
- ✅ Mature implementation (ported from JAX-Privacy)

**Cons**:
- ❌ Single device only (for now)
- ❌ Stateful (state must be managed)
- ❌ More complex than independent noise

**Use case**: Single-GPU training when utility is critical

---

### 3. MF Distributed - Naive Approach (❌ WRONG - Do Not Implement)

```python
# WRONG: Each device with independent MF state
noise_fn_0, state_0 = band_mf_noise(...)  # Device 0
noise_fn_1, state_1 = band_mf_noise(...)  # Device 1

# Each device adds different correlated noise
noisy_grads_0, state_0 = noise_fn_0(grads_0, state_0)
noisy_grads_1, state_1 = noise_fn_1(grads_1, state_1)

# AllReduce
dist.all_reduce(noisy_grads, op=ReduceOp.SUM)
```

**Why this is wrong**:
1. ❌ Each device has different RNG state → different noise sequences
2. ❌ Correlation structure is per-device, not global
3. ❌ Privacy accounting breaks (noise doesn't correlate correctly)
4. ❌ Loses most of MF's utility advantage

**Do not implement this approach!**

---

### 4. MF Distributed - State Synchronization (❌ BAD - Performance Killer)

```python
# BAD: Synchronize state after each step
noisy_grads, state = noise_fn(grads, state)

# Broadcast state to all devices
state_tensor = serialize(state)
dist.broadcast(state_tensor, src=0)
state = deserialize(state_tensor)

# AllReduce
dist.all_reduce(noisy_grads, op=ReduceOp.SUM)
```

**Why this is bad**:
1. ❌ Adds extra communication round per step
2. ❌ State is not trivially serializable (Generator, StreamingMatrix)
3. ❌ Negates MF's utility gains due to communication overhead
4. ❌ Doesn't scale beyond a few GPUs

**Avoid this approach!**

---

### 5. MF Distributed - Synchronized Seeding (⏳ CORRECT - Phase 5)

```python
# Setup (once)
if rank == 0:
    global_seed = torch.tensor([generator.initial_seed()])
else:
    global_seed = torch.zeros(1, dtype=torch.long)
dist.broadcast(global_seed, src=0)  # ← One-time communication

# Each step (no communication)
device_gen = torch.Generator()
step_seed = global_seed.item() + step * world_size + rank
device_gen.manual_seed(step_seed)

# Generate correlated noise (deterministic from seed)
noisy_grads, state = noise_fn(grads, state, generator=device_gen)

# AllReduce (standard DDP)
dist.all_reduce(noisy_grads, op=ReduceOp.SUM)
```

**Why this is correct**:
1. ✅ Same matrix parameters across devices
2. ✅ Deterministic noise from synchronized seeds
3. ✅ Correlation structure preserved globally
4. ✅ Zero communication overhead (beyond AllReduce)
5. ✅ Privacy accounting same as single-device

**Pros**:
- ✅ 10-50% utility improvement (same as single-device MF)
- ✅ No additional communication
- ✅ Transparent to users (auto-detect)
- ✅ Linear scaling up to 8+ GPUs expected
- ✅ Same privacy guarantee

**Cons**:
- ⏳ Not yet implemented (Phase 5)
- ⏳ Needs validation with FSDP, multi-epoch, etc.

**Use case**: Multi-GPU training when utility is critical (Phase 5+)

**Scientific basis**: McKenna et al. (2024), arxiv.org/abs/2405.15913

---

## Memory & Communication Overhead

| Approach | Per-Step Memory | Per-Step Communication | Total Overhead |
|----------|----------------|------------------------|----------------|
| DP-SGD Single | O(model_size) | None | None |
| DP-SGD DDP | O(model_size/k) | 1× AllReduce | ~2× grads transferred |
| MF Single (BandMF) | O(model_size + bands) | None | ~O(bands) extra |
| MF Single (BLT) | O(model_size + buffers) | None | ~O(buffers) extra |
| MF DDP (Synced Seeds) | O(model_size/k + bands) | 1× AllReduce | ~2× grads (same as DP-SGD) ✅ |
| MF DDP (State Sync) ❌ | O(model_size/k + bands) | 1× AllReduce + 1× Broadcast(state) | ~2× grads + state (BAD) |

**Key insight**: Synchronized seeding has **same communication as standard DDP**, just better utility!

---

## Privacy Accounting Comparison

| Approach | Epsilon Calculation | Delta | Amplification |
|----------|---------------------|-------|---------------|
| DP-SGD Single | Moments accountant | Standard | Sampling |
| DP-SGD DDP | Same (per-example accounting) | Standard | Sampling |
| MF Single | Same as DP-SGD | Standard | Sampling |
| MF DDP (Wrong) ❌ | ❌ Breaks (correlation wrong) | ❌ | ❌ |
| MF DDP (Synced) ✅ | Same as DP-SGD | Standard | Sampling |

**Critical**: Privacy accounting happens at **gradient level**, not device level. As long as:
1. Each example is clipped (✅)
2. Sufficient noise is added (✅)
3. Noise correlations are correct (✅ with synced seeds)

Then privacy is preserved.

**Reference**: Abadi et al. (2016) - "Deep Learning with Differential Privacy"

---

## Implementation Complexity

| Approach | Lines of Code | Complexity | Maintenance |
|----------|--------------|------------|-------------|
| DP-SGD Single | ~50 | Low | Low |
| DP-SGD DDP | +10 (auto-detect) | Low | Low |
| MF Single (BandMF) | ~200 (per mechanism) | Medium | Medium |
| MF DDP (Synced Seeds) | +30 (seed broadcast) | Medium | Medium ✅ |
| MF DDP (State Sync) ❌ | +100 (serialize/sync) | High | High ❌ |

**Key insight**: Synchronized seeding adds minimal complexity (~30 LOC) compared to alternatives.

---

## When to Use Each Approach

### Use Standard DP-SGD when:
- ✅ Training on single GPU with limited resources
- ✅ Simplicity is paramount
- ✅ Utility requirements are moderate
- ✅ Quick prototyping

### Use Single-Device MF when:
- ✅ Training on single GPU with good resources
- ✅ Utility is critical (want 10-50% improvement)
- ✅ Training for many epochs (correlation benefits accumulate)
- ✅ Can afford stateful API

### Use Distributed MF (Phase 5+) when:
- ✅ Training on multiple GPUs
- ✅ Utility is critical (need every % improvement)
- ✅ Large models (>1B parameters)
- ✅ Willing to use cutting-edge techniques

### Do NOT use:
- ❌ Naive per-device MF (wrong)
- ❌ State synchronization MF (kills performance)

---

## Testing Strategy Comparison

| Approach | Unit Tests | Integration Tests | GPU Tests | Privacy Auditing |
|----------|-----------|------------------|-----------|-----------------|
| DP-SGD | ✅ Done | ✅ Done | ✅ Done | ✅ Done |
| MF Single | ✅ Done | ✅ Done | ✅ Done | ✅ Done |
| MF DDP | ⏳ Phase 5 | ⏳ Phase 5 | ⏳ Phase 5 | ⏳ Phase 5 |

**Phase 5 Testing Plan**:
- Unit: Seed synchronization, noise correlation verification
- Integration: Full training loops with DDP/FSDP
- GPU: 2, 4, 8 GPU validation on MNIST, CIFAR-10, LLaMA-7B
- Auditing: Verify epsilon/delta match single-device

---

## Performance Benchmarks (Projected)

| Setup | DP-SGD | MF Single | MF DDP (Synced) | Speedup |
|-------|--------|-----------|-----------------|---------|
| MNIST (1 GPU) | 30s | 31s (+3%) | N/A | 1.0× |
| CIFAR-10 (1 GPU) | 5min | 5.2min (+4%) | N/A | 1.0× |
| CIFAR-10 (4 GPU) | 1.5min | N/A | 1.6min (+7%) | ~3.3× |
| LLaMA-7B (8 GPU) | 2hr | N/A | 2.1hr (+5%) | ~7.6× |

**Notes**:
- MF overhead: 3-7% (noise generation)
- DDP scaling: ~Linear up to 8 GPUs (expected)
- Utility gain: 10-50% better final accuracy (not shown in timing)

**Trade-off**: Spend 5% more time, get 10-50% better utility → **Worth it!**

---

## Conclusion: The Right Choice

```
┌─────────────────────────────────────────────────────────────┐
│                       Decision Tree                          │
│                                                              │
│                 Multiple GPUs Available?                     │
│                        ╱           ╲                        │
│                      NO             YES                      │
│                     ╱                 ╲                     │
│          Utility Critical?      Phase 5 Complete?           │
│             ╱        ╲              ╱         ╲            │
│           NO          YES          NO           YES         │
│           │            │           │             │          │
│       DP-SGD      MF Single    DP-SGD       MF DDP         │
│      (simple)    (best single)  (wait)   (best multi) ✅   │
└─────────────────────────────────────────────────────────────┘
```

**Today (Phase 1-3)**:
- Single GPU + Utility critical? → Use MF Single ✅
- Multiple GPUs? → Use DP-SGD DDP ✅

**Future (Phase 5+)**:
- Multiple GPUs + Utility critical? → Use MF DDP ✅

---

## References

- **Standard DP-SGD**: Abadi et al. (2016), arxiv.org/abs/1607.00133
- **DP-FTRL**: Kairouz et al. (2021), arxiv.org/abs/2103.00039
- **BandMF**: Choquette-Choo et al. (2023), arxiv.org/abs/2306.08153
- **BLT**: McMahan et al. (2024), arxiv.org/abs/2404.16706
- **Scaling MF**: McKenna et al. (2024), arxiv.org/abs/2405.15913

**Full details**: See DISTRIBUTED_MF_PLAN.md and DISTRIBUTED_MF_QUICKREF.md
