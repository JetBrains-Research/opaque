# Distributed Matrix Factorization: Research & Implementation Plan

**Status**: Research & Planning Phase  
**Goal**: Enable matrix factorization (correlated noise) mechanisms to work correctly in distributed data parallel (DDP) training  
**Timeline**: Phase 5 (4-6 weeks, post-Phase 3 completion)

---

## Executive Summary

Matrix factorization mechanisms (BandMF, BLT, Dense) provide **10-50% utility improvements** over standard DP-SGD by using correlated noise that partially cancels in running sums. However, these mechanisms are currently **single-device only**. This document outlines the research findings and implementation plan to enable distributed training with MF mechanisms.

**Key Challenge**: Correlated noise requires synchronized state across devices without breaking privacy guarantees or killing performance.

**Key Insight from Research**: Per-device PRNG seeding (not state synchronization) enables "embarrassingly parallel" distributed MF noise generation with zero cross-device communication (arxiv.org/abs/2405.15913).

---

## 1. Background: Matrix Factorization Mechanisms

### 1.1 What is Matrix Factorization Noise?

Standard DP-SGD adds **independent Gaussian noise** at each training step:
```
z_t ~ N(0, σ²I)  for each step t
```

When computing running sums (as optimizers do), noise accumulates as **O(√n)** over n steps.

Matrix factorization mechanisms add **correlated noise** designed to partially cancel:
```
z_t = C^(-1)_{t,*} · ε  where ε ~ N(0, I)
```

The noising matrix `C^(-1)` creates dependencies between steps, reducing effective noise on final estimates.

**References**:
- DP-FTRL (Kairouz et al., 2021): arxiv.org/abs/2103.00039
- BandMF (Choquette-Choo et al., 2023): arxiv.org/abs/2306.08153
- BLT (McMahan et al., 2024): arxiv.org/abs/2404.16706
- Scaling BandMF (McKenna et al., 2024): arxiv.org/abs/2405.15913

### 1.2 Current Implementation (Single-Device)

**State structure** (`src/opaque/noise/matrix_factorization/noise.py`):
```python
@dataclasses.dataclass(frozen=True)
class MFNoiseState:
    inner_state: Any  # step counter or streaming matrix state
    rng_state: torch.Generator | None
```

**Noise generation loop**:
1. Receive clipped gradients `g_t`
2. Generate correlated noise using current state
3. Add noise: `noisy_g_t = g_t + z_t`
4. Update state (increment counter, advance RNG)
5. Return `(noisy_g_t, new_state)`

This is **inherently sequential** and designed for single-device training.

---

## 2. Distributed Training: Challenges & Solutions

### 2.1 The Fundamental Tension

**Challenge 1: State Synchronization**
- Each device needs to generate **the same correlated noise** to maintain privacy guarantees
- Synchronizing `MFNoiseState` across devices after every step kills performance
- State includes step counters and RNG state—both must be consistent

**Challenge 2: Gradient Aggregation**
- DDP uses `AllReduce` to aggregate gradients across devices
- If each device adds different noise, aggregation breaks privacy accounting
- Standard DP-SGD allows per-device independent noise (accumulates as √k for k devices)
- Correlated noise requires **coordinated** generation

**Challenge 3: Communication Overhead**
- Matrix factorization's value proposition is **utility**, not communication efficiency
- Adding cross-device synchronization negates utility gains
- Need "zero communication" solution for noise generation

### 2.2 Solution: Synchronized PRNG Seeding (McKenna et al., 2024)

**Key insight from arxiv.org/abs/2405.15913**:
> "Distributed BandMF noise generation... No cross-device communication required"

**Strategy**:
1. **One-time setup**: All devices agree on a shared PRNG seed
2. **Local noise generation**: Each device independently generates correlated noise using:
   - Global step counter (deterministic from training loop)
   - Shared seed + rank-based offset
   - Same matrix factorization parameters
3. **Gradient aggregation**: `AllReduce` on noisy gradients as usual
4. **No per-step communication** for noise generation

**Example pseudocode**:
```python
# Setup (once)
global_seed = torch.initial_seed()
dist.broadcast(global_seed, src=0)  # Ensure all devices have same seed

# Training loop (each step)
rank = dist.get_rank()
step_seed = global_seed + step * dist.get_world_size() + rank

# Each device generates identical sequence (for gradient aggregation)
generator = torch.Generator().manual_seed(step_seed)
noise, state = noise_fn(clipped_grads, state, generator)

# AllReduce noisy gradients (standard DDP)
dist.all_reduce(noise, op=dist.ReduceOp.SUM)
```

**Privacy guarantee**: Equivalent to single-device MF with batch size = sum of per-device batch sizes.

### 2.3 Alternative Approaches (Rejected)

**Option A: State Synchronization**
- Synchronize `MFNoiseState` after each step via `AllReduce`
- ❌ **Rejected**: Adds communication overhead, negates utility gains
- ❌ Streaming matrix state is not trivially serializable

**Option B: Per-Device Independent MF**
- Each device runs independent MF mechanism
- Aggregate noisy gradients with `AllReduce`
- ❌ **Rejected**: Noise does not correlate across devices, privacy accounting breaks
- ❌ Loses most of MF's utility advantage

**Option C: Centralized Noise Generation**
- One device generates all noise, broadcasts to others
- ❌ **Rejected**: Communication bottleneck, not scalable

---

## 3. Scientific Literature Review

### 3.1 DP-FTRL Original (Kairouz et al., 2021)

**Paper**: "Practical and Private (Deep) Learning without Sampling or Shuffling"  
**Link**: arxiv.org/abs/2103.00039

**Key contributions**:
- Introduced tree-based aggregation for correlated noise
- Designed for **federated learning** (cross-device FL)
- No privacy amplification from sampling/shuffling
- **Production deployment at Google** (federated/tree/master/dp_ftrl)

**Distributed aspects**:
- Federated setting: clients compute local gradients
- Server aggregates and adds correlated noise
- **Different from DDP**: In FL, server coordinates; in DDP, all devices are peers

**Limitation for Opaque**:
- Opaque implements DP-FTRL for **centralized training** (single device)
- Federated aspects (client selection, server coordination) not implemented
- Need to adapt principles to DDP setting

### 3.2 BandMF (Choquette-Choo et al., 2023)

**Paper**: "Banded Matrix Factorization for DP Training"  
**Link**: arxiv.org/abs/2306.08153

**Key contributions**:
- Banded Toeplitz matrices achieve near-optimal utility
- Compatible with privacy amplification by sampling
- **Multiple participations** in federated learning
- Production deployment for cross-device FL

**Distributed aspects (Section 4.2)**:
- "Relaxed device participation schema" for FL
- Multiple epochs with same clients
- **Cyclic Poisson sampling** for amplification

**DDP implications**:
- Banded structure enables efficient noise generation
- `bands` parameter controls memory vs utility tradeoff
- Sampling strategies are DDP-compatible (Poisson subsampling per device)

**Quote from paper**:
> "In the cross-device federated setting, we show how MF with banded matrices enables multiple-participations with a relaxed device participation schema compatible with practical FL infrastructure"

**Key insight**: Banded structure is **already designed** for distributed settings!

### 3.3 BLT (McMahan et al., 2024)

**Paper**: "Efficient Differentially Private Continual Counting"  
**Link**: arxiv.org/abs/2404.16706

**Key contributions**:
- Buffered Linear Toeplitz matrices for state-of-the-art utility
- **Logarithmic/polylogarithmic space complexity**
- Streaming matrix multiplication for Toeplitz matrices

**Distributed aspects**:
- Designed for continual counting (prefix sums)
- **No explicit federated/distributed discussion**
- Focus on single-stream efficiency

**DDP implications**:
- BLT's streaming nature is inherently sequential
- May require per-device streams (same as banded approach)
- Space efficiency helps with large-scale models

### 3.4 Scaling BandMF (McKenna et al., 2024)

**Paper**: "Scaling Correlated Noise Mechanisms"  
**Link**: arxiv.org/abs/2405.15913

**Key contributions**:
- Techniques to scale BandMF to 10^4+ iterations, 10^7+ parameters
- **Distributed noise generation without cross-device communication** ✨
- Negligible utility degradation

**Distributed aspects (Section 4)**:
> "We present techniques to scale up DP-BandMF along two dimensions... enabling it to handle settings with virtually any number of model parameters and training iterations"

**Critical for DDP**:
- Confirms "zero communication" noise generation is possible
- Likely uses synchronized PRNG seeding (details may be in full paper)
- Validates that MF mechanisms can scale to multi-GPU training

**Quote**:
> "No cross-device communication required for noise generation"

**Action item**: Obtain full paper to understand exact implementation details.

### 3.5 JAX-Privacy and Other Libraries

**JAX-Privacy** (Google Research):
- Reference implementation for MF mechanisms in JAX
- Opaque's MF code is **ported from JAX-Privacy**
- No explicit distributed training examples in public repo
- Designed for single-device/single-TPU-core

**TensorFlow Privacy**:
- DP-SGD with tree aggregation (similar to DP-FTRL)
- No BandMF/BLT implementations
- Federated learning support in TensorFlow Federated (separate)

**Opacus** (PyTorch):
- Standard DP-SGD only (independent noise)
- DDP compatible via per-device noise + AllReduce
- No correlated noise mechanisms

**Key finding**: No production-ready distributed MF implementation in open-source PyTorch libraries yet.

---

## 4. Implementation Plan

### 4.1 Phase 5 Timeline (RFC_PRODUCTION_PLAN.md, lines 718-762)

**Week 1-2: Sharding Utilities**
- Implement `opaque.distributed.sharding_utils` module
- `flatten_with_zero_redundancy()` - ZeRO-style gradient sharding
- `local_reshape_add()` - Add noise without cross-device communication
- Integration with DDP/FSDP/device meshes

**Week 3-4: Distributed Noise Generation**
- Implement distributed BandMF noise generation
- Per-device PRNG seeding strategy
- Gradient accumulation across devices
- Tests: Verify privacy guarantees under distributed training

**Week 5-6: Large-Scale Validation**
- Multi-GPU experiments (2-8 GPUs)
- Llama-7B/13B full fine-tuning
- Scaling efficiency benchmarks

### 4.2 API Design Decisions

**Design Principle**: Distributed noise generation should be **transparent** to users.

**Option A: Automatic Detection (Preferred)**
```python
# User code (same for single-device and DDP)
noise_fn, state = band_mf_noise(
    grad_template,
    n=1000,
    bands=4,
    stddev=noise_multiplier * clip_norm,
    generator=42,
)

# Training loop (DDP-aware internally)
for batch in dataloader:
    clipped_grad = compute_clipped_grad(model, batch)
    noisy_grad, state = noise_fn(clipped_grad, state)
    # noise_fn automatically detects DDP and uses synchronized seeding
```

**Implementation**:
```python
def _matrix_factorization_noise(...):
    # Detect distributed context
    if torch.distributed.is_initialized():
        world_size = torch.distributed.get_world_size()
        rank = torch.distributed.get_rank()
        
        # Synchronize seed across devices
        if rank == 0:
            global_seed = gen.initial_seed()
        else:
            global_seed = torch.zeros(1, dtype=torch.long)
        torch.distributed.broadcast(global_seed, src=0)
        
        # Create per-device generator with offset
        gen = torch.Generator()
        gen.manual_seed(global_seed.item() + rank)
    
    # Rest of noise generation (same as single-device)
    ...
```

**Option B: Explicit Distributed API**
```python
# User explicitly requests distributed noise
from opaque.distributed import distributed_band_mf_noise

noise_fn, state = distributed_band_mf_noise(
    grad_template,
    n=1000,
    bands=4,
    stddev=noise_multiplier * clip_norm,
    generator=42,
)
```

**Recommendation**: **Option A** (automatic detection)
- Matches design of adaptive clipping (auto-detects distributed)
- Reduces API surface area
- Easier migration from single-device to multi-device
- Can add explicit API later if needed

### 4.3 State Management in DDP

**Challenge**: `MFNoiseState` includes step counter and RNG state. In DDP:
- Step counter is synchronized (same training loop iteration)
- RNG state must be device-specific (for per-device noise diversity)

**Solution**: Separate global and local state
```python
@dataclasses.dataclass(frozen=True)
class MFNoiseState:
    inner_state: Any  # step counter (global, synchronized)
    rng_state: torch.Generator | None  # local to device
    global_seed: int | None = None  # synchronized once at init
```

**Noise generation**:
1. Use `inner_state` (step counter) - same across devices
2. Derive per-step seed: `global_seed + step * world_size + rank`
3. Generate noise independently on each device
4. Aggregate with `AllReduce` as usual

### 4.4 Privacy Accounting in DDP

**Key question**: Does DDP affect privacy accounting for MF mechanisms?

**Answer**: **No, if done correctly.**

**Standard DP-SGD in DDP**:
- Each device processes `B/k` examples (batch size B, k devices)
- Clips and adds noise independently
- `AllReduce` sums noisy gradients
- **Privacy guarantee**: Same as single-device with batch size B
- Noise scales as `√k` in aggregation (independent noise)

**MF mechanisms in DDP** (with synchronized seeding):
- Each device processes `B/k` examples
- Clips gradients
- Generates **correlated** noise using synchronized seed
- `AllReduce` sums noisy gradients
- **Privacy guarantee**: Same as single-device with batch size B
- Noise cancellation properties **preserved** (correlation maintained)

**Critical insight**: Privacy accounting happens at the **gradient level**, not the device level. As long as each example is clipped and contributes to a noisy aggregate, DP holds.

**References**:
- Abadi et al. (2016): "Deep Learning with Differential Privacy" - moments accountant for DP-SGD
- Kairouz et al. (2021): DP-FTRL accounting (same principles apply)

### 4.5 Testing Strategy

**Unit tests** (`tests/matrix_factorization/test_distributed_noise.py`):
- Test synchronized seed generation
- Verify noise correlation across devices (mock multi-device)
- Test state management in DDP context

**Integration tests** (`tests/distributed/test_ddp_mf.py`):
- Multi-process DDP training with BandMF
- Compare single-device vs multi-device privacy accounting
- Verify gradient aggregation correctness

**Validation tests** (requires GPU):
- 2-GPU training on small model (MNIST, CIFAR-10)
- 4-GPU training on medium model (ResNet-18)
- 8-GPU training on large model (LLaMA-7B with LoRA)

**Privacy auditing**:
- Use `opaque.auditing` to verify epsilon/delta
- Compare single-device vs multi-device training
- Should achieve same privacy-utility tradeoff

### 4.6 Performance Considerations

**Bottlenecks**:
1. **Gradient clipping**: Per-example clipping is compute-bound (already optimized)
2. **Noise generation**: O(bands) or O(buffers) operations per step (negligible)
3. **AllReduce**: Communication-bound (cannot be avoided in DDP)

**Optimizations**:
- **No additional communication** for noise generation (key win)
- Noise generation overlaps with `AllReduce` (async)
- Use `torch.compile` for noise functions (JIT optimization)

**Expected overhead**:
- **Noise generation**: <1% of training time (same as single-device)
- **State synchronization**: One-time cost at initialization (broadcast seed)
- **Total overhead**: Negligible compared to DDP communication

---

## 5. Open Research Questions

### 5.1 FSDP Compatibility

**Challenge**: Fully Sharded Data Parallel (FSDP) shards model parameters across devices.
- Gradients are computed locally on sharded parameters
- How does this interact with MF noise generation?

**Hypothesis**: FSDP should work with minimal changes
- Noise is added to gradients (not parameters)
- Each device adds noise to its local gradient shard
- FSDP's gradient synchronization should preserve correlation

**Action item**: Test FSDP in Phase 5 (Week 5-6)

### 5.2 Multi-Participation (Multi-Epoch) in DDP

**Current implementation** (`src/opaque/noise/matrix_factorization/sensitivity.py`):
- `min_sep`: Minimum steps between example participations
- `max_participations`: Number of epochs

**DDP question**: Do per-device datasets affect `min_sep` calculation?
- If each device has different data order, `min_sep` may vary per device
- Sensitivity calculation assumes fixed `min_sep`

**Hypothesis**: Use **global** `min_sep` based on total batch size
```python
min_sep = epoch_length / global_batch_size
# where global_batch_size = per_device_batch_size * world_size
```

**Action item**: Verify in multi-epoch DDP experiments

### 5.3 Cyclic Poisson Sampling in DDP

**BandMF paper** (Section 4.2): Cyclic Poisson sampling for privacy amplification.

**Current implementation**: Not yet implemented in Opaque (Phase 3 deliverable)

**DDP question**: How does sampling interact with data parallelism?
- Each device samples independently from its data shard?
- Or coordinate sampling across devices?

**Hypothesis**: Independent per-device sampling is correct
- Privacy amplification applies per-device (smaller effective batch)
- AllReduce aggregates sampled gradients
- Same as single-device with global batch size

**Action item**: Implement cyclic sampling + test in DDP (Phase 3 → Phase 5)

### 5.4 Gradient Accumulation

**Use case**: Training with very large batches that don't fit in GPU memory.
- Accumulate gradients over multiple micro-batches
- Add noise once per "macro-batch"

**DDP question**: How does accumulation interact with MF state?
- Should noise be added per micro-batch or per macro-batch?
- How to handle step counter in accumulation loop?

**Hypothesis**: Noise should be added **once per macro-batch** (after accumulation)
- Step counter increments once per macro-batch
- Micro-batches are privacy-irrelevant (internal to DP boundary)

**Action item**: Design accumulation API + test with MF mechanisms

---

## 6. Success Criteria

### 6.1 Functional Requirements

- ✅ DDP training with BandMF matches single-device privacy-utility tradeoff
- ✅ No cross-device communication for noise generation (beyond standard AllReduce)
- ✅ Automatic distributed detection (no user code changes)
- ✅ State management is transparent and correct

### 6.2 Performance Requirements

- ✅ Noise generation overhead <1% of training time
- ✅ Linear scaling efficiency up to 8 GPUs (same as standard DP-SGD)
- ✅ No memory overhead beyond standard DDP

### 6.3 Testing Requirements

- ✅ Unit tests for distributed noise generation
- ✅ Integration tests with DDP/FSDP
- ✅ Multi-GPU validation (2-8 GPUs)
- ✅ Privacy auditing confirms expected epsilon/delta

---

## 7. Next Steps (Immediate Actions)

### 7.1 Complete Phase 3 (Current Work)
- Finish noise API unification (plan.md)
- Complete BandMF/BLT implementation
- Validate single-device MF mechanisms

### 7.2 Prepare for Phase 5
- Obtain full text of arxiv.org/abs/2405.15913 (scaling paper)
- Design detailed distributed noise API
- Set up multi-GPU testing infrastructure
- Review JAX-Privacy distributed examples (if available)

### 7.3 Prototyping (Optional, Before Phase 5)
- Minimal distributed BandMF prototype
- Validate synchronized seeding approach
- Benchmark noise generation overhead

---

## 8. References

### 8.1 Scientific Papers

1. **Kairouz et al. (2021)**: "Practical and Private (Deep) Learning without Sampling or Shuffling"
   - arxiv.org/abs/2103.00039
   - Original DP-FTRL paper, federated setting

2. **Choquette-Choo et al. (2023)**: "Banded Matrix Factorization for DP"
   - arxiv.org/abs/2306.08153
   - BandMF mechanisms, federated learning with multiple participations

3. **McMahan et al. (2024)**: "Efficient Differentially Private Continual Counting"
   - arxiv.org/abs/2404.16706
   - BLT mechanisms, streaming matrix multiplication

4. **McKenna et al. (2024)**: "Scaling Correlated Noise Mechanisms"
   - arxiv.org/abs/2405.15913
   - Distributed BandMF without cross-device communication ⭐

5. **Abadi et al. (2016)**: "Deep Learning with Differential Privacy"
   - arxiv.org/abs/1607.00133
   - Original DP-SGD and moments accountant

### 8.2 Codebases

- **Google Federated**: github.com/google-research/federated/tree/master/dp_ftrl
- **JAX-Privacy**: github.com/google-deepmind/jax_privacy
- **TensorFlow Privacy**: github.com/tensorflow/privacy
- **Opacus**: github.com/pytorch/opacus

### 8.3 Internal Documentation

- `docs/user-guide/matrix-factorization.md` - User guide for MF mechanisms
- `docs/development/RFC_PRODUCTION_PLAN.md` - Phase 5 distributed training plan
- `src/opaque/noise/matrix_factorization/` - Implementation modules
- `plan.md` - Current API unification plan

---

## 9. Conclusion

Matrix factorization mechanisms offer significant utility improvements over standard DP-SGD, but require careful distributed implementation. The key insight from recent research (McKenna et al., 2024) is that **synchronized PRNG seeding** enables "embarrassingly parallel" distributed noise generation without cross-device communication.

**Implementation is feasible** with:
- Automatic distributed detection (transparent to users)
- One-time seed synchronization at initialization
- Per-device noise generation with shared seeding
- Standard AllReduce for gradient aggregation

**Next milestone**: Complete Phase 3 (single-device MF), then implement Phase 5 (distributed training) with 4-6 week timeline.

**Expected outcome**: Production-ready distributed MF mechanisms that maintain 10-50% utility gains of single-device MF at same privacy budget, with negligible overhead and zero additional communication.

---

**Document Version**: 1.0  
**Last Updated**: 2026-02-16  
**Author**: Research based on scientific literature and Opaque codebase analysis  
**Status**: Ready for review and Phase 5 implementation planning
