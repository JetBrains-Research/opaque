# Distributed Matrix Factorization: Quick Reference

**TL;DR**: Use synchronized PRNG seeding across devices. No state synchronization needed. Zero communication overhead beyond standard DDP.

---

## The Core Problem

Matrix factorization (MF) mechanisms add **correlated noise** across training steps to improve utility. In distributed training:
- Each device needs to generate noise that maintains correlation
- Synchronizing state after every step kills performance
- Per-device independent noise breaks privacy accounting

---

## The Solution (McKenna et al., 2024)

**Key insight**: "Distributed BandMF noise generation... No cross-device communication required" (arxiv.org/abs/2405.15913)

**Strategy**:
1. **Setup (once)**: Broadcast shared seed to all devices
2. **Training loop**: Each device generates noise using `global_seed + step * world_size + rank`
3. **Aggregation**: Standard `AllReduce` on noisy gradients

**Result**: Correlated noise preserved, no communication overhead.

---

## Implementation Checklist

### Week 1-2: Sharding Utilities
- [ ] Create `opaque.distributed.sharding_utils` module
- [ ] Implement `flatten_with_zero_redundancy()` - ZeRO-style sharding
- [ ] Implement `local_reshape_add()` - Local noise addition
- [ ] DDP/FSDP integration helpers

### Week 3-4: Distributed Noise Generation
- [ ] Add distributed detection to `_matrix_factorization_noise()`
- [ ] Implement synchronized seed broadcast
- [ ] Per-device generator with rank offset
- [ ] Test: Verify noise correlation across devices
- [ ] Test: Privacy accounting matches single-device

### Week 5-6: Validation
- [ ] Multi-GPU experiments (2, 4, 8 GPUs)
- [ ] Llama-7B/13B full fine-tuning
- [ ] Scaling efficiency benchmarks
- [ ] Privacy auditing validation

---

## API Design (Auto-Detect Distributed)

```python
# User code (same for single-device and multi-device)
noise_fn, state = band_mf_noise(
    grad_template,
    n=1000,
    bands=4,
    stddev=noise_multiplier * clip_norm,
    generator=42,
)

# Training loop - noise_fn automatically handles DDP
for batch in dataloader:
    clipped_grad = compute_clipped_grad(model, batch)
    noisy_grad, state = noise_fn(clipped_grad, state)  # DDP-aware internally
    # Standard DDP aggregation happens inside noise_fn
```

**Implementation** (in `src/opaque/noise/matrix_factorization/noise.py`):
```python
def _matrix_factorization_noise(grad_template, noising, *, stddev, gen, dtype=None):
    # Detect distributed context
    is_distributed = torch.distributed.is_available() and torch.distributed.is_initialized()
    
    if is_distributed:
        rank = torch.distributed.get_rank()
        world_size = torch.distributed.get_world_size()
        
        # Synchronize seed (once)
        if rank == 0:
            global_seed = torch.tensor([gen.initial_seed()], dtype=torch.long)
        else:
            global_seed = torch.zeros(1, dtype=torch.long)
        torch.distributed.broadcast(global_seed, src=0)
        
        # Create per-device generator with offset
        device_gen = torch.Generator()
        device_gen.manual_seed(global_seed.item() + rank)
        gen = device_gen
    
    # Continue with standard noise generation
    ...
```

---

## Key Research Findings

### What Works
✅ Synchronized PRNG seeding (McKenna et al., 2024)  
✅ Banded matrices are DDP-compatible by design (Choquette-Choo et al., 2023)  
✅ Privacy accounting unchanged from single-device  
✅ No additional communication beyond AllReduce  

### What Doesn't Work
❌ State synchronization (kills performance)  
❌ Per-device independent MF (breaks privacy)  
❌ Centralized noise generation (communication bottleneck)  

### Open Questions
❓ FSDP compatibility (needs testing)  
❓ Multi-epoch min_sep calculation in DDP  
❓ Cyclic Poisson sampling coordination  
❓ Gradient accumulation with MF state  

---

## Testing Strategy

**Unit tests** (`tests/matrix_factorization/test_distributed_noise.py`):
```python
def test_synchronized_seeding():
    """Verify all devices generate correlated noise."""
    pass

def test_privacy_accounting_ddp():
    """Compare single-device vs multi-device privacy."""
    pass

def test_state_management_ddp():
    """Verify step counter and RNG state handling."""
    pass
```

**Integration tests** (`tests/distributed/test_ddp_mf.py`):
```python
def test_bandmf_ddp_training():
    """Full training loop with 2-GPU DDP + BandMF."""
    pass

def test_blt_fsdp_training():
    """Full training loop with 4-GPU FSDP + BLT."""
    pass
```

**Validation** (requires GPU cluster):
- MNIST (2 GPUs, quick smoke test)
- CIFAR-10 + ResNet-18 (4 GPUs, medium scale)
- LLaMA-7B + LoRA (8 GPUs, large scale)

---

## Success Criteria

- ✅ DDP training with MF matches single-device utility
- ✅ Noise generation overhead <1%
- ✅ Linear scaling up to 8 GPUs
- ✅ Privacy auditing confirms expected epsilon/delta
- ✅ Zero additional communication

---

## References

**Primary**:
- McKenna et al. (2024): arxiv.org/abs/2405.15913 - **Scaling distributed BandMF** ⭐
- Choquette-Choo et al. (2023): arxiv.org/abs/2306.08153 - BandMF for FL
- Kairouz et al. (2021): arxiv.org/abs/2103.00039 - DP-FTRL foundations

**Implementation guides**:
- `DISTRIBUTED_MF_PLAN.md` - Full research & planning document
- `docs/development/RFC_PRODUCTION_PLAN.md` - Phase 5 timeline
- `docs/user-guide/matrix-factorization.md` - User guide

**Code**:
- `src/opaque/noise/matrix_factorization/noise.py` - Core implementation
- `src/opaque/clipping/adaptive.py` - Example of auto-detecting distributed

---

## Next Immediate Actions

1. **Read full paper**: Obtain arxiv.org/abs/2405.15913 (scaling BandMF)
2. **Prototype**: Implement minimal synchronized seeding in `noise.py`
3. **Test**: Unit test for seed synchronization
4. **Document**: Update user guide with distributed examples
5. **Prepare**: Set up multi-GPU testing infrastructure

---

**Quick Start**: See `DISTRIBUTED_MF_PLAN.md` for comprehensive research and detailed implementation plan.
