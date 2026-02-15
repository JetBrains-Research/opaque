# Testing Distributed Training on 4x L4 GPUs

This machine has **4x NVIDIA L4 GPUs** (24GB each), perfect for testing Opaque's distributed training features.

## Quick Start

### Run Full Test Suite (Distributed)

```bash
# Run all distributed tests with 4 GPUs
torchrun --nproc_per_node=4 -m pytest tests/distributed/test_ddp_integration.py -v

# Run specific test class
torchrun --nproc_per_node=4 -m pytest tests/distributed/test_ddp_integration.py::TestGradientAggregation -v

# Run with 2 GPUs (for comparison)
torchrun --nproc_per_node=2 -m pytest tests/distributed/test_ddp_integration.py -v
```

### Run Example Training Script

```bash
# Full distributed training example
torchrun --nproc_per_node=4 examples/distributed_dp_training.py

# With fewer GPUs
torchrun --nproc_per_node=2 examples/distributed_dp_training.py
```

## What Gets Tested

The DDP integration tests validate:

1. **Core Utilities**
   - ✅ Distributed initialization detection
   - ✅ Rank and world size reporting (each GPU gets rank 0-3)

2. **Gradient Aggregation**
   - ✅ `all_reduce_gradients()` - sum across all 4 GPUs
   - ✅ `average_gradients()` - average across all 4 GPUs
   - Each GPU computes local gradients, then aggregates with all-reduce

3. **State Synchronization**
   - ✅ `sync_scalar()` - synchronize single values (e.g., loss)
   - ✅ `sync_state()` - synchronize AdaptiveClipState across all GPUs
   - Ensures all GPUs use same clip_norm after adaptation

4. **Deterministic Noise**
   - ✅ Each GPU generates different noise (seed + rank)
   - ✅ Reproducible across runs
   - Critical for privacy: different noise on each device

5. **End-to-End DP Training**
   - ✅ Clipped gradients (per-device with vmap)
   - ✅ Cross-device aggregation (all-reduce)
   - ✅ Noise injection (deterministic)
   - ✅ Adaptive clipping with sync

## Hardware Verification

Check GPU availability:

```bash
# List GPUs
nvidia-smi

# Check PyTorch can see all 4 GPUs
python -c "import torch; print(f'GPUs available: {torch.cuda.device_count()}')"
```

## Expected Behavior

### Single GPU (baseline)
```bash
uv run pytest tests/distributed/test_ddp_integration.py -v
# → 2 passed, 7 skipped (distributed tests skip)
```

### Multi-GPU (distributed)
```bash
torchrun --nproc_per_node=4 -m pytest tests/distributed/test_ddp_integration.py -v
# → 9 passed, 0 skipped (all tests run)
```

Each GPU runs the same test in parallel, coordinating via NCCL backend.

## Debugging

### Verbose Logging

```bash
# See per-rank output
torchrun --nproc_per_node=4 -m pytest tests/distributed/test_ddp_integration.py -v -s

# See distributed logs
TORCH_DISTRIBUTED_DEBUG=DETAIL torchrun --nproc_per_node=4 -m pytest tests/distributed/test_ddp_integration.py -v
```

### Run Single Test

```bash
torchrun --nproc_per_node=4 -m pytest \
    tests/distributed/test_ddp_integration.py::TestGradientAggregation::test_all_reduce_gradients_sum \
    -v -s
```

## Performance Notes

With 4 GPUs:
- **Data parallelism**: Each GPU processes different batch
- **Gradient aggregation**: ~4x throughput (embarrassingly parallel until all-reduce)
- **Memory**: Each GPU has 24GB, models up to ~20GB per GPU
- **Communication**: NCCL backend optimized for NVIDIA GPUs

## Troubleshooting

### NCCL Errors

If you see NCCL initialization errors:

```bash
# Check network connectivity
torchrun --nproc_per_node=4 python -c "import torch; torch.distributed.init_process_group('nccl'); print('OK')"

# Use Gloo backend (slower, more compatible)
NCCL_SOCKET_IFNAME=lo torchrun --nproc_per_node=4 -m pytest tests/distributed/test_ddp_integration.py -v
```

### Port Already in Use

```bash
# Specify different port
torchrun --nproc_per_node=4 --master_port=29501 -m pytest tests/distributed/test_ddp_integration.py -v
```

### GPU Out of Memory

```bash
# Use fewer GPUs
torchrun --nproc_per_node=2 -m pytest tests/distributed/test_ddp_integration.py -v

# Or smaller batch sizes in example
torchrun --nproc_per_node=4 examples/distributed_dp_training.py  # Already uses small batches
```
