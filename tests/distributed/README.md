# Distributed Training Tests

This directory contains tests for Opaque's distributed training primitives.

## Test Structure

- `test_core_utilities.py` - Core distributed utilities (is_initialized, get_rank, etc.)
- `test_gradient_aggregation.py` - Gradient aggregation (all_reduce_gradients, average_gradients)
- `test_state_synchronization.py` - State synchronization (sync_scalar, sync_state)
- `test_noise_determinism.py` - Deterministic noise generation in distributed mode
- `test_ddp_integration.py` - Full DDP integration tests (requires GPU)

## Running Tests

### Single-Device Tests

Most tests work in single-device mode (without torch.distributed initialization):

```bash
# Run all distributed tests (single device)
uv run pytest tests/distributed/ -v

# Run specific test file
uv run pytest tests/distributed/test_core_utilities.py -v
```

### Multi-GPU Tests

The DDP integration tests require actual multi-GPU setup. On the machine with 4 L4 GPUs:

```bash
# Run with all 4 GPUs
torchrun --nproc_per_node=4 -m pytest tests/distributed/test_ddp_integration.py -v

# Run with 2 GPUs
torchrun --nproc_per_node=2 -m pytest tests/distributed/test_ddp_integration.py -v

# Run specific test
torchrun --nproc_per_node=4 -m pytest tests/distributed/test_ddp_integration.py::TestGradientAggregation::test_all_reduce_gradients_sum -v
```

## Test Coverage

✅ **Core Utilities** (8 tests)
- Distributed initialization detection
- Rank and world size reporting
- All-reduce operation validation
- Barrier synchronization

✅ **Gradient Aggregation** (11 tests)
- PyTree gradient summing/averaging
- Nested structure handling
- Device and dtype preservation
- Edge cases (empty, single tensor, lists, tuples)

✅ **State Synchronization** (14 tests)
- Scalar synchronization
- Dataclass state synchronization
- AdaptiveClipState integration
- Field type handling (float, int, bool)

✅ **Noise Determinism** (13 tests)
- Seed-based reproducibility
- Distributed mode noise generation
- Device preservation
- PyTree support

✅ **DDP Integration** (6 tests) - requires GPU
- Multi-GPU gradient aggregation
- State synchronization validation
- Deterministic noise per rank
- End-to-end DP training step

## Hardware Requirements

- **Single-device tests**: Any device (CPU/GPU)
- **DDP integration tests**: Multiple CUDA GPUs
- **Recommended**: 4x L4 GPUs (24GB each)

## Notes

- Tests automatically skip when requirements are not met (e.g., no GPU, no distributed)
- Use `-m "not gpu"` to skip GPU tests: `pytest tests/distributed/ -m "not gpu"`
- DDP tests validate actual cross-device communication and synchronization
- All tests pass in both single-device and multi-device modes
