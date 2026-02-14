# Opaque Test Suite Organization

This document describes the organization and purpose of different test directories.

## Test Structure

```
tests/
├── compat/                         # Compatibility tests for patches
│   ├── test_attention.py           # Attention implementation tests
│   ├── test_features.py            # Training feature tests
│   ├── test_peft.py                # PEFT method tests
│   └── test_architectures.py       # Multi-architecture tests
├── validation/                     # End-to-end DP training validation
│   └── test_lora_dp_training.py
├── integration/                    # Integration tests (planned)
├── clipping/                       # Core clipping functionality
├── noise/                          # Noise mechanisms
├── sampling/                       # Privacy sampling
└── utils/                          # Utility functions
```

## Test Categories

### 1. Compatibility Tests (`tests/compat/`)

**Purpose**: Verify that HuggingFace Transformers patches work correctly with various configurations.

**Files**:
- `test_attention.py` - Attention implementation tests
- `test_features.py` - Training feature tests
- `test_peft.py` - PEFT method tests
- `test_architectures.py` - Multi-architecture tests
- `conftest.py` - Shared fixtures and helpers

**Dependencies**: Install with `uv sync --group compat`
- Requires: `transformers>=4.57.0`, `peft>=0.18.0`

**What it tests**:
- ✅ Attention implementations (eager, SDPA)
- ❌ flash_attention_2 (incompatible with vmap - uses torch.nonzero)
- ❌ flex_attention (incompatible with vmap - tensor metadata issues)
- ✅ Mixed precision (fp16, bfloat16)
- ✅ CUDA/MPS/CPU support (cross-platform)
- ✅ torch.compile integration
- ✅ Multiple model architectures (Qwen2, Gemma2, DeepSeek, Phi-2)
- ✅ PEFT methods (LoRA, IA3, Prefix tuning, P-tuning, Prompt tuning)
- ❌ Gradient checkpointing (known incompatibility with vmap)

**Test Classes** (17 tests total):
- `TestAttentionImplementations` - Different attention backends (4 tests)
- `TestTrainingFeatures` - Checkpointing, mixed precision, compile (4 tests)
- `TestPEFTMethods` - LoRA, IA3, Prefix/P/Prompt tuning (5 tests)
- `TestMultiArchitectureCompatibility` - Architecture smoke tests (4 tests)

**Run with**:
```bash
# Standard run
pytest tests/compat/ -v

# Run only compat tests (skip others)
pytest -m compat -v

# Skip compat tests (for CI without transformers installed)
pytest -m "not compat" -v
```

**Results**: 15 passing, 2 skipped (large model downloads)

---

### 2. Validation Tests (`tests/validation/`)

**Purpose**: End-to-end validation of differential privacy training workflows.

**File**: `test_lora_dp_training.py`

**What it tests**:
- GPT-2 LoRA + DP training workflows
- Functional conversion and parameter partitioning
- Multiple training steps
- Privacy accounting integration
- Mellum-specific model tests

**Test Classes**:
- `TestGPT2LoRADPTraining` - GPT-2 specific workflows
- `TestMellumLoRADPTraining` - Mellum model tests
- `TestEndToEndDPTraining` - Complete training scenarios
- `TestMultiArchitectureModels` - Architecture-specific tests (to be reorganized)

**Run with**:
```bash
pytest tests/validation/ -v
```

---

### 3. Core Functionality Tests

#### `tests/clipping/`
- `test_clipped_fun.py` - Clipping function primitives
- `test_clipped_grad.py` - Gradient clipping API
- `test_adaptive.py` - Adaptive clipping

#### `tests/noise/`
- `test_noise.py` - Noise mechanisms (Gaussian, etc.)

#### `tests/sampling/`
- `test_poisson.py` - Poisson sampling for amplification

#### `tests/utils/`
- Utility function tests

---

## Running Tests

### Quick smoke test (compatibility)
```bash
pytest tests/compat/ -v
```

### Full compatibility suite (includes large model downloads)
```bash
pytest tests/compat/ -v --run-slow
```

### Validation tests (end-to-end)
```bash
pytest tests/validation/ -v
```

### All tests
```bash
pytest tests/ -v
```

### Specific test class
```bash
pytest tests/compat/test_transformers_patches.py::TestAttentionImplementations -v
```

---

## Test Development Guidelines

### Compatibility Tests
- **Purpose**: Fast-running tests that verify patch correctness
- **Models**: Use small configs (1-2 layers)
- **Coverage**: Test different configurations, not full training
- **Duration**: Each test should run in < 30 seconds

### Validation Tests
- **Purpose**: Verify real-world training scenarios work correctly
- **Models**: Can use larger models, more layers
- **Coverage**: Test complete workflows, multi-step training
- **Duration**: Can be slower (minutes per test)

### Adding New Tests

**For new attention implementation**:
- Add to `tests/compat/test_transformers_patches.py::TestAttentionImplementations`

**For new model architecture**:
- Add to `tests/compat/test_transformers_patches.py::TestMultiArchitectureCompatibility`
- Mark with `@pytest.mark.skip()` if it requires large downloads

**For new training feature**:
- Add end-to-end test to `tests/validation/test_lora_dp_training.py`

---

## Continuous Integration

The test suite is organized to support different CI strategies:

- **PR checks**: Run fast compat tests only (`tests/compat/`)
- **Nightly builds**: Run full validation suite including large models
- **Release checks**: Run everything with `--run-slow`

---

## Test Status Summary

| Category | Passing | Skipped | Notes |
|----------|---------|---------|-------|
| Compatibility | 15 | 2 | 17 total tests, 2 skipped (large downloads) |
| Validation | TBD | TBD | Being reorganized |
| Core | TBD | TBD | Existing tests |

---

## Dependency Groups

The project uses dependency groups to keep optional test dependencies separate:

```bash
# Core development (required for basic tests)
uv sync --group dev

# Compatibility testing (HuggingFace ecosystem)
uv sync --group compat

# Documentation building
uv sync --group docs

# Install everything
uv sync --all-groups
```

---

## Future Work

### Completed ✅
- ~~Add more PEFT method tests (IA3, prefix tuning)~~ ✅
- ~~Add Flash Attention 2 tests~~ ✅ (documented as incompatible)
- ~~Add flex_attention tests~~ ✅ (documented as incompatible)

### Planned Reorganization
1. Move API-specific tests (microbatching, noise, return values) to `tests/integration/`
2. Simplify `test_lora_dp_training.py` to focus only on end-to-end validation
3. Add CI configuration to skip compat tests when dependencies not installed
