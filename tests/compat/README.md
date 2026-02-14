# Compatibility Tests

Tests for HuggingFace Transformers compatibility with Opaque's vmap-based DP-SGD.

## Overview

These tests verify that our automatic vmap patches work correctly with various HuggingFace models, attention implementations, and PEFT methods.

## Installation

```bash
uv sync --group compat
```

This installs:
- `transformers>=4.57.0`
- `peft>=0.18.0`
- `pytest>=7.0.0`
- `pytest-typeguard>=4.0.0`

## Test Files

### `conftest.py`
Shared fixtures and helpers for all compat tests:
- `device` - Inherited from global conftest (auto-selects CUDA > MPS > CPU)
- `qwen2_config` - Small Qwen2 config for fast testing
- `qwen2_tokenizer` - Qwen2 tokenizer
- `prepare_lora_model()` - Helper to create LoRA models
- `run_clipped_grad_test()` - Helper to run clipped gradient tests

### `test_attention.py` (4 tests)
Test different attention implementations:
- ✅ eager (works on all devices)
- ✅ sdpa (works on all devices)
- ❌ flash_attention_2 (CUDA-only, incompatible - uses torch.nonzero)
- ❌ flex_attention (works on all devices, incompatible - tensor metadata issues)

### `test_features.py` (4 tests)
Test training features:
- ❌ Gradient checkpointing (incompatible - autograd.Function)
- ✅ Mixed precision (fp16, bfloat16) - 2 tests
- ✅ torch.compile

### `test_peft.py` (5 tests)
Test PEFT methods (all work on all devices):
- ✅ LoRA
- ✅ IA3
- ✅ Prefix Tuning
- ✅ P-Tuning
- ✅ Prompt Tuning

### `test_architectures.py` (4 tests, 2 skipped)
Test different model architectures:
- ✅ Qwen2 (standard MHA/GQA)
- ✅ Gemma2 (custom sliding window attention)
- ⏭️ DeepSeek (skipped - large download)
- ⏭️ Phi-2 (skipped - large download)

## Running Tests

```bash
# All compat tests
pytest tests/compat/ -v

# Specific test file
pytest tests/compat/test_attention.py -v

# Specific test
pytest tests/compat/test_peft.py::TestPEFTMethods::test_lora -v

# With markers
pytest -m compat -v                    # Only compat tests
pytest -m "compat and not slow" -v     # Fast compat tests only

# Include skipped tests (large downloads)
pytest tests/compat/ -v --run-slow
```

## Test Results

**Total**: 17 tests
- **Passing**: 15
- **Skipped**: 2 (large model downloads)

## Platform Support

Tests automatically adapt to your hardware using the global `device` fixture:

| Platform | Device | Tests Run | Tests Skipped | Notes |
|----------|--------|-----------|---------------|-------|
| **macOS Intel** | CPU | 16 | 1 (flash_attn) | Full CPU coverage |
| **macOS Apple Silicon** | MPS | 16 | 1 (flash_attn) | GPU acceleration via MPS |
| **Linux/Windows + NVIDIA** | CUDA | 17 | 0 | Full coverage including flash_attn |
| **Linux/Windows (no GPU)** | CPU | 16 | 1 (flash_attn) | Same as macOS Intel |

**Device priority**: CUDA > MPS > CPU (automatic selection)

## What Gets Tested

Each test:
1. Creates a small model with specific configuration
2. Applies LoRA or other PEFT method
3. Converts model to functional form
4. Runs `clipped_grad()` with per-example gradients
5. Verifies gradients are computed correctly

## Known Incompatibilities

### Flash Attention 2 ❌
- **Why**: Uses `torch.nonzero()` for unpadding, which produces variable-length outputs incompatible with vmap
- **Can it be fixed?**: No - would require rewriting the entire kernel, defeating the performance purpose
- **Workaround**: Use `eager` or `sdpa` attention instead (both work on CPU, CUDA, and MPS)

### flex_attention ❌
- **Why**: Tensor metadata assertion failures under vmap (upstream PyTorch issue)
- **Can it be fixed?**: Maybe in future PyTorch versions as flex_attention matures
- **Workaround**: Use `eager` or `sdpa` attention instead

### Gradient Checkpointing ❌
- **Why**: Uses `autograd.Function` which requires `setup_context` staticmethod for vmap compatibility
- **Can it be fixed?**: Theoretically yes, but requires changes to PyTorch's checkpointing implementation
- **Workaround**: Don't use gradient checkpointing with DP-SGD (memory-compute tradeoff)

**Recommendation**: Use **eager** or **sdpa** attention for maximum compatibility across all devices.

## Adding New Tests

### New Attention Implementation
Add to `test_attention.py`:
```python
def test_new_attention(self, qwen2_config, qwen2_tokenizer, device):
    qwen2_config._attn_implementation = "new_attention"
    model = prepare_lora_model(qwen2_config).to(device)
    grads, _ = run_clipped_grad_test(model, qwen2_tokenizer)
    assert len(grads) > 0
```

**Note**: Always include `device` fixture parameter and call `.to(device)` for cross-platform compatibility.

### New PEFT Method
Add to `test_peft.py`:
```python
def test_new_peft_method(self, qwen2_config, qwen2_tokenizer, device):
    from peft import NewPEFTConfig

    qwen2_config._attn_implementation = "eager"
    model = AutoModelForCausalLM.from_config(qwen2_config)

    config = NewPEFTConfig(...)
    model = get_peft_model(model, config).to(device)
    grads, _ = run_clipped_grad_test(model, qwen2_tokenizer)
    assert len(grads) > 0
```

### New Architecture
Add to `test_architectures.py`:
```python
def test_new_architecture(self, device):
    config = AutoConfig.from_pretrained("org/model")
    config.num_hidden_layers = 1
    config._attn_implementation = "eager"

    model = AutoModelForCausalLM.from_config(config)
    tokenizer = AutoTokenizer.from_pretrained("org/model")

    model = prepare_lora_model(config).to(device)
    grads, _ = run_clipped_grad_test(model, tokenizer)
    assert len(grads) > 0
```

**Key Pattern**: Always use `device` fixture and call `.to(device)` on models.

## References

- [HuggingFace Transformers](https://huggingface.co/docs/transformers)
- [PEFT Documentation](https://huggingface.co/docs/peft)
- [PyTorch vmap](https://pytorch.org/docs/stable/generated/torch.vmap.html)
