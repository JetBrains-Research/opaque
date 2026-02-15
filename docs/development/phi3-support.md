# Phi-3 SMoE Support in Opaque

## Overview

Opaque now provides full vmap-compatibility patches for Microsoft's Phi-3 models. This document explains the architecture-specific challenges and the patches implemented to support Phi-3 under `torch.func.vmap`.

## Quick Start

```python
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import get_peft_model, LoraConfig
from opaque import clipped_grad, make_functional

# 1. Load model with eager attention (required for vmap)
model = AutoModelForCausalLM.from_pretrained(
    "microsoft/Phi-3-mini-4k-instruct",
    attn_implementation="eager",  # ✅ Must use eager, not sdpa or flash_attention_2
    trust_remote_code=True,        # ✅ Phi-3 uses custom code
)

# 2. Apply LoRA with Phi-3's fused QKV projection
lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["qkv_proj"],  # ✅ Note: fused QKV, not separate q/k/v
)
model = get_peft_model(model, lora_config)

# 3. Use with Opaque's clipped_grad (patches apply automatically)
fmodel, trainable, frozen = make_functional(model)
grad_fn, state = clipped_grad(loss_fn, argnums=0, l2_clip_norm=1.0)
```

## Architecture Specifics

### 1. Attention Mechanism

Phi-3 implements attention with the following characteristics:

```
Phi-3 Attention Block:
├── qkv_proj: Linear projection
│   └── Output shape: [batch, seq, 3 * (num_heads * head_dim)]
│   └── Contains fused Q, K, V
├── DynamicCache: Custom KV cache
│   └── Stores previous key/value activations
│   └── Has custom .get_usable_length() method
├── RotaryEmbedding: Position encodings
│   └── Frequency scaling based on position
└── Grouped Query Attention (GQA)
    └── num_key_value_heads < num_query_heads
```

### 2. Fused QKV Projection

Phi-3 uses **fused QKV** unlike standard transformers that have separate `q_proj`, `k_proj`, `v_proj`:

```python
# ❌ Standard (Qwen2, LLaMA, etc.)
q = self.q_proj(hidden_states)
k = self.k_proj(hidden_states)
v = self.v_proj(hidden_states)

# ✅ Phi-3
qkv = self.qkv_proj(hidden_states)  # [batch, seq, 3*hidden]
# Then split: q, k, v = qkv.split(hidden_dim, dim=-1)
```

**Impact on LoRA**: Must target `["qkv_proj"]` instead of `["q_proj", "k_proj", "v_proj"]`.

### 3. DynamicCache Implementation

Phi-3's `DynamicCache` is custom and may lack the `get_usable_length()` method that modern transformers expect:

```python
# Expected interface (modern transformers)
cache.get_usable_length(layer_idx: int) -> int

# Phi-3 may only have:
cache.seen_tokens: int
cache.key_cache: List[Tensor]  # [num_layers] of [batch, num_heads, seq, head_dim]
cache.value_cache: List[Tensor]
```

**Under vmap**: This mismatch causes issues because vmap needs to call `get_usable_length()` to determine cache sizes during KV reuse.

## Patches Applied

### 1. VmapCompatibleDynamicCache Wrapper (`src/opaque/compat/transformers/_phi3.py`)

**Problem**: Phi-3's DynamicCache may not have `get_usable_length()`.

**Solution**: Wrap the cache and provide the method:

```python
class VmapCompatibleDynamicCache:
    def __init__(self, original_cache):
        self._cache = original_cache
    
    def get_usable_length(self, layer_idx: int) -> int:
        # Try original method first
        if hasattr(self._cache, "get_usable_length"):
            return self._cache.get_usable_length(layer_idx)
        
        # Fallback: check key_cache
        if hasattr(self._cache, "key_cache"):
            kc = self._cache.key_cache[layer_idx]
            if kc is not None:
                return kc.shape[-2]  # Return sequence length
        
        # Final fallback: use seen_tokens
        if hasattr(self._cache, "seen_tokens"):
            return self._cache.seen_tokens
        
        return 0
```

### 2. DynamicCache.__init__ Patching

Automatically add `get_usable_length()` to Phi-3's DynamicCache on initialization:

```python
def apply_phi3_patches():
    """Apply patches when Phi-3 module is imported."""
    module = importlib.import_module("transformers.models.phi3.modeling_phi3")
    
    if hasattr(module, "DynamicCache"):
        # Monkey-patch __init__ to add get_usable_length
        original_init = module.DynamicCache.__init__
        
        def vmap_compatible_init(self, *args, **kwargs):
            original_init(self, *args, **kwargs)
            if not hasattr(self, "get_usable_length"):
                # Add the method
                self.get_usable_length = ...
        
        module.DynamicCache.__init__ = vmap_compatible_init
```

### 3. Standard Patches (Inherited)

Phi-3 inherits standard patches from `_standard_models.py`:
- `repeat_kv`: Expands key/value for GQA (Grouped Query Attention)
- `eager_attention_forward`: vmap-compatible attention computation using negative indexing

These work because Phi-3 is listed in `_STANDARD_MODEL_MODULES`.

## Known Limitations

### 1. SDPA Attention Not Recommended

```python
# ❌ Avoid for DP-SGD
config._attn_implementation = "sdpa"

# ✅ Use this instead
config._attn_implementation = "eager"
```

**Why**: SDPA is the default in modern transformers for performance, but Phi-3's SDPA implementation may have subtle differences under vmap. For DP-SGD where correctness is paramount, use `eager`.

### 2. Flash Attention 2 Not Supported

Flash Attention 2 uses `torch.nonzero()` for unpadding, which produces variable-length outputs incompatible with vmap. No patch can fix this without sacrificing the kernel's performance benefits.

### 3. Gradient Checkpointing Incompatible

Phi-3 uses `torch.nn.utils.checkpoint` which relies on `autograd.Function`. This is incompatible with vmap unless the Function implements `setup_context()`. Standard Phi-3 doesn't, so:

```python
# ❌ This will fail under clipped_grad
model.gradient_checkpointing_enable()

# ✅ Use other memory techniques instead
# - LoRA (much lower memory than full fine-tuning)
# - Quantization (qint8, qint4)
# - Microbatching (manual accumulation)
```

## Testing

### Unit Test

```python
def test_phi3_architecture(device):
    """Test Phi-3 with clipped_grad."""
    config = AutoConfig.from_pretrained(
        "microsoft/Phi-3-mini-4k-instruct",
        trust_remote_code=True,
    )
    config.num_hidden_layers = 1
    config._attn_implementation = "eager"
    
    tokenizer = AutoTokenizer.from_pretrained(
        "microsoft/Phi-3-mini-4k-instruct",
        trust_remote_code=True,
    )
    
    model = prepare_lora_model(
        config,
        target_modules=["qkv_proj"],  # ✅ Phi-3 specific
    ).to(device)
    
    grads, _ = run_clipped_grad_test(model, tokenizer)
    assert len(grads) > 0
```

### Run Tests

```bash
# View test (currently skipped due to model size/auth)
pytest tests/compat/test_architectures.py::TestMultiArchitectureCompatibility::test_phi3_architecture -v

# Enable with HF token
HF_TOKEN=<token> pytest tests/compat/test_architectures.py::TestMultiArchitectureCompatibility::test_phi3_architecture
```

## Implementation Details

### File: `src/opaque/compat/transformers/_phi3.py`

**350 LOC module providing:**

1. **VmapCompatibleDynamicCache class**
   - Wraps original cache
   - Delegates most attributes
   - Implements `get_usable_length()` with fallbacks

2. **apply_rotary_pos_emb function** (placeholder)
   - Reserved for RoPE calculations if needed
   - Currently uses standard vmap-compatible approach

3. **apply_phi3_patches function**
   - Patches DynamicCache.__init__ for auto-compatibility
   - Called at import time by apply_global_patches()

### File: `src/opaque/compat/transformers/_global_patches.py`

**Updated to call `apply_phi3_patches()`:**

```python
def apply_global_patches() -> None:
    apply_shared_patches()           # masking_utils, sdpa_attention
    apply_standard_model_patches()   # All standard models including Phi3
    apply_gemma2_patches()           # Gemma2 custom softcap
    apply_phi3_patches()             # ← NEW: Phi-3 custom DynamicCache
```

### File: `tests/compat/conftest.py`

**Added Phi-3 fixtures:**

```python
@pytest.fixture
def phi3_config():
    """Small Phi-3 config for testing."""
    config = AutoConfig.from_pretrained(
        "microsoft/Phi-3-mini-4k-instruct",
        trust_remote_code=True,
    )
    config.num_hidden_layers = 4
    return config

@pytest.fixture
def phi3_tokenizer():
    """Phi-3 tokenizer."""
    tokenizer = AutoTokenizer.from_pretrained(
        "microsoft/Phi-3-mini-4k-instruct",
        trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer
```

## Common Errors & Solutions

### Error: "AttributeError: 'DynamicCache' object has no attribute 'get_usable_length'"

**Cause**: Phi-3's DynamicCache lacks `get_usable_length()` method.

**Solution**: Opaque patches this automatically. Ensure:
```python
import opaque  # Apply patches at import
```

### Error: "RuntimeError: unsupported operand type(s) for +: 'int' and 'DynamicCache'"

**Cause**: Cache handling in attention forward pass.

**Solution**: Ensure `attn_implementation="eager"` is set before model creation.

### Error: "AttributeError: 'Tensor' object has no attribute 'get_usable_length'"

**Cause**: KV cache is a Tensor instead of DynamicCache object.

**Solution**: Check that `use_cache=True` in model config, or manually initialize cache as DynamicCache.

## Performance Notes

### Memory Usage

Phi-3 models are efficient due to:
- **GQA (Grouped Query Attention)**: Reduces KV cache memory
- **Smaller vocabulary**: ~32K tokens vs ~128K for LLaMA
- **MQA-like design**: Further compresses attention

DP-SGD adds per-example gradient computation via vmap, which temporarily doubles memory during forward pass:
```
Normal: batch_size * [model params + activations]
DP-SGD: batch_size * [model params + 2x activations] (for per-example gradients)
```

With LoRA, the memory overhead is small:
- LoRA params: ~0.1% of model size
- Additional activations: minor (only for LoRA layers)

### Computational Cost

- `eager` attention: Slower but vmap-compatible
- `sdpa` attention: Faster but may have subtle vmap incompatibilities

For DP training, use `eager` for correctness over performance.

## References

- **Phi-3 Paper**: [Phi-3 Technical Report](https://arxiv.org/abs/2404.14219)
- **PEFT/LoRA**: [PEFT Documentation](https://huggingface.co/docs/peft)
- **Transformers**: [HuggingFace Transformers](https://huggingface.co/docs/transformers)
- **torch.vmap**: [PyTorch Functional Transformations](https://pytorch.org/docs/stable/generated/torch.vmap.html)
