# Known Limitations

This document describes current limitations of Opaque and recommended workarounds.

## Gradient Checkpointing Incompatibility

**Status:** Known PyTorch limitation
**Impact:** Cannot use `torch.utils.checkpoint` with Opaque
**Workaround:** Use `microbatch_size` parameter instead

### Problem

Gradient checkpointing (`torch.utils.checkpoint.checkpoint`) is incompatible with `torch.func.vmap`, which Opaque uses for per-example gradient computation.

**Error you'll see:**
```python
RuntimeError: You tried to vmap over _NoopSaveInputs, but it does not have
vmap support. Please override and implement the vmap staticmethod or set
generate_vmap_rule=True.
```

### When This Happens

1. **Explicit checkpoint use:**
   ```python
   from torch.utils.checkpoint import checkpoint

   def forward(self, x):
       x = checkpoint(self.layer, x, use_reentrant=False)  # ← Breaks!
   ```

2. **HuggingFace Transformers:**
   ```python
   model = AutoModel.from_pretrained("gpt2")
   model.gradient_checkpointing_enable()  # ← Breaks!
   ```

3. **Libraries with built-in checkpointing:**
   - Some models enable checkpointing by default
   - Check model documentation

### Solution: Use Microbatching

Opaque provides **manual microbatch accumulation** that achieves similar memory savings without the vmap incompatibility.

**Basic Usage:**
```python
from opaque.clipping import clipped_grad

grad_fn, state = clipped_grad(
    loss_fn,
    argnums=0,
    batch_argnums=(1, 2),
    l2_clip_norm=1.0,
    microbatch_size=16,  # ← Process batch in chunks of 16
)
```

**Automatic Tuning (Recommended):**
```python
from opaque.profiling import find_max_microbatch_size
from opaque.clipping import clipped_grad

# Automatically find optimal microbatch size
optimal_size = find_max_microbatch_size(
    model=model,
    sample_batch=(sample_x, sample_y),
    loss_fn=loss_fn,
    l2_clip_norm=1.0,
    safety_margin=0.1,  # Keep 10% memory free
)

print(f"Using microbatch_size={optimal_size}")

# Use the optimal size
grad_fn, state = clipped_grad(
    loss_fn,
    microbatch_size=optimal_size,
    ...
)
```

### Disabling Gradient Checkpointing

**HuggingFace Transformers:**
```python
from transformers import AutoModel

model = AutoModel.from_pretrained("gpt2")

# ❌ DON'T: Enable checkpointing
# model.gradient_checkpointing_enable()

# ✅ DO: Keep it disabled (default)
# model.gradient_checkpointing is False by default

# Use Opaque with microbatching instead
grad_fn, state = clipped_grad(
    loss_fn,
    microbatch_size=8,  # Memory-efficient!
    ...
)
```

**Custom Models:**
Before:
```python
class MyModel(nn.Module):
    def forward(self, x):
        # ❌ This breaks with Opaque
        x = checkpoint(self.layer1, x, use_reentrant=False)
        x = checkpoint(self.layer2, x, use_reentrant=False)
        return x
```

After:
```python
class MyModel(nn.Module):
    def forward(self, x):
        # ✅ Regular forward pass
        x = self.layer1(x)
        x = self.layer2(x)
        return x

# Use microbatching in clipped_grad instead
grad_fn, state = clipped_grad(
    loss_fn,
    microbatch_size=16,  # ← Achieves similar memory savings
    ...
)
```

### Memory Comparison

| Technique | Memory Usage | Compute Cost | Opaque Compatible |
|-----------|-------------|--------------|-------------------|
| No optimization | O(n) | 1x | ✅ Yes |
| Gradient checkpointing | O(√n) | ~2x | ❌ No |
| Microbatching (size=m) | O(m) | 1x | ✅ Yes |

Where:
- n = batch size
- m = microbatch size (user controlled)

**Key insight:** Microbatching gives you similar memory control as checkpointing, without the vmap incompatibility!

### Why This Happens

- **PyTorch's checkpoint** uses `autograd.Function` internally
- **torch.func.vmap** requires functions to implement vmap rules
- The checkpoint `autograd.Function` doesn't have vmap support

This is tracked in [PyTorch Issue #165880](https://github.com/pytorch/pytorch/issues/165880).

### Future

When PyTorch fixes vmap + checkpoint compatibility:
- Opaque will automatically support checkpointing
- No code changes needed
- You can choose between checkpointing and microbatching

## See Also

- [Poisson Sampling & Microbatching Guide](user-guide/sampling.md)
- [API Reference - Clipping](api/core/clipping.md)
- [Tutorial - Sampling and Microbatching](tutorials/05_sampling_and_microbatching.ipynb)
