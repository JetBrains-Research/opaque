# Gradient Checkpointing Quick Reference

> **TL;DR**: Don't use `torch.utils.checkpoint` with Opaque. Use microbatching instead.

---

## ❌ What Doesn't Work

```python
import torch.utils.checkpoint as checkpoint

model.gradient_checkpointing_enable()
grads = clipped_grad(loss_fn, ...)(params, batch)
# RuntimeError: autograd.Function incompatible with vmap
```

**Why**: PyTorch's checkpointing uses `autograd.Function` which lacks vmap batching rules.

---

## ✅ What to Use Instead

```python
# Microbatching (coming in Phase 1B)
grad_fn = clipped_grad(
    loss_fn,
    l2_clip_norm=1.0,
    microbatch_size=32,  # Process 32 examples at a time
)
grads = grad_fn(params, large_batch)
```

**Why**: Reduces memory by processing batches in chunks. More effective for DP-SGD.

---

## 📊 Memory Usage Comparison

| Method | Memory Usage | Works with vmap? | Best for |
|--------|--------------|------------------|----------|
| Full batch | High (all gradients) | ✅ Yes | Small batches |
| Checkpointing | Medium (recompute activations) | ❌ No | N/A in Opaque |
| Microbatching | Low (chunk-sized) | ✅ Yes | Large batches |

---

## 🔍 Why Does JAX Work?

JAX-Privacy can use checkpointing:
```python
import jax

checkpointed_fn = jax.checkpoint(loss_fn)
grads = jax.vmap(jax.grad(checkpointed_fn))(params, batch)  # Works!
```

**Reason**: Functional composition - both are just function transformations.

---

## 📚 Learn More

- **Full analysis**: [GRADIENT_CHECKPOINTING_PLAN.md](GRADIENT_CHECKPOINTING_PLAN.md)
- **Executive summary**: [GRADIENT_CHECKPOINTING_SUMMARY.md](GRADIENT_CHECKPOINTING_SUMMARY.md)
- **Test explanation**: `tests/compat/test_features.py::TestGradientCheckpointing`

---

## 🎯 When to Use Microbatching

Use microbatching when:
- ✅ Training on large batches (>128 examples)
- ✅ Running into OOM errors
- ✅ Training 7B+ models with LoRA
- ✅ Using limited GPU memory

Don't need microbatching when:
- Small batches (<64 examples)
- Plenty of GPU memory
- Small models (<1B parameters)

---

## 🚀 Coming in Phase 1B

Full microbatching implementation with:
- Automatic chunk size selection
- Memory profiling integration
- Performance benchmarks
- Tutorial and examples

---

**Status**: ✅ Research complete | 📅 Implementation planned for Phase 1B
