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

## ✅ What to Use Instead (Already Available!)

```python
# Microbatching is already implemented!
grad_fn, clip_state = clipped_grad(
    loss_fn,
    l2_clip_norm=1.0,
    microbatch_size=32,  # Process 32 examples at a time
)
grads, new_state = grad_fn(params, large_batch, state=clip_state)
```

**Why**: Reduces memory by processing batches in chunks. More effective for DP-SGD.

**Where to find it**:
- See `examples/microbatching_demo.py` for comprehensive tutorial
- Used in `examples/train_causal_lm.py` and `examples/train_qwen.py`
- Implemented via PyTorch's `vmap(..., chunk_size=microbatch_size)`

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

## 🚀 Status: Already Available!

Microbatching has been implemented and is ready to use:
- ✅ Available in `clipped_grad()` and `clipped_fun()`
- ✅ Thoroughly tested (tests/clipping/test_clipped_fun.py)
- ✅ Used in production examples
- ✅ Tutorial available: examples/microbatching_demo.py

```bash
# Try the demo
python examples/microbatching_demo.py

# Or use in training
python examples/train_causal_lm.py --microbatch_size 4
```

For more details, see:
- examples/microbatching_demo.py (comprehensive tutorial)
- docs/development/GRADIENT_CHECKPOINTING_PLAN.md (technical analysis)
- docs/development/GRADIENT_CHECKPOINTING_SUMMARY.md (quick reference)
- examples/train_causal_lm.py (real-world usage with LLMs)

---

**Status**: ✅ Implementation complete | 📚 Ready to use today
