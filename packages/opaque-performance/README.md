# opaque-performance

Fused Triton kernels (with pure-PyTorch fallbacks) and PyTorch-version
performance patches for vmap-style per-example training in Opaque.

## Install

```bash
pip install opaque-performance                # fallbacks only
pip install "opaque-performance[kernels]"     # + Triton fused kernels
```

## Quick start

```python
import opaque.performance as perf

perf.patch_checkpoint()                  # fix torch.utils.checkpoint under vmap(grad(...))
assert perf.is_checkpoint_patched()

from opaque.performance.kernels.swiglu import opaque_swiglu
```

## Layout

- `opaque.performance.kernels` — `opaque_swiglu`, `opaque_geglu_*`,
  `opaque_rms_norm`, `opaque_rope*`, `opaque_cross_entropy_loss`,
  `opaque_linear_cross_entropy_loss`, `opaque_lora_*` (fall back to pure
  PyTorch when Triton is unavailable).
- `opaque.performance.torch.checkpoint` — `patch_checkpoint`,
  `is_checkpoint_patched`, `unpatch_checkpoint` (not currently supported).

Independent of DP — usable as a plain Triton kernel library.
