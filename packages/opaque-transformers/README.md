# opaque-transformers

HuggingFace Transformers integration for Opaque: vmap-compatibility and
fused-kernel patches for LLaMA / Llama 3, Mistral, Qwen2/3, Phi-3,
Gemma(2), Granite, Cohere(2), and GPT-2.

## Install

```bash
pip install opaque-transformers                 # patches only
pip install "opaque-transformers[kernels]"      # + Triton fused kernels
pip install "opaque-transformers[peft]"         # + PEFT / LoRA support
```

Depends on `opaque-core`, `opaque-patches`, `opaque-dpsgd`, and `transformers>=4.57`.

## Quick start

Runtime compat patches (vmap-safe masking, collator / checkpoint hooks) are
applied when you construct :class:`opaque.transformers.trainer.Trainer`, or
when you call :func:`opaque.patches.apply_runtime_patches` explicitly (e.g. in
a notebook that uses HF primitives without the trainer).

```python
from opaque.patches import apply_runtime_patches, is_runtime_patched
from opaque.transformers import Trainer

apply_runtime_patches(compat=True)     # global runtime shims — idempotent
trainer = Trainer(model, args, ...)  # runtime compat + apply_model_patches on the model

assert is_runtime_patched()
```

## Layout

- **`opaque.api.transformers.trainer`** — Trainer implementation
  (`_trainer.py`, `_config.py`, `_state.py`, `_optim.py`,
  `_scheduler.py`, `_checkpoint.py`, `_distributed.py`,
  `_performance_kernels.py`, …).
- **`opaque.transformers`** / **`opaque.transformers.trainer`** — thin
  re-export façades (same pattern as `opaque-engine`: `opaque.api.*` for
  implementation, `opaque.*` for stable imports).
- **`opaque.patches.transformers`** — vmap-safe runtime patches and optional
  Triton kernel hooks (see `opaque.patches`).
