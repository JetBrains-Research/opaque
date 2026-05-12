# opaque-transformers

HuggingFace Transformers integration for Opaque: vmap-compatibility and
fused-kernel patches for LLaMA / Llama 3, Mistral, Qwen2/3, Phi-3,
Gemma(2), Granite, Cohere(2), and GPT-2.

## Install

```bash
pip install opaque-transformers                 # patches only
pip install "opaque-transformers[kernels]"      # + Triton fused kernels
pip install "opaque-transformers[peft]"         # + PEFT / LoRA support
pip install "opaque-transformers[hpo]"          # + Optuna, W&B, Ray Tune (DPTrainer HPO)
```

For Ray Tune only, `opaque-transformers[ray-hpo]` is enough.

Depends on `opaque-core`, `opaque-patches`, `opaque-dpsgd`, and `transformers>=4.57`.

## Quick start

Runtime compat patches (vmap-safe masking, collator / checkpoint hooks) are
applied when you construct :class:`opaque.transformers.trainer.DPTrainer`, or
when you call :func:`opaque.transformers.patch_all` explicitly (e.g. in a
notebook that uses HF primitives without the trainer).

```python
import opaque.transformers as hf

hf.patch_all()          # global runtime shims — idempotent
trainer = DPTrainer(model, args, ...)  # runtime compat + apply_model_patches on the model

assert hf.is_patched()
```

## Layout

- **`opaque.api.transformers.trainer`** — DPTrainer implementation
  (`_dp_trainer.py`, `_config.py`, `_hpo.py`, …).
- **`opaque.transformers`** / **`opaque.transformers.trainer`** — thin
  re-export façades (same pattern as `opaque-engine`: `opaque.api.*` for
  implementation, `opaque.*` for stable imports).
- **`opaque.patches.transformers`** — vmap-safe runtime patches and optional
  Triton kernel hooks (see `opaque.patches`).
