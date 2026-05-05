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

Depends on `opaque-core`, `opaque-performance`, and `transformers>=4.57`.

## Quick start

Patches are applied **automatically on import**. To opt out, set
`OPAQUE_SKIP_TRANSFORMERS_PATCHES=all` (or a comma-separated subset; see
`opaque.transformers.patches`) before importing `transformers`:

```python
import opaque.transformers as hf    # patches applied here

assert hf.is_patched()
hf.patch_all()                     # idempotent — safe to call again
```

## Layout

- `opaque.transformers.patches` — model-specific vmap + kernel patches
  (LLaMA, Mistral, Qwen2/3, Phi-3, Gemma/Gemma2, Granite, Cohere/Cohere2).
- `opaque.transformers.trainer` / `callbacks` / `integrations` / `data` /
  `models` — reserved for the upcoming DP-aware Trainer API.
