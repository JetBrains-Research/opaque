# opaque-huggingface

HuggingFace Transformers compatibility patches for Opaque:

- `opaque.compat.transformers` — vmap compatibility + kernel patches for LLaMA, Mistral, Qwen2/3, Phi3, Gemma(2), Granite, Cohere(2)

Patches are opt-in: call `opaque.compat.transformers.apply_transformers_patches()` explicitly.

Depends on `opaque-core`, `opaque-performance`, and `transformers>=4.57`.
