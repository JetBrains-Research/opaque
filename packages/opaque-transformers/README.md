# opaque-transformers

Hugging Face trainer integration for Opaque: DP-aware training loop
(`opaque.transformers.trainer.DPTrainer`) with TRL-style SFT/DPO support
(`opaque.transformers.trl`) and Hugging Face compatibility layers.

## Install

Install the root package as described in the [repository installation guide](https://github.com/JetBrains-Research/opaque#installation).
Use its `transformers` extra for trainer integration or `trl` for TRL config
conversion.

Depends on `opaque-engine`, `opaque-kernels`, `opaque-dpsgd`, `opaque-dpftrl`,
`opaque-accounting`, `opaque-optimizers`, `opaque-alignment`, plus
`transformers>=4.57`, `peft>=0.13`, and `datasets>=2.0`.

For Triton fused kernels (RoPE, RMSNorm, activation, cross-entropy), install
`opaque-kernels[transformers]` — kernels are a dependency of `opaque-kernels`
and gate on CUDA + Triton at runtime.

## Quick start

Runtime compat patches (vmap-safe masking, collator / checkpoint hooks) are
applied when you construct :class:`opaque.transformers.trainer.DPTrainer`, or
when you call :func:`opaque.transformers.patches.apply_runtime_patches` explicitly (e.g. in
a notebook that uses HF primitives without the trainer).

```python
from opaque.transformers.patches import apply_runtime_patches, is_runtime_patched
from opaque.transformers import DPTrainer

apply_runtime_patches(compat=True)  # global runtime shims — idempotent
trainer = DPTrainer(
    model, args, ...
)  # runtime compat + apply_model_patches on the model

assert is_runtime_patched()
```

## Layout

- **`opaque.api.transformers.trainer`** — DPTrainer implementation
  (`_dp_trainer.py`, `_config.py`, `_state.py`, `_optim.py`,
  `_scheduler.py`, `_checkpoint.py`, `_distributed.py`,
  `_performance_kernels.py`, …).
- **`opaque.transformers`** / **`opaque.transformers.trainer`** — thin
  re-export façades (same pattern as `opaque-engine`: `opaque.api.*` for
  implementation, `opaque.*` for stable imports).
- **`opaque.transformers.patches.families`** — vmap-safe runtime patches and optional
  Triton kernel hooks (see `opaque.transformers.patches`).
