# Opaque: Agent Briefing

**Project**: Functional DP-SGD library for PyTorch
**Status**: Production-ready, active optimization work
**Hardware**: H200 80GB GPU
**Python**: 3.11, package manager: `uv`

---

## What This Is

Opaque provides composable primitives for differentially private model training
in PyTorch. Built on `torch.func` (vmap, grad), every component uses explicit
state — no hooks, no subclassing, no hidden mutation.

Core DP-SGD pipeline: per-example gradient clipping (`clipped_grad`) + Gaussian
noise injection + privacy accounting (Rust PLD engine) + Poisson sampling.

Includes fused Triton kernels for memory-efficient DP training of large language
models with LoRA.

---

## Repository Structure

The library uses PEP 420 implicit namespace packages. `opaque`, `opaque.noise`,
`opaque.optimizers`, and `opaque.compat` are namespace packages (no `__init__.py`)
contributed to by multiple distributions.

```
packages/
├── opaque/                      # Meta-package (deps + accounting shim)
│   ├── src/opaque/accounting.py # Convenience re-export of opaque_accounting
│   └── tests/                   # Cross-mechanism + validation tests
├── opaque-core/                 # Shared primitives
│   └── src/opaque/
│       ├── _env.py, random/, utils/, clipping/, sampling/,
│       │   distributed/, profiling/
│       └── noise/types.py       # NoiseState base
├── opaque-dpsgd/                # DP-SGD mechanism
│   └── src/opaque/
│       ├── noise/gaussian.py, truncated_gaussian.py, per_group_noise.py
│       └── optimizers/adamw_bc.py
├── opaque-mf/                   # Matrix-factorization (DP-FTRL) mechanisms
│   └── src/opaque/
│       ├── noise/mf/            # BLT, BSR, BISR, JME, lambda-CGD, etc.
│       └── optimizers/adamw_jme.py
├── opaque-auditing/             # Empirical privacy auditing
│   └── src/opaque/auditing/     # one_run, coin_flip, attacks
├── opaque-performance/          # Fused Triton kernels + checkpoint patches
│   └── src/opaque/compat/       # kernels/, pytorch/
├── opaque-huggingface/          # HuggingFace Transformers patches
│   └── src/opaque/compat/transformers/
└── opaque-accounting/           # Rust-backed PLD accounting (maturin)

docs/
├── index.md                     # Main documentation entry
├── user-guide/                  # Per-component guides
├── api/                         # API reference
└── development/                 # Internal docs

examples/
├── train_causal_lm.py           # End-to-end DP-SGD LoRA training
└── train_dp_ftrl.py             # DP-FTRL training with MF mechanisms
```

---

## Key Architecture

### Kernel Optimization Pattern

All Triton kernels use a two-level `autograd.Function` pattern for `vmap(grad())` support:

1. `Opaque_Foo` — main entry. `forward()`/`backward()` for standard autograd, `vmap()` for functorch.
2. `_FooBackward` — backward wrapper with its own `vmap()` for batched backward.

When `vmap(grad(fn))` runs, functorch intercepts `.apply()` and routes to `vmap()` methods where tensors are regular and Triton kernels work directly.

### Auto-Patching (now opt-in)

Patches are **no longer applied on import**. Call them explicitly:

```python
from opaque.compat.pytorch import apply_pytorch_patches
from opaque.compat.transformers import apply_transformers_patches

apply_pytorch_patches()        # gradient-checkpointing under vmap
apply_transformers_patches()   # vmap + kernel + KV-cache patches for HF models
```

Control with env vars (each accepts `all` or comma-separated names):
- `OPAQUE_SKIP_TRANSFORMERS_PATCHES=all` or `=vmap,kernels`
- `OPAQUE_SKIP_TRANSFORMERS_VMAP_PATCHES=all` or `=shared,standard,gemma2,phi3`
- `OPAQUE_SKIP_TRANSFORMERS_KERNEL_PATCHES=all` or `=swiglu,rope,ce,fused_ce,lora`

### Namespace Packages (PEP 420)

`opaque`, `opaque.noise`, `opaque.optimizers`, `opaque.compat` are namespace
packages — no top-level `opaque/__init__.py`. Always import from the specific
submodule:

```python
from opaque.clipping import clipped_grad       # not "from opaque import clipped_grad"
from opaque.noise.gaussian import gaussian_noise  # not "from opaque.noise import gaussian_noise"
from opaque.optimizers.adamw_bc import adamw_bc   # not "from opaque.optimizers import adamw_bc"
```

### Supported Models

LLaMA, Mistral, Qwen2, Qwen3, Phi3, Gemma, Gemma2, Granite, Cohere, Cohere2.

### New-Style autograd.Function

All `autograd.Function` classes use the new-style API with `setup_context()` instead of passing `ctx` to `forward()`. This means `@torch.amp.custom_fwd`/`@custom_bwd` decorators are NOT compatible (PyTorch issue #132388). The default autocast behavior is correct: forward runs under caller's autocast, backward has autocast OFF.

---

## Common Commands

```bash
# Run ALL tests (every package)
uv run pytest packages/

# Run one package's tests
uv run pytest packages/opaque-core/tests/ -v
uv run pytest packages/opaque-dpsgd/tests/ -v
uv run pytest packages/opaque-mf/tests/ -v
uv run pytest packages/opaque-auditing/tests/ -v
uv run pytest packages/opaque-performance/tests/ -v
uv run pytest packages/opaque-huggingface/tests/ -v

# Cross-mechanism / validation tests (meta package)
uv run pytest packages/opaque/tests/ -v

# Lint and format
uv run ruff check packages/
uvx ruff format packages/

# Run Rust tests (accounting engine)
cargo test --workspace

# Training (Mellum-4b with DP-SGD LoRA)
uv run python examples/train_causal_lm.py --preset mellum-kstack --max-steps 100

# Training without kernel patches (baseline)
OPAQUE_SKIP_TRANSFORMERS_KERNEL_PATCHES=all uv run python examples/train_causal_lm.py --preset mellum-kstack --max-steps 100
```

---

## Experiment Tracking (W&B)

- Entity: `federated-compute`, Project: `opaque`
- Instance: `https://jetbrains.wandb.io`
- Always use `PYTHONUNBUFFERED=1` for real-time output
- Offline by default; goes online when `WANDB_API_KEY` is set
- Env vars: `WANDB_API_KEY`, `WANDB_BASE_URL`, `WANDB_ENTITY`, `WANDB_PROJECT`, `WANDB_NAME`

---

## Key Design Decisions

1. **Functional-first**: Immutable state, pure functions, explicit RNG keys
2. **vmap-native**: All operations support `torch.func.vmap` for per-example gradients
3. **Gradient checkpointing**: Supported via `compat.pytorch` patches (first-order only); combinable with microbatching
4. **Triton kernels**: Operation-level fusion instead of graph-level optimization
5. **Pre-shift in CE kernels**: Labels shifted in Python, not kernel, for trivial vmap merge
6. **Skip frozen weight gradients**: LoRA training skips `dC` for `lm_head` (~1/3 backward saved)

---

## Current Optimization Work

See `docs/development/MEMORY_OPTIMIZATION_PLAN.md` for full details.

**Completed:**
- Phase 1: Fused Triton kernels (SwiGLU, GeGLU, RoPE, CE)
- Phase 2: Fused LoRA operations (W, QKV, MLP)
- Phase 3: Chunked cross-entropy
- Phase 3b: Fused linear cross-entropy (ported from Apple CCE)
- Phase 5: Gradient checkpointing under vmap (~81% savings on Mellum-4b)

**Attempted / no code changes needed:**
- Phase 4: Dtype precision guards (new-style `autograd.Function` is incompatible with `@torch.amp.custom_fwd` / `@custom_bwd`, PyTorch #132388)

**Not started:**
- Phase 6: CPU offloading (frozen embeddings, activation offload)
- Phase 7: Research (4-bit quantization)

---

## Documentation

Human-facing docs: `docs/` (MkDocs, Material theme)
Development docs: `docs/development/`
This file is agent-only — users should read `docs/index.md` or `README.md`.
