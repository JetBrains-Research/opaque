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

```
packages/opaque/
├── src/opaque/
│   ├── __init__.py              # Public API, auto-patching on import
│   ├── accounting.py            # PLD-based privacy accounting (Rust engine)
│   ├── kernels/                 # Fused Triton kernels (4800+ LOC)
│   │   ├── swiglu.py            # SwiGLU activation
│   │   ├── geglu.py             # GeGLU Exact + Approx
│   │   ├── rope_embedding.py    # Rotary position embeddings
│   │   ├── cross_entropy.py     # Chunked cross-entropy
│   │   ├── linear_cross_entropy.py  # Fused linear + CE (from Apple CCE)
│   │   ├── lora.py              # Fused LoRA W/QKV/MLP
│   │   └── utils.py             # Triton helpers
│   ├── compat/transformers/     # HuggingFace model patching
│   │   ├── __init__.py          # Transformers patch orchestration
│   │   ├── _kernel_patches.py   # Triton kernel optimization patches
│   │   ├── _vmap_patches.py     # Vmap compatibility patch orchestration
│   │   ├── _shared.py           # Shared attention patches
│   │   └── _standard_models.py  # Model-specific patches
│   ├── clipping/                # Per-example gradient clipping
│   ├── noise/                   # Gaussian + correlated noise (MF)
│   ├── auditing/                # Empirical privacy auditing
│   │   ├── __init__.py          # Re-exports: coin_flip, one_run, loss_scores
│   │   ├── coin_flip.py         # CoinFlip class + coin_flip() factory
│   │   ├── one_run/             # One-run estimator (Steinke et al. 2023)
│   │   │   ├── estimate.py      # one_run() + OneRunEstimate
│   │   │   ├── roc.py           # Pareto frontier, TN/FN counts, TPR@FPR
│   │   │   └── stats.py         # p-value, epsilon binary search
│   │   └── attacks/             # Membership inference attacks
│   │       └── loss.py          # loss_scores() via vmap
│   ├── distributed/             # DDP utilities
│   ├── sampling/                # Poisson samplers
│   ├── profiling/               # Memory profiling
│   ├── random/                  # RNG key utilities
│   └── utils/                   # PyTree ops, make_functional
├── tests/                       # ~766 tests
│   ├── kernels/                 # Kernel tests (forward, backward, vmap, perf)
│   ├── compat/                  # HuggingFace compatibility tests
│   ├── clipping/                # Clipping tests
│   ├── noise/                   # Noise tests
│   ├── accounting/              # Privacy accounting tests
│   ├── auditing/                # Auditing tests
│   ├── distributed/             # DDP tests
│   ├── sampling/                # Sampler tests
│   ├── matrix_factorization/    # MF noise tests
│   ├── validation/              # End-to-end DP training validation
│   └── profiling/               # Profiling tests
└── README.md

docs/
├── index.md                     # Main documentation entry
├── getting-started/             # Installation, quickstart
├── user-guide/                  # Per-component guides
│   ├── clipping.md, noise.md, accounting.md, sampling.md
│   ├── huggingface.md           # HF model integration
│   ├── kernel-optimizations.md  # Triton kernel documentation
│   ├── memory-profiling.md      # Memory profiling guide
│   └── distributed.md, auditing.md, optimizers.md
├── api/                         # API reference
├── mechanisms/                  # Privacy mechanism details
├── development/                 # Internal docs
│   └── MEMORY_OPTIMIZATION_PLAN.md  # Kernel optimization plan
└── tutorials/                   # Jupyter notebooks

examples/
└── train_causal_lm.py           # End-to-end DP-SGD LoRA training script
```

---

## Key Architecture

### Kernel Optimization Pattern

All Triton kernels use a two-level `autograd.Function` pattern for `vmap(grad())` support:

1. `Opaque_Foo` — main entry. `forward()`/`backward()` for standard autograd, `vmap()` for functorch.
2. `_FooBackward` — backward wrapper with its own `vmap()` for batched backward.

When `vmap(grad(fn))` runs, functorch intercepts `.apply()` and routes to `vmap()` methods where tensors are regular and Triton kernels work directly.

### Auto-Patching

On `import opaque`, patches are applied to HuggingFace Transformers models automatically. Control with env vars (each accepts `all` or comma-separated names):
- `OPAQUE_SKIP_COMPAT_PATCHES=all` — disable all patching
- `OPAQUE_SKIP_TRANSFORMERS_PATCHES=all` or `=vmap,kernels`
- `OPAQUE_SKIP_TRANSFORMERS_VMAP_PATCHES=all` or `=shared,standard,gemma2,phi3`
- `OPAQUE_SKIP_TRANSFORMERS_KERNEL_PATCHES=all` or `=swiglu,rope,ce,fused_ce,lora`

### Supported Models

LLaMA, Mistral, Qwen2, Qwen3, Phi3, Gemma, Gemma2, Granite, Cohere, Cohere2.

### New-Style autograd.Function

All `autograd.Function` classes use the new-style API with `setup_context()` instead of passing `ctx` to `forward()`. This means `@torch.amp.custom_fwd`/`@custom_bwd` decorators are NOT compatible (PyTorch issue #132388). The default autocast behavior is correct: forward runs under caller's autocast, backward has autocast OFF.

---

## Common Commands

```bash
# Run all tests
uv run pytest packages/opaque/tests/ -v

# Run kernel tests only
uv run pytest packages/opaque/tests/kernels/ -v

# Lint and format
uv run ruff check packages/
uvx ruff format packages/

# Run Rust tests (accounting engine)
cargo test --workspace

# Training (Mellum-4b with DP-SGD LoRA)
uv run python examples/train_causal_lm.py --preset mellum-kstack --max_steps 100

# Training without kernel patches (baseline)
OPAQUE_SKIP_TRANSFORMERS_KERNEL_PATCHES=all uv run python examples/train_causal_lm.py --preset mellum-kstack --max_steps 100
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
