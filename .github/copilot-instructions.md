# Opaque — Copilot Instructions

## What This Is

Opaque is a PyTorch library for differentially private (DP) LLM fine-tuning with LoRA. It implements DP-SGD: per-example gradient clipping, calibrated Gaussian noise, and privacy accounting. The API is functional (not OOP), mirroring JAX-Privacy's design.

## Architecture

```
src/opaque/
├── clipping/          # Gradient clipping (clip_pytree, clipped_fun, clipped_grad, adaptive)
├── noise/             # Noise injection (gaussian, bounded, matrix factorization variants)
│   └── matrix_factorization/  # Streaming matrix, Toeplitz, sensitivity math
├── accounting/        # Privacy accounting — Python bindings to Rust crate
├── auditing/          # Empirical privacy auditing (score_by_loss, bootstrap)
├── sampling/          # Batch sampling (CyclicPoissonSampling)
├── profiling/         # Memory profiling utilities
├── utils/             # PyTree ops (optree), functional conversion
└── compat/            # HuggingFace transformers/peft integration

crates/dp-accounting/  # Rust privacy accounting (PyO3 → maturin wheels)
tests/                 # Mirrors src/ structure
examples/              # Training scripts (train_causal_lm.py)
```

## Key Patterns

- **Functional API**: Functions return `(result, state)` tuples. State is immutable. No classes with mutable state.
- **PyTrees**: Parameters are nested dicts of tensors, manipulated via `optree`. Use `tree_map`, `tree_leaves`.
- **vmap**: Per-example gradients via `torch.func.vmap` and `torch.func.grad`.
- **Noise API**: `noise_fn, state = gaussian_noise(...)` returns a callable + immutable state.
- **Accounting API**: `state = acc.create()` → `state = acc.compose_*(state, ...)` → `eps = acc.get_epsilon(state, delta)`.
- **Clipping**: `clipped_grad(loss_fn, ...)` is the high-level API. It delegates to `clipped_fun` internally.

## Build & Test

```bash
uv sync --group dev                    # Install dev dependencies
uv run pytest                          # Run tests (parallel via pytest-xdist)
uv run pytest -n 0                     # Sequential mode
uv run pytest -m "not slow"            # Skip slow tests
uv run pytest -m jax_validation        # JAX cross-validation only (needs --group jax-validation)
uv run ruff check src/ tests/          # Lint
uv run ruff format src/ tests/         # Format
```

The Rust accounting crate requires maturin: `cd crates/dp-accounting && maturin develop`.

## Conventions

- Python 3.11+, line length 88, Ruff for lint/format.
- Google-style docstrings with usage examples.
- Test files mirror source: `src/opaque/noise/gaussian_noise.py` → `tests/noise/test_gaussian_noise.py`.
- Tests use `pytest` with `hypothesis` for property-based testing where appropriate.
- Numerical validation against JAX-Privacy uses `atol=1e-5`.
- Prefer `torch.testing.assert_close` over manual tolerance checks.

## What NOT To Do

- Do not add mutable state to API functions. This is a functional library.
- Do not use `torch.utils._pytree` — use `optree` (it's the public, stable alternative).
- Do not import from `_helpers.py` or `_internal` modules in public API.
- Do not suppress errors silently. Fail-fast with descriptive exceptions.
- This is security-critical code (DP guarantees). Do not approximate or skip edge cases in clipping/noise.
