# AGENTS.md

## Cursor Cloud specific instructions

This is a Python + Rust monorepo for **Opaque**, a functional DP-SGD library for PyTorch. See `README.md` and `CONTRIBUTING.md` for full docs.

### Packages

| Package | Path | Language | Build |
|---|---|---|---|
| `opaque-dp` | `packages/opaque/` | Python | setuptools |
| `opaque-accounting` | `packages/opaque-accounting/` | Rust + Python | maturin (PyO3) |

### Prerequisites

- Python 3.11+ (< 3.13), Rust stable (>= 1.70), `uv` package manager.
- `uv` is installed to `~/.local/bin`; ensure `PATH` includes it (the update script handles this).

### Key commands

All commands are documented in `README.md` and `CONTRIBUTING.md`. Quick reference:

```bash
uv sync --group dev --group compat     # Install all dev + HuggingFace deps (builds Rust extension)
uv run pytest -m "not gpu"             # Run non-GPU Python tests (~936 tests, ~13 min on CPU)
uv run ruff check packages/            # Lint
uv run ruff format --check packages/   # Format check
cargo test --workspace                 # Rust tests (~255 tests, ~3 min)
```

### Non-obvious notes

- `uv sync` triggers a full Rust build of `opaque-accounting` via maturin. First run takes ~30s; subsequent runs are cached.
- There is no application server or database. This is a pure library — testing is entirely via `pytest` and `cargo test`.
- GPU/CUDA tests are marked `@pytest.mark.gpu` and auto-skip when no GPU is present. No manual marker exclusion needed beyond `-m "not gpu"`.
- HuggingFace compat tests (`--group compat`) auto-skip via `pytest.importorskip()` if transformers/peft aren't installed.
- The `test_deep_heterogeneous_tree_no_recursion_error` test in accounting can be slow (~2 min on CPU).
- `OPAQUE_SKIP_COMPAT_PATCHES=all` env var disables auto-patching of HuggingFace models on `import opaque` (useful for debugging import issues).
