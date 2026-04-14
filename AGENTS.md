# Agents

## Cursor Cloud specific instructions

### Overview

Opaque is a functional DP-SGD library for PyTorch (monorepo: `packages/opaque` Python lib + `packages/opaque-accounting` Rust/PyO3 accounting engine). There are no services to run — it is a pure library. See `README.md` and `CLAUDE.MD` for architecture details.

### Prerequisites

- **Python 3.12** (system default; `>=3.11,<3.13` required)
- **Rust stable** (≥1.70, for building `opaque-accounting` via maturin)
- **uv** package manager (install with `pip install uv`; ensure `~/.local/bin` is on `PATH`)

### Common commands

Documented in `README.md` § Development and `CONTRIBUTING.md` § Testing. Quick reference:

| Task | Command |
|---|---|
| Install deps | `uv sync --group dev --group compat` |
| Lint | `uv run ruff check packages/` |
| Format check | `uv run ruff format --check packages/` |
| Python tests (CPU) | `uv run pytest packages/opaque/tests packages/opaque-accounting/tests -m "not gpu"` |
| Rust tests | `cargo test --workspace` |

### Non-obvious caveats

- `uv sync` builds the Rust extension (`opaque-accounting`) from source via maturin. If the Rust toolchain is missing, `uv sync` will fail during the build step. Make sure `rustc` and `cargo` are available.
- The `test_deep_heterogeneous_tree_no_recursion_error` test in `packages/opaque-accounting/tests/test_composition.py` is very CPU-intensive and can take several minutes on its own.
- `test_sdpa_eager_gradient_parity` in `packages/opaque/tests/compat/test_attention.py` is flaky under high system load (numerical precision). It passes when run in isolation.
- 136 tests are auto-skipped due to missing CUDA/GPU or optional dependency groups (`cross-validation`, `examples`). This is expected on CPU-only VMs.
- The `PATH` must include `~/.local/bin` for `uv` when installed via pip. The update script handles this.
