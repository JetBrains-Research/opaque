# Opaque – Agent Instructions

See `CLAUDE.MD` for architecture overview, project structure, and key design decisions.

## Cursor Cloud specific instructions

### Environment

- **Python 3.12**, **Rust stable** (≥1.70), **uv** package manager.
- No external services (databases, Docker, etc.) are needed — this is a pure computation library.
- `uv` is installed at `~/.local/bin/uv`; the update script ensures it is on `PATH`.

### Development commands

Standard commands are documented in `README.md` and `CONTRIBUTING.md`. Quick reference:

```bash
uv sync --group dev --group compat --all-packages   # Install all dev deps (builds Rust extension)
uv run pytest packages/opaque/tests -m "not gpu" -v  # Python tests (CPU-only)
uv run pytest packages/opaque-accounting/tests -v     # Accounting tests
cargo test --workspace                                 # Rust tests
uv run ruff check packages/                            # Lint
uv run ruff format --check packages/                   # Format check
```

### Gotchas

- The Rust native extension (`opaque-accounting`) is compiled automatically by `uv sync` via maturin. If you modify Rust code in `packages/opaque-accounting/src/`, re-run `uv sync` to rebuild.
- GPU/CUDA tests are marked `@pytest.mark.gpu` and are skipped automatically when no GPU is available. Do not pass `-m gpu` in a CPU-only environment.
- HuggingFace compatibility tests (`packages/opaque/tests/compat/`) require `--group compat` dependencies (transformers, peft). They are installed by default in this environment.
- `opaque_accounting.calibrate()` requires `param_max` within the mechanism's valid range (e.g., `noise_multiplier` ≤ 1.2 for Poisson-Gaussian). Check the error message for the accepted range.
- The `CalibrateResult` object uses `.param` and `.achieved` fields (not `.value`).
