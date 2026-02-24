# Contributing to Opaque

Thank you for your interest in contributing to Opaque!

**For comprehensive development documentation, see [docs/development/](docs/development/).**

---

## Quick Start

### Setup

```bash
# Clone repository
git clone https://github.com/JetBrains-Research/opaque.git
cd opaque

# Install dependencies
uv sync

# Verify installation
uv run pytest
```

---

## Development Philosophy

1. **Correctness First**: Validate against JAX-Privacy before optimizing
2. **Explicit > Implicit**: Fail fast rather than hide issues
3. **Documentation Required**: Code without docs is incomplete
4. **Security-Critical**: DP guarantees depend on correct implementation

---

## TDD Workflow

Opaque follows a Test-Driven Development workflow:

1. **Test First**: Write failing test defining the API
2. **Implement**: Make test pass (minimal code to pass)
3. **Document**: Add docstrings with usage examples
4. **Refactor**: Improve code quality and structure
5. **Verify**: Run full test suite with coverage

**See**: [Contributing Guide](docs/development/contributing.md)

---

## Testing

### Dependency Groups

Opaque uses `uv` dependency groups to separate test dependencies:

```bash
uv sync                          # Core only (clipping, noise, accounting)
uv sync --group dev              # + pytest, ruff, scipy
uv sync --group compat           # + transformers, peft (HuggingFace tests)
uv sync --group cross-validation # + dp-accounting, riskcal (reference comparison)
uv sync --group examples         # + datasets, torchopt, jupyter, matplotlib
uv sync --group docs             # + mkdocs
uv sync --all-groups             # Everything
```

### Running Tests

```bash
# Run all unit tests
uv run pytest

# With coverage
uv run pytest --cov=opaque --cov-report=html

# Specific test file
uv run pytest packages/opaque/tests/clipping/test_clipped_fun.py -v
```

### Test Markers and Filtering

Tests use pytest markers for filtering:

```bash
# GPU tests (requires CUDA or MPS)
uv run pytest -m gpu

# Non-GPU tests only
uv run pytest -m "not gpu"
```

Other tests use `pytest.importorskip()` for automatic dependency handling:
- HuggingFace tests: Skip if `transformers` not installed (requires `--group compat`)
- Cross-validation: Skip if `dp-accounting` not installed (requires `--group cross-validation`)

No manual marker exclusion needed - tests skip automatically when dependencies are missing.

### GPU and Multi-GPU Tests

Some tests require a CUDA GPU. These are located in `packages/opaque/tests/distributed/` and
use `torch.distributed` with NCCL backend:

```bash
# Run GPU tests (requires CUDA or MPS GPU)
uv run pytest -m gpu -v

# Run distributed tests (requires 2+ GPUs)
uv run pytest packages/opaque/tests/distributed/ -v
```

GPU tests are marked with `@pytest.mark.gpu` and are automatically filtered in CI.

---

## Code Quality

```bash
# Format code
uv run ruff format packages/

# Check linting
uv run ruff check packages/

# Fix auto-fixable issues
uv run ruff check --fix packages/
```

**Standards**:
- Line length: 100 characters
- Type hints for public APIs
- Google-style docstrings

---

## Pull Request Process

### Before Submitting

1. **Run tests**: `uv run pytest packages/opaque/tests packages/opaque-accounting/tests`
2. **Format code**: `uv run ruff format --check packages/`
3. **Check linting**: `uv run ruff check packages/`
4. **Update docs**: Ensure docstrings are complete

### PR Checklist

- [ ] Tests pass locally
- [ ] Code is formatted with Ruff
- [ ] Docstrings follow Google style
- [ ] Type hints added for public APIs
- [ ] Linked to relevant issue(s)

---

## Commit Messages

**Format**:
```
<type>: <subject>

<body>
```

**Types**: `feat`, `fix`, `docs`, `test`, `refactor`, `perf`, `chore`

**Example**:
```
feat: implement basic clipped_grad

- Add clipped_grad function to opaque.core.clipping
- Support single parameter and batch data
- Include tests comparing against JAX-Privacy

Closes #5
```

---

## Documentation

### Build Docs Locally

```bash
# Install docs dependencies
uv sync --group docs

# Serve with live reload
uv run mkdocs serve

# Build static site
uv run mkdocs build
```

### Writing Docs

- **User guides**: `docs/user-guide/`
- **API reference**: Auto-generated from docstrings
- **Development**: `docs/development/`

---

## Getting Help

- **Questions**: Open a [Discussion](https://github.com/JetBrains-Research/opaque/discussions)
- **Bugs**: Open an [Issue](https://github.com/JetBrains-Research/opaque/issues)

---

## Detailed Documentation

**For comprehensive guides, see:**

- [Contributing Guide](docs/development/contributing.md) - Full contribution guidelines

---

## License

By contributing, you agree that your contributions will be licensed under the Apache 2.0 License.

---

## Acknowledgments

Thank you for contributing to differential privacy research and making LLM fine-tuning more privacy-preserving!

**Key Resources**:
- [JAX-Privacy](https://github.com/google-deepmind/jax_privacy) - Reference implementation
- [Full Opaque Documentation](docs/)
