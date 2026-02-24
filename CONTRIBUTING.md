# Contributing to Opaque

Thank you for your interest in contributing to Opaque!

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

## Types of Contributions

We welcome all kinds of contributions:

- **Bug fixes**: Found something broken? Open an issue or submit a fix
- **Features**: New DP mechanisms, optimization, distributed training support
- **Documentation**: Clarifications, examples, tutorial notebooks
- **DP validation**: Cross-validation against JAX-Privacy, numerical comparisons
- **Performance**: Profiling, optimization, memory efficiency improvements
- **Examples**: Real-world use cases (LoRA fine-tuning, classification, etc.)

No contribution is too small!

---

## Repository Structure

The monorepo contains two main packages:

```
packages/
├── opaque/              # PyTorch DP-SGD library (Python)
│   ├── src/opaque/      # Source code (clipping, noise, accounting, sampling, optimizers)
│   └── tests/           # Test suite (111 tests, ~90% coverage)
└── opaque-accounting/   # Privacy accounting engine (Rust + Python bindings)
    ├── src/             # Rust implementation (PLD, mechanisms, composition)
    └── tests/           # Rust test suite (182 tests)

docs/                    # User documentation (getting-started, guides, tutorials, API)
examples/                # Example scripts and notebooks
```

**For Python changes**: Edit `packages/opaque/src/opaque/` and add tests to `packages/opaque/tests/`

**For accounting changes**: If it's just using the existing Rust API from Python, work in `packages/opaque-accounting/opaque_accounting/`. If you need to modify the Rust core, work in `packages/opaque-accounting/src/`.

---

## Finding Issues to Work On

**Good starting points**:
- [Issues labeled `good-first-issue`](https://github.com/JetBrains-Research/opaque/labels/good-first-issue)
- [Issues labeled `help-wanted`](https://github.com/JetBrains-Research/opaque/labels/help-wanted)
- [Open Discussions](https://github.com/JetBrains-Research/opaque/discussions)

**Before starting**:
1. Comment on the issue to say you're working on it (avoid duplicated effort)
2. Ask for clarification if the requirements aren't clear
3. Check existing tests to understand the expected behavior

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
- **Tutorials**: `docs/tutorials/`
- **API reference**: Auto-generated from docstrings

---

## Creating a Release

When you're ready to release a new version:

### Step 1: Trigger Release Preparation

```bash
# Via GitHub CLI
gh workflow run release.md --field version=0.1.0

# Or via GitHub UI: Actions → "Automated Release" → Run workflow
```

The workflow will:
- Analyze commits since the last release
- Generate release notes with AI assistance
- Update version numbers in `pyproject.toml` and documentation
- Create a Pull Request with all changes

### Step 2: Review and Merge

- Review the PR for accuracy
- Verify the AI-generated release notes
- Make any edits if needed
- Merge the PR when ready

### Step 3: Create and Push Tag

Once merged, create the release tag:

```bash
git checkout main && git pull
git tag v0.1.0
git push origin v0.1.0
```

### Step 4: Automatic Publishing

The `publish.yml` workflow runs automatically when the tag is pushed:
- Builds wheels for both `opaque-dp` and `opaque-accounting`
- Publishes to GCP Artifact Registry
- Creates a GitHub Release with notes and artifacts

**Monitor the release:**
```bash
gh run watch
```

---

## Getting Help

- **Questions**: Open a [Discussion](https://github.com/JetBrains-Research/opaque/discussions)
- **Bugs**: Open an [Issue](https://github.com/JetBrains-Research/opaque/issues)

---

## License

By contributing, you agree that your contributions will be licensed under the Apache 2.0 License.

---

## Acknowledgments

Thank you for contributing to differential privacy research and making LLM fine-tuning more privacy-preserving!

**Key Resources**:
- [JAX-Privacy](https://github.com/google-deepmind/jax_privacy) - Reference implementation
- [Full Opaque Documentation](docs/)
