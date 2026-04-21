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

### Dependency Groups and Extras

Opaque keeps only two root `uv` dependency groups; everything else lives in
per-package `[project.optional-dependencies]`:

```bash
uv sync --group dev --all-packages          # Tests + lint: pytest, ruff, scipy, all workspace packages
uv sync --group examples --all-packages        # Examples runtime: torchopt, datasets, wandb (plus opaque packages)
uv sync --group docs                           # + mkdocs stack

# Package extras (compose with --extra):
#   opaque-huggingface[peft]        — HuggingFace + PEFT
#   opaque-huggingface[kernels]     — + Triton kernels
#   opaque-performance[kernels]     — Triton kernels
#   opaque-dpsgd[optimizers]        — torchopt bindings
#   opaque-dpftrl[optimizers]           — torchopt bindings
#   opaque-accounting[cross-validation] — dp-accounting, riskcal
uv sync --group dev --all-packages --extra peft --extra kernels
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

Three orthogonal markers are declared in the root `pyproject.toml`:

- `cuda` — requires CUDA; auto-skipped on non-CUDA hosts.
- `mps` — requires Apple Metal (MPS); auto-skipped on non-MPS hosts.
- `slow` — takes >5 s on CPU; excluded from PR CI, run on pushes to
  `main`.

```bash
# PR-equivalent lane (matches CPU CI)
uv run pytest -m "not cuda and not mps and not slow"

# CUDA tests only (requires a GPU)
uv run pytest -m cuda

# MPS tests only (requires Apple Metal)
uv run pytest -m mps

# Slow tests (run on push to main)
uv run pytest -m slow
```

Gated HuggingFace models use the `@requires_hf_auth` skipif helper from
`packages/opaque-huggingface/tests/huggingface/_helpers.py`. Set
`HF_TOKEN` / `HUGGINGFACEHUB_API_TOKEN` / `HUGGINGFACE_TOKEN` to run
those tests; otherwise they skip automatically.

Other tests use `pytest.importorskip()` for automatic dependency handling:
- HuggingFace tests: Skip if `transformers` not installed (install via `--extra huggingface` on the umbrella or `opaque-huggingface[peft]`)
- Cross-validation: Skip if `dp-accounting` not installed (install via `opaque-accounting[cross-validation]`)

No manual marker exclusion needed - tests skip automatically when dependencies are missing.

### GPU and Multi-GPU Tests

Some tests require a CUDA GPU. They live under each package's
`tests/distributed/` directory (e.g. `packages/opaque-dpsgd/tests/distributed/`,
`packages/opaque-huggingface/tests/distributed/`) and use `torch.distributed`
with the NCCL backend:

```bash
# Run CUDA tests (requires CUDA)
uv run pytest -m cuda -v

# Run distributed tests (requires 2+ GPUs)
uv run pytest packages/opaque-dpsgd/tests/distributed/ \
              packages/opaque-huggingface/tests/distributed/ -v
```

`@pytest.mark.cuda` tests auto-skip on hosts without CUDA; `@pytest.mark.mps`
tests auto-skip on hosts without Apple Metal.

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

### Documentation Versioning

Docs are deployed with `mike` and versioned on `gh-pages`.

- **Push to `main`** updates the rolling docs alias: `latest`
- **Push tag `vX.Y.Z`** publishes immutable docs version: `X.Y.Z`
- Default docs version remains `latest`

This keeps release docs stable while allowing continuous docs updates on `main`.

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

### Step 3: Automatic Publishing

After the PR is merged, the release pipeline automatically continues:
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
