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

Opaque uses **lockstep versioning**: all eight distributions (`opaque`,
`opaque-core`, `opaque-dpsgd`, `opaque-dpftrl`, `opaque-auditing`,
`opaque-performance`, `opaque-huggingface`, `opaque-accounting`) release
at the same version. The version is derived from git tags via
[`setuptools-scm`](https://setuptools-scm.readthedocs.io/) — there is
no `version = "..."` literal to bump in the Python `pyproject.toml`
files.

### Pipeline states

| State | Version | Where |
|---|---|---|
| PR / local dev | `0.X.Y.devN+g<sha>` (fallback `0.0.0` if no tag) | not published |
| Push to `main` | `0.X.Y.devN+g<sha>` (N = commits since last tag) | GCP Artifact Registry (dev channel) |
| Tag `v0.X.Y` | `0.X.Y` (clean, no dev/local suffix) | GCP + GitHub Release |

### Releasing

1. **Open a release branch from `main`** and let the team review the
   automatically-generated changelog:

   ```bash
   git checkout main && git pull
   git checkout -b release/0.2.0
   uvx git-cliff --tag v0.2.0 --unreleased --config cliff.toml \
     --prepend CHANGELOG.md
   # Edit CHANGELOG.md as needed, commit, push, open PR against main.
   git add CHANGELOG.md
   git commit -m "docs: prepare v0.2.0 changelog"
   git push -u origin release/0.2.0
   ```

2. **Merge the release PR into `main`.** Do not tag yet — merge first so
   the tag points at the merge commit.

3. **Tag and push:**

   ```bash
   git checkout main && git pull
   git tag v0.2.0
   git push origin v0.2.0
   ```

4. **`release.yml` takes over.** On tag push CI:
   - Runs `.github/scripts/set_build_versions.sh` to pin the accounting
     Cargo/Python version and rewrite the umbrella's sub-package pins.
   - Builds wheels for all 7 Python packages + native wheels for
     `opaque-accounting` across linux-{amd64,arm64} and macos-arm64.
   - Uploads every wheel to the GCP Artifact Registry at the clean
     `0.2.0` version.
   - Creates a GitHub Release, body auto-filled from `git-cliff` output,
     wheels attached as release assets.

5. **Seed the next dev cycle.** To make `setuptools-scm` resolve future
   `main` commits to `0.3.0.devN` instead of `0.2.1.devN`, add a dev
   anchor tag:

   ```bash
   git tag v0.3.0.dev0
   git push origin v0.3.0.dev0
   ```

   Optional — without it, dev builds read as `0.2.1.devN` (still
   PEP 440-valid and installable).

### Manual local release smoke-test

Before tagging, verify builds work end-to-end:

```bash
uv sync --group dev --all-packages --extra all

# Dry-run the preflight script
bash .github/scripts/set_build_versions.sh 0.2.0
grep -E '^version|opaque-core' packages/opaque/pyproject.toml \
                               packages/opaque-accounting/pyproject.toml \
                               Cargo.toml

# Build every Python wheel
rm -rf dist
for pkg in opaque opaque-core opaque-dpsgd opaque-dpftrl opaque-auditing \
           opaque-performance opaque-huggingface; do
  (cd "packages/$pkg" && uv build --wheel --out-dir ../../dist)
done

# Build the accounting native wheel
(cd packages/opaque-accounting && uv build --wheel --out-dir ../../dist)

ls dist/   # expect 8 wheels, all at 0.2.0

# Inspect a wheel's metadata
unzip -p dist/opaque_core-*.whl '*/METADATA' | grep '^Version:'

# Revert the preflight's in-tree edits
git checkout -- packages/opaque/pyproject.toml \
               packages/opaque-accounting/pyproject.toml Cargo.toml
```

### Changelog conventions

We lean on [Conventional Commits](https://www.conventionalcommits.org/):

- `feat:` → **Added**
- `fix:` → **Fixed**
- `refactor:` / `change:` / `perf:` → **Changed**
- `docs:` → **Documentation**
- `test:` → **Tests**
- `ci:` / `build:` → **CI/CD**
- `delete:` → **Removed**
- `chore:` / `style:` → skipped from the public changelog
- Any commit with `!` after the type (`feat!:`) or a `BREAKING CHANGE:`
  footer → **Breaking**

`git-cliff` groups commits into those sections. Edit `CHANGELOG.md`
before the release PR merges if any entries need polish.

### Yanking a bad release

If a release is broken, delete the tag + release, then recut:

```bash
git tag -d v0.2.0
git push --delete origin v0.2.0
# Delete the GitHub Release via gh CLI or UI, delete the GCP artifact.
```

Then cut `v0.2.1` with the fix.

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
