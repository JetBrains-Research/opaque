# Contributing to Opaque

Thank you for your interest in contributing to Opaque!

Please read our [Code of Conduct](CODE_OF_CONDUCT.md) before participating. It
explains the standards that keep the community welcoming and how to confidentially
report concerns.

## Quick Start

### Setup

```bash
# Clone repository
git clone https://github.com/JetBrains-Research/opaque.git
cd opaque

# Install the complete contributor environment
uv sync --group dev --all-packages --extra all

# Verify the CPU PR lane
uv run pytest -m "not cuda and not mps and not slow"
```

---

## Types of Contributions

We welcome all kinds of contributions:

- **Bug fixes**: Found something broken? Open an issue or submit a fix
- **Features**: New DP mechanisms, optimization, distributed training support
- **Documentation**: Clarifications, examples, tutorial notebooks
- **DP validation**: Cross-validation against JAX-Privacy, numerical comparisons
- **Performance**: Profiling, optimization, memory efficiency improvements
- **Examples**: Real-world training pipelines (Hugging Face LLMs, vision classification, etc.)

No contribution is too small!

---

## Repository Structure

The monorepo ships multiple distributions sharing the `opaque.*`
PEP 420 namespace. User-facing installation should go through the root
`opaque` package; implementation packages live under `packages/`:

```
pyproject.toml           # opaque — pins the curated sub-package bundle
README.md                # top-level description

packages/
├── opaque-base/         # Pure-Python serialization registry
├── opaque-engine/       # Torch substrate: pytree, clipping, scheduling, distributed support
├── opaque-optimizers/   # Functional optimizer chain
├── opaque-accounting/   # Rust/PyO3 PLD privacy accounting
├── opaque-dpsgd/        # DP-SGD noise, adaptive clipping, and sampling
├── opaque-dpftrl/       # Correlated-noise DP-FTRL mechanisms and sampling
├── opaque-auditing/     # Empirical privacy auditing
├── opaque-patches/      # PyTorch, Transformers, and Triton patches
├── opaque-transformers/ # Hugging Face trainer integration
└── opaque-alignment/    # DP-safe SFT and DPO primitives

docs/                    # User documentation (getting-started, guides, tutorials, API)
examples/                # Example scripts and notebooks
```

**For Python changes**: edit the relevant implementation under
`packages/opaque-<name>/src/opaque/api/` and add tests under the matching
`packages/opaque-<name>/tests/`. Public façades under `src/opaque/` contain
only re-exports; user-facing examples should import through those façades.

**For accounting changes**: Python façade and wrapper code lives under
`packages/opaque-accounting/src/opaque/`; the Rust crate is in
`packages/opaque-accounting/src/` with its manifest at
`packages/opaque-accounting/Cargo.toml`.

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
2. **Implement**: Make the test pass (minimal code to pass)
3. **Document**: Add docstrings with usage examples
4. **Refactor**: Improve code quality and structure
5. **Verify**: Run the full test suite with coverage

---

## Testing

### Dependency Groups and Extras

Opaque keeps three root `uv` dependency groups; everything else lives in
per-package `[project.optional-dependencies]`:

```bash
uv sync --group dev --all-packages --extra all       # Tests + lint: pytest, ruff, scipy, workspace extras
uv sync --group examples --all-packages --extra all  # Training examples: datasets, W&B, and workspace extras
uv sync --group docs --all-packages                  # MkDocs stack and documented package sources

# Package extras (compose with --extra):
#   opaque[transformers]             — Hugging Face + patching stack
#   opaque[dpftrl]                   — DP-FTRL mechanisms
#   opaque[auditing]                 — empirical privacy auditing
#   opaque-accounting[cross-validation] — dp-accounting, riskcal
uv sync --group dev --all-packages --extra transformers
```

### Running Tests

```bash
# Run all unit tests
uv run pytest

# With coverage
uv run pytest --cov=opaque --cov-report=html

# Specific test file
uv run pytest packages/opaque-engine/tests/clipping/test_clipped_fun.py -v
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

Gated Hugging Face models use the `@requires_hf_auth` skip-if helper from
`packages/opaque-transformers/tests/opaque_transformers/_helpers.py`. Set
`HF_TOKEN` / `HUGGINGFACEHUB_API_TOKEN` / `HUGGINGFACE_TOKEN` to run
those tests; otherwise they skip automatically.

Other tests use `pytest.importorskip()` for automatic dependency handling:
- Hugging Face tests: Skip if `transformers` is not installed (install via `opaque[transformers]`)
- Cross-validation: Skip if `dp-accounting` is not installed (install via `opaque-accounting[cross-validation]`)

No manual marker exclusion is needed—tests skip automatically when dependencies are missing.

### GPU and Multi-GPU Tests

Some tests require a CUDA GPU. They live under each package's
`tests/ddp/` directory (e.g. `packages/opaque-dpsgd/tests/ddp/`,
`packages/opaque-transformers/tests/ddp/`) and use `torch.distributed`
with the NCCL backend:

```bash
# Run CUDA tests (requires CUDA)
uv run pytest -m cuda -v

# Run distributed tests (requires 2+ GPUs)
uv run pytest packages/opaque-dpsgd/tests/ddp/ \
              packages/opaque-transformers/tests/ddp/ -v
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
- Line length: 88 characters
- Type hints for public APIs
- Google-style docstrings

---

## Pull Request Process

### Before Submitting

1. **Run tests**: `uv run pytest -m "not cuda and not mps and not slow"`
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
<type>(scope): <imperative subject>

<body>
```

**Types**: `feat` / `add`, `fix`, `refactor` / `change` / `perf`,
`docs`, `test`, `ci` / `build`, `chore` / `style`, `delete`. Append `!`
to mark a breaking change (e.g. `feat!:`). See [Changelog
conventions](#changelog-conventions) for how each type maps to a
release-notes section.

Since the repo squash-merges, the PR title is the commit message on
`main`. The PR gate runs [`action-semantic-pull-request`] to enforce
the above shape — no merge if the title doesn't parse.

[`action-semantic-pull-request`]: https://github.com/amannn/action-semantic-pull-request

**Example**:
```
feat(dpsgd): add clipped-grad example

- Add clipped_grad to opaque.dpsgd.clipping
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

Opaque uses **lockstep versioning**: the root distribution and all workspace
sub-packages release
at the same version. Python sub-package versions come from
[`setuptools-scm`](https://setuptools-scm.readthedocs.io/) — no
`version = "..."` literal to bump in `pyproject.toml` files.

Release notes live on the [GitHub Releases](https://github.com/JetBrains-Research/opaque/releases)
page. There is no `CHANGELOG.md` to maintain.

### Pipeline states

| State | Version | Where |
|---|---|---|
| PR push | `0.X.Y.devN+pr.<num>.g<sha>` | workflow artifacts on the PR's Actions run (14-day retention) |
| Push to `main` | `0.X.Y.devN+g<sha>` | JetBrains Packages (dev channel) |
| Published Release `vX.Y.Z` | `X.Y.Z` | JetBrains Packages + GitHub Release assets |

### Releasing

1. **Let the draft maintain itself.** On every main merge, `ci.yml`
   upserts a single open draft Release for the next-patch version
   (`v0.2.2` if last stable was `v0.2.1`). The draft body has three
   fenced sections — AI-summarized highlights, `git-cliff`'s "What's
   changed", and contributors — all refreshed on each main merge.
2. **Prep in the GitHub Releases UI.** Open the draft, edit the title
   or tag if you want a different bump (`v0.3.0` for a minor release,
   `v0.3.0.rc1` for a release candidate), and type hand-written prose
   anywhere outside the `<!-- *:begin --> ... <!-- *:end -->` fences
   — those survive the next refresh.
3. **Click Publish.** `release.yml` runs:
   - `tag-guard` verifies the tag's commit is reachable from `origin/main`.
   - Matrix builds: 7 Python wheels + `opaque-accounting` for
     linux-{amd64, arm64} and macos-arm64.
   - Wheels publish to JetBrains Packages at the clean version.
   - Wheels attach to the Release as assets.

### Release-note conventions

The draft body is generated from Conventional Commit messages via
`git-cliff`; keeping PR titles conventional is what makes this
pipeline work. See the [Commit Messages](#commit-messages) section
above for the accepted types and their mapping to release-note
sections.

### Manual local release smoke-test

Before publishing, verify builds work end-to-end:

```bash
uv sync --group dev --all-packages --extra all

# Dry-run the preflight script at a specific version
bash .github/scripts/set_build_versions.sh 0.2.0
grep -E '^version|opaque-(base|engine|optimizers|dpsgd|dpftrl|auditing|patches|transformers|alignment|accounting)' pyproject.toml \
                               packages/opaque-accounting/pyproject.toml \
                               Cargo.toml

# Build the opaque wheel (workspace root)
rm -rf dist
uv build --wheel --out-dir dist

# Build every sub-package wheel
for pkg in opaque-base opaque-engine opaque-optimizers opaque-dpsgd opaque-dpftrl \
                opaque-auditing opaque-patches opaque-transformers opaque-alignment; do
  (cd "packages/$pkg" && uv build --wheel --out-dir ../../dist)
done

# Build the accounting native wheel
(cd packages/opaque-accounting && uv build --wheel --out-dir ../../dist)

ls dist/   # expect 11 wheels, all at 0.2.0

# Inspect a wheel's metadata
unzip -p dist/opaque_engine-*.whl '*/METADATA' | grep '^Version:'

# Revert the preflight's in-tree edits
git checkout -- pyproject.toml \
               packages/opaque-accounting/pyproject.toml Cargo.toml
```

### Yanking a bad release

If a release is broken, delete the tag + GitHub Release, then recut:

```bash
git tag -d v0.2.0
git push --delete origin v0.2.0
# Delete the GitHub Release via gh CLI or UI; delete the wheel from
# JetBrains Packages (delete the published version from the package registry).
```

Then prep a fresh draft for `v0.2.1` with the fix.

---

## Getting Help

- **Questions**: Open a [Discussion](https://github.com/JetBrains-Research/opaque/discussions)
- **Bugs**: Open an [Issue](https://github.com/JetBrains-Research/opaque/issues)

---

## License

By contributing, you agree that your contributions will be licensed under the Apache 2.0 License.

---

## Acknowledgments

Thank you for contributing to differential privacy research and making private machine learning more accessible!

**Key Resources**:
- [JAX-Privacy](https://github.com/google-deepmind/jax_privacy) - Reference implementation
- [Full Opaque Documentation](docs/)
