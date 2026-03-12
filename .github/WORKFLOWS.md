# GitHub Actions Workflows

This directory contains automated workflows for the Opaque project.

## Workflows

### 🧪 CI (`ci.yml`)

**Triggers**: Push to `main`, Pull Requests to `main`

**Jobs**:

- **Test**: Run pytest on Python 3.11 and 3.12
  - Executes full test suite with coverage
  - Uploads coverage to Codecov
- **Lint**: Check code formatting and linting with Ruff
  - Validates code style
  - Ensures code quality standards
- **Build**: Build Python package
  - Verifies package can be built
  - Uploads build artifacts

**Status**: ![CI](https://github.com/JetBrains-Research/opaque/actions/workflows/ci.yml/badge.svg)

### 📚 Documentation Deployment (`docs.yml`)

**Triggers**: Push to `main`, Manual workflow dispatch

**Jobs**:

- **Build**: Build MkDocs documentation
  - Installs dependencies with uv
  - Builds static site with `mkdocs build --strict`
  - Uploads site artifact
- **Deploy**: Deploy to GitHub Pages
  - Deploys built site to GitHub Pages
  - Available at: https://jetbrains-research.github.io/opaque

**Status**: ![Docs](https://github.com/JetBrains-Research/opaque/actions/workflows/docs.yml/badge.svg)

### ✅ Documentation Check (`docs-check.yml`)

**Triggers**: Pull Requests that modify documentation

**Jobs**:

- **Build Docs**: Verify documentation builds successfully
  - Runs on PRs touching docs, mkdocs.yml, or source code
  - Ensures PRs don't break documentation
  - Reports build errors early

**Status**: ![Docs Check](https://github.com/JetBrains-Research/opaque/actions/workflows/docs-check.yml/badge.svg)

## Setup Instructions

### For Repository Maintainers

#### 1. Enable GitHub Pages

1. Go to repository **Settings** → **Pages**
2. Under "Build and deployment":
  - **Source**: GitHub Actions
3. Save changes

The documentation will be automatically deployed to `https://jetbrains-research.github.io/opaque` on every push to `main`.

#### 2. Configure Codecov (Optional)

1. Sign up at [codecov.io](https://codecov.io)
2. Add the repository
3. Copy the upload token
4. Add as repository secret: `CODECOV_TOKEN`

#### 3. Branch Protection Rules

Recommended settings for `main` branch:

- ✅ Require a pull request before merging
- ✅ Require status checks to pass:
  - `Test on Python 3.11`
  - `Test on Python 3.12`
  - `Lint and Format Check`
  - `Build package`
  - `Build documentation (PR check)` (for docs changes)
- ✅ Require branches to be up to date

### For Contributors

No additional setup needed! Workflows run automatically on:

- Pull requests
- Pushes to main (after merge)

## Workflow Details

### Dependencies

All workflows use:

- **uv**: Fast Python package installer
- **Python 3.11/3.12**: Test matrix covers both versions
- **GitHub Actions**: v4/v5 for latest features

### Caching

uv automatically caches dependencies for faster builds.

### Artifacts

- **Test coverage**: Uploaded to Codecov
- **Build artifacts**: Available in workflow runs (30 days)
- **Documentation site**: Deployed to GitHub Pages

## Troubleshooting

### Documentation build fails

Check:

1. `mkdocs.yml` syntax is valid
2. All linked files exist
3. No broken cross-references
4. Source code docstrings are properly formatted

Run locally:

```bash
uv run --group docs mkdocs build --strict
```

### Tests fail in CI but pass locally

Check:

1. Python version matches (3.11 or 3.12)
2. All dependencies are in `pyproject.toml`
3. Tests don't depend on local files/environment
4. No hardcoded paths

### Lint failures

Run locally to fix:

```bash
uv run ruff format src/ tests/
uv run ruff check --fix src/ tests/
```

## Manual Workflow Triggers

Documentation can be manually deployed:

1. Go to **Actions** → **Deploy Documentation**
2. Click **Run workflow**
3. Select branch (usually `main`)
4. Click **Run workflow**

Useful for:

- Testing documentation changes
- Forcing a redeploy
- Recovering from deployment issues

## Performance

Typical workflow times:

- **CI (test)**: ~2-3 minutes
- **CI (lint)**: ~30 seconds
- **CI (build)**: ~1 minute
- **Docs build**: ~1 minute
- **Docs deploy**: ~30 seconds

Total PR check time: ~3-4 minutes

## Security

Workflows follow security best practices:

- ✅ Pin action versions (`@v4`, `@v5`)
- ✅ Minimal permissions (read-only where possible)
- ✅ No secrets in logs
- ✅ Dependabot updates for actions

## Badge URLs

For README.md:

```markdown
![CI](https://github.com/JetBrains-Research/opaque/actions/workflows/ci.yml/badge.svg)
![Docs](https://github.com/JetBrains-Research/opaque/actions/workflows/docs.yml/badge.svg)
![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)
![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)
```

---

**Questions?** See [CONTRIBUTING.md](../CONTRIBUTING.md) or open an issue.
