# Release Process

This document describes the automated release process for `opaque-dp` and `opaque-accounting` packages.

## Overview

The project uses a three-tier release strategy optimized for early-stage rapid iteration:

1. **Per-Commit Dev Builds** (automatic on merge to `main`)
2. **AI-Assisted Release Preparation** (manual trigger)
3. **Release Publishing** (automatic on git tag)

## Versioning Strategy

**Development (default):**
- Local version: `0.0.0.dev0` (in `pyproject.toml`)
- Merge to main → builds as: `0.0.0.dev0+<github.run_number>` (e.g., `0.0.0.dev0+12345`)
- Published to: GCP Artifact Registry (dev channel)
- Purpose: Fast internal testing, no version management overhead

**Release:**
- Explicit version: e.g., `0.1.0`, `0.2.0`, `1.0.0`
- Published to: GCP Artifact Registry (production channel)
- Tagged in git as: `v0.1.0`, `v0.2.0`, etc.

## Package Names

- **PyPI/Registry name**: `opaque-dp` (main library), `opaque-accounting` (accounting engine)
- **Import name**: `opaque` (no `-dp` suffix)

```python
# Installation
pip install opaque-dp==0.1.0

# Usage
import opaque
from opaque import clipped_grad, gaussian_noise
```

## Automated Release Workflow

### Step 1: Prepare Release (AI-Assisted)

Trigger the agentic workflow to prepare release materials:

```bash
# Via GitHub CLI
gh workflow run release.md --field version=0.1.0

# Or via GitHub UI: Actions → "Automated Release Preparation" → Run workflow
```

**What it does:**
1. Analyzes git commits since last release
2. Generates user-friendly release notes with AI (using GitHub Copilot)
3. Updates `CHANGELOG.md` with new version section
4. Updates `docs/getting-started/installation.md` with new version
5. Updates `pyproject.toml` files with release version
6. Creates a Pull Request with all changes

**Review the PR:**
- Check the AI-generated release notes for accuracy
- Verify changelog formatting
- Ensure version numbers are correct
- Make any necessary edits

### Step 2: Merge and Tag

Once the PR looks good:

```bash
# Merge the PR (via GitHub UI or CLI)
gh pr merge <PR-NUMBER> --squash

# Pull the merged changes
git checkout main && git pull

# Create and push the release tag
git tag v0.1.0
git push origin v0.1.0
```

### Step 3: Publish (Automatic)

The `publish.yml` workflow triggers automatically on tag push:

1. Extracts version from git tag (e.g., `v0.1.0` → `0.1.0`)
2. Updates `pyproject.toml` with release version
3. Builds wheels for both packages
4. Authenticates to GCP via Workload Identity Federation
5. Publishes to GCP Artifact Registry
6. Creates GitHub Release with:
   - Release notes from `CHANGELOG.md`
   - Wheel artifacts attached

**Monitor the release:**
```bash
# Watch the workflow
gh run watch

# Or via GitHub UI: Actions → "Publish Release"
```

## Manual Release (Alternative)

If you prefer not to use git tags, you can manually dispatch the publish workflow:

```bash
gh workflow run publish.yml --field version=0.1.0
```

This will build and publish without requiring a git tag.

## Installation from GCP Artifact Registry

### For Development (Unstable)

```bash
# Configure pip to use GCP registry
pip install --index-url https://europe-west4-python.pkg.dev/jetbrains-ml4se-fed/jbr-fed-python/simple/ \
  opaque-dp==0.0.0.dev0+12345
```

### For Production (Stable)

```bash
# Install specific release
pip install --index-url https://europe-west4-python.pkg.dev/jetbrains-ml4se-fed/jbr-fed-python/simple/ \
  opaque-dp==0.1.0
```

### For Both Packages

```bash
# Install with accounting
pip install --index-url https://europe-west4-python.pkg.dev/jetbrains-ml4se-fed/jbr-fed-python/simple/ \
  opaque-dp==0.1.0 opaque-accounting==0.1.0
```

## Workflow Files

| File | Purpose | Trigger |
|------|---------|---------|
| `ci.yml` | Build dev wheels, run tests | Push/PR to main |
| `release.md` | AI-assisted release prep | Manual dispatch |
| `release.lock.yml` | Compiled agentic workflow | Auto-generated |
| `publish.yml` | Build and publish release | Git tag `v*` or manual |

## Architecture

```
┌─────────────────────────────────────────────────────┐
│ Developer commits to main                            │
└────────────────┬────────────────────────────────────┘
                 │
                 ▼
        ┌────────────────┐
        │   ci.yml       │  Builds 0.0.0.dev0+12345
        └────────┬───────┘  Publishes to GCP (dev)
                 │
                 │  (Daily iteration continues...)
                 │
                 ▼
        ┌────────────────┐
        │ release.md     │  Manual: "Prepare 0.1.0"
        └────────┬───────┘  AI generates notes, updates docs
                 │           Creates PR
                 ▼
        ┌────────────────┐
        │ Review PR      │  Human review & merge
        └────────┬───────┘
                 │
                 ▼
        ┌────────────────┐
        │ Create tag     │  git tag v0.1.0 && push
        └────────┬───────┘
                 │
                 ▼
        ┌────────────────┐
        │ publish.yml    │  Builds 0.1.0 wheels
        └────────┬───────┘  Publishes to GCP (prod)
                 │           Creates GitHub Release
                 ▼
        ┌────────────────┐
        │ Released! 🎉   │
        └────────────────┘
```

## Security

**GCP Authentication:**
- Uses Workload Identity Federation (WIF) - no secrets stored
- Service account: `jbr-fed-github@jetbrains-ml4se-fed.iam.gserviceaccount.com`
- WIF Provider: `federated-compute` (in `github` workload identity pool)
- Configured via repository variables:
  - `GCP_WORKLOAD_IDENTITY_PROVIDER`
  - `GCP_SERVICE_ACCOUNT_EMAIL`

**AI in Workflows:**
- GitHub Agentic Workflows uses GitHub Copilot (secure, sandboxed)
- Read-only access to repository by default
- Write operations via safe-outputs (PR creation only)
- All changes reviewed before merge

## Troubleshooting

**Build fails in ci.yml:**
```bash
# Check logs
gh run view --log

# Test locally
cd packages/opaque && uv build --wheel
cd ../opaque-accounting && uv build --wheel
```

**GCP authentication fails:**
```bash
# Check WIF configuration
gh variable list | grep GCP

# Verify service account has permissions
gcloud projects get-iam-policy jetbrains-ml4se-fed \
  --flatten="bindings[].members" \
  --filter="bindings.members:github-devcontainer-builder"
```

**AI workflow compilation fails:**
```bash
# Recompile
gh aw compile

# Check for errors in release.md frontmatter
```

**Version conflict:**
```bash
# Check existing tags
git tag -l

# Delete bad tag if needed
git tag -d v0.1.0
git push origin :refs/tags/v0.1.0
```

## Future Enhancements

Potential additions as the project matures:

1. **Nightly validation** - Full test suite + GPU tests daily
2. **TestPyPI staging** - Test releases before production
3. **Multi-platform wheel builds** - ARM64, Windows support
4. **Changelog validation** - Enforce conventional commits
5. **Version bump automation** - Semantic versioning from commit messages

## Questions?

See [CONTRIBUTING.md](../../CONTRIBUTING.md) or open an issue.
