---
on:
  workflow_dispatch:
    inputs:
      version:
        description: "Release version (e.g., 0.1.0, 0.1.1, 0.2.0)"
        required: true

permissions:
  contents: read

safe-outputs:
  create-pull-request:
    title-prefix: "Release: "
    base-branch: main
---

# Automated Release

Prepare and publish version `${{ github.event.inputs.version }}` of the `opaque-dp` Python library.

## Context

**Repository structure:**
- `packages/opaque/` - Main DP-SGD library (pure Python, package name: opaque-dp)
- `packages/opaque-accounting/` - Privacy accounting engine (Rust + Python bindings)
- `CHANGELOG.md` - Project changelog
- `docs/getting-started/installation.md` - Installation documentation

**Version strategy:**
- Repository always keeps `version = "0.0.0.dev0"` in pyproject.toml files
- Version is updated during build time only (not committed to repo)
- This ensures dev builds and releases use the same base files
- `opaque-dp` and `opaque-accounting` are published with the same release version
- During publish, dependency metadata is rewritten transiently so `opaque-dp` depends on `opaque-accounting==<release-version>`

## Your Tasks (Phase 1: Preparation)

### 0. Validate and Normalize Version

First, validate the user-provided version and normalize it to PEP 440:

1. Take the version: `${{ github.event.inputs.version }}`
2. Validate it's a proper version (e.g., `1.2.3`, `1.2.3rc1`, `1.2.3a1`, `1.2.3.dev1`)
3. If invalid, STOP and explain what formats are accepted
4. If valid but needs normalization (e.g., `1.2.3.rc1` → `1.2.3rc1`), normalize it
5. If you fixed user's input, inform them: "ℹ️ Normalizing your version: X → Y"
6. Use the normalized version for all subsequent steps (branch name, tag, etc.)

Valid formats (PEP 440):
- `1.2.3` - stable release
- `1.2.3rc1` - release candidate (no period before rc)
- `1.2.3a1` - alpha (use `a`, not `alpha`)
- `1.2.3b1` - beta (use `b`, not `beta`)
- `1.2.3.dev1` - dev version (period required before dev)

**Hard requirement:**
- The PR branch name MUST be `release/v<normalized-version>` (use normalized, not user input)
- The version MUST be PEP 440 compliant or you MUST fail with a clear error

### 1. Generate Release Notes

Analyze commits since the last release tag (or all commits if this is the first release). Create comprehensive, user-friendly release notes that include:

- **What's New**: Major features and improvements
- **Bug Fixes**: Issues resolved
- **Breaking Changes**: API changes that require user action (if any)
- **Performance**: Notable performance improvements
- **Documentation**: Significant documentation updates

Use: `git log $(git describe --tags --match 'v*' --abbrev=0 2>/dev/null || git rev-list --max-parents=0 HEAD)..HEAD --oneline`

Format the release notes in clean, engaging Markdown suitable for GitHub Releases.

**Hard requirements (do not skip):**
- You MUST derive release notes from Git history in this run, not from pre-existing text in `CHANGELOG.md`.
- Build an explicit commit range from the latest release tag (`v*`) to `HEAD`.
- Categorize commits by intent (Added/Changed/Fixed/Docs/Breaking).
- Exclude merge noise and bot-only maintenance commits unless they changed user-visible behavior.
- Include a `Commit Evidence` section in the PR body with short SHAs and subjects used to build the notes.

### 2. Update CHANGELOG.md

Prepend a new section to CHANGELOG.md:

```markdown
## [${{ github.event.inputs.version }}] - YYYY-MM-DD

### Added
- List new features

### Changed
- List changes to existing features

### Fixed
- List bug fixes
```

Populate this section from the commit analysis above. If an existing unreleased section conflicts with commit history, prefer commit history and update entries accordingly.

### 3. Update Installation Documentation

Update `docs/getting-started/installation.md` to reference the normalized version in installation commands.

**Important:** Do NOT update version in pyproject.toml files. They should remain as `version = "0.0.0.dev0"`.
Do NOT commit dependency pin updates either; publish handles release dependency pinning transiently.

### 4. Create Pull Request

Create a Pull Request with:
- Title: `Release v<normalized-version>` (use the normalized version from step 0, e.g., `1.2.3rc1`)
- Branch: `release/v<normalized-version>` (MUST use normalized version, not user input)
- Body: Include the generated release notes
- Base: `main`

Example: If user entered `1.2.3.rc1`, use `release/v1.2.3rc1` (normalized)

Before creating the PR, verify there is no existing open PR whose head branch starts with `release/v<normalized-version>`.

The PR description should include:
1. Generated release notes
2. Instructions for maintainer:
   ```
   📋 Release v${{ github.event.inputs.version }}
   
   Review changes:
   ✅ CHANGELOG.md updated
   ✅ Documentation updated with new version
   ⚠️  Version NOT changed in pyproject.toml (stays 0.0.0.dev0)
   
   After merging this PR:
   🤖 Phase 2 will automatically trigger (based on branch name release/v${{ github.event.inputs.version }}):
      - Build wheels with version ${{ github.event.inputs.version }}
      - Publish to GCP Artifact Registry
      - Create GitHub Release with tag v${{ github.event.inputs.version }}
   ```

---

## Phase 2: Build and Publish (Automatic after PR merge)

After the PR with branch starting with `release/v` is merged, the workflow automatically continues with version normalization and PEP 440 validation.

```bash
# Set version during build only (not committed)
cd packages/opaque && uv version ${{ github.event.inputs.version }} && uv build --wheel
cd packages/opaque-accounting && uv version ${{ github.event.inputs.version }} && uv build --wheel
```

### 6. Publish to GCP Artifact Registry

Authenticate and publish all wheels to:
`https://europe-west4-python.pkg.dev/jetbrains-ml4se-fed/jbr-fed-python/`

### 7. Create GitHub Release

Create GitHub Release with:
- Tag: `v${{ github.event.inputs.version }}`
- Title: `opaque-dp v${{ github.event.inputs.version }}`
- Body: Release notes from CHANGELOG.md
- Artifacts: All built wheel files

## Safety Checks

Before proceeding:
- ✅ Verify version is a stable semantic version `X.Y.Z` (no `dev`, `rc`, `alpha`, `beta`, suffixes)
- ✅ Confirm version doesn't already exist as a git tag
- ✅ Ensure CHANGELOG.md exists

If any check fails, create an issue instead explaining the problem.

