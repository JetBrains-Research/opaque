# Cutting a release

Opaque uses **lockstep versioning**: all eight distributions
(`opaque`, `opaque-core`, `opaque-dpsgd`, `opaque-dpftrl`,
`opaque-auditing`, `opaque-performance`, `opaque-huggingface`,
`opaque-accounting`) release at the same version.

The version is **derived from git tags** via
[`setuptools-scm`](https://setuptools-scm.readthedocs.io/). There is no
`version = "..."` literal to bump in the Python `pyproject.toml` files —
the tag is the source of truth.

## Pipeline states

| State | Version | Where |
|---|---|---|
| PR / local dev | `0.X.Y.devN+g<sha>` (fallback `0.0.0` if no tag) | not published |
| Push to `main` | `0.X.Y.devN+g<sha>` (N = commits since last tag) | GCP Artifact Registry (dev channel) |
| Tag `v0.X.Y` | `0.X.Y` (clean, no dev/local suffix) | GCP + GitHub Release |

## Releasing

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

   This is optional — without it, dev builds read as `0.2.1.devN` (still
   PEP 440-valid and installable).

## Manual local release smoke-test

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

## Changelog conventions

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

## Yanking a bad release

If a release is broken, yank the GCP artifact and point users at the
previous version:

```bash
# No GitHub Release revocation; delete the tag + release, then recut.
git tag -d v0.2.0
git push --delete origin v0.2.0
# Delete the GitHub Release via gh CLI or UI, delete the GCP artifact.
```

Then cut `v0.2.1` with the fix.
