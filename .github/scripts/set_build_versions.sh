#!/usr/bin/env bash
#
# set_build_versions.sh — write the current git-derived version into every
# build artifact that can't be handled by setuptools-scm natively:
#
#   1. `packages/opaque-accounting/pyproject.toml` — maturin reads the wheel
#      version from `[project] version`; maturin doesn't know about
#      setuptools-scm.
#   2. `packages/opaque-accounting/Cargo.toml` — the Rust crate version must
#      match the Python wheel version for the PyO3 extension.
#   3. `packages/opaque/pyproject.toml` — the umbrella metadata pins its
#      sub-packages with `opaque-*==<version>` so `pip install opaque`
#      resolves consistently. These pins need rewriting for dev/release
#      builds because dynamic versioning can't expand them.
#
# For Python sub-packages the script also exports
# `SETUPTOOLS_SCM_PRETEND_VERSION` via `$GITHUB_ENV` so every wheel carries
# the same version — setuptools-scm's own git-describe would otherwise pick
# up the dirty tree (this script mutates pyprojects/Cargo.toml before the
# build) and produce a drifted `.dev0+g<sha>.d<date>` string.
#
# Intended to run in CI as a preflight before `uv build`. On local dev,
# leave pyprojects / Cargo.toml untouched — uv workspace resolution and
# setuptools-scm fallbacks handle that case.
#
# Usage: bash .github/scripts/set_build_versions.sh [VERSION]
#
# If VERSION is omitted, derive it from `git describe --tags --match 'v*'`
# and normalize to PEP 440:
#
#   v0.2.0                         → 0.2.0                       (release tag)
#   v0.2.0-dirty                   → 0.2.0+d20260422             (release + dirty)
#   v0.2.0-5-g810b6b2              → 0.2.1.dev5+g810b6b2         (post-release)
#   v0.2.0-5-g810b6b2-dirty        → 0.2.1.dev5+g810b6b2.d20260422
#   v0.3.0.dev0-5-g810b6b2         → 0.3.0.dev5+g810b6b2         (pre-release anchor)

set -euo pipefail

VERSION="${1:-}"
if [[ -z "$VERSION" ]]; then
  if ! RAW=$(git describe --tags --match 'v*' --dirty 2>/dev/null); then
    echo "ERROR: no v* tag found and no explicit VERSION given" >&2
    exit 1
  fi
  RAW="${RAW#v}"

  DIRTY_SUFFIX=""
  if [[ "$RAW" == *-dirty ]]; then
    DIRTY_SUFFIX=".d$(date -u +%Y%m%d)"
    RAW="${RAW%-dirty}"
  fi

  # git-describe emits `<tag>-<distance>-g<sha>` when HEAD is past the tag.
  # Normalize to PEP 440, mirroring setuptools-scm's `guess-next-dev` scheme.
  if [[ "$RAW" =~ ^(.+)-([0-9]+)-g([0-9a-f]+)$ ]]; then
    TAG_PART="${BASH_REMATCH[1]}"
    DISTANCE="${BASH_REMATCH[2]}"
    SHA="${BASH_REMATCH[3]}"

    if [[ "$TAG_PART" =~ ^([0-9]+)\.([0-9]+)\.([0-9]+)(\.(dev|alpha|beta|rc)[0-9]+)?$ ]]; then
      MAJOR="${BASH_REMATCH[1]}"
      MINOR="${BASH_REMATCH[2]}"
      PATCH="${BASH_REMATCH[3]}"
      PRERELEASE="${BASH_REMATCH[4]}"

      if [[ -n "$PRERELEASE" ]]; then
        # Pre-release anchor (e.g. 0.3.0.dev0): keep base, use distance as dev N.
        VERSION="${MAJOR}.${MINOR}.${PATCH}.dev${DISTANCE}+g${SHA}${DIRTY_SUFFIX}"
      else
        # Release tag (e.g. 0.2.0): bump patch, use distance as dev N.
        VERSION="${MAJOR}.${MINOR}.$((PATCH + 1)).dev${DISTANCE}+g${SHA}${DIRTY_SUFFIX}"
      fi
    else
      echo "ERROR: cannot parse tag '$TAG_PART' as PEP 440" >&2
      exit 1
    fi
  else
    VERSION="${RAW}${DIRTY_SUFFIX:++${DIRTY_SUFFIX#.}}"
  fi
fi

echo "opaque build version → $VERSION"

# --- opaque-accounting (maturin) --------------------------------------------
# Python wheel version lives in the package's pyproject.toml and accepts the
# full PEP 440 string. Cargo.toml demands SemVer, which rejects PEP 440 dev
# markers (`0.2.0.dev42`). Transform PEP 440 → SemVer for the Rust side:
#
#   0.2.0                     → 0.2.0
#   0.2.0.dev42               → 0.2.0-dev.42
#   0.2.0.dev42+g<sha>        → 0.2.0-dev.42+g<sha>
#   0.2.0.dev42+g<sha>.dYMD   → 0.2.0-dev.42+g<sha>.dYMD
#
# The `+` build-metadata segment is valid in both PEP 440 and SemVer.
CARGO_VERSION=$(echo "$VERSION" | sed -E 's/\.dev([0-9]+)/-dev.\1/')

# Python wheel metadata
sed -i.bak -E "s%^version = \"[^\"]+\"%version = \"$VERSION\"%" \
  packages/opaque-accounting/pyproject.toml
rm -f packages/opaque-accounting/pyproject.toml.bak

# Rust crate version (workspace-wide, inherited by opaque-accounting/Cargo.toml)
sed -i.bak -E "s%^version = \"[^\"]+\"%version = \"$CARGO_VERSION\"%" Cargo.toml
rm -f Cargo.toml.bak

# --- opaque umbrella: pin sub-packages to the same version ------------------
# Use `%` as sed delimiter so regex alternation `|` doesn't collide.
sed -i.bak -E \
  -e "s%opaque-([a-z-]+)==0\.0\.0\.dev0%opaque-\1==$VERSION%g" \
  -e "s%opaque-([a-z-]+)(\[[a-z,-]+\])==0\.0\.0\.dev0%opaque-\1\2==$VERSION%g" \
  -e "s%opaque-([a-z-]+)>=0\.0\.0\.dev0%opaque-\1==$VERSION%g" \
  -e "s%opaque-([a-z-]+)(\[[a-z,-]+\])>=0\.0\.0\.dev0%opaque-\1\2==$VERSION%g" \
  packages/opaque/pyproject.toml
rm -f packages/opaque/pyproject.toml.bak

echo "Updated version pins:"
grep -E "^version = \"|opaque-[a-z-]+" packages/opaque-accounting/pyproject.toml packages/opaque-accounting/Cargo.toml packages/opaque/pyproject.toml | sed 's|^|  |'

# --- export for downstream build steps --------------------------------------
# setuptools-scm would otherwise re-derive the version from the now-dirty
# tree and drift from what we've written into opaque-accounting/pyproject.toml.
if [[ -n "${GITHUB_ENV:-}" ]]; then
  echo "SETUPTOOLS_SCM_PRETEND_VERSION=$VERSION" >> "$GITHUB_ENV"
fi
