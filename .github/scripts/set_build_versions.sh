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
# Intended to run in CI as a preflight before `uv build`. On local dev,
# leave pyprojects / Cargo.toml untouched — uv workspace resolution and
# setuptools-scm fallbacks handle that case.
#
# Usage: bash .github/scripts/set_build_versions.sh [VERSION]
#
# If VERSION is omitted, derive it from `git describe --tags --match 'v*'`.

set -euo pipefail

VERSION="${1:-}"
if [[ -z "$VERSION" ]]; then
  if ! VERSION=$(git describe --tags --match 'v*' --dirty 2>/dev/null); then
    echo "ERROR: no v* tag found and no explicit VERSION given" >&2
    exit 1
  fi
  VERSION="${VERSION#v}"
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
