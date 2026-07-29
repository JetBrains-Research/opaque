#!/bin/sh
set -e

# Opaque is a Python (uv) + Rust monorepo. This script bootstraps a cloud
# development environment: it installs the system build deps, uv, and a Rust
# toolchain (needed by the opaque-accounting PyO3/maturin build), then syncs
# the full contributor workspace. All steps are idempotent.

# 1. System packages for curl-based installers and the Rust/maturin build.
if command -v apt-get >/dev/null 2>&1; then
    apt-get update
    apt-get install -y --no-install-recommends \
        curl ca-certificates build-essential pkg-config git
fi

# 2. Install uv if it is missing (installs into ~/.local/bin).
if ! command -v uv >/dev/null 2>&1; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
fi
export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"

# 3. Ensure a Rust toolchain is available for opaque-accounting.
if ! command -v cargo >/dev/null 2>&1; then
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
fi
export PATH="$HOME/.cargo/bin:$PATH"

# 4. Persist the PATH additions and unbuffered output for the agent's shells.
grep -q 'opaque uv/cargo PATH' "$HOME/.bashrc" 2>/dev/null || \
    printf '\n# opaque uv/cargo PATH\nexport PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"\n' >> "$HOME/.bashrc"
grep -q 'PYTHONUNBUFFERED' "$HOME/.bashrc" 2>/dev/null || \
    echo 'export PYTHONUNBUFFERED=1' >> "$HOME/.bashrc"

# 5. Install the complete contributor environment. The first sync also builds
#    the opaque-accounting Rust extension via maturin (~30s cold, cached after).
uv sync --group dev --all-packages --extra all
