#!/usr/bin/env bash
# Install the dependency resolution selected by an explicit Python test group.

set -euo pipefail

: "${DEPENDENCY_SELECTION:=locked}"
: "${PYTORCH_BACKEND:=default}"

if [[ "$PYTORCH_BACKEND" == "cpu" ]]; then
  requirements_file="$(mktemp)"
  trap 'rm -f "$requirements_file"' EXIT

  export_args=(
    uv export --group dev --all-packages --extra all --no-hashes
    --format requirements.txt
  )
  case "$DEPENDENCY_SELECTION" in
    locked)
      export_args+=(--locked)
      ;;
    minimum)
      export_args+=(--upgrade --resolution lowest-direct)
      ;;
    latest)
      export_args+=(--upgrade --resolution highest)
      ;;
    *)
      echo "::error::Unknown dependency selection: $DEPENDENCY_SELECTION"
      exit 1
      ;;
  esac

  # The universal lock exports CUDA-only transitive requirements. CPU Torch
  # does not need them, and CUDA lanes retain the locked CUDA installation.
  "${export_args[@]}" | grep -Ev '^(nvidia-|triton==)' > "$requirements_file"
  uv venv --clear
  uv pip sync --torch-backend cpu "$requirements_file"
  uv run --no-sync python - <<'PY'
import torch

if torch.version.cuda is not None:
    raise RuntimeError(f"expected CPU PyTorch, got CUDA {torch.version.cuda}")

print(f"Using CPU PyTorch {torch.__version__}")
PY
  exit 0
fi

if [[ "$PYTORCH_BACKEND" != "default" ]]; then
  echo "::error::Unknown PyTorch backend: $PYTORCH_BACKEND"
  exit 1
fi

case "$DEPENDENCY_SELECTION" in
  locked)
    uv sync --locked --group dev --all-packages --extra all
    ;;
  minimum)
    uv sync --upgrade --resolution lowest-direct \
      --group dev --all-packages --extra all
    ;;
  latest)
    uv sync --upgrade --resolution highest \
      --group dev --all-packages --extra all
    ;;
  *)
    echo "::error::Unknown dependency selection: $DEPENDENCY_SELECTION"
    exit 1
    ;;
esac
