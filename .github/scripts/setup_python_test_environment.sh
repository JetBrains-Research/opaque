#!/usr/bin/env bash
# Install the dependency resolution selected by an explicit Python test group.

set -euo pipefail

: "${DEPENDENCY_SELECTION:=locked}"
: "${OPAQUE_EXPECTED_PROVIDERS:?Expected providers must be set explicitly}"

expected_providers=()
IFS=',' read -r -a requested_providers <<< "$OPAQUE_EXPECTED_PROVIDERS"
for provider in "${requested_providers[@]}"; do
  provider="${provider//[[:space:]]/}"
  if [[ -z "$provider" ]]; then
    continue
  fi
  case "$provider" in
    torch|mlx)
      ;;
    *)
      echo "::error::Unknown expected provider: $provider"
      exit 1
      ;;
  esac
  if (( ${#expected_providers[@]} > 0 )); then
    for seen_provider in "${expected_providers[@]}"; do
      if [[ "$provider" == "$seen_provider" ]]; then
        echo "::error::Duplicate expected provider: $provider"
        exit 1
      fi
    done
  fi
  expected_providers+=("$provider")
done

if (( ${#expected_providers[@]} == 0 )); then
  echo "::error::Expected at least one provider"
  exit 1
fi

sync_args=(--group dev --group test --all-packages)

case "$DEPENDENCY_SELECTION" in
  locked)
    uv sync --locked "${sync_args[@]}"
    ;;
  minimum)
    uv sync --upgrade --resolution lowest-direct \
      "${sync_args[@]}"
    ;;
  latest)
    uv sync --upgrade --resolution highest \
      "${sync_args[@]}"
    ;;
  *)
    echo "::error::Unknown dependency selection: $DEPENDENCY_SELECTION"
    exit 1
    ;;
esac

uv run python - "${expected_providers[@]}" <<'PY'
import importlib
import sys

PROVIDER_MODULES = {
    "torch": ("torch", "opaque.api.torch"),
    "mlx": ("mlx.core", "opaque.api.mlx"),
}

for provider in sys.argv[1:]:
    runtime_module, provider_module = PROVIDER_MODULES[provider]
    try:
        importlib.import_module(runtime_module)
        importlib.import_module(provider_module)
    except ImportError as exc:
        raise SystemExit(
            f"Expected provider {provider!r} is unavailable. "
            f"Install its runtime and {provider_module!r}."
        ) from exc
PY
