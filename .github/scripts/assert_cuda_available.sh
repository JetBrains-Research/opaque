#!/usr/bin/env bash
# Prove that a CUDA test lane can run CUDA tests before it reports a result.
#
# conftest.py skips every `cuda`-marked test when torch reports no device, so a
# CUDA lane on a host that lost CUDA collects its tests, skips all of them, and
# exits 0 — the lane goes green while validating nothing. Asserting the
# capability up front turns that silent pass into a loud failure.
#
# The lane is recognised by its pytest marker expression rather than by the
# runner label. A lane whose green light means "CUDA tests passed" has to prove
# CUDA works whichever runner it was scheduled on, a CUDA lane accidentally
# pointed at a CPU runner fails loudly instead of skipping quietly, and a new
# GPU runner label inherits the check without touching this script.

set -euo pipefail

: "${PYTEST_MARKER:?PYTEST_MARKER must be set}"

# Every non-CUDA lane spells its exclusion `not cuda`, so dropping the negative
# terms first leaves a bare `cuda` only in the expressions that positively
# select CUDA-marked tests. The word boundaries keep an unrelated marker that
# merely starts with `cuda` from matching.
positive_marker_terms="${PYTEST_MARKER//not cuda/}"
if [[ ! "$positive_marker_terms" =~ (^|[^[:alnum:]_])cuda([^[:alnum:]_]|$) ]]; then
  echo "Marker expression '$PYTEST_MARKER' does not select CUDA tests; no CUDA preflight needed."
  exit 0
fi

echo "Marker expression '$PYTEST_MARKER' selects CUDA tests; asserting a usable CUDA device."

if uv run python - <<'PY'
import sys

import torch


def fail(reason: str) -> None:
    print(
        f"{reason} (torch {torch.__version__}, torch.version.cuda={torch.version.cuda})",
        file=sys.stderr,
    )
    sys.exit(1)


if not torch.cuda.is_available():
    fail("torch reports no available CUDA device")

devices = torch.cuda.device_count()
if devices < 1:
    fail("torch reports CUDA as available but exposes no device")

# A driver that answers queries is not a runtime that computes, and a host
# update can break the second while the first still responds. Read a value back
# off the device instead of trusting `is_available()` on its own.
probe = torch.zeros(1, device="cuda")
probe += 1
if probe.item() != 1.0:
    fail("arithmetic on a CUDA tensor returned an unexpected value")

print(
    f"CUDA is usable: {devices} device(s), device 0 is "
    f"{torch.cuda.get_device_name(0)} "
    f"(torch {torch.__version__}, CUDA {torch.version.cuda})"
)
PY
then
  exit 0
fi

echo "::error::CUDA preflight failed on a lane selecting '$PYTEST_MARKER'. Every test this lane selects is CUDA-marked, and pytest skips those silently on a host without CUDA, so the lane fails here instead of reporting a green result that validated nothing."
exit 1
