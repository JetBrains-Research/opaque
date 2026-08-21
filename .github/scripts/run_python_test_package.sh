#!/usr/bin/env bash
# Run one explicit package test step while appending to the job's coverage data.

set -euo pipefail

: "${ALLOW_EMPTY_TEST_SELECTION:=false}"
: "${ALLOW_TEST_FAILURE:=false}"
: "${COVERAGE_FILE:?COVERAGE_FILE must be set}"
: "${PYTEST_MARKER:?PYTEST_MARKER must be set}"
: "${REPORT_DURATIONS:=false}"
: "${TEST_PATH:?TEST_PATH must be set}"
: "${TEST_RESULTS_FILE:?TEST_RESULTS_FILE must be set}"

pytest_args=(
  uv run pytest "$TEST_PATH"
  -m "$PYTEST_MARKER"
)
if [[ -n "${JUNIT_DIR:-}" ]]; then
  # One report per path, named after it, so `check_executed_tests.py` can sum
  # executed counts across a leg. pytest overwrites `--junitxml`, so a shared
  # file would report only the last path.
  mkdir -p "$JUNIT_DIR"
  pytest_args+=(--junitxml "$JUNIT_DIR/${TEST_PATH//\//_}.xml")
fi
if [[ -n "${PYTEST_XDIST:-}" ]]; then
  read -r -a xdist_args <<< "$PYTEST_XDIST"
  pytest_args+=("${xdist_args[@]}")
fi
pytest_args+=(
  --cov=opaque
  --cov-append
  --cov-report=
  -q
)
if [[ "$REPORT_DURATIONS" == "true" ]]; then
  pytest_args+=(--durations=0 --durations-min=5)
fi

set +e
"${pytest_args[@]}"
status=$?
set -e

if [[ "$status" -eq 0 ]]; then
  printf '%s\n' "$TEST_PATH" >> "$TEST_RESULTS_FILE"
  exit 0
fi
if [[ "$status" -eq 5 && "$ALLOW_EMPTY_TEST_SELECTION" == "true" ]]; then
  echo "::notice::No tests matched $TEST_PATH for $PYTEST_MARKER."
  exit 0
fi
if [[ "$status" -ne 5 ]]; then
  printf '%s\n' "$TEST_PATH" >> "$TEST_RESULTS_FILE"
fi
if [[ "$ALLOW_TEST_FAILURE" == "true" ]]; then
  echo "::warning::pytest exited with status $status for $TEST_PATH in this advisory lane."
  exit 0
fi
exit "$status"
