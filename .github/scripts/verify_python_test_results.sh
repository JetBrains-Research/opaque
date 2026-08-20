#!/usr/bin/env bash
# Fail a test group only after every package step has had a chance to run.

set -euo pipefail

: "${RESULTS:?RESULTS must be set}"

failures=()
while IFS='=' read -r package outcome; do
  if [[ "$outcome" == "failure" ]]; then
    failures+=("$package")
  fi
done <<< "$RESULTS"

if [[ "${#failures[@]}" -gt 0 ]]; then
  printf '::error::Package test steps failed: %s\n' "$(IFS=', '; echo "${failures[*]}")"
  exit 1
fi
