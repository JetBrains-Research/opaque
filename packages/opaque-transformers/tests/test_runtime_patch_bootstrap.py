# Copyright (c) 2025 Opaque Authors
# SPDX-License-Identifier: Apache-2.0
"""Explicit runtime patch bootstrap (no import side-effects)."""

from __future__ import annotations

import subprocess
import sys


def _run(code: str) -> str:
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    return result.stdout.strip()


def test_subprocess_import_alone_not_patched():
    out = _run(
        "import opaque.transformers\n"
        "from opaque.patches import is_runtime_patched\n"
        "print(is_runtime_patched())"
    )
    assert out == "False"


def test_subprocess_apply_runtime_patches_sets_state():
    out = _run(
        "from opaque.patches import apply_runtime_patches, is_runtime_patched\n"
        "apply_runtime_patches(compat=True)\n"
        "print(is_runtime_patched())"
    )
    assert out == "True"
