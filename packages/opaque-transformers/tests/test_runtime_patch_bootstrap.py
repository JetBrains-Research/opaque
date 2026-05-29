# Copyright (c) 2025 Opaque Authors
# SPDX-License-Identifier: Apache-2.0
"""Explicit runtime patch bootstrap (no env-driven toggles)."""

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
    out = _run("import opaque.transformers as t; print(t.is_patched())")
    assert out == "False"


def test_subprocess_patch_all_applies_runtime_compat():
    out = _run(
        "import opaque.transformers as t\n"
        "t.patch_all()\n"
        "print(t.is_patched(), t.is_vmap_patched())"
    )
    assert out == "True True"
