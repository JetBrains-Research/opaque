# Copyright (c) 2025 Opaque Authors
# SPDX-License-Identifier: Apache-2.0
"""Verify `opaque.huggingface` auto-patching respects OPAQUE_SKIP_* env vars.

Module-level auto-patching only runs once per interpreter, so we spawn a
subprocess for each scenario to get a fresh import.
"""

from __future__ import annotations

import os
import subprocess
import sys


def _run(script: str, env_overrides: dict[str, str]) -> str:
    env = os.environ.copy()
    env.update(env_overrides)
    for key in list(env):
        if key.startswith("OPAQUE_SKIP_") and key not in env_overrides:
            del env[key]
    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        env=env,
    )
    assert result.returncode == 0, f"subprocess failed: {result.stderr}"
    return result.stdout.strip()


def test_default_import_applies_compat_patches():
    """Without any OPAQUE_SKIP_* env var, compat patches fire on import."""
    out = _run(
        "import opaque.huggingface\nprint(opaque.huggingface.is_patched())",
        env_overrides={},
    )
    assert out == "True"


def test_skip_transformers_patches_all_reports_no_patches_landed():
    """``OPAQUE_SKIP_TRANSFORMERS_PATCHES=all`` skips every sub-patch.

    ``is_patched()`` reflects actually-landed state — it must return False
    when no sub-patch was dispatched. Also verify the vmap layer wasn't
    applied as a cross-check.
    """
    out = _run(
        "import opaque.huggingface\n"
        "print(opaque.huggingface.is_patched(), opaque.huggingface.is_vmap_patched())",
        env_overrides={"OPAQUE_SKIP_TRANSFORMERS_PATCHES": "all"},
    )
    assert out == "False False"


def test_skip_only_vmap_leaves_other_patches():
    """``OPAQUE_SKIP_TRANSFORMERS_PATCHES=vmap`` skips only the vmap layer."""
    out = _run(
        "import opaque.huggingface\n"
        "print(opaque.huggingface.is_patched(), opaque.huggingface.is_vmap_patched())",
        env_overrides={"OPAQUE_SKIP_TRANSFORMERS_PATCHES": "vmap"},
    )
    # Other sub-patches (kv_cache, data) still landed → is_patched() is True.
    assert out == "True False"


def test_unknown_skip_token_raises():
    """Typos in OPAQUE_SKIP_* env vars fail loudly instead of silently no-op."""
    env = os.environ.copy()
    env["OPAQUE_SKIP_TRANSFORMERS_PATCHES"] = "vamp"
    result = subprocess.run(
        [sys.executable, "-c", "import opaque.huggingface"],
        capture_output=True,
        text=True,
        env=env,
    )
    assert result.returncode != 0
    assert "unknown token(s)" in result.stderr
    assert "'vamp'" in result.stderr
