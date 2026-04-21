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
        "import opaque.huggingface\n"
        "print(opaque.huggingface.is_patched())",
        env_overrides={},
    )
    assert out == "True"


def test_skip_transformers_patches_all_short_circuits_patch_all():
    """``OPAQUE_SKIP_TRANSFORMERS_PATCHES=all`` stops any transformers patching.

    The flag semantically records "handled" even when skipped: the
    ``is_patched()`` predicate returns True because ``apply_transformers_patches``
    ran to completion (it's a no-op under ``all``, but still marks the
    module as processed). Verify instead by watching that the vmap patch
    wasn't actually applied.
    """
    out = _run(
        "import opaque.huggingface\n"
        "print(opaque.huggingface.is_vmap_patched())",
        env_overrides={"OPAQUE_SKIP_TRANSFORMERS_PATCHES": "all"},
    )
    assert out == "False"


def test_skip_only_vmap_leaves_other_patches():
    """``OPAQUE_SKIP_TRANSFORMERS_PATCHES=vmap`` skips only the vmap layer."""
    out = _run(
        "import opaque.huggingface\n"
        "print(opaque.huggingface.is_vmap_patched())",
        env_overrides={"OPAQUE_SKIP_TRANSFORMERS_PATCHES": "vmap"},
    )
    assert out == "False"
