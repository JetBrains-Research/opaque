# Copyright (c) 2025 Opaque Authors
# SPDX-License-Identifier: Apache-2.0
"""Verify `opaque.performance` auto-patching respects OPAQUE_SKIP_* env vars.

Module-level auto-patching only runs once per interpreter, so we spawn a
subprocess for each scenario to get a fresh import.
"""

from __future__ import annotations

import subprocess
import sys


def _run(script: str, env_overrides: dict[str, str]) -> str:
    """Run ``script`` in a subprocess with ``env_overrides`` applied."""
    import os

    env = os.environ.copy()
    env.update(env_overrides)
    # Drop any ambient OPAQUE_SKIP_* that would leak from the parent.
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


def test_default_import_applies_all_patches():
    """Without any OPAQUE_SKIP_* env var, both patch layers fire on import."""
    out = _run(
        "import opaque.performance\n"
        "from opaque.performance import is_checkpoint_patched\n"
        "from opaque.performance.huggingface import is_kernel_patched\n"
        "print(is_checkpoint_patched(), is_kernel_patched())",
        env_overrides={},
    )
    assert out == "True True"


def test_skip_pytorch_patches_all_disables_everything():
    """``OPAQUE_SKIP_PYTORCH_PATCHES=all`` short-circuits `patch_all()`."""
    out = _run(
        "import opaque.performance\n"
        "from opaque.performance import is_checkpoint_patched\n"
        "from opaque.performance.huggingface import is_kernel_patched\n"
        "print(is_checkpoint_patched(), is_kernel_patched())",
        env_overrides={"OPAQUE_SKIP_PYTORCH_PATCHES": "all"},
    )
    assert out == "False False"


def test_skip_checkpoint_only_leaves_kernels_applied():
    """``OPAQUE_SKIP_PYTORCH_PATCHES=checkpoint`` skips only the torch patch."""
    out = _run(
        "import opaque.performance\n"
        "from opaque.performance import is_checkpoint_patched\n"
        "from opaque.performance.huggingface import is_kernel_patched\n"
        "print(is_checkpoint_patched(), is_kernel_patched())",
        env_overrides={"OPAQUE_SKIP_PYTORCH_PATCHES": "checkpoint"},
    )
    assert out == "False True"


def test_skip_kernel_patches_does_not_error():
    """``OPAQUE_SKIP_TRANSFORMERS_KERNEL_PATCHES=all`` is honored without error.

    The ``is_kernel_patched()`` predicate is a "handled-once" guard rather
    than a "did anything" indicator — it goes True even on skip so that
    ``apply_kernel_patches()`` stays idempotent. What this test guards
    against is the env var causing a crash / breaking ``patch_all()``.
    """
    out = _run(
        "import opaque.performance\n"
        "from opaque.performance import is_checkpoint_patched\n"
        "print(is_checkpoint_patched())",
        env_overrides={"OPAQUE_SKIP_TRANSFORMERS_KERNEL_PATCHES": "all"},
    )
    assert out == "True"
