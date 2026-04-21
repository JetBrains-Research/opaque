# Copyright (c) 2025 Opaque Authors
# SPDX-License-Identifier: Apache-2.0
"""Verify `opaque.performance` auto-patching respects OPAQUE_SKIP_* env vars.

Module-level auto-patching only runs once per interpreter, so we spawn a
subprocess for each scenario to get a fresh import.

``is_kernel_patched()`` reflects actually-landed kernel patches — it is only
True when CUDA + Triton are available AND at least one target class was
patched. On CPU-only CI runners it returns False, so tests either gate on
``torch.cuda.is_available() and importlib.util.find_spec('triton')`` or
assert on the checkpoint flag alone.
"""

from __future__ import annotations

import importlib.util
import os
import subprocess
import sys

import pytest
import torch

_KERNELS_AVAILABLE = (
    torch.cuda.is_available() and importlib.util.find_spec("triton") is not None
)
_requires_kernels = pytest.mark.skipif(
    not _KERNELS_AVAILABLE,
    reason="is_kernel_patched() is only True with CUDA + Triton installed",
)


def _run(script: str, env_overrides: dict[str, str]) -> str:
    """Run ``script`` in a subprocess with ``env_overrides`` applied."""
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


def test_default_import_applies_checkpoint_patch():
    """Without any OPAQUE_SKIP_* env var, the checkpoint patch lands."""
    out = _run(
        "import opaque.performance\n"
        "from opaque.performance import is_checkpoint_patched\n"
        "print(is_checkpoint_patched())",
        env_overrides={},
    )
    assert out == "True"


@_requires_kernels
def test_default_import_applies_kernel_patches_with_triton():
    """On CUDA+Triton hosts, at least one kernel patch lands on default import."""
    out = _run(
        "import opaque.performance\n"
        "from opaque.performance.huggingface import is_kernel_patched\n"
        "print(is_kernel_patched())",
        env_overrides={},
    )
    assert out == "True"


def test_default_import_skips_kernels_without_triton():
    """On CPU-only hosts ``is_kernel_patched()`` is False — no CUDA/Triton."""
    if _KERNELS_AVAILABLE:
        pytest.skip("CUDA+Triton available on this host")
    out = _run(
        "import opaque.performance\n"
        "from opaque.performance.huggingface import is_kernel_patched\n"
        "print(is_kernel_patched())",
        env_overrides={},
    )
    assert out == "False"


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


def test_skip_checkpoint_only_leaves_kernels_available():
    """``OPAQUE_SKIP_PYTORCH_PATCHES=checkpoint`` skips only the torch patch."""
    out = _run(
        "import opaque.performance\n"
        "from opaque.performance import is_checkpoint_patched\n"
        "from opaque.performance.huggingface import is_kernel_patched\n"
        "print(is_checkpoint_patched(), is_kernel_patched())",
        env_overrides={"OPAQUE_SKIP_PYTORCH_PATCHES": "checkpoint"},
    )
    # Kernel flag depends on CUDA+Triton; check checkpoint half unconditionally.
    checkpoint_flag, kernel_flag = out.split()
    assert checkpoint_flag == "False"
    expected_kernel = "True" if _KERNELS_AVAILABLE else "False"
    assert kernel_flag == expected_kernel


def test_skip_kernel_patches_all_reports_kernels_off():
    """``OPAQUE_SKIP_TRANSFORMERS_KERNEL_PATCHES=all`` → is_kernel_patched() is False."""
    out = _run(
        "import opaque.performance\n"
        "from opaque.performance import is_checkpoint_patched\n"
        "from opaque.performance.huggingface import is_kernel_patched\n"
        "print(is_checkpoint_patched(), is_kernel_patched())",
        env_overrides={"OPAQUE_SKIP_TRANSFORMERS_KERNEL_PATCHES": "all"},
    )
    assert out == "True False"


def test_unknown_skip_token_raises():
    """Typos in OPAQUE_SKIP_* env vars fail loudly instead of silently no-op."""
    env = os.environ.copy()
    env["OPAQUE_SKIP_PYTORCH_PATCHES"] = "checkpont"  # typo
    result = subprocess.run(
        [sys.executable, "-c", "import opaque.performance"],
        capture_output=True,
        text=True,
        env=env,
    )
    assert result.returncode != 0
    assert "unknown token(s)" in result.stderr
    assert "'checkpont'" in result.stderr
