# Copyright (c) 2025 Opaque Authors
# SPDX-License-Identifier: Apache-2.0
"""The provider's runtime patches stand on their own.

``apply_runtime_patches`` here covers the Torch-core concerns and nothing else:
no higher layer need be installed for it to work, and installing one must not be
required for its probe to tell the truth. The cross-layer half of that contract
— that ``opaque.transformers.patches.apply_runtime_patches()`` also satisfies this probe — is
tested where the delegation lives.

These run in subprocesses: the patches are process-global and idempotent, so a
probe is only meaningful in an interpreter that has not been patched yet.
"""

from __future__ import annotations

import subprocess
import sys
import textwrap


def _run(body: str) -> str:
    result = subprocess.run(
        [sys.executable, "-c", textwrap.dedent(body)],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    return result.stdout.strip()


def test_provider_entry_point_patches_torch_on_its_own():
    out = _run("""
        from opaque.torch import apply_runtime_patches
        from opaque.torch.checkpoint import is_checkpoint_patched

        before = is_checkpoint_patched()
        apply_runtime_patches()
        print(before, is_checkpoint_patched())
    """)
    assert out == "False True"


def test_compat_false_leaves_torch_unpatched():
    out = _run("""
        from opaque.torch import apply_runtime_patches
        from opaque.torch.checkpoint import is_checkpoint_patched

        apply_runtime_patches(compat=False)
        print(is_checkpoint_patched())
    """)
    assert out == "False"


def test_checkpoint_entry_point_and_umbrella_agree():
    """The single-concern entry point sets the same state as the umbrella."""
    out = _run("""
        from opaque.torch.checkpoint import (
            apply_checkpoint_patch,
            is_checkpoint_patched,
        )

        apply_checkpoint_patch()
        print(is_checkpoint_patched())
    """)
    assert out == "True"
