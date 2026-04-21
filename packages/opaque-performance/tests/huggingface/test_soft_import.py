# Copyright (c) 2025 Opaque Authors
# SPDX-License-Identifier: Apache-2.0
"""`opaque.performance.huggingface.patch_all()` must no-op without transformers.

The subprocess blocks the `transformers` import by inserting ``None`` into
``sys.modules`` before touching `opaque.performance.huggingface`. This
simulates the scenario where `opaque-performance` is installed but
`transformers` is not, which should be handled gracefully (log, skip
patching) rather than raising.
"""

from __future__ import annotations

import os
import subprocess
import sys


SOFT_IMPORT_SCRIPT = """
import sys
# Block `import transformers` before opaque.performance touches it.
sys.modules["transformers"] = None

from opaque.performance.huggingface import patch_all, is_kernel_patched

# Should not raise and should not have patched anything.
patch_all()
print(is_kernel_patched())
"""


def test_patch_all_no_ops_without_transformers():
    env = os.environ.copy()
    # Force-skip torch-level patching so we isolate the HF layer.
    env["OPAQUE_SKIP_PYTORCH_PATCHES"] = "all"
    result = subprocess.run(
        [sys.executable, "-c", SOFT_IMPORT_SCRIPT],
        capture_output=True,
        text=True,
        env=env,
    )
    assert result.returncode == 0, (
        f"soft-import subprocess failed: stderr={result.stderr!r}"
    )
    assert result.stdout.strip() == "False"
