# Copyright (c) 2025 Opaque Authors
# SPDX-License-Identifier: Apache-2.0
"""`opaque.patches.transformers.components.apply_transformers_model_patches()` must handle missing transformers."""

from __future__ import annotations

import os
import subprocess
import sys

SOFT_IMPORT_SCRIPT = """
import sys
# Block `import transformers` before opaque touches it.
sys.modules["transformers"] = None

from opaque.api.patches.transformers._router import apply_transformers_model_patches

# Should not raise and should not have patched anything.
try:
    import torch.nn as nn
    apply_transformers_model_patches(nn.Module())
    print("False")
except Exception as e:
    print(repr(e))
"""


def test_apply_kernels_no_ops_without_transformers():
    env = os.environ.copy()
    result = subprocess.run(
        [sys.executable, "-c", SOFT_IMPORT_SCRIPT],
        capture_output=True,
        text=True,
        env=env,
    )
    assert result.returncode == 0, f"subprocess failed: stderr={result.stderr!r}"
    assert result.stdout.strip() == "False"
