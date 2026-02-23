# Copyright (c) 2025 Opaque Authors
# SPDX-License-Identifier: Apache-2.0
"""Compatibility patches for various libraries.

Currently supports:
- transformers: HuggingFace Transformers models

Patches are applied automatically at `import opaque` time.
No user action required.

Disable with: OPAQUE_NO_PATCH=1
"""

from opaque.compat.transformers import (
    apply_global_patches,
    is_globally_patched,
)

__all__ = [
    "apply_global_patches",
    "is_globally_patched",
]
