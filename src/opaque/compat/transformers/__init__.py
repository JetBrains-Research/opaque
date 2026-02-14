# Copyright (c) 2025 Opaque Authors
# SPDX-License-Identifier: Apache-2.0
"""vmap compatibility patches for HuggingFace Transformers models.

Patches are applied automatically at `import opaque` time.
No user action required - just import opaque and use clipped_grad with any
supported HuggingFace model.

Disable auto-patching with: OPAQUE_NO_PATCH=1

Supported models:
- GPT-2
- LLaMA (and LLaMA-based: Mistral, DeepSeek, etc.)
- Qwen2
- Phi, Phi-3
- OLMo
- Gemma, Gemma2
"""

from opaque.compat.transformers._global_patches import (
    apply_global_patches,
    is_globally_patched,
)

__all__ = [
    "apply_global_patches",
    "is_globally_patched",
]
