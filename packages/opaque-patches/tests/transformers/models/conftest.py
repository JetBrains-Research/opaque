# Copyright (c) 2025 Opaque Authors
# SPDX-License-Identifier: Apache-2.0
"""Model-family test config.

The suite runs with ``attn_implementation="sdpa"`` (the transformers production
default). Under ``vmap(grad)`` the fused SDPA backends (efficient/cudnn) have no
batching rule yet and fall back to a slow per-example loop; the MATH backend is
vmap-native. Prefer MATH so the SDPA *integration* (masking shim, kv-repeat,
mask-ignore) is exercised cleanly and fast — the backend kernel itself is
PyTorch's concern, not Opaque's.
"""

import pytest


@pytest.fixture(autouse=True)
def _prefer_math_sdpa():
    try:
        from torch.nn.attention import sdpa_kernel, SDPBackend
    except Exception:
        yield
        return
    with sdpa_kernel(
        [SDPBackend.MATH, SDPBackend.EFFICIENT_ATTENTION, SDPBackend.FLASH_ATTENTION]
    ):
        yield
