# Copyright (c) 2025 Opaque Authors
# SPDX-License-Identifier: Apache-2.0
"""RoPE module-level patches: extended model families share Llama-style RoPE."""

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
_requires_cuda_triton = pytest.mark.skipif(
    not _KERNELS_AVAILABLE,
    reason="RoPE Triton patches apply only with CUDA + Triton",
)


def _run(script: str, env_overrides: dict[str, str] | None = None) -> str:
    env = os.environ.copy()
    if env_overrides:
        env.update(env_overrides)
    for key in list(env):
        if key.startswith("OPAQUE_SKIP_") and (
            env_overrides is None or key not in env_overrides
        ):
            del env[key]
    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        env=env,
    )
    assert result.returncode == 0, f"subprocess failed: {result.stderr}"
    return result.stdout.strip()


@_requires_cuda_triton
def test_cohere_cohere2_apply_rotary_uses_opaque_kernel():
    """Cohere / Cohere2 use the same module-level API as Llama; patch must land."""
    out = _run(
        "import opaque.performance\n"
        "from opaque.performance.huggingface import kernel_patches as kp\n"
        "import transformers.models.cohere.modeling_cohere as mc\n"
        "import transformers.models.cohere2.modeling_cohere2 as mc2\n"
        "assert mc.apply_rotary_pos_emb is kp._opaque_apply_rotary_pos_emb\n"
        "assert mc2.apply_rotary_pos_emb is kp._opaque_apply_rotary_pos_emb\n"
        "print('ok')",
        env_overrides={},
    )
    assert out == "ok"


@_requires_cuda_triton
def test_skip_rope_token_leaves_cohere_stock():
    """OPAQUE_SKIP_TRANSFORMERS_KERNEL_PATCHES=rope skips module-level RoPE swap."""
    out = _run(
        "import opaque.performance\n"
        "import transformers.models.cohere.modeling_cohere as mc\n"
        "import transformers.models.cohere2.modeling_cohere2 as mc2\n"
        "assert mc.apply_rotary_pos_emb.__name__ == 'apply_rotary_pos_emb'\n"
        "assert mc2.apply_rotary_pos_emb.__name__ == 'apply_rotary_pos_emb'\n"
        "print('ok')",
        env_overrides={"OPAQUE_SKIP_TRANSFORMERS_KERNEL_PATCHES": "rope"},
    )
    assert out == "ok"
