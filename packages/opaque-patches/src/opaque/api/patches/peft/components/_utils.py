# Copyright (c) 2025 Opaque Authors
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import torch


def _active_lora_dtype(x: torch.Tensor) -> torch.dtype:
    """Dtype that LoRA A/B weights must use for the fused kernel call.

    When ``torch.autocast(device_type="cuda")`` is active the kernel's
    autocast-intercepted matmuls (``F.linear``, ``@``) produce tensors
    in the autocast dtype, so the still-fp32 LoRA weights would mismatch
    at the subsequent ``addmm_``. Mirror the public ``opaque_*`` wrappers'
    :func:`follow_autocast` behaviour by casting to the autocast dtype
    when active; otherwise honour the input dtype (mixed-precision via
    explicit model cast).
    """
    if x.is_cuda and torch.is_autocast_enabled("cuda"):
        return torch.get_autocast_dtype("cuda")
    return x.dtype


def _extract_lora_params(lora_linear):
    """Extract (W, A, B, scaling) from a peft LoRA Linear module.

    Returns (W, A, B, scaling) or (W, None, None, 0.0) if no active adapter.
    """
    W = lora_linear.base_layer.weight

    if (
        lora_linear.disable_adapters
        or not lora_linear.active_adapters
        or lora_linear.active_adapters[0] not in lora_linear.lora_A
    ):
        return W, None, None, 0.0

    active = lora_linear.active_adapters[0]
    # PEFT stores lora_A as (rank, in_features), kernel expects (in_features, rank)
    A = lora_linear.lora_A[active].weight.T
    # PEFT stores lora_B as (out_features, rank), kernel expects (rank, out_features)
    B = lora_linear.lora_B[active].weight.T
    scaling = lora_linear.scaling[active]
    return W, A, B, scaling


def _has_lora(module, proj_name):
    """Check if a module's sub-module has active LoRA adapters."""
    proj = getattr(module, proj_name, None)
    if proj is None:
        return False
    return hasattr(proj, "lora_A") and len(getattr(proj, "lora_A", {})) > 0


def _no_lora_dropout(module, proj_name):
    """Check that a projection has no active LoRA dropout (p=0 / Identity).

    Fused QKV/MLP kernels bypass per-projection forwards, so dropout would
    be silently skipped. Only fuse when dropout is a no-op.
    """
    proj = getattr(module, proj_name, None)
    if proj is None:
        return True

    if not hasattr(proj, "lora_dropout") or not getattr(proj, "active_adapters", []):
        return True

    active = proj.active_adapters[0]
    if active not in proj.lora_dropout:
        return True

    dropout = proj.lora_dropout[active]
    if isinstance(dropout, torch.nn.Identity):
        return True
    if isinstance(dropout, torch.nn.Dropout) and dropout.p == 0.0:
        return True
    return False


def _no_bias(module, proj_name):
    """Check that a projection has no bias (required for fused QKV kernel)."""
    proj = getattr(module, proj_name, None)
    if proj is None:
        return False
    base = getattr(proj, "base_layer", proj)
    return base.bias is None
