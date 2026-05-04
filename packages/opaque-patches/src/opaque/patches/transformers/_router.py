# Copyright (c) 2025 Opaque Authors
# SPDX-License-Identifier: Apache-2.0
"""Central patching orchestration for HuggingFace Transformers."""

from __future__ import annotations

import logging
import types
from typing import Callable

import torch
import torch.nn as nn

from opaque.patches.transformers._registry import detect_family, get_family_apply_fn

logger = logging.getLogger(__name__)


# When CUDA + Triton aren't available the kernel patches can't run; force
# them off even if the caller passed an explicit ``True``.  Listed here
# (rather than computed by prefix) so the Liger-aligned flag set is
# explicit and grep-able.
_KERNEL_KWARGS = ("rope", "rms_norm", "activation", "cross_entropy")


def _patch_forward(
    target_cls: type[nn.Module] | None,
    factory: Callable | None,
    model: nn.Module | None,
) -> bool:
    """Helper to apply Liger-style global + instance forward replacement.

    Returns True if any patch was applied (either global or instance).
    """
    if target_cls is None or factory is None:
        return False

    patched = False

    # 1. Global class-level patching
    if hasattr(target_cls, "forward") and not hasattr(
        target_cls.forward, "__opaque_patched__"
    ):
        new_fwd = factory(target_cls.forward)
        new_fwd.__opaque_patched__ = True
        target_cls.forward = new_fwd
        patched = True

    # 2. Instance-level fallback patching
    if model is not None:
        for module in model.modules():
            fwd_fn = getattr(module.forward, "__func__", module.forward)
            if type(module) is target_cls and not hasattr(fwd_fn, "__opaque_patched__"):
                new_fwd = factory(type(module).forward)
                new_fwd.__opaque_patched__ = True
                module.forward = types.MethodType(new_fwd, module)
                patched = True

    return patched


def apply_transformers_model_patches(
    model: nn.Module, *, performance: bool = True, compat: bool = True, **kwargs
) -> None:
    """Apply Liger-style global + instance patching for kernels and compat wrappers."""
    # Check CUDA for kernels, but compat patches can run without CUDA
    has_kernels = True
    if not torch.cuda.is_available():
        has_kernels = False
    else:
        try:
            import triton  # noqa: F401
        except ImportError:
            has_kernels = False

    # Force disable kernels if dependencies are missing
    if not has_kernels:
        performance = False
        for key in _KERNEL_KWARGS:
            if key in kwargs:
                kwargs[key] = False

    family = detect_family(model)
    if family is None:
        logger.debug("opaque: model has no detectable family; skipping patches")
        return

    apply_fn = get_family_apply_fn(family)
    if apply_fn is None:
        logger.debug(
            "opaque: no apply function registered for family %s; "
            "register one via opaque.patches.transformers.register_family",
            family,
        )
        return

    apply_fn(model, performance=performance, compat=compat, **kwargs)
    logger.debug("opaque: Applied model patches for %s", family)

    batchify = kwargs.get("batchify", compat)
    # Apply batchify to PeftModel classes if needed
    if batchify and model is not None:
        try:
            import peft

            if isinstance(model, peft.PeftModel):
                cls = type(model)
                from opaque.patches.transformers.components.batchify import (
                    apply_batchify_patch,
                )

                apply_batchify_patch(cls, model)
        except ImportError:
            pass


__all__ = [
    "apply_transformers_model_patches",
]
