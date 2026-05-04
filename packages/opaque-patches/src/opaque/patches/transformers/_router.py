# Copyright (c) 2025 Opaque Authors
# SPDX-License-Identifier: Apache-2.0
"""Central patching orchestration for HuggingFace Transformers."""

from __future__ import annotations

import logging
import types
from typing import Callable

import torch
import torch.nn as nn

from opaque.patches.transformers._registry import SUPPORTED_FAMILIES, detect_family

logger = logging.getLogger(__name__)


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
        # Overwrite all specific kernel kwargs if they were set
        for key in list(kwargs.keys()):
            if key.startswith("fuse_"):
                kwargs[key] = False

    family = detect_family(model)

    if family and family in SUPPORTED_FAMILIES:
        import importlib

        try:
            models_module = importlib.import_module(
                "opaque.patches.transformers.models." + family
            )
            patch_fn = getattr(
                models_module, "apply_" + family.replace("-", "_") + "_patches"
            )
        except (ImportError, AttributeError) as e:
            logger.warning(
                "opaque: Could not load patch function for %s: %s", family, e
            )
            patch_fn = None

        if patch_fn:
            patch_fn(model, performance=performance, compat=compat, **kwargs)
            logger.debug(f"opaque: Applied model patches for {family}")

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
