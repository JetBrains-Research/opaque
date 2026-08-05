# Copyright (c) 2025 Opaque Authors
# SPDX-License-Identifier: Apache-2.0
"""Central patching orchestration for HuggingFace Transformers."""

from __future__ import annotations

import logging
import types
from typing import TYPE_CHECKING

from opaque.api.patches.transformers._registry import detect_family, get_family_apply_fn

if TYPE_CHECKING:
    from collections.abc import Callable

    import torch.nn as nn

logger = logging.getLogger(__name__)


# Per-concern kwargs whose default bucket is ``kernels`` — they install
# Triton kernels that require CUDA + Triton at runtime, so the router
# auto-forces them off when those aren't available. Listed here
# (rather than computed by prefix) so the flag set is explicit and
# grep-able.
_KERNEL_KWARGS = (
    "rope",
    "rms_norm",
    "activation",
    "cross_entropy",
    "fused_linear_cross_entropy",
)


def _has_kernel_runtime() -> bool:
    """Return True iff CUDA + Triton are importable on this host — the runtime the
    Triton-only kernels (rope / rms_norm / activation / cross_entropy) need.

    Delegates to :func:`opaque.api.engine.device.fused_kernels_available` so the
    "what runs the fused kernels?" question has a single source of truth. Portable
    accelerations (grouped-GEMM MoE, chunked CE) do NOT gate on this — they run on
    MPS/CPU and default from ``performance``.
    """
    from opaque.api.engine.device import fused_kernels_available

    return fused_kernels_available()


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
    model: nn.Module,
    *,
    performance: bool = True,
    compat: bool = True,
    kernels: bool | None = None,
    fused_linear_cross_entropy: bool = False,
    **kwargs,
) -> None:
    """Apply Liger-style global + instance patching for kernels and compat wrappers.

    Three umbrella flags drive the per-concern kwargs:

    - ``performance`` — memory-efficiency patches that run on any host
      (currently ``kv_cache``).
    - ``compat`` — vmap-safety wrappers (``eager_attention``, ``batchify``).
    - ``kernels`` — "use accelerated kernels" (``rope``, ``rms_norm``,
      ``activation``, ``cross_entropy``, ``grouped_moe``). Defaults to
      ``performance`` when ``None``. The flag is unconditional; the **per-kernel
      install** checks the environment (see the factory / family): the Triton-only
      kernels gate on :func:`_has_kernel_runtime` so they're never installed
      off-CUDA, while the portable ones (grouped-GEMM MoE on ``torch._grouped_mm``,
      chunked CE) run on any host.

    ``fused_linear_cross_entropy`` (the fused lm_head+CE kernel) is promoted out
    of ``**kwargs`` because it defaults to ``False`` rather than inheriting from
    ``kernels``: the fused forward returns ``logits=None``, incompatible with
    callers that read logits.
    """
    if kernels is None:
        kernels = performance
    dropout = kwargs.get("dropout", compat)
    batchify = kwargs.get("batchify", compat)

    family = detect_family(model)
    if family is None:
        if dropout or batchify:
            raise ValueError(
                "opaque: dropout/batchify patches require a registered transformers "
                f"family; got {type(model).__name__}"
            )
        logger.debug(
            "opaque: model family for %s is unknown; no model-level patches applied.",
            type(model).__name__,
        )
        return

    apply_fn = get_family_apply_fn(family)
    if apply_fn is None:
        logger.debug(
            "opaque: no apply function registered for family %s; "
            "register one via opaque.patches.transformers.register_family",
            family,
        )
        return

    apply_fn(
        model,
        performance=performance,
        compat=compat,
        kernels=kernels,
        fused_linear_cross_entropy=fused_linear_cross_entropy,
        **kwargs,
    )
    logger.debug("opaque: Applied model patches for %s", family)

    # ``dropout`` (compat): zero the model's dropout. DP-SGD trains without it,
    # and SDPA's fused dropout breaks vmap(grad). Model-wide traversal, so it
    # runs here rather than in the per-class factory. Opt out with
    # ``dropout=False`` (keeps the model's dropout).
    if dropout and model is not None:
        from opaque.api.patches.transformers.components.dropout import disable_dropout

        disable_dropout(model)

    # Apply batchify to PeftModel classes if needed
    if batchify and model is not None:
        try:
            import peft

            if isinstance(model, peft.PeftModel):
                cls = type(model)
                from opaque.api.patches.transformers.components.batchify import (
                    apply_batchify_patch,
                )

                apply_batchify_patch(cls, model)
        except ImportError:
            pass


__all__ = [
    "apply_transformers_model_patches",
]
