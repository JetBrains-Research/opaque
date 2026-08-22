# Copyright (c) 2025 Opaque Authors
# SPDX-License-Identifier: Apache-2.0
"""Family registry — built-in + user-registered HuggingFace model families.

Dispatch flow (called from
:func:`apply_transformers_model_patches`):

1. ``detect_family(model)`` → short family name from
   ``model.config.model_type``.
2. ``get_family_apply_fn(name)`` → the registered apply function.
3. The apply function is what
   :func:`opaque.transformers.patches.families.make_apply_model_patches` returns.

Both built-in and user-defined families share the same registration path:
``register_family(name, apply_fn)``.  Built-in families register
themselves at import time from
``opaque.transformers.patches.families.models.X``; downstream users register
their own families the same way (typically right after building
``apply_X_patches`` via :func:`make_apply_model_patches`).

User registration:

    from opaque.api.transformers.patches.families import (make_apply_family_patches, make_apply_model_patches, register_family)

    apply_my_fam_family_patches = make_apply_family_patches(
        family="my_fam", module_path="my_pkg.modeling_my_fam",
    )
    apply_my_fam_patches = make_apply_model_patches(
        family="my_fam",
        family_apply=apply_my_fam_family_patches,
        module_path="my_pkg.modeling_my_fam",
        classes={...},
        activation_kind="swiglu",
    )
    register_family("my_fam", apply_my_fam_patches)

Now ``opaque.transformers.patches.apply_model_patches(my_fam_instance)`` routes to it.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch.nn as nn

    from opaque.api.transformers.patches.types import ModelPatchFn

_FAMILY_REGISTRY: dict[str, ModelPatchFn] = {}

_builtin_families_loaded = False


def _ensure_builtin_families() -> None:
    """Import the shipped family modules so their registrations land.

    Each ``...patches.models.X`` module calls :func:`register_family` at import
    time, so the registry is only populated as a side effect of importing that
    package. Loading it here — on the first lookup rather than from some
    package initializer — keeps dispatch correct no matter which module the
    caller reached first: a lookup that found an empty registry would report
    the family as unsupported and silently apply no model patches at all.

    The shipped modules only build factory closures; the ``transformers``
    modeling modules they target are imported lazily, inside the patch
    functions, on first use.
    """
    global _builtin_families_loaded
    if _builtin_families_loaded:
        return
    from opaque.api.transformers.patches import models  # noqa: F401

    _builtin_families_loaded = True


def register_family(name: str, apply_fn: ModelPatchFn) -> None:
    """Register a HuggingFace model family.

    The dispatcher will route models with
    ``model.config.model_type == name`` to ``apply_fn``.  Used both by
    opaque's shipped families (each ``opaque.transformers.patches.families.models.X``
    calls this on import) and by downstream users adding their own.

    Args:
        name: The HF ``model_type`` string (matches
            :func:`opaque.transformers.patches.families.family_name`).  Re-registering
            an existing name overwrites the previous registration.
        apply_fn: Callable with signature
            ``apply(model, *, performance=True, compat=True, **kwargs) -> None``,
            typically the return value of
            :func:`opaque.transformers.patches.families.make_apply_model_patches`.
    """
    _FAMILY_REGISTRY[name] = apply_fn


def get_family_apply_fn(name: str) -> ModelPatchFn | None:
    """Return the registered apply function for a family, or ``None``."""
    _ensure_builtin_families()
    return _FAMILY_REGISTRY.get(name)


def supported_families() -> list[str]:
    """List currently registered families (built-ins + user-registered)."""
    _ensure_builtin_families()
    return sorted(_FAMILY_REGISTRY)


def detect_family(model: nn.Module) -> str | None:
    """Detect the model family from the model config."""
    config = getattr(model, "config", None)
    if config:
        model_type = getattr(config, "model_type", None)
        if model_type == "gemma3_text":
            return "gemma3"
        return model_type
    return None


__all__ = [
    "detect_family",
    "get_family_apply_fn",
    "register_family",
    "supported_families",
]
