"""Transformers runtime patch bootstrap (explicit, no import side-effects).

Global HF runtime shims (vmap-safe masking, collator empty-batch handling,
gradient-checkpointing hooks) are **not** applied when this package is
imported. Call :func:`patch_all` or construct
:class:`~opaque.transformers.trainer.DPTrainer`, which invokes
:func:`apply_transformers_runtime_compat_patches` then applies per-model
patches via :func:`opaque.patches.apply_model_patches` with ``compat=True``
and ``performance=args.use_liger_kernel``.
"""

from __future__ import annotations

_COMPAT_PATCHES_LANDED: bool = False
_VMAP_MASKING_LANDED: bool = False


def apply_transformers_runtime_compat_patches() -> None:
    """Install global HF runtime compat shims (masking, collator, checkpoint).

    Always passes ``compat=True`` into :func:`opaque.patches.apply_runtime_patches`.
    Idempotent for a given interpreter in practice (underlying shims guard
    re-application).
    """
    global _COMPAT_PATCHES_LANDED, _VMAP_MASKING_LANDED

    try:
        from opaque.patches import apply_runtime_patches

        apply_runtime_patches(compat=True)
        _COMPAT_PATCHES_LANDED = True
        _VMAP_MASKING_LANDED = True
    except ImportError:
        pass


def is_patched() -> bool:
    """``True`` if any opaque Transformers runtime patch has been applied."""
    return _COMPAT_PATCHES_LANDED


def is_vmap_patched() -> bool:
    """``True`` if the masking / SDPA vmap-safety layer is active."""
    return _VMAP_MASKING_LANDED


def patch_all() -> None:
    """Apply global runtime compat patches (masking, collator, checkpoint).

    Does **not** apply model-level patches; use
    :func:`opaque.patches.apply_model_patches` on your module, or train with
    :class:`~opaque.transformers.trainer.DPTrainer`.
    """
    apply_transformers_runtime_compat_patches()


__all__ = [
    "apply_transformers_runtime_compat_patches",
    "is_patched",
    "is_vmap_patched",
    "patch_all",
]
