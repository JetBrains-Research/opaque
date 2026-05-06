"""Opaque HuggingFace Transformers integration.

Houses the DP-SGD-aware :class:`opaque.transformers.trainer.DPTrainer` shim.
HuggingFace compatibility patches and Triton kernels live in
:mod:`opaque.patches.transformers`; import them via
``opaque.patches.apply_model_patches`` / ``apply_runtime_patches``.

Runtime masking / collator / checkpoint shims install on first import of
this package (unless ``OPAQUE_SKIP_TRANSFORMERS_PATCHES`` opts out).
"""

from __future__ import annotations

import os
from importlib.metadata import PackageNotFoundError, version as _pkg_version

try:
    __version__ = _pkg_version("opaque-transformers")
except PackageNotFoundError:
    __version__ = "0.0.0"

_VALID_SKIP_TOKENS = frozenset({"all", "vmap"})

# Whether ``apply_runtime_patches`` ran and landed any subsystem (collator,
# checkpoint, and/or masking).
_COMPAT_PATCHES_LANDED: bool = False
# Whether the masking / vmap-safety layer was enabled inside that call.
_VMAP_MASKING_LANDED: bool = False


def _parse_skip_transformers_patches() -> frozenset[str]:
    raw = os.environ.get("OPAQUE_SKIP_TRANSFORMERS_PATCHES", "").strip()
    if not raw:
        return frozenset()
    tokens = frozenset(part.strip() for part in raw.split(",") if part.strip())
    unknown = sorted(tokens - _VALID_SKIP_TOKENS)
    if unknown:
        raise ValueError(
            "OPAQUE_SKIP_TRANSFORMERS_PATCHES: unknown token(s) "
            + ", ".join(repr(t) for t in unknown)
        )
    return tokens


def _apply_transformers_runtime_patches() -> None:
    """Install vmap-safe HF runtime shims (masking, collator, checkpoint).

    Respects ``OPAQUE_SKIP_TRANSFORMERS_PATCHES``:

    - ``all`` — skip every runtime patch (both flags stay ``False``).
    - ``vmap`` — skip only the masking / SDPA shim layer; collator + checkpoint
      patches still apply so :func:`is_patched` remains ``True``.
    """
    global _COMPAT_PATCHES_LANDED, _VMAP_MASKING_LANDED

    tokens = _parse_skip_transformers_patches()
    if "all" in tokens:
        return

    skip_vmap = "vmap" in tokens
    try:
        from opaque.patches import apply_runtime_patches

        apply_runtime_patches(vmap_masking=not skip_vmap)
        _COMPAT_PATCHES_LANDED = True
        _VMAP_MASKING_LANDED = not skip_vmap
    except ImportError:
        pass


def is_patched() -> bool:
    """``True`` if any opaque Transformers runtime patch landed on import."""
    return _COMPAT_PATCHES_LANDED


def is_vmap_patched() -> bool:
    """``True`` if the masking / SDPA vmap-safety layer is active."""
    return _VMAP_MASKING_LANDED


def patch_all() -> None:
    """Re-run runtime patch orchestration (idempotent where underlying shims allow)."""
    _apply_transformers_runtime_patches()


_apply_transformers_runtime_patches()

__all__ = [
    "__version__",
    "is_patched",
    "is_vmap_patched",
    "patch_all",
]
