"""Opaque HuggingFace integration.

Curated facade over the HuggingFace Transformers compatibility layer.

Patching is **opt-in**: module import does not touch ``transformers``.
Call :func:`patch_all` explicitly (or the umbrella :func:`opaque.patch_all`)
before training with a supported HuggingFace model under ``vmap(grad(...))``.

Quick start::

    import opaque.huggingface as hf

    hf.patch_all()                 # apply all vmap/kernel/data/kv-cache patches
    assert hf.is_patched()

Future subpackages (``trainer``, ``callbacks``, ``integrations``,
``data``, ``models``) are scaffolded but currently empty — see
individual module docstrings for the planned API.
"""

from opaque.huggingface.patches import (
    apply_transformers_patches as _apply,
    is_kernel_patched,
    is_transformers_patched as _is_patched,
    is_vmap_patched,
    patch_lora_model,
)

__version__ = "0.0.0.dev0"


def patch_all() -> None:
    """Apply all HuggingFace Transformers compatibility patches (idempotent).

    Honors the ``OPAQUE_SKIP_TRANSFORMERS_*`` environment variables
    documented in :mod:`opaque.huggingface.patches`.
    """
    _apply()


def is_patched() -> bool:
    """Check whether :func:`patch_all` has already been applied."""
    return _is_patched()


__all__ = [
    "__version__",
    "patch_all",
    "is_patched",
    "is_kernel_patched",
    "is_vmap_patched",
    "patch_lora_model",
]
