# Copyright (c) 2025 Opaque Authors
# SPDX-License-Identifier: Apache-2.0
"""Registry for model-specific patchers and auto-detection utilities."""

from contextlib import contextmanager
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from opaque.compat.transformers.base import BasePatcher

# Registry maps architecture names to patcher classes
_PATCHER_REGISTRY: dict[str, type["BasePatcher"]] = {}

# Track patchers applied to models (by id)
_ACTIVE_PATCHERS: dict[int, "BasePatcher"] = {}


def register_patcher(architecture_name: str):
    """Decorator to register a patcher class for an architecture."""

    def decorator(cls: type["BasePatcher"]) -> type["BasePatcher"]:
        _PATCHER_REGISTRY[architecture_name.lower()] = cls
        return cls

    return decorator


def get_model_architecture(model) -> Optional[str]:
    """Detect model architecture from config.

    Returns the architecture name (lowercase) or None if not detected.
    """
    config = getattr(model, "config", None)
    if config is None:
        return None

    # Check architectures list (most reliable)
    architectures = getattr(config, "architectures", None)
    if architectures:
        # Extract base name: "LlamaForCausalLM" -> "llama"
        arch = architectures[0]
        # Common suffixes to strip
        for suffix in ("ForCausalLM", "ForSequenceClassification", "Model", "LMHeadModel"):
            if arch.endswith(suffix):
                arch = arch[: -len(suffix)]
                break
        return arch.lower()

    # Fallback: model_type from config
    model_type = getattr(config, "model_type", None)
    if model_type:
        return model_type.lower()

    return None


def get_patcher_for_model(model) -> Optional["BasePatcher"]:
    """Get an appropriate patcher instance for the given model.

    Returns None if model architecture is not supported.
    """
    arch = get_model_architecture(model)
    if arch is None:
        return None

    # Try direct match
    if arch in _PATCHER_REGISTRY:
        return _PATCHER_REGISTRY[arch]()

    # Try aliases (e.g., "qwen2" might be registered as "qwen")
    for registered_arch, patcher_cls in _PATCHER_REGISTRY.items():
        if arch.startswith(registered_arch) or registered_arch.startswith(arch):
            return patcher_cls()

    return None


def patch_model(model) -> bool:
    """Apply vmap-compatible patches to a model.

    Auto-detects the model architecture and applies appropriate patches.

    Args:
        model: A HuggingFace transformers model

    Returns:
        True if patches were applied, False if model not supported or already patched
    """
    model_id = id(model)

    # Already patched
    if model_id in _ACTIVE_PATCHERS:
        return False

    patcher = get_patcher_for_model(model)
    if patcher is None:
        return False

    patcher.patch()
    _ACTIVE_PATCHERS[model_id] = patcher
    return True


def unpatch_model(model) -> bool:
    """Remove vmap-compatible patches from a model.

    Args:
        model: A HuggingFace transformers model

    Returns:
        True if patches were removed, False if model wasn't patched
    """
    model_id = id(model)

    if model_id not in _ACTIVE_PATCHERS:
        return False

    patcher = _ACTIVE_PATCHERS.pop(model_id)
    patcher.unpatch()
    return True


def is_patched(model) -> bool:
    """Check if a model has been patched."""
    return id(model) in _ACTIVE_PATCHERS


@contextmanager
def vmap_compat(model):
    """Context manager for temporarily applying vmap-compatible patches.

    Usage:
        with vmap_compat(model):
            grads, state = clipped_grad(...)
    """
    was_patched = is_patched(model)

    if not was_patched:
        patch_model(model)

    try:
        yield
    finally:
        if not was_patched:
            unpatch_model(model)


def list_supported_architectures() -> list[str]:
    """Return list of supported architecture names."""
    return list(_PATCHER_REGISTRY.keys())


# Expose registry for introspection
SUPPORTED_ARCHITECTURES = _PATCHER_REGISTRY


# Import patchers to trigger registration
# This must be at the end to avoid circular imports
def _register_all_patchers():
    """Import all patcher modules to register them."""
    # Import each patcher module - they auto-register via decorator
    try:
        from opaque.compat.transformers import llama  # noqa: F401
    except ImportError:
        pass
    try:
        from opaque.compat.transformers import qwen2  # noqa: F401
    except ImportError:
        pass
    try:
        from opaque.compat.transformers import phi  # noqa: F401
    except ImportError:
        pass
    try:
        from opaque.compat.transformers import olmo  # noqa: F401
    except ImportError:
        pass


_register_all_patchers()
