# Copyright (c) 2025 Opaque Authors
# SPDX-License-Identifier: Apache-2.0
"""Phi-3-specific vmap compatibility patches.

Phi-3 requires minimal patching for vmap compatibility. This module only patches:
1. DynamicCache.get_usable_length() - Adds method if not present
2. repeat_kv - Uses shared vmap-compatible implementation

Note: Phi-3's attention forward methods are already vmap-compatible with the
shared patches, so no attention patching is needed.
"""

import importlib
from typing import Any

from opaque.compat.transformers._shared import vmap_repeat_kv

_PHI3_MODULE = "transformers.models.phi3.modeling_phi3"


class VmapCompatibleDynamicCache:
    """vmap-compatible wrapper for Phi-3's DynamicCache.

    Phi-3 uses a custom DynamicCache that may not have get_usable_length() method.
    This wrapper ensures compatibility under vmap.
    """

    def __init__(self, original_cache):
        """Initialize with original cache object."""
        self._cache = original_cache

    def __getattr__(self, name: str) -> Any:
        """Delegate to original cache for all other attributes."""
        return getattr(self._cache, name)

    def get_usable_length(self, layer_idx: int) -> int:
        """Get usable sequence length for given layer."""
        if hasattr(self._cache, "get_usable_length"):
            return self._cache.get_usable_length(layer_idx)

        # Fallback for caches that store KV in key_cache/value_cache attributes
        if hasattr(self._cache, "key_cache") and len(self._cache.key_cache) > layer_idx:
            key_cache = self._cache.key_cache[layer_idx]
            if key_cache is not None:
                # key_cache shape: (batch, num_heads, seq_len, head_dim)
                # Under vmap: (num_heads, seq_len, head_dim)
                return key_cache.shape[-2]

        # Fallback for caches with seen_tokens attribute
        if hasattr(self._cache, "seen_tokens"):
            return self._cache.seen_tokens

        # Last resort: return 0 (no cached tokens)
        return 0


# =============================================================================
# Patch application
# =============================================================================


def apply_phi3_patches() -> None:
    """Apply Phi-3-specific vmap patches.

    Patches Phi-3 with cache-compatible attention implementation.

    Note: Requires apply_shared_patches() from _shared to be called first.
    """
    try:
        module = importlib.import_module(_PHI3_MODULE)

        # Patch DynamicCache if it exists
        if hasattr(module, "DynamicCache"):
            original_init = module.DynamicCache.__init__

            def vmap_compatible_init(self, *args, **kwargs):
                """Initialize with vmap compatibility."""
                original_init(self, *args, **kwargs)

                # Add get_usable_length method if not present
                if not hasattr(self, "get_usable_length"):

                    def get_usable_length(layer_idx: int) -> int:
                        if (
                            hasattr(self, "key_cache")
                            and len(self.key_cache) > layer_idx
                        ):
                            kc = self.key_cache[layer_idx]
                            if kc is not None:
                                return kc.shape[-2]
                        if hasattr(self, "seen_tokens"):
                            return self.seen_tokens
                        return 0

                    self.get_usable_length = get_usable_length

            module.DynamicCache.__init__ = vmap_compatible_init

        # Patch repeat_kv with base implementation
        if hasattr(module, "repeat_kv"):
            module.repeat_kv = vmap_repeat_kv

    except ImportError:
        # Phi-3 not available in this transformers version
        pass
