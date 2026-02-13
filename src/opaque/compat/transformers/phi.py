# Copyright (c) 2025 Opaque Authors
# SPDX-License-Identifier: Apache-2.0
"""Phi model patcher for vmap compatibility.

Applies to: Phi-2, Phi-3
"""

from opaque.compat.transformers.base import (
    BasePatcher,
    vmap_repeat_kv,
    vmap_eager_attention_forward,
)
from opaque.compat.transformers.registry import register_patcher


@register_patcher("phi")
class PhiPatcher(BasePatcher):
    """Patcher for Phi-2 models."""

    architecture_name = "phi"
    transformers_module_path = "transformers.models.phi.modeling_phi"

    def _patch_module(self, module) -> None:
        """Apply Phi-specific patches."""
        self._store_and_patch(module, "repeat_kv", vmap_repeat_kv)
        if hasattr(module, "eager_attention_forward"):
            self._store_and_patch(
                module, "eager_attention_forward", vmap_eager_attention_forward
            )

    def _unpatch_module(self, module) -> None:
        self._restore(module, "repeat_kv")
        self._restore(module, "eager_attention_forward")


@register_patcher("phi3")
class Phi3Patcher(BasePatcher):
    """Patcher for Phi-3 models."""

    architecture_name = "phi3"
    transformers_module_path = "transformers.models.phi3.modeling_phi3"

    def _patch_module(self, module) -> None:
        self._store_and_patch(module, "repeat_kv", vmap_repeat_kv)
        if hasattr(module, "eager_attention_forward"):
            self._store_and_patch(
                module, "eager_attention_forward", vmap_eager_attention_forward
            )

    def _unpatch_module(self, module) -> None:
        self._restore(module, "repeat_kv")
        self._restore(module, "eager_attention_forward")
