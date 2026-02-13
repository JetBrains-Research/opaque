# Copyright (c) 2025 Opaque Authors
# SPDX-License-Identifier: Apache-2.0
"""OLMo model patcher for vmap compatibility."""

from opaque.compat.transformers.base import (
    BasePatcher,
    vmap_repeat_kv,
    vmap_eager_attention_forward,
)
from opaque.compat.transformers.registry import register_patcher


@register_patcher("olmo")
class OLMoPatcher(BasePatcher):
    """Patcher for OLMo models."""

    architecture_name = "olmo"
    transformers_module_path = "transformers.models.olmo.modeling_olmo"

    def _patch_module(self, module) -> None:
        """Apply OLMo-specific patches."""
        self._store_and_patch(module, "repeat_kv", vmap_repeat_kv)
        if hasattr(module, "eager_attention_forward"):
            self._store_and_patch(
                module, "eager_attention_forward", vmap_eager_attention_forward
            )

    def _unpatch_module(self, module) -> None:
        self._restore(module, "repeat_kv")
        self._restore(module, "eager_attention_forward")
