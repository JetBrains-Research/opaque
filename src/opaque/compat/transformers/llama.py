# Copyright (c) 2025 Opaque Authors
# SPDX-License-Identifier: Apache-2.0
"""LLaMA-family model patcher for vmap compatibility.

Applies to: LLaMA, Mistral, DeepSeek (based on LLaMA architecture)
"""

from opaque.compat.transformers.base import (
    BasePatcher,
    vmap_repeat_kv,
    vmap_eager_attention_forward,
)
from opaque.compat.transformers.registry import register_patcher


@register_patcher("llama")
class LlamaPatcher(BasePatcher):
    """Patcher for LLaMA and LLaMA-based models."""

    architecture_name = "llama"
    transformers_module_path = "transformers.models.llama.modeling_llama"

    def _patch_module(self, module) -> None:
        """Apply LLaMA-specific patches."""
        # Patch repeat_kv - uses hardcoded 4D unpacking
        self._store_and_patch(module, "repeat_kv", vmap_repeat_kv)

        # Patch eager_attention_forward if it exists
        if hasattr(module, "eager_attention_forward"):
            self._store_and_patch(
                module, "eager_attention_forward", vmap_eager_attention_forward
            )

    def _unpatch_module(self, module) -> None:
        """Restore original implementations."""
        self._restore(module, "repeat_kv")
        self._restore(module, "eager_attention_forward")


@register_patcher("mistral")
class MistralPatcher(BasePatcher):
    """Patcher for Mistral models (LLaMA-based)."""

    architecture_name = "mistral"
    transformers_module_path = "transformers.models.mistral.modeling_mistral"

    def _patch_module(self, module) -> None:
        self._store_and_patch(module, "repeat_kv", vmap_repeat_kv)
        if hasattr(module, "eager_attention_forward"):
            self._store_and_patch(
                module, "eager_attention_forward", vmap_eager_attention_forward
            )

    def _unpatch_module(self, module) -> None:
        self._restore(module, "repeat_kv")
        self._restore(module, "eager_attention_forward")


@register_patcher("deepseek")
class DeepSeekPatcher(BasePatcher):
    """Patcher for DeepSeek models (LLaMA-based)."""

    architecture_name = "deepseek"
    # DeepSeek uses LLaMA modeling code
    transformers_module_path = "transformers.models.llama.modeling_llama"

    def _patch_module(self, module) -> None:
        self._store_and_patch(module, "repeat_kv", vmap_repeat_kv)
        if hasattr(module, "eager_attention_forward"):
            self._store_and_patch(
                module, "eager_attention_forward", vmap_eager_attention_forward
            )

    def _unpatch_module(self, module) -> None:
        self._restore(module, "repeat_kv")
        self._restore(module, "eager_attention_forward")
