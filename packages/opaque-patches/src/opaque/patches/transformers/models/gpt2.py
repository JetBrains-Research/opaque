# Copyright (c) 2025 Opaque Authors
# SPDX-License-Identifier: Apache-2.0
"""Patches for GPT-2 — vmap-safety only.

GPT-2 uses GELU activations and standard LayerNorm (not SwiGLU/GeGLU or
RMSNorm), so it has no MLP or normalization kernel to patch.  However, it
still needs the standard vmap-safety patches:

- ``batchify``: transformers ≥ 4.47 added a ``logits_to_keep`` slice
  (``hidden_states[:, slice_indices, :]``) that assumes a 3-D batch dim.
  Under ``vmap`` the batch dim is absent, making hidden_states 2-D and
  causing an IndexError.  The batchify patch adds the missing dim on entry
  and strips it on exit.
- ``kv_cache``: prevents DynamicCache allocation per-example under vmap.
"""

from __future__ import annotations

from opaque.patches.transformers._registry import register_family
from opaque.patches.transformers.components.batchify import apply_batchify_patch
from opaque.patches.transformers.components.kv_cache import apply_kv_cache_patch


def apply_gpt2_patches(
    model=None,
    *,
    performance: bool = True,
    compat: bool = True,
    **kwargs,
) -> None:
    try:
        from transformers.models.gpt2.modeling_gpt2 import GPT2LMHeadModel
    except ImportError:
        return

    if kwargs.get("batchify", compat):
        apply_batchify_patch(GPT2LMHeadModel, model)
    if kwargs.get("kv_cache", compat):
        apply_kv_cache_patch(GPT2LMHeadModel, model)


register_family("gpt2", apply_gpt2_patches)

__all__ = ["apply_gpt2_patches"]
