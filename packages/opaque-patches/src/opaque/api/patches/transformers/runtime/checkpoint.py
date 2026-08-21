# Copyright (c) 2025 Opaque Authors
# SPDX-License-Identifier: Apache-2.0
"""Hugging Face glue for gradient checkpointing under vmap(grad(...)).

The torch-core half of this concern belongs to the Torch provider
(:func:`opaque.torch.checkpoint.apply_checkpoint_patch`); what is left here
rebinds ``transformers`` and so is applied on every torch regime, whenever
``transformers`` imports.
"""

from __future__ import annotations

import logging

from opaque.torch import under_functorch_transform

logger = logging.getLogger(__name__)

__all__ = ["apply_checkpoint_patches"]


def apply_checkpoint_patches(*, vmap_checkpointing: bool = True) -> None:
    """Install the Hugging Face checkpointing compatibility patches.

    No-op when transformers is not importable — the torch-side checkpoint+vmap
    composition does not depend on it. Idempotent.
    """
    if vmap_checkpointing is False:
        return

    try:
        import transformers
    except ImportError:
        logger.info(
            "opaque: transformers not importable; HuggingFace checkpoint glue "
            "skipped. The torch-side checkpoint+vmap composition does not depend "
            "on it."
        )
        return

    _force_non_reentrant(transformers)
    _make_input_require_grads_vmap_safe(transformers)


def _force_non_reentrant(transformers) -> None:
    """Force ``use_reentrant=False`` in ``gradient_checkpointing_enable``.

    HF defaults to the reentrant path, which is incompatible with functorch
    transforms. Also flips ``.training`` on the checkpoint-carrying modules (PEFT
    keeps base layers in eval, which otherwise makes checkpoint a no-op).
    """
    orig_enable = transformers.PreTrainedModel.gradient_checkpointing_enable
    if getattr(orig_enable, "__opaque_patched__", False):
        return

    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None):
        if gradient_checkpointing_kwargs is None:
            gradient_checkpointing_kwargs = {}
        if gradient_checkpointing_kwargs.get("use_reentrant", False):
            logger.warning(
                "opaque: overriding use_reentrant=True to False; the reentrant "
                "checkpoint path is incompatible with vmap(grad(...))."
            )
        gradient_checkpointing_kwargs["use_reentrant"] = False
        orig_enable(self, gradient_checkpointing_kwargs=gradient_checkpointing_kwargs)
        for m in self.modules():
            if getattr(m, "gradient_checkpointing", False):
                m.training = True

    gradient_checkpointing_enable.__opaque_patched__ = True
    transformers.PreTrainedModel.gradient_checkpointing_enable = (
        gradient_checkpointing_enable
    )


def _make_input_require_grads_vmap_safe(transformers) -> None:
    """Make ``enable_input_require_grads``' forward hook a no-op under a transform.

    The hook calls ``output.requires_grad_(True)`` on input embeddings; under
    vmap(grad(...)) that both raises and is redundant (the transform already
    tracks differentiability). Handles 4.x (``_require_grads_hook``) and 5.x
    (``_require_grads_hooks``).
    """
    orig_enable = transformers.PreTrainedModel.enable_input_require_grads
    if getattr(orig_enable, "__opaque_patched__", False):
        return

    def _vmap_safe(hook):
        def wrapped(module, args, output):
            if under_functorch_transform():
                return output
            return hook(module, args, output)

        return wrapped

    def enable_input_require_grads(self):
        orig_enable(self)
        handles = getattr(self, "_require_grads_hooks", None)
        if not handles:
            single = getattr(self, "_require_grads_hook", None)
            handles = [single] if single is not None else []
        for handle in handles:
            # Rewrap in place so handle.remove() keeps working.
            hooks_dict = handle.hooks_dict_ref()
            if hooks_dict is None or handle.id not in hooks_dict:
                continue
            hooks_dict[handle.id] = _vmap_safe(hooks_dict[handle.id])

    enable_input_require_grads.__opaque_patched__ = True
    transformers.PreTrainedModel.enable_input_require_grads = enable_input_require_grads
