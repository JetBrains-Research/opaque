# Copyright (c) 2025 Opaque Authors
# SPDX-License-Identifier: Apache-2.0
"""Enable gradient checkpointing under vmap(grad(...)).

PyTorch's gradient checkpointing (torch.utils.checkpoint) is incompatible with
functorch transforms out of the box due to three issues:

1. saved_tensors_hooks are blocked inside grad/vjp by a safety decorator
2. _NoopSaveInputs (checkpoint internal) lacks a vmap batching rule
3. _CheckpointFrame tensor-count validation fails under vmap

Additionally, functorch's grad uses create_graph=True for its internal backward
pass, which causes recomputed tensors to be captured in the backward graph —
completely defeating checkpoint's memory savings. Since opaque only uses
first-order differentiation (vmap(grad)), we can safely use create_graph=False,
which both fixes the memory issue and slightly reduces overhead.

After applying these patches, HuggingFace models can use
model.gradient_checkpointing_enable() with vmap(grad(...)).

Skip with: OPAQUE_SKIP_PYTORCH_CHECKPOINT_PATCHES=all
"""

from __future__ import annotations

import logging

from opaque._env import parse_skip_env

logger = logging.getLogger(__name__)

_is_checkpoint_patched = False


def apply_checkpoint_patches() -> None:
    """Patch PyTorch to allow gradient checkpointing under vmap(grad(...)).

    Applies four patches:
    1. Remove doesnt_support_saved_tensors_hooks from grad/vjp internals
    2. Add vmap batching rule to checkpoint's _NoopSaveInputs
    3. Disable checkpoint tensor-count validation (fails under vmap)
    4. Use create_graph=False in _autograd_grad (safe for first-order only)

    No-op when OPAQUE_SKIP_PYTORCH_CHECKPOINT_PATCHES=all.
    """
    global _is_checkpoint_patched

    if _is_checkpoint_patched:
        return

    skip = parse_skip_env("OPAQUE_SKIP_PYTORCH_CHECKPOINT_PATCHES")
    if "all" in skip:
        _is_checkpoint_patched = True
        return

    import torch._functorch.eager_transforms as eager_transforms
    from torch.utils.checkpoint import _CheckpointFrame, _NoopSaveInputs

    # Patch 1: Remove saved_tensors_hooks restriction.
    # grad_and_value_impl and _vjp_with_argnums are wrapped with
    # @doesnt_support_saved_tensors_hooks which blocks checkpoint's hooks.
    if hasattr(eager_transforms.grad_and_value_impl, "__wrapped__"):
        eager_transforms.grad_and_value_impl = (
            eager_transforms.grad_and_value_impl.__wrapped__
        )
    if hasattr(eager_transforms._vjp_with_argnums, "__wrapped__"):
        eager_transforms._vjp_with_argnums = (
            eager_transforms._vjp_with_argnums.__wrapped__
        )

    # Patch 2: Add vmap batching rule to _NoopSaveInputs.
    # Checkpoint uses this autograd.Function internally; vmap needs a rule.
    # _NoopSaveInputs is a bookkeeping node that doesn't transform data.
    @staticmethod  # type: ignore[misc]
    def _noop_save_inputs_vmap(info, in_dims, *args):  # type: ignore[no-untyped-def]
        return _NoopSaveInputs.apply(*args), None

    _NoopSaveInputs.vmap = _noop_save_inputs_vmap  # type: ignore[attr-defined]

    # Patch 3: Disable tensor-count validation in _CheckpointFrame.
    # Under vmap, recomputation produces different tensor counts due to
    # batched operations expanding differently than unbatched ones.
    _CheckpointFrame.check_recomputed_tensors_match = lambda self, gid: None  # type: ignore[assignment]

    # Patch 4: Use create_graph=False in _autograd_grad.
    # With create_graph=True (the default), backward builds a computation
    # graph whose saved tensors trap recomputed activations — defeating
    # checkpoint. create_graph=False avoids this entirely.
    #
    # SAFETY: create_graph=False is safe for first-order differentiation
    # (grad, vjp, jacrev, hessian). It breaks higher-order-through-backward
    # transforms like jacrev(jacrev()) or grad(hessian()). Opaque only uses
    # first-order (vmap(grad)).
    _orig_autograd_grad = eager_transforms._autograd_grad

    def _autograd_grad_no_create_graph(
        outputs,
        inputs,
        grad_outputs=None,
        retain_graph=False,
        create_graph=True,
    ):  # type: ignore[no-untyped-def]
        return _orig_autograd_grad(
            outputs,
            inputs,
            grad_outputs,
            retain_graph=retain_graph,
            create_graph=False,
        )

    eager_transforms._autograd_grad = _autograd_grad_no_create_graph

    # Patch 5: Force use_reentrant=False in HuggingFace's
    # gradient_checkpointing_enable(). HF defaults to use_reentrant=True,
    # which is fundamentally incompatible with functorch transforms.
    try:
        import transformers

        _orig_gc_enable = transformers.PreTrainedModel.gradient_checkpointing_enable

        def _gradient_checkpointing_enable_nonreentrant(
            self, gradient_checkpointing_kwargs=None
        ):  # type: ignore[no-untyped-def]
            if gradient_checkpointing_kwargs is None:
                gradient_checkpointing_kwargs = {}
            if gradient_checkpointing_kwargs.get("use_reentrant", False):
                logger.warning(
                    "opaque: Overriding use_reentrant=True to False. "
                    "The reentrant checkpoint path is incompatible with "
                    "vmap(grad(...))."
                )
            gradient_checkpointing_kwargs["use_reentrant"] = False
            _orig_gc_enable(
                self, gradient_checkpointing_kwargs=gradient_checkpointing_kwargs
            )
            # HuggingFace's checkpointing is gated on `self.training` per
            # decoder layer. PEFT intentionally keeps base-model layers in
            # eval mode (to disable Dropout / BatchNorm updates), so
            # checkpoint silently becomes a no-op.  Fix: flip `.training`
            # only on the modules that carry the checkpoint flag, leaving
            # their children (Dropout, etc.) untouched.
            for m in self.modules():
                if getattr(m, "gradient_checkpointing", False):
                    m.training = True

        transformers.PreTrainedModel.gradient_checkpointing_enable = (
            _gradient_checkpointing_enable_nonreentrant
        )
    except ImportError:
        pass

    logger.debug("opaque: Applied checkpoint compatibility patches")
    _is_checkpoint_patched = True


def is_checkpoint_patched() -> bool:
    """Check if checkpoint compatibility patches have been applied."""
    return _is_checkpoint_patched
