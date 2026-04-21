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

Patches 4, 7 implement a post-set / restore protocol between functional_call
and _autograd_grad: functional_call post-sets params on the module after its
context manager restores originals (so backward sees correct params);
_autograd_grad restores originals after backward completes.

After applying these patches, HuggingFace models can use
model.gradient_checkpointing_enable() with vmap(grad(...)).

Skip with: OPAQUE_SKIP_PYTORCH_CHECKPOINT_PATCHES=all
"""

from __future__ import annotations

import logging
import threading

from opaque.core._env import parse_skip_env

logger = logging.getLogger(__name__)

_is_checkpoint_patched = False


def apply_checkpoint_patches() -> None:
    """Patch PyTorch to allow gradient checkpointing under vmap(grad(...)).

    Applies eight patches:
    1. Remove doesnt_support_saved_tensors_hooks from grad/vjp internals
    2. Add vmap batching rule to checkpoint's _NoopSaveInputs
    3. Disable checkpoint tensor-count validation (fails under vmap)
    4. Use create_graph=False in _autograd_grad + restore params after backward
    5. Fix save_on_cpu to use empty_like (vmap-compatible async pinned transfers)
    6. Force use_reentrant=False in HuggingFace's gradient_checkpointing_enable
    7. Post-set params after functional_call for backward recomputation
    8. Transparent checkpoint wrapper for HF binding compatibility

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

    # Shared imports and thread-local for the functional_call ↔ _autograd_grad
    # protocol (Patches 4, 7, 8).
    from opaque.core.utils.functional import _set_module_params

    _param_ctx = threading.local()

    # Patch 4: Use create_graph=False in _autograd_grad + restore params.
    # With create_graph=True (the default), backward builds a computation
    # graph whose saved tensors trap recomputed activations — defeating
    # checkpoint. create_graph=False avoids this entirely.
    #
    # Also restores original params after backward completes. Patch 7
    # post-sets new params on the module for backward; this hook undoes
    # that so the module is clean afterwards.
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
        result = _orig_autograd_grad(
            outputs,
            inputs,
            grad_outputs,
            retain_graph=retain_graph,
            create_graph=False,
        )
        # Restore original params after backward completes (undoes Patch 7 post-set).
        pending = getattr(_param_ctx, "pending_restore", None)
        if pending:
            for mod, orig in pending:
                _set_module_params(mod, orig)
            pending.clear()
        return result

    eager_transforms._autograd_grad = _autograd_grad_no_create_graph

    # Patch 5: Fix save_on_cpu to work under vmap.
    # PyTorch's save_on_cpu uses torch.empty(tensor.size(), ...) which
    # returns the logical (unbatched) shape under vmap. The subsequent
    # copy_() then fails because the destination is unbatched while the
    # source is batched. Using empty_like preserves vmap batch dimensions
    # via its EXISTING_BDIM batching rule, so both tensors are batched
    # and copy_() works. Async pinned memory transfers are preserved.
    import torch
    import torch.autograd.graph as autograd_graph

    _OrigSaveOnCpu = autograd_graph.save_on_cpu

    class _VmapSaveOnCpu(_OrigSaveOnCpu):
        def __init__(self, pin_memory: bool = False, device_type: str = "cuda") -> None:
            device_module = getattr(torch, device_type, torch.cuda)

            def pack_to_cpu(tensor):  # type: ignore[no-untyped-def]
                if not pin_memory:
                    return (tensor.device, tensor.cpu())
                is_pinnable = device_module.is_available() and not tensor.is_sparse
                packed = torch.empty_like(tensor, device="cpu", pin_memory=is_pinnable)
                packed.copy_(tensor, non_blocking=is_pinnable)
                return (tensor.device, packed)

            def unpack_from_cpu(packed):  # type: ignore[no-untyped-def]
                device, tensor = packed
                return tensor.to(device, non_blocking=pin_memory)

            # Skip _OrigSaveOnCpu.__init__ (it would install the broken
            # hooks); go directly to the grandparent saved_tensors_hooks.
            torch.autograd.graph.saved_tensors_hooks.__init__(
                self, pack_to_cpu, unpack_from_cpu
            )

    autograd_graph.save_on_cpu = _VmapSaveOnCpu

    # Patches 7-8: Post-set / restore protocol for functional_call + checkpoint.
    #
    # Problem: torch.func.functional_call replaces module._parameters via
    # a context manager, then restores originals on exit.  Checkpoint
    # recomputation happens during backward — after the context manager has
    # exited — so sublayers see stale (original) parameters.
    #
    # Solution (Patch 7): After functional_call completes and restores
    # originals, immediately re-apply the new params ("post-set") so the
    # module has them for backward / checkpoint recomputation.  Originals
    # are captured and queued for restore after backward (Patch 4).
    #
    # Patch 8 is a transparent checkpoint wrapper kept only so Patch 6 can
    # update HF's stale checkpoint binding.

    # Patch 7: Post-set params after functional_call for backward.
    _orig_functional_call = torch.func.functional_call

    def _functional_call_with_param_ctx(  # type: ignore[no-untyped-def]
        module,
        parameter_and_buffer_dicts,
        args=(),
        kwargs=None,
        **kw,
    ):
        stack = getattr(_param_ctx, "stack", None)
        if stack is None:
            _param_ctx.stack = stack = []
        is_outermost = len(stack) == 0

        # Before the outermost call: restore any stale pending params from
        # a previous forward that had no backward (e.g. inference).
        if is_outermost:
            pending = getattr(_param_ctx, "pending_restore", None)
            if pending:
                for mod, orig in pending:
                    _set_module_params(mod, orig)
                pending.clear()

            # Snapshot originals (what functional_call will restore on exit).
            all_named = dict((*module.named_parameters(), *module.named_buffers()))
            if isinstance(parameter_and_buffer_dicts, dict):
                keys = parameter_and_buffer_dicts.keys()
            else:
                keys: set[str] = set()  # type: ignore[no-redef]
                for d in parameter_and_buffer_dicts:
                    keys.update(d.keys())
            originals = {k: all_named[k] for k in keys if k in all_named}

        stack.append((module, parameter_and_buffer_dicts))
        try:
            result = _orig_functional_call(
                module, parameter_and_buffer_dicts, args, kwargs, **kw
            )
        finally:
            stack.pop()

        # After outermost: re-apply params so backward sees them.
        if is_outermost:
            if isinstance(parameter_and_buffer_dicts, dict):
                _set_module_params(module, parameter_and_buffer_dicts)
            else:
                for d in parameter_and_buffer_dicts:
                    _set_module_params(module, d)
            # Queue originals for restore after backward (Patch 4).
            pending = getattr(_param_ctx, "pending_restore", None)
            if pending is None:
                _param_ctx.pending_restore = pending = []
            pending.append((module, originals))

        return result

    torch.func.functional_call = _functional_call_with_param_ctx

    # Patch 8: Transparent checkpoint wrapper.
    # With post-set (Patch 7), checkpoint recomputation already sees the
    # correct params on the module — no per-segment _set_module_params needed.
    # This wrapper exists so Patch 6 can point HF's stale binding to it.
    import torch.utils.checkpoint as _ckpt_mod

    _orig_checkpoint = _ckpt_mod.checkpoint

    def _checkpoint_with_param_ctx(function, *args, **kwargs):  # type: ignore[no-untyped-def]
        return _orig_checkpoint(function, *args, **kwargs)

    _ckpt_mod.checkpoint = _checkpoint_with_param_ctx

    # Patch 6: Force use_reentrant=False in HuggingFace's
    # gradient_checkpointing_enable(). HF defaults to use_reentrant=True,
    # which is fundamentally incompatible with functorch transforms.
    #
    # Also fixes the stale ``checkpoint`` binding: HF does
    #   ``from torch.utils.checkpoint import checkpoint``
    # at module load time.  If HF was imported before Patch 8 (e.g. by
    # ``pytest.importorskip``), that binding points to the unpatched
    # function.  We overwrite it so Patch 8's param-context protocol works.
    try:
        import transformers
        import transformers.modeling_utils as _hf_modeling

        if hasattr(_hf_modeling, "checkpoint"):
            _hf_modeling.checkpoint = _checkpoint_with_param_ctx

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
