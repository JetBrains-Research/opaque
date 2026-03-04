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

Patches 7-8 implement a protocol between functional_call and checkpoint:
functional_call records (module, params) on a thread-local stack; checkpoint
captures that context and replays it during recomputation. This ensures
checkpoint sees the correct parameters even after functional_call's context
manager has restored the originals.

After applying these patches, HuggingFace models can use
model.gradient_checkpointing_enable() with vmap(grad(...)).

Skip with: OPAQUE_SKIP_PYTORCH_CHECKPOINT_PATCHES=all
"""

from __future__ import annotations

import logging
import threading

from opaque._env import parse_skip_env

logger = logging.getLogger(__name__)

_is_checkpoint_patched = False


def apply_checkpoint_patches() -> None:
    """Patch PyTorch to allow gradient checkpointing under vmap(grad(...)).

    Applies eight patches:
    1. Remove doesnt_support_saved_tensors_hooks from grad/vjp internals
    2. Add vmap batching rule to checkpoint's _NoopSaveInputs
    3. Disable checkpoint tensor-count validation (fails under vmap)
    4. Use create_graph=False in _autograd_grad (safe for first-order only)
    5. Fix save_on_cpu to use empty_like (vmap-compatible async pinned transfers)
    6. Force use_reentrant=False in HuggingFace's gradient_checkpointing_enable
    7. Record (module, params) on a thread-local in functional_call
    8. Capture param context in checkpoint, replay before recomputation

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
        def __init__(
            self, pin_memory: bool = False, device_type: str = "cuda"
        ) -> None:
            device_module = getattr(torch, device_type, torch.cuda)

            def pack_to_cpu(tensor):  # type: ignore[no-untyped-def]
                if not pin_memory:
                    return (tensor.device, tensor.cpu())
                is_pinnable = (
                    device_module.is_available() and not tensor.is_sparse
                )
                packed = torch.empty_like(
                    tensor, device="cpu", pin_memory=is_pinnable
                )
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

    # Patches 7-8: Parameter-context-aware checkpoint protocol.
    #
    # Problem: torch.func.functional_call replaces module._parameters via
    # a context manager, then restores originals on exit.  Checkpoint
    # recomputation happens during backward — after the context manager has
    # exited — so sublayers see stale (original) parameters.
    #
    # Solution: functional_call records (module, params) on a thread-local
    # stack (Patch 7).  checkpoint captures that stack at forward time and
    # replays it (via _set_module_params) before every recomputation
    # (Patch 8).  During the original forward _set_module_params is a
    # harmless no-op (the same tensors are already on the module).
    from opaque.utils.functional import _set_module_params

    # Shared thread-local between Patch 7 and 8.
    _param_ctx = threading.local()

    # Patch 7: Record active parameter substitutions in functional_call.
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
        stack.append((module, parameter_and_buffer_dicts))
        try:
            return _orig_functional_call(
                module, parameter_and_buffer_dicts, args, kwargs, **kw
            )
        finally:
            stack.pop()

    torch.func.functional_call = _functional_call_with_param_ctx

    # Patch 8: Capture param context in checkpoint, replay on recompute.
    import torch.utils.checkpoint as _ckpt_mod

    _orig_checkpoint = _ckpt_mod.checkpoint

    def _checkpoint_with_param_ctx(function, *args, **kwargs):  # type: ignore[no-untyped-def]
        stack = getattr(_param_ctx, "stack", None)
        captured = list(stack) if stack else None
        if captured is None:
            return _orig_checkpoint(function, *args, **kwargs)

        orig_fn = function

        def fn_with_params(*a, **kw):  # type: ignore[no-untyped-def]
            for module, params in captured:
                if isinstance(params, dict):
                    _set_module_params(module, params)
                else:
                    for d in params:
                        _set_module_params(module, d)
            return orig_fn(*a, **kw)

        return _orig_checkpoint(fn_with_params, *args, **kwargs)

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
