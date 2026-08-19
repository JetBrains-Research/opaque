# Copyright (c) 2025 Opaque Authors
# SPDX-License-Identifier: Apache-2.0
"""Torch-core checkpoint/functorch compatibility patches.

This module lives in ``opaque-torch`` so the Torch provider can apply the
necessary shims without depending on ``opaque-patches``.  The public entry
points ``apply_checkpoint_patch`` and ``is_checkpoint_patched`` are re-exported
by ``opaque.api.patches.torch.checkpoint`` together with Hugging Face glue.
"""

from __future__ import annotations

import contextlib
import logging
import threading

logger = logging.getLogger(__name__)

_is_checkpoint_patched = False

# Idempotency flags for individual patches so repeated calls are harmless even
# when the top-level orchestrator is bypassed.
_SAVED_TENSOR_HOOKS_GUARD_APPLIED = False
_SAVE_ON_CPU_APPLIED = False
_NOOP_SAVE_INPUTS_APPLIED = False
_CREATE_GRAPH_APPLIED = False
_REPARAMETRIZE_RECOMPUTE_APPLIED = False

_orig_reparametrize = None
_active = threading.local()

_HIGHER_ORDER_MESSAGE = (
    "torch.func transforms (grad, vjp, jacrev, hessian) don't support saved "
    "tensor hooks (e.g. torch.autograd.graph.save_on_cpu) under higher-order "
    "differentiation such as grad(grad) or hessian: the saved-tensor "
    "round-trip severs the higher-order graph and would silently produce "
    "incorrect gradients. First-order use is supported."
)
_COMPILE_MESSAGE = (
    "torch.func transforms (grad, vjp, jacrev, hessian) don't support saved "
    "tensor hooks (e.g. torch.autograd.graph.save_on_cpu) under torch.compile, "
    "where saved-tensor hooks are managed by AOTAutograd. Use eager execution "
    "for saved-tensor hooks with torch.func."
)


def prev_grad_mode() -> bool:
    """Return the grad mode captured when the innermost ``grad`` was entered."""
    import torch

    if torch.compiler.is_compiling():
        return torch.is_grad_enabled()
    try:
        from torch._C._functorch import (
            CGradInterpreterPtr,
            TransformType,
            peek_interpreter_stack,
        )

        interpreter = peek_interpreter_stack()
        if interpreter is None or interpreter.key() != TransformType.Grad:
            return torch.is_grad_enabled()
        return CGradInterpreterPtr(interpreter).prevGradMode()
    except Exception:  # pragma: no cover - API moved/unavailable
        return torch.is_grad_enabled()


# ---------------------------------------------------------------------------
# Capability probes
# ---------------------------------------------------------------------------


def native_checkpoint_support() -> bool:
    """True when torch supports ``vmap(grad(checkpoint(...)))`` natively.

    Sentinel: the parameter-lifetime fix records active reparametrizations on a
    thread-local so checkpoint recomputation re-binds the functional parameters.
    It ships together with the create_graph conditioning, so this one symbol
    gates the whole checkpoint-side patch set.
    """
    try:
        from torch.nn.utils import stateless
    except Exception:  # pragma: no cover - torch layout moved
        return False
    return hasattr(stateless, "_active_reparametrizations")


def saved_tensor_hooks_guard_scoped() -> bool:
    """True when torch scopes the saved-tensor-hooks guard to higher-order only.

    Sentinel: the old blanket ``doesnt_support_saved_tensors_hooks`` was renamed
    to ``disable_saved_tensors_hooks_for_higher_order``. When present, a single
    first-order transform already permits saved-tensor hooks (so checkpoint and
    save_on_cpu work) while higher-order differentiation still raises.
    """
    try:
        from torch._functorch import vmap
    except Exception:  # pragma: no cover - torch layout moved
        return False
    return hasattr(vmap, "disable_saved_tensors_hooks_for_higher_order")


# ---------------------------------------------------------------------------
# Per-capability patches
# ---------------------------------------------------------------------------


def _disable_saved_tensor_hooks_for_higher_order(fn):
    """Apply PyTorch's scoped hook guard to a pre-upstream transform.

    Replaces torch's blanket guard with the scoped one: hooks stay
    disabled where they would silently produce wrong gradients (under
    higher-order differentiation, and under ``torch.compile`` where
    AOTAutograd owns them) and are permitted for first-order use.
    """
    import functools

    import torch

    @functools.wraps(fn)
    def dispatch(*args, **kwargs):
        if torch.compiler.is_compiling():
            with torch.autograd.graph.disable_saved_tensors_hooks(_COMPILE_MESSAGE):
                return fn(*args, **kwargs)
        try:
            from torch._C._functorch import TransformType, get_interpreter_stack

            stack = get_interpreter_stack()
            higher_order = bool(stack) and any(
                interpreter.key() in (TransformType.Grad, TransformType.Jvp)
                for interpreter in stack
            )
        except Exception:  # pragma: no cover - private API moved/unavailable
            higher_order = True
        if higher_order:
            with torch.autograd.graph.disable_saved_tensors_hooks(
                _HIGHER_ORDER_MESSAGE
            ):
                return fn(*args, **kwargs)
        return fn(*args, **kwargs)

    return dispatch


def apply_saved_tensor_hooks_guard() -> None:
    """Scope torch's blanket saved-tensor-hooks guard to higher-order use."""
    global _SAVED_TENSOR_HOOKS_GUARD_APPLIED
    if _SAVED_TENSOR_HOOKS_GUARD_APPLIED:
        return
    import torch._functorch.eager_transforms as eager

    for name in ("grad_and_value_impl", "_vjp_with_argnums"):
        unguarded = getattr(getattr(eager, name, None), "__wrapped__", None)
        if unguarded is not None:
            setattr(
                eager, name, _disable_saved_tensor_hooks_for_higher_order(unguarded)
            )
    _SAVED_TENSOR_HOOKS_GUARD_APPLIED = True


def apply_save_on_cpu() -> None:
    """Install a vmap-safe pinned-memory ``save_on_cpu`` implementation."""
    global _SAVE_ON_CPU_APPLIED
    if _SAVE_ON_CPU_APPLIED:
        return
    import torch
    import torch.autograd.graph as autograd_graph

    orig = autograd_graph.save_on_cpu

    class _VmapSaveOnCpu(orig):
        def __init__(self, pin_memory: bool = False, device_type: str = "cuda") -> None:
            device_module = getattr(torch, device_type, torch.cuda)

            def pack_to_cpu(tensor):
                if not pin_memory:
                    return (tensor.device, tensor.cpu())
                is_pinnable = device_module.is_available() and not tensor.is_sparse
                packed = torch.empty_like(tensor, device="cpu", pin_memory=is_pinnable)
                packed.copy_(tensor, non_blocking=is_pinnable)
                return (tensor.device, packed)

            def unpack_from_cpu(packed):
                device, tensor = packed
                return tensor.to(device, non_blocking=pin_memory)

            # Skip orig.__init__ (it installs the unbatched-buffer hooks); go
            # straight to the grandparent saved_tensors_hooks.
            torch.autograd.graph.saved_tensors_hooks.__init__(
                self, pack_to_cpu, unpack_from_cpu
            )

    autograd_graph.save_on_cpu = _VmapSaveOnCpu
    _SAVE_ON_CPU_APPLIED = True


def apply_noop_save_inputs() -> None:
    """Install a vmap rule for legacy checkpoint input bookkeeping."""
    global _NOOP_SAVE_INPUTS_APPLIED
    if _NOOP_SAVE_INPUTS_APPLIED:
        return
    try:
        from torch.utils.checkpoint import _NoopSaveInputs
    except ImportError:
        return  # torch >= 2.12: symbol removed, no rule needed

    @staticmethod
    def _vmap(info, in_dims, *args):
        return _NoopSaveInputs.apply(*args), None

    _NoopSaveInputs.vmap = _vmap
    _NOOP_SAVE_INPUTS_APPLIED = True


def apply_create_graph() -> None:
    """Install the first-order checkpoint backward patch."""
    global _CREATE_GRAPH_APPLIED
    if _CREATE_GRAPH_APPLIED:
        return
    import torch._functorch.eager_transforms as eager

    orig = eager._autograd_grad

    # ``grad_and_value_impl``'s internal backward is the only caller with no
    # ``grad_outputs``; condition just that one on the grad mode captured
    # when the transform was entered, so a genuine higher-order composition
    # (grad(grad), hessian) still builds its graph.
    def _autograd_grad(
        outputs, inputs, grad_outputs=None, retain_graph=False, create_graph=True
    ):
        if create_graph and grad_outputs is None:
            create_graph = prev_grad_mode()
        return orig(
            outputs,
            inputs,
            grad_outputs,
            retain_graph=retain_graph,
            create_graph=create_graph,
        )

    eager._autograd_grad = _autograd_grad
    _CREATE_GRAPH_APPLIED = True


def _stack() -> list:
    s = getattr(_active, "stack", None)
    if s is None:
        s = _active.stack = []
    return s


def apply_reparametrize_recompute() -> None:
    """Install functional-parameter rebinding for checkpoint recomputation."""
    global _REPARAMETRIZE_RECOMPUTE_APPLIED, _orig_reparametrize
    if _REPARAMETRIZE_RECOMPUTE_APPLIED:
        return
    _wrap_reparametrize_module()
    _wrap_checkpoint_frame()
    _REPARAMETRIZE_RECOMPUTE_APPLIED = True


def _wrap_reparametrize_module() -> None:
    global _orig_reparametrize
    import torch.nn.utils.stateless as stateless

    _orig_reparametrize = stateless._reparametrize_module

    @contextlib.contextmanager
    def _reparametrize_module(
        module,
        parameters_and_buffers,
        tie_weights=False,
        strict=False,
        stack_weights=False,
    ):
        # Record the *untied* map: recomputation re-binds from this
        # snapshot with ``tie_weights=False``, so untying twice would
        # rebind the wrong names.
        params = (
            stateless._untie_named_tensors_map(module, parameters_and_buffers)
            if tie_weights
            else parameters_and_buffers
        )
        _stack().append((module, params))
        try:
            with _orig_reparametrize(
                module,
                parameters_and_buffers,
                tie_weights=tie_weights,
                strict=strict,
                stack_weights=stack_weights,
            ):
                yield
        finally:
            _stack().pop()

    stateless._reparametrize_module = _reparametrize_module


def _wrap_checkpoint_frame() -> None:
    from torch.utils.checkpoint import _CheckpointFrame

    orig_init = _CheckpointFrame.__init__

    def __init__(self, recompute_fn, *args, **kwargs):
        snapshot = list(_stack())
        if snapshot:

            def rebinding_recompute(*a, _fn=recompute_fn, _snapshot=snapshot):
                with contextlib.ExitStack() as stack:
                    for module, params in _snapshot:
                        stack.enter_context(
                            _orig_reparametrize(module, params, tie_weights=False)
                        )
                    return _fn(*a)

            recompute_fn = rebinding_recompute
        orig_init(self, recompute_fn, *args, **kwargs)

    _CheckpointFrame.__init__ = __init__


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------


def apply_checkpoint_patch(*, vmap_checkpointing: bool = True) -> None:
    """Patch PyTorch (as needed) to allow gradient checkpointing under
    ``vmap(grad(...))``. Idempotent; a no-op when ``vmap_checkpointing`` is False.
    """
    global _is_checkpoint_patched
    if _is_checkpoint_patched or not vmap_checkpointing:
        return

    # The two capability probes are independent: a torch may have the
    # param-lifetime fix without the guard scoping, or vice versa. Gate each
    # backport on its own probe rather than short-circuiting on native support.
    if not saved_tensor_hooks_guard_scoped():
        apply_saved_tensor_hooks_guard()
        apply_save_on_cpu()

    if not native_checkpoint_support():
        apply_noop_save_inputs()  # self-skips on torch >= 2.12
        apply_create_graph()
        apply_reparametrize_recompute()

    _is_checkpoint_patched = True
    logger.debug("opaque: applied checkpoint+functorch compatibility patches.")


def is_checkpoint_patched() -> bool:
    """True once :func:`apply_checkpoint_patch` has run."""
    return _is_checkpoint_patched


__all__ = [
    "apply_checkpoint_patch",
    "apply_create_graph",
    "apply_noop_save_inputs",
    "apply_reparametrize_recompute",
    "apply_save_on_cpu",
    "apply_saved_tensor_hooks_guard",
    "is_checkpoint_patched",
    "native_checkpoint_support",
    "saved_tensor_hooks_guard_scoped",
]
