"""Torch gradient-checkpointing compatibility surface.

Public entry points for the vmap-safe checkpoint patches the Torch provider
carries: the ``apply_*`` installers rebind the corresponding
``torch.utils.checkpoint`` / ``torch.func`` internals (idempotently), and the
probes report what the running torch already supports. ``opaque-patches``
composes these into :func:`opaque.patches.apply_runtime_patches`; they are
exposed here so integrations can install a single patch without pulling in
the whole patch set.
"""

from opaque.api.torch.backend._checkpoint_compat import (
    apply_create_graph,
    apply_noop_save_inputs,
    apply_reparametrize_recompute,
    apply_save_on_cpu,
    apply_saved_tensor_hooks_guard,
    native_checkpoint_support,
    saved_tensor_hooks_guard_scoped,
)

__all__ = [
    "apply_create_graph",
    "apply_noop_save_inputs",
    "apply_reparametrize_recompute",
    "apply_save_on_cpu",
    "apply_saved_tensor_hooks_guard",
    "native_checkpoint_support",
    "saved_tensor_hooks_guard_scoped",
]
