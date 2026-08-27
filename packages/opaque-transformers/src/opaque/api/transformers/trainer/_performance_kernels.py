"""``use_performance_kernels`` / ``performance_kernels_config`` integration.

DPTrainer drives the model patches via two split umbrellas:

* ``use_compat_patches`` → ``compat`` (vmap-safety: ``eager_attention``,
  ``batchify``, vmap-safe masking / collator / checkpoint hooks).
* ``use_performance_kernels`` → ``kernels`` (CUDA + Triton kernel group:
  ``rope``, ``rms_norm``, ``activation``, ``cross_entropy``, and the
  opt-in ``fused_linear_cross_entropy``).

The ``performance`` bucket — currently only ``kv_cache`` — stays on by
default regardless of ``use_performance_kernels``: ``kv_cache`` is a pure
Python patch that disables HF's ``DynamicCache`` allocation, which
otherwise leaks vmap references and inflates training memory.  Users who
want to keep the cache (e.g. for an HF model whose forward depends on
it) opt out explicitly via ``performance_kernels_config={"kv_cache":
False}``.

``performance_kernels_config`` is a flat ``dict[str, bool]`` forwarded
as-is to ``opaque.transformers.patches.apply_model_patches`` kwargs — no key
translation.  Supported keys mirror the patch surface:
``rope``, ``rms_norm``, ``activation``, ``cross_entropy``,
``fused_linear_cross_entropy``, ``kv_cache``, ``eager_attention``,
``batchify``.
"""

from __future__ import annotations

from typing import Any


def apply_performance_kernels(
    model: Any,
    kernel_config: dict[str, Any] | None = None,
) -> None:
    """Apply the fused-kernel model patches with a flat opaque-shaped config.

    Mutates ``model`` in place.  When ``kernel_config`` is ``None`` every
    supported kernel for the model family is enabled (full performance
    set) while compat wrappers stay on.

    **DPTrainer** applies the same stack internally — callers rarely
    need this function directly.
    """
    from opaque.transformers.patches import apply_model_patches

    apply_model_patches(
        model,
        performance=True,
        compat=True,
        kernels=True,
        **(kernel_config or {}),
    )


__all__ = [
    "apply_performance_kernels",
]
