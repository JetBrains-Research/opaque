"""``use_performance_kernels`` / ``performance_kernels_config`` integration.

When users set ``use_performance_kernels=True`` we apply opaque's
vmap-safe high-performance kernels via ``opaque-patches``. The flag
shape (per-component booleans like ``rope``, ``rms_norm``, ``swiglu``,
``cross_entropy``, …) mirrors what HF accepts in
``transformers.integrations.liger.apply_liger_kernel`` so configs port
1:1, but the kernel implementations are ours (vmap-compatible
reimplementations, not a Liger dependency).

Opaque ships the same fusion concepts under different
flag names (``fuse_rope``, ``fuse_rms_norm``, ``fuse_swiglu``,
``fuse_cross_entropy``).  This module translates HF-shaped keys to
opaque's flag names so HF-style configs are accepted unchanged.

The translation table below is **temporary** — a follow-up PR will
rename opaque's per-component flags to match the HF naming directly,
at which point this becomes a pass-through.

Notes on the mapping
--------------------
* Both ``cross_entropy`` and ``fused_linear_cross_entropy`` map to the
  same opaque flag (``fuse_cross_entropy``); opaque's CE patch
  auto-dispatches between materialize-logits and fused-linear paths
  based on dtype/device at call time.  Setting either Liger key is
  treated as a request to enable the unified opaque flag.
* ``geglu`` maps to ``fuse_swiglu`` for now — Gemma family already
  reads ``fuse_swiglu`` and dispatches to the GeGLU kernel internally.
  The Liger-naming-alignment PR will split this.
* ``layer_norm`` has no opaque equivalent (no LayerNorm Triton kernel).
  The mapping drops the key with an info log.
"""

from __future__ import annotations

import logging
from typing import Any


log = logging.getLogger(__name__)

_PERF_KERNEL_KEY_TO_OPAQUE: dict[str, str] = {
    "rope": "fuse_rope",
    "rms_norm": "fuse_rms_norm",
    "swiglu": "fuse_swiglu",
    "cross_entropy": "fuse_cross_entropy",
    "fused_linear_cross_entropy": "fuse_cross_entropy",
    "geglu": "fuse_swiglu",
}


def translate_performance_kernels_config(cfg: dict[str, Any] | None) -> dict[str, bool]:
    """Translate a Liger ``kernel_config`` dict to opaque-patches kwargs.

    Unknown keys are dropped with an INFO log; mapping collisions
    (``cross_entropy`` and ``fused_linear_cross_entropy`` both targeting
    ``fuse_cross_entropy``) are resolved by OR-ing — if either is True,
    the unified opaque flag is enabled.
    """
    out: dict[str, bool] = {}
    if not cfg:
        return out
    for key, value in cfg.items():
        opaque_key = _PERF_KERNEL_KEY_TO_OPAQUE.get(key)
        if opaque_key is None:
            log.info(
                "performance_kernels_config: %r has no opaque-patches equivalent; ignored.",
                key,
            )
            continue
        out[opaque_key] = out.get(opaque_key, False) or bool(value)
    return out


def apply_performance_kernels_via_opaque_patches(
    model: Any,
    kernel_config: dict[str, Any] | None = None,
) -> None:
    """Apply opaque-patches kernels using HF-shaped configuration.

    Invoked once at ``__init__``-time when ``use_performance_kernels=True``;
    mutates ``model`` in place. Config keys match the HF
    :func:`transformers.integrations.liger.apply_liger_kernel` surface for
    config portability.

    When ``kernel_config`` is ``None`` (HF default), this applies every
    supported kernel flag for the model family (full performance set) while
    keeping compat wrappers on.

    **DPTrainer** applies the same stack internally: compat-only by default,
    and performance + translated ``performance_kernels_config`` when
    ``use_performance_kernels=True`` — callers rarely need this function directly.
    """
    from opaque.patches import apply_model_patches

    apply_model_patches(
        model,
        performance=True,
        compat=True,
        **translate_performance_kernels_config(kernel_config),
    )


__all__ = [
    "apply_performance_kernels_via_opaque_patches",
    "translate_performance_kernels_config",
]
