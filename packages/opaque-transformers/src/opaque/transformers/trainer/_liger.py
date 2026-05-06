"""HF ``use_liger_kernel`` / ``liger_kernel_config`` integration.

HF's ``Trainer.use_liger_kernel`` calls ``apply_liger_kernel(model, kernel_config)``
which dispatches to per-model functions accepting per-component booleans
(``rope``, ``rms_norm``, ``swiglu``, ``cross_entropy``,
``fused_linear_cross_entropy``, ``geglu``, ``layer_norm``, …).

Opaque ships the same Triton kernels in ``opaque-patches`` under different
flag names (``fuse_rope``, ``fuse_rms_norm``, ``fuse_swiglu``,
``fuse_cross_entropy``).  This module translates the Liger key set to
opaque's, keeping ``use_liger_kernel=True`` a drop-in flag for HF users.

The translation table below is **temporary** — a follow-up PR will
rename opaque's per-component flags to match Liger directly, at which
point this becomes a pass-through.

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

_LIGER_KEY_TO_OPAQUE: dict[str, str] = {
    "rope": "fuse_rope",
    "rms_norm": "fuse_rms_norm",
    "swiglu": "fuse_swiglu",
    "cross_entropy": "fuse_cross_entropy",
    "fused_linear_cross_entropy": "fuse_cross_entropy",
    "geglu": "fuse_swiglu",
}


def translate_liger_config(cfg: dict[str, Any] | None) -> dict[str, bool]:
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
        opaque_key = _LIGER_KEY_TO_OPAQUE.get(key)
        if opaque_key is None:
            log.info(
                "liger_kernel_config: %r has no opaque-patches equivalent; ignored.",
                key,
            )
            continue
        out[opaque_key] = out.get(opaque_key, False) or bool(value)
    return out


def apply_liger_kernel_via_opaque_patches(
    model: Any,
    kernel_config: dict[str, Any] | None = None,
) -> None:
    """Apply opaque-patches kernels using Liger-style configuration.

    Mirrors HF's :func:`transformers.integrations.liger.apply_liger_kernel`
    contract: invoked once at ``__init__``-time when ``use_liger_kernel=True``,
    mutates ``model`` in place.

    When ``kernel_config`` is ``None`` (HF default), this falls back to
    ``apply_model_patches(model, performance=True, compat=True)`` — i.e.
    the full opaque-patches default set (every supported kernel for the
    model's family).
    """
    from opaque.patches import apply_model_patches

    apply_model_patches(
        model,
        performance=True,
        compat=True,
        **translate_liger_config(kernel_config),
    )


__all__ = [
    "apply_liger_kernel_via_opaque_patches",
    "translate_liger_config",
]
