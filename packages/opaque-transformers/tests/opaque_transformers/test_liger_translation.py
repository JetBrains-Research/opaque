"""Tests for HF ``liger_kernel_config`` translation.

Validates that ``opaque.api.transformers.trainer._liger.translate_liger_config``
maps Liger's per-component keys to the equivalent ``opaque-patches`` flag
names, with two non-trivial behaviors that affect HF parity:

1. Both ``cross_entropy`` and ``fused_linear_cross_entropy`` collapse to the
   single ``fuse_cross_entropy`` opaque flag (opaque auto-dispatches between
   materialize-logits and fused-linear paths internally).  When either is
   True, the unified flag is enabled.

2. Unknown keys (``layer_norm`` and any future Liger key with no opaque
   equivalent) are silently dropped — matching HF's behavior of filtering
   kwargs by signature inside ``_apply_liger_kernel_to_instance``.

Related Trainer-level wiring is covered by ``test_kernel_patches.py``.
"""

from __future__ import annotations

import logging


from opaque.api.transformers.trainer._liger import translate_liger_config


def test_empty_config_yields_empty_dict():
    assert translate_liger_config(None) == {}
    assert translate_liger_config({}) == {}


def test_llama_style_keys_translate():
    """Standard Llama Liger config: rope/rms_norm/swiglu/cross_entropy."""
    out = translate_liger_config(
        {
            "rope": True,
            "rms_norm": True,
            "swiglu": True,
            "cross_entropy": True,
        }
    )
    assert out == {
        "fuse_rope": True,
        "fuse_rms_norm": True,
        "fuse_swiglu": True,
        "fuse_cross_entropy": True,
    }


def test_gemma_style_keys_geglu_maps_to_fuse_swiglu():
    """Gemma family uses GeGLU; opaque's Gemma patch reads fuse_swiglu but
    dispatches to the GeGLU kernel.  The Liger-naming-alignment PR will
    split these; until then geglu→fuse_swiglu is correct."""
    out = translate_liger_config({"rope": True, "geglu": True, "rms_norm": True})
    assert out == {
        "fuse_rope": True,
        "fuse_swiglu": True,
        "fuse_rms_norm": True,
    }


def test_cross_entropy_and_fused_linear_collapse_with_or_semantics():
    """Both Liger CE flags target the same opaque flag.  OR semantics: if
    either is True, the unified opaque flag is enabled."""
    assert translate_liger_config(
        {"cross_entropy": True, "fused_linear_cross_entropy": False}
    ) == {"fuse_cross_entropy": True}
    assert translate_liger_config(
        {"cross_entropy": False, "fused_linear_cross_entropy": True}
    ) == {"fuse_cross_entropy": True}
    # Both False → False.
    assert translate_liger_config(
        {"cross_entropy": False, "fused_linear_cross_entropy": False}
    ) == {"fuse_cross_entropy": False}
    # Both True → True (no double-on).
    assert translate_liger_config(
        {"cross_entropy": True, "fused_linear_cross_entropy": True}
    ) == {"fuse_cross_entropy": True}


def test_unknown_keys_are_dropped_with_info_log(caplog):
    """layer_norm has no opaque kernel; future Liger keys without opaque
    equivalents must be silently dropped (HF parity: kwargs filtered by
    signature)."""
    with caplog.at_level(logging.INFO, logger="opaque.api.transformers.trainer._liger"):
        out = translate_liger_config(
            {"rope": True, "layer_norm": True, "flash_attn": True}
        )
    assert out == {"fuse_rope": True}
    # Both unknown keys reported, but only as INFO (not error).
    messages = [rec.message for rec in caplog.records if rec.levelno == logging.INFO]
    assert any("layer_norm" in m for m in messages)
    assert any("flash_attn" in m for m in messages)


def test_falsy_values_pass_through_as_false():
    """A user might pass ``rope=False`` to disable a kernel; that must
    translate as a False flag, not be dropped."""
    assert translate_liger_config({"rope": False, "rms_norm": False}) == {
        "fuse_rope": False,
        "fuse_rms_norm": False,
    }


def test_truthy_non_bool_values_become_true():
    """HF's filter accepts truthy values; we coerce to bool."""
    assert translate_liger_config({"rope": 1, "rms_norm": "yes"}) == {
        "fuse_rope": True,
        "fuse_rms_norm": True,
    }
