"""Fused Triton kernels — SwiGLU, GeGLU, RoPE, RMSNorm, fused CE, MoE, LoRA.

The ``ACTIVATION_*`` selector constants ``opaque_lora_mlp`` dispatches on
live in :mod:`opaque.kernels.types`.
"""

from opaque.api.kernels import (
    opaque_cross_entropy_loss,
    opaque_fused_add_rms_norm,
    opaque_geglu_approx,
    opaque_geglu_exact,
    opaque_linear_cross_entropy_loss,
    opaque_lora_mlp,
    opaque_lora_qkv,
    opaque_lora_w,
    opaque_moe,
    opaque_rms_norm,
    opaque_rope,
    opaque_rope_qk,
    opaque_slow_rope,
    opaque_swiglu,
)
from opaque.kernels import types

__all__ = [
    "opaque_cross_entropy_loss",
    "opaque_fused_add_rms_norm",
    "opaque_geglu_approx",
    "opaque_geglu_exact",
    "opaque_linear_cross_entropy_loss",
    "opaque_lora_mlp",
    "opaque_lora_qkv",
    "opaque_lora_w",
    "opaque_moe",
    "opaque_rms_norm",
    "opaque_rope",
    "opaque_rope_qk",
    "opaque_slow_rope",
    "opaque_swiglu",
    "types",
]
