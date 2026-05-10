"""Fused Triton kernels — SwiGLU, GeGLU, RoPE, RMSNorm, fused CE, LoRA."""

from opaque.api.patches.kernels import (
    ACTIVATION_GEGLU_APPROX,
    ACTIVATION_GEGLU_EXACT,
    ACTIVATION_SWIGLU,
    opaque_cross_entropy_loss,
    opaque_fused_add_rms_norm,
    opaque_geglu_approx,
    opaque_geglu_exact,
    opaque_linear_cross_entropy_loss,
    opaque_lora_mlp,
    opaque_lora_qkv,
    opaque_lora_w,
    opaque_rms_norm,
    opaque_rope,
    opaque_rope_qk,
    opaque_slow_rope,
    opaque_swiglu,
)

__all__ = [
    "opaque_cross_entropy_loss",
    "opaque_linear_cross_entropy_loss",
    "opaque_swiglu",
    "opaque_geglu_exact",
    "opaque_geglu_approx",
    "opaque_rms_norm",
    "opaque_fused_add_rms_norm",
    "opaque_rope",
    "opaque_rope_qk",
    "opaque_slow_rope",
    "opaque_lora_w",
    "opaque_lora_qkv",
    "opaque_lora_mlp",
    "ACTIVATION_SWIGLU",
    "ACTIVATION_GEGLU_EXACT",
    "ACTIVATION_GEGLU_APPROX",
]
