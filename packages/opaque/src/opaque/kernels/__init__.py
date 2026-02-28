"""
Opaque Kernels: vmap-compatible Triton kernels for DP-SGD training.

All kernels use new-style PyTorch autograd.Function API with custom vmap rules.
"""

# Loss functions
from .cross_entropy import Opaque_CrossEntropyLoss, opaque_cross_entropy_loss
from .linear_cross_entropy import (
    Opaque_LinearCrossEntropyLoss,
    opaque_linear_cross_entropy_loss,
)

# Activation functions
from .swiglu import Opaque_SwiGLU, opaque_swiglu
from .geglu import (
    Opaque_GeGLU_Exact,
    Opaque_GeGLU_Approx,
    opaque_geglu_exact,
    opaque_geglu_approx,
)

# Position embeddings
from .rope_embedding import (
    Opaque_RoPE,
    Opaque_RoPE_QK,
    Opaque_SlowRoPE,
    opaque_rope,
    opaque_rope_qk,
    opaque_slow_rope,
)

# LoRA kernels
from .lora import (
    Opaque_LoRA_W,
    Opaque_LoRA_QKV,
    Opaque_LoRA_MLP,
    opaque_lora_w,
    opaque_lora_qkv,
    opaque_lora_mlp,
    ACTIVATION_SWIGLU,
    ACTIVATION_GEGLU_EXACT,
    ACTIVATION_GEGLU_APPROX,
)

__all__ = [
    # Loss
    "Opaque_CrossEntropyLoss",
    "opaque_cross_entropy_loss",
    "Opaque_LinearCrossEntropyLoss",
    "opaque_linear_cross_entropy_loss",
    # Activations
    "Opaque_SwiGLU",
    "opaque_swiglu",
    "Opaque_GeGLU_Exact",
    "Opaque_GeGLU_Approx",
    "opaque_geglu_exact",
    "opaque_geglu_approx",
    # Position embeddings
    "Opaque_RoPE",
    "Opaque_RoPE_QK",
    "Opaque_SlowRoPE",
    "opaque_rope",
    "opaque_rope_qk",
    "opaque_slow_rope",
    # LoRA
    "Opaque_LoRA_W",
    "Opaque_LoRA_QKV",
    "Opaque_LoRA_MLP",
    "opaque_lora_w",
    "opaque_lora_qkv",
    "opaque_lora_mlp",
    "ACTIVATION_SWIGLU",
    "ACTIVATION_GEGLU_EXACT",
    "ACTIVATION_GEGLU_APPROX",
]
