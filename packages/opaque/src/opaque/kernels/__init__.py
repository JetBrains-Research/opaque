"""
Opaque Kernels: vmap-compatible Triton kernels for DP-SGD training.

All kernels use new-style PyTorch autograd.Function API with custom vmap rules.
"""

# Normalization layers
from .layernorm import NewStyleLayerNorm, layernorm_vmap
from .rms_layernorm import RMSLayerNorm, rms_layernorm

# Loss functions
from .cross_entropy import NewStyleCrossEntropy, cross_entropy_vmap

# Activation functions
from .swiglu import (
    NewStyleSwiGLU,
    swiglu_vmap,
    triton_swiglu_forward,
    triton_swiglu_backward,
)
from .geglu import (
    NewStyleGeGLUExact,
    NewStyleGeGLUApprox,
    geglu_exact_vmap,
    geglu_approx_vmap,
    triton_geglu_exact_forward,
    triton_geglu_exact_backward,
    triton_geglu_approx_forward,
    triton_geglu_approx_backward,
)

# Position embeddings
from .rope_embedding import (
    NewStyleRoPEEmbedding,
    NewStyleRoPEEmbeddingQK,
    NewStyleSlowRoPEEmbedding,
    rope_embedding_vmap,
    rope_embedding_qk_vmap,
    slow_rope_embedding_vmap,
)

# LoRA kernels
from .lora import (
    NewStyleLoRAW,
    NewStyleLoRAQKV,
    NewStyleLoRAMLP,
    lora_linear_vmap,
    lora_qkv_vmap,
    lora_mlp_vmap,
    ACTIVATION_SWIGLU,
    ACTIVATION_GEGLU_EXACT,
    ACTIVATION_GEGLU_APPROX,
)

__all__ = [
    # Normalization
    "NewStyleLayerNorm",
    "layernorm_vmap",
    "RMSLayerNorm",
    "rms_layernorm",
    # Loss
    "NewStyleCrossEntropy",
    "cross_entropy_vmap",
    # Activations
    "NewStyleSwiGLU",
    "swiglu_vmap",
    "triton_swiglu_forward",
    "triton_swiglu_backward",
    "NewStyleGeGLUExact",
    "NewStyleGeGLUApprox",
    "geglu_exact_vmap",
    "geglu_approx_vmap",
    "triton_geglu_exact_forward",
    "triton_geglu_exact_backward",
    "triton_geglu_approx_forward",
    "triton_geglu_approx_backward",
    # Position embeddings
    "NewStyleRoPEEmbedding",
    "NewStyleRoPEEmbeddingQK",
    "NewStyleSlowRoPEEmbedding",
    "rope_embedding_vmap",
    "rope_embedding_qk_vmap",
    "slow_rope_embedding_vmap",
    # LoRA
    "NewStyleLoRAW",
    "NewStyleLoRAQKV",
    "NewStyleLoRAMLP",
    "lora_linear_vmap",
    "lora_qkv_vmap",
    "lora_mlp_vmap",
    "ACTIVATION_SWIGLU",
    "ACTIVATION_GEGLU_EXACT",
    "ACTIVATION_GEGLU_APPROX",
]
