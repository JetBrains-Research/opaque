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
from .swiglu import NewStyleSwiGLU, swiglu_vmap
from .geglu import (
    NewStyleGeGLUExact,
    NewStyleGeGLUApprox,
    geglu_exact_vmap,
    geglu_approx_vmap,
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
    "NewStyleGeGLUExact",
    "NewStyleGeGLUApprox",
    "geglu_exact_vmap",
    "geglu_approx_vmap",
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
]
