# Copyright (c) 2025 Opaque Authors
# SPDX-License-Identifier: Apache-2.0
"""Automatic Triton kernel patches for HuggingFace Transformers models.

Replaces model components with Opaque's vmap-compatible Triton kernels at class
level. Applied at `import opaque` time when CUDA and Triton are available.

Patched components:
- RMSNorm: Standard (LLaMA, Mistral, Qwen2, Phi3) and Gemma (weight+1 trick)
- MLP activations: SwiGLU (LLaMA, Mistral, Qwen2, Phi3) and GeGLU (Gemma, Gemma2)
- RoPE: apply_rotary_pos_emb for all supported models
- Cross-entropy loss: ForCausalLM loss via LOSS_MAPPING
- LoRA: peft.tuners.lora.Linear forward

Disable with: OPAQUE_NO_KERNEL_PATCH=1
"""

from __future__ import annotations

import importlib
import logging
import os

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

# Track patching state
_is_kernel_patched = False
_disabled = os.environ.get("OPAQUE_NO_KERNEL_PATCH", "0") == "1"


# =============================================================================
# Patch targets
# =============================================================================

# Standard RMSNorm: self.weight, self.variance_epsilon
_STANDARD_RMSNORM = [
    ("transformers.models.llama.modeling_llama", "LlamaRMSNorm"),
    ("transformers.models.mistral.modeling_mistral", "MistralRMSNorm"),
    ("transformers.models.qwen2.modeling_qwen2", "Qwen2RMSNorm"),
    ("transformers.models.phi3.modeling_phi3", "Phi3RMSNorm"),
]

# Gemma RMSNorm: self.eps, effective_weight = 1.0 + self.weight
_GEMMA_RMSNORM = [
    ("transformers.models.gemma.modeling_gemma", "GemmaRMSNorm"),
    ("transformers.models.gemma2.modeling_gemma2", "Gemma2RMSNorm"),
]

# SwiGLU MLP: separate gate_proj, up_proj, down_proj
_SWIGLU_MLP = [
    ("transformers.models.llama.modeling_llama", "LlamaMLP"),
    ("transformers.models.mistral.modeling_mistral", "MistralMLP"),
    ("transformers.models.qwen2.modeling_qwen2", "Qwen2MLP"),
]

# Phi3 MLP: combined gate_up_proj, needs chunk(2)
_PHI3_MLP = [
    ("transformers.models.phi3.modeling_phi3", "Phi3MLP"),
]

# GeGLU exact MLP (Gemma): gelu activation
_GEGLU_EXACT_MLP = [
    ("transformers.models.gemma.modeling_gemma", "GemmaMLP"),
]

# GeGLU approx MLP (Gemma2): gelu_pytorch_tanh activation
_GEGLU_APPROX_MLP = [
    ("transformers.models.gemma2.modeling_gemma2", "Gemma2MLP"),
]

# RoPE: module-level apply_rotary_pos_emb function
_ROPE_MODELS = [
    "transformers.models.llama.modeling_llama",
    "transformers.models.mistral.modeling_mistral",
    "transformers.models.qwen2.modeling_qwen2",
    "transformers.models.phi3.modeling_phi3",
    "transformers.models.gemma.modeling_gemma",
    "transformers.models.gemma2.modeling_gemma2",
]


# =============================================================================
# Replacement forward methods
# =============================================================================

def _opaque_rmsnorm_forward(self, hidden_states):
    """Standard RMSNorm forward using Opaque Triton kernel."""
    from opaque.kernels import Opaque_RMSNorm

    result = Opaque_RMSNorm.apply(hidden_states, self.weight, self.variance_epsilon)
    return result[0] if isinstance(result, tuple) else result


def _opaque_gemma_rmsnorm_forward(self, hidden_states):
    """Gemma RMSNorm forward using Opaque Triton kernel (weight+1 trick)."""
    from opaque.kernels import Opaque_RMSNorm

    effective_weight = (1.0 + self.weight).float()
    result = Opaque_RMSNorm.apply(hidden_states, effective_weight, self.eps)
    output = result[0] if isinstance(result, tuple) else result
    return output.type_as(hidden_states)


def _opaque_swiglu_mlp_forward(self, x):
    """SwiGLU MLP forward using Opaque Triton kernel."""
    from opaque.kernels import Opaque_SwiGLU

    return self.down_proj(Opaque_SwiGLU.apply(self.gate_proj(x), self.up_proj(x)))


def _opaque_phi3_mlp_forward(self, hidden_states):
    """Phi3 MLP forward (combined gate_up_proj) using Opaque Triton kernel."""
    from opaque.kernels import Opaque_SwiGLU

    gate, up = self.gate_up_proj(hidden_states).chunk(2, dim=-1)
    return self.down_proj(Opaque_SwiGLU.apply(gate, up))


def _opaque_geglu_exact_mlp_forward(self, x):
    """Gemma MLP forward using Opaque GeGLU exact kernel."""
    from opaque.kernels import Opaque_GeGLU_Exact

    return self.down_proj(Opaque_GeGLU_Exact.apply(self.gate_proj(x), self.up_proj(x)))


def _opaque_geglu_approx_mlp_forward(self, x):
    """Gemma2 MLP forward using Opaque GeGLU approx kernel."""
    from opaque.kernels import Opaque_GeGLU_Approx

    return self.down_proj(Opaque_GeGLU_Approx.apply(self.gate_proj(x), self.up_proj(x)))


def _rotate_half(x):
    """Rotates half the hidden dims of the input (standard HF rotate_half)."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def _opaque_apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    """RoPE using Opaque Triton kernel.

    Replaces HF's apply_rotary_pos_emb at module level. Uses Opaque_RoPE_QK
    which processes Q and K together with GQA support.

    Falls back to PyTorch when cos/sin cannot be reduced to 2D (e.g., when
    position_ids create truly per-batch-element cos/sin).
    """
    from opaque.kernels import Opaque_RoPE_QK

    # HF provides cos/sin as (batch, seq_len, head_dim) or (seq_len, head_dim).
    # The kernel needs 2D (seq_len, head_dim) after squeeze.
    cos_2d = cos.unsqueeze(unsqueeze_dim).squeeze()
    sin_2d = sin.unsqueeze(unsqueeze_dim).squeeze()

    if cos_2d.dim() == 2:
        return Opaque_RoPE_QK.apply(q, k, cos_2d, sin_2d, None)

    # Fallback to PyTorch for batched cos/sin (e.g., variable position_ids)
    cos_u = cos.unsqueeze(unsqueeze_dim)
    sin_u = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos_u) + (_rotate_half(q) * sin_u)
    k_embed = (k * cos_u) + (_rotate_half(k) * sin_u)
    return q_embed, k_embed


def _opaque_causal_lm_loss(
    logits,
    labels,
    vocab_size: int,
    num_items_in_batch=None,
    ignore_index: int = -100,
    shift_labels=None,
    **kwargs,
) -> torch.Tensor:
    """CausalLM loss using Opaque cross-entropy Triton kernel.

    Supports all vocab sizes via chunked computation for vocab > 65536.
    """
    from opaque.kernels import Opaque_CrossEntropy

    logits = logits.float()

    if shift_labels is None:
        # Shift so that tokens < n predict n (same as HF ForCausalLMLoss)
        labels = nn.functional.pad(labels, (0, 1), value=ignore_index)
        shift_labels = labels[..., 1:].contiguous()

    logits_flat = logits.view(-1, vocab_size)
    shift_labels_flat = shift_labels.view(-1)
    shift_labels_flat = shift_labels_flat.to(logits_flat.device)

    losses, _ = Opaque_CrossEntropy.apply(logits_flat, shift_labels_flat)

    # Mask out ignored positions so they get zero upstream gradient
    mask = shift_labels_flat != ignore_index
    masked_losses = losses * mask.float()

    # Match HF fixed_cross_entropy reduction behavior
    if num_items_in_batch is not None:
        if torch.is_tensor(num_items_in_batch):
            num_items_in_batch = num_items_in_batch.to(losses.device)
        return masked_losses.sum() / num_items_in_batch
    else:
        return masked_losses.sum() / mask.sum().clamp(min=1)


def _opaque_lora_linear_forward(self, x, *args, **kwargs):
    """LoRA linear forward using Opaque kernel (vmap-compatible).

    Replaces peft.tuners.lora.Linear.forward. Uses Opaque_LoRA_W which
    computes base projection + LoRA delta in a single call.
    """
    from opaque.kernels import Opaque_LoRA_W

    if self.disable_adapters or not self.active_adapters:
        return self.base_layer(x)

    active = self.active_adapters[0]
    if active not in self.lora_A:
        return self.base_layer(x)

    dropout = self.lora_dropout[active]
    x_input = dropout(x)

    W = self.base_layer.weight
    # PEFT stores lora_A as (rank, in_features), kernel expects (in_features, rank)
    # Cast to input dtype for mixed precision compatibility
    A = self.lora_A[active].weight.T.to(x_input.dtype)
    # PEFT stores lora_B as (out_features, rank), kernel expects (rank, out_features)
    B = self.lora_B[active].weight.T.to(x_input.dtype)
    scaling = self.scaling[active]

    result = Opaque_LoRA_W.apply(x_input, W, A, B, scaling)

    # Add base layer bias if present (kernel does F.linear without bias)
    if self.base_layer.bias is not None:
        result = result + self.base_layer.bias

    # Handle additional active adapters (rare, but support it)
    for adapter in self.active_adapters[1:]:
        if adapter in self.lora_A:
            dropout_i = self.lora_dropout[adapter]
            x_i = dropout_i(x)
            A_i = self.lora_A[adapter].weight.T
            B_i = self.lora_B[adapter].weight.T
            scaling_i = self.scaling[adapter]
            lora_out = (x_i @ A_i) @ B_i * scaling_i
            result = result + lora_out

    return result


# =============================================================================
# Patching helpers
# =============================================================================

def _patch_forward(module_path: str, class_name: str, new_forward) -> bool:
    """Patch a class's forward method if the module is available."""
    try:
        module = importlib.import_module(module_path)
        cls = getattr(module, class_name, None)
        if cls is None:
            return False

        # Store original forward for potential restoration
        if not hasattr(cls, "_opaque_original_forward"):
            cls._opaque_original_forward = cls.forward

        cls.forward = new_forward
        return True

    except (ImportError, RuntimeError):
        return False


def _patch_rope_functions(patched: list) -> None:
    """Patch apply_rotary_pos_emb at module level for each model."""
    for module_path in _ROPE_MODELS:
        try:
            module = importlib.import_module(module_path)
            if hasattr(module, "apply_rotary_pos_emb"):
                if not hasattr(module, "_opaque_original_apply_rotary_pos_emb"):
                    module._opaque_original_apply_rotary_pos_emb = module.apply_rotary_pos_emb
                module.apply_rotary_pos_emb = _opaque_apply_rotary_pos_emb
                patched.append(f"{module_path.split('.')[-1]}.apply_rotary_pos_emb")
        except (ImportError, RuntimeError):
            pass


def _patch_cross_entropy_loss(patched: list) -> None:
    """Patch LOSS_MAPPING to use Opaque cross-entropy kernel."""
    try:
        from transformers.loss.loss_utils import LOSS_MAPPING

        for key in ("ForCausalLM", "ForConditionalGeneration"):
            if key in LOSS_MAPPING:
                LOSS_MAPPING[key] = _opaque_causal_lm_loss
                patched.append(f"LOSS_MAPPING[{key}]")

    except (ImportError, RuntimeError):
        pass


def _patch_lora_forward(patched: list) -> None:
    """Patch peft LoRA Linear forward with Opaque kernel."""
    try:
        from peft.tuners.lora import Linear as PeftLoRALinear

        if not hasattr(PeftLoRALinear, "_opaque_original_forward"):
            PeftLoRALinear._opaque_original_forward = PeftLoRALinear.forward

        PeftLoRALinear.forward = _opaque_lora_linear_forward
        patched.append("peft.LoRA.Linear")

    except (ImportError, RuntimeError):
        pass


# =============================================================================
# Public API
# =============================================================================

def apply_kernel_patches() -> None:
    """Replace HF model components with Opaque Triton kernels.

    Patches at class/module level for:
    - RMSNorm: LLaMA, Mistral, Qwen2, Phi3, Gemma, Gemma2
    - MLP activations: SwiGLU and GeGLU variants
    - RoPE: apply_rotary_pos_emb for all supported models
    - Cross-entropy loss: ForCausalLM via LOSS_MAPPING
    - LoRA: peft.tuners.lora.Linear forward

    No-op when CUDA/Triton unavailable or OPAQUE_NO_KERNEL_PATCH=1.
    """
    global _is_kernel_patched

    if _is_kernel_patched:
        return

    if _disabled or not torch.cuda.is_available():
        _is_kernel_patched = True
        return

    try:
        import triton  # noqa: F401
    except ImportError:
        _is_kernel_patched = True
        return

    patched = []

    # Standard RMSNorm
    for path, cls_name in _STANDARD_RMSNORM:
        if _patch_forward(path, cls_name, _opaque_rmsnorm_forward):
            patched.append(cls_name)

    # Gemma RMSNorm (weight+1)
    for path, cls_name in _GEMMA_RMSNORM:
        if _patch_forward(path, cls_name, _opaque_gemma_rmsnorm_forward):
            patched.append(cls_name)

    # SwiGLU MLP
    for path, cls_name in _SWIGLU_MLP:
        if _patch_forward(path, cls_name, _opaque_swiglu_mlp_forward):
            patched.append(cls_name)

    # Phi3 MLP (combined gate_up_proj)
    for path, cls_name in _PHI3_MLP:
        if _patch_forward(path, cls_name, _opaque_phi3_mlp_forward):
            patched.append(cls_name)

    # GeGLU exact MLP (Gemma)
    for path, cls_name in _GEGLU_EXACT_MLP:
        if _patch_forward(path, cls_name, _opaque_geglu_exact_mlp_forward):
            patched.append(cls_name)

    # GeGLU approx MLP (Gemma2)
    for path, cls_name in _GEGLU_APPROX_MLP:
        if _patch_forward(path, cls_name, _opaque_geglu_approx_mlp_forward):
            patched.append(cls_name)

    # RoPE
    _patch_rope_functions(patched)

    # Cross-entropy loss
    _patch_cross_entropy_loss(patched)

    # LoRA
    _patch_lora_forward(patched)

    if patched:
        logger.debug(f"opaque: Applied Triton kernel patches to: {', '.join(patched)}")

    _is_kernel_patched = True


def is_kernel_patched() -> bool:
    """Check if Triton kernel patches have been applied."""
    return _is_kernel_patched
