# Copyright (c) 2025 Opaque Authors
# SPDX-License-Identifier: Apache-2.0
"""Automatic Triton kernel patches for HuggingFace Transformers models.

Replaces model components with Opaque's vmap-compatible Triton kernels at class
level. Applied at `import opaque` time when CUDA and Triton are available.

Patched components:
- MLP activations: SwiGLU (LLaMA, Mistral, Qwen2, Qwen3, Phi3, Granite, Cohere, Cohere2) and GeGLU (Gemma, Gemma2)
- RoPE: apply_rotary_pos_emb for all supported models (standard half-split rotation)
- Cross-entropy loss: ForCausalLM loss via LOSS_MAPPING (fp32 fallback)
- Fused linear + CE: ForCausalLM.forward replaced to skip lm_head materialization (bf16/fp16)
- LoRA: peft.tuners.lora.Linear forward + auto-fused QKV (Opaque_LoRA_QKV) and MLP (Opaque_LoRA_MLP) via get_peft_model hook

Disable all with: OPAQUE_SKIP_TRANSFORMERS_KERNEL_PATCHES=all
Skip specific kernels: OPAQUE_SKIP_TRANSFORMERS_KERNEL_PATCHES=swiglu,rope,ce,fused_ce,lora
"""

from __future__ import annotations

import importlib
import logging
import sys
import types

import torch
import torch.nn as nn

from opaque._env import parse_skip_env

logger = logging.getLogger(__name__)

# Track patching state
_is_kernel_patched = False


# =============================================================================
# Patch targets
# =============================================================================

# SwiGLU MLP: separate gate_proj, up_proj, down_proj
_SWIGLU_MLP = [
    ("transformers.models.llama.modeling_llama", "LlamaMLP"),
    ("transformers.models.mistral.modeling_mistral", "MistralMLP"),
    ("transformers.models.qwen2.modeling_qwen2", "Qwen2MLP"),
    ("transformers.models.qwen3.modeling_qwen3", "Qwen3MLP"),
    ("transformers.models.granite.modeling_granite", "GraniteMLP"),
    ("transformers.models.cohere.modeling_cohere", "CohereMLP"),
    ("transformers.models.cohere2.modeling_cohere2", "Cohere2MLP"),
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
    "transformers.models.qwen3.modeling_qwen3",
    "transformers.models.phi3.modeling_phi3",
    "transformers.models.gemma.modeling_gemma",
    "transformers.models.gemma2.modeling_gemma2",
    "transformers.models.granite.modeling_granite",
]


# =============================================================================
# Replacement forward methods
# =============================================================================


def _make_swiglu_mlp_forward(original):
    """SwiGLU MLP forward using Opaque Triton kernel."""
    def forward(self, x):
        if not x.is_cuda:
            return original(self, x)
        from opaque.compat.kernels.swiglu import Opaque_SwiGLU

        return self.down_proj(Opaque_SwiGLU.apply(self.gate_proj(x), self.up_proj(x)))
    return forward


def _make_phi3_mlp_forward(original):
    """Phi3 MLP forward (combined gate_up_proj) using Opaque Triton kernel."""
    def forward(self, hidden_states):
        if not hidden_states.is_cuda:
            return original(self, hidden_states)
        from opaque.compat.kernels.swiglu import Opaque_SwiGLU

        gate, up = self.gate_up_proj(hidden_states).chunk(2, dim=-1)
        return self.down_proj(Opaque_SwiGLU.apply(gate, up))
    return forward


def _make_geglu_exact_mlp_forward(original):
    """Gemma MLP forward using Opaque GeGLU exact kernel."""
    def forward(self, x):
        if not x.is_cuda:
            return original(self, x)
        from opaque.compat.kernels.geglu import Opaque_GeGLU_Exact

        return self.down_proj(Opaque_GeGLU_Exact.apply(self.gate_proj(x), self.up_proj(x)))
    return forward


def _make_geglu_approx_mlp_forward(original):
    """Gemma2 MLP forward using Opaque GeGLU approx kernel."""
    def forward(self, x):
        if not x.is_cuda:
            return original(self, x)
        from opaque.compat.kernels.geglu import Opaque_GeGLU_Approx

        return self.down_proj(Opaque_GeGLU_Approx.apply(self.gate_proj(x), self.up_proj(x)))
    return forward


def _rotate_half(x):
    """Rotates half the hidden dims of the input (standard HF rotate_half)."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def _opaque_apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    """RoPE using Opaque Triton kernel.

    Replaces HF's apply_rotary_pos_emb at module level. Uses Opaque_RoPE_QK
    which processes Q and K together with GQA support.

    Falls back to PyTorch when:
    - cos/sin cannot be reduced to 2D (e.g., variable position_ids)
    - tensors are not on CUDA (Triton requires CUDA)
    """
    # Triton kernels require CUDA
    if not q.is_cuda:
        cos_u = cos.unsqueeze(unsqueeze_dim)
        sin_u = sin.unsqueeze(unsqueeze_dim)
        q_embed = (q * cos_u) + (_rotate_half(q) * sin_u)
        k_embed = (k * cos_u) + (_rotate_half(k) * sin_u)
        return q_embed, k_embed

    from opaque.compat.kernels.rope_embedding import Opaque_RoPE_QK

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


def _pytorch_causal_lm_loss(
    logits,
    labels,
    vocab_size: int,
    num_items_in_batch=None,
    ignore_index: int = -100,
    shift_labels=None,
    **kwargs,
) -> torch.Tensor:
    """Standard PyTorch cross-entropy loss for non-CUDA devices."""
    logits = logits.float()

    if shift_labels is None:
        labels = nn.functional.pad(labels, (0, 1), value=ignore_index)
        shift_labels = labels[..., 1:].contiguous()

    logits_flat = logits.view(-1, vocab_size)
    shift_labels_flat = shift_labels.view(-1)

    if num_items_in_batch is not None:
        loss = nn.functional.cross_entropy(
            logits_flat, shift_labels_flat, ignore_index=ignore_index, reduction="sum",
        )
        if torch.is_tensor(num_items_in_batch):
            num_items_in_batch = num_items_in_batch.to(loss.device)
        return loss / num_items_in_batch
    else:
        return nn.functional.cross_entropy(
            logits_flat, shift_labels_flat, ignore_index=ignore_index,
        )


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
    Falls back to PyTorch cross-entropy on non-CUDA devices.
    """
    # Triton kernels require CUDA — fall back to standard CE on CPU/MPS
    if not logits.is_cuda:
        return _pytorch_causal_lm_loss(
            logits, labels, vocab_size, num_items_in_batch, ignore_index,
            shift_labels, **kwargs,
        )

    from opaque.compat.kernels.cross_entropy import Opaque_CrossEntropyLoss

    logits = logits.float()

    if shift_labels is None:
        # Shift so that tokens < n predict n (same as HF ForCausalLMLoss)
        labels = nn.functional.pad(labels, (0, 1), value=ignore_index)
        shift_labels = labels[..., 1:].contiguous()

    logits_flat = logits.view(-1, vocab_size)
    shift_labels_flat = shift_labels.view(-1)
    shift_labels_flat = shift_labels_flat.to(logits_flat.device)

    losses, _ = Opaque_CrossEntropyLoss.apply(logits_flat, shift_labels_flat)

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


# ForCausalLM classes eligible for fused linear + cross-entropy loss.
# All share identical structure: self.model(backbone) → self.lm_head → loss.
_FUSED_CE_CAUSAL_LM = [
    ("transformers.models.llama.modeling_llama", "LlamaForCausalLM"),
    ("transformers.models.mistral.modeling_mistral", "MistralForCausalLM"),
    ("transformers.models.qwen2.modeling_qwen2", "Qwen2ForCausalLM"),
    ("transformers.models.qwen3.modeling_qwen3", "Qwen3ForCausalLM"),
    ("transformers.models.gemma.modeling_gemma", "GemmaForCausalLM"),
    ("transformers.models.gemma2.modeling_gemma2", "Gemma2ForCausalLM"),
    ("transformers.models.granite.modeling_granite", "GraniteForCausalLM"),
    ("transformers.models.cohere.modeling_cohere", "CohereForCausalLM"),
    ("transformers.models.cohere2.modeling_cohere2", "Cohere2ForCausalLM"),
]


def _make_fused_ce_causal_lm_forward(original):
    """ForCausalLM forward with fused linear + cross-entropy loss.

    When labels are provided and hidden_states are bf16/fp16, skips lm_head
    projection and computes loss directly from hidden_states @ lm_head.weight.T
    using CCE Triton kernels. Avoids materializing the full (B, S, V) logit
    tensor — saves ~1 GB per sample for 128K vocab models.
    """
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        position_ids=None,
        past_key_values=None,
        inputs_embeds=None,
        labels=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        cache_position=None,
        num_logits_to_keep=0,
        **kwargs,
    ):
        # No labels → inference → use original forward
        if labels is None:
            return original(
                self,
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                labels=labels,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                cache_position=cache_position,
                num_logits_to_keep=num_logits_to_keep,
                **kwargs,
            )

        # Resolve config defaults
        output_attentions = (
            output_attentions
            if output_attentions is not None
            else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        # Call backbone
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position,
            **kwargs,
        )
        hidden_states = outputs[0]

        # Fused path requires half precision on CUDA (CCE backward constraint)
        if hidden_states.is_cuda and hidden_states.dtype in (torch.bfloat16, torch.float16):
            from opaque.compat.kernels.linear_cross_entropy import (
                Opaque_LinearCrossEntropyLoss,
            )

            weight = self.lm_head.weight

            # Cohere-style multiplicative logit scaling: logits * scale
            logit_scale = getattr(self.config, "logit_scale", None)
            if logit_scale is not None and logit_scale != 1.0:
                weight = weight * logit_scale

            # Granite divisive scaling: logits / logits_scaling
            # Applied to weight before kernel (same as Cohere) so autograd
            # correctly chains the gradient back to the original weight.
            logits_scaling = getattr(self.config, "logits_scaling", None)
            if logits_scaling is not None and logits_scaling != 1.0:
                weight = weight / logits_scaling

            # Gemma2 softcapping: softcap * tanh(logits / softcap)
            softcap = getattr(self.config, "final_logit_softcapping", 0) or 0

            # Kernel returns nll_sum (unreduced) — reduce here
            nll_sum = Opaque_LinearCrossEntropyLoss.apply(
                hidden_states,
                weight,
                labels,
                -100,
                softcap,
            )

            num_items_in_batch = kwargs.get("num_items_in_batch")
            if num_items_in_batch is not None:
                if torch.is_tensor(num_items_in_batch):
                    num_items_in_batch = num_items_in_batch.to(nll_sum.device)
                loss = nll_sum / num_items_in_batch
            else:
                shifted_labels = labels[..., 1:].contiguous().flatten()
                n_valid = (shifted_labels != -100).sum().float().clamp(min=1)
                loss = nll_sum / n_valid

            logits = None
        else:
            # fp32 fallback: materialize logits, use existing CE kernel via LOSS_MAPPING
            logits = self.lm_head(hidden_states[..., -num_logits_to_keep:, :])
            loss = self.loss_function(logits, labels, self.vocab_size, **kwargs)

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        from transformers.modeling_outputs import CausalLMOutputWithPast

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    return forward


def _make_lora_linear_forward(original):
    """LoRA linear forward using Opaque kernel (vmap-compatible).

    Replaces peft.tuners.lora.Linear.forward. Uses Opaque_LoRA_W which
    computes base projection + LoRA delta in a single call.
    Falls back to PEFT's original forward on non-CUDA devices.
    """
    def forward(self, x, *args, **kwargs):
        if not x.is_cuda:
            return original(self, x, *args, **kwargs)

        from opaque.compat.kernels.lora import Opaque_LoRA_W

        if self.disable_adapters or not self.active_adapters:
            return self.base_layer(x)

        active = self.active_adapters[0]
        if active not in self.lora_A:
            return self.base_layer(x)

        dropout = self.lora_dropout[active]
        x_input = dropout(x)

        W = self.base_layer.weight
        # Conv1D stores weight as (in_features, out_features); F.linear expects
        # (out_features, in_features).  PEFT sets fan_in_fan_out=True for Conv1D.
        if getattr(self, "fan_in_fan_out", False):
            W = W.T
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

    return forward


# =============================================================================
# Patching helpers
# =============================================================================


def _patch_forward(module_path: str, class_name: str, forward_factory) -> bool:
    """Patch a class's forward method if the module is available.

    ``forward_factory`` is called with the original forward and must return
    the replacement.  This captures the original via closure — no attribute
    is stashed on the class.
    """
    try:
        module = importlib.import_module(module_path)
        cls = getattr(module, class_name, None)
        if cls is None:
            return False

        cls.forward = forward_factory(cls.forward)
        return True

    except (ImportError, RuntimeError):
        return False


def _patch_rope_functions(patched: list) -> None:
    """Patch apply_rotary_pos_emb at module level for each model."""
    for module_path in _ROPE_MODELS:
        try:
            module = importlib.import_module(module_path)
            if hasattr(module, "apply_rotary_pos_emb"):
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


def _patch_fused_ce(patched: list) -> None:
    """Patch ForCausalLM.forward to use fused linear + cross-entropy loss."""
    for path, cls_name in _FUSED_CE_CAUSAL_LM:
        if _patch_forward(path, cls_name, _make_fused_ce_causal_lm_forward):
            patched.append(f"{cls_name}(fused_ce)")


def _patch_lora_forward(patched: list) -> None:
    """Patch peft LoRA Linear forward and get_peft_model for fused LoRA."""
    try:
        from peft.tuners.lora import Linear as PeftLoRALinear

        PeftLoRALinear.forward = _make_lora_linear_forward(PeftLoRALinear.forward)
        patched.append("peft.LoRA.Linear")

    except (ImportError, RuntimeError):
        pass

    # Hook get_peft_model for automatic fused LoRA MLP patching
    try:
        import peft

        _original_get_peft_model = peft.get_peft_model

        def _patched_get_peft_model(model, peft_config=None, *args, **kwargs):
            result = _original_get_peft_model(
                model, peft_config, *args, **kwargs
            )
            try:
                _auto_fuse_lora(result)
            except Exception as e:
                logger.debug(f"opaque: Fused LoRA MLP auto-patch skipped: {e}")
            return result

        peft.get_peft_model = _patched_get_peft_model
        patched.append("peft.get_peft_model(auto-fuse)")

    except (ImportError, RuntimeError):
        pass


# =============================================================================
# Fused LoRA MLP patching
# =============================================================================

# Map MLP class names to activation types for fused LoRA MLP
_MLP_ACTIVATION_MAP = {
    "LlamaMLP": 0,  # ACTIVATION_SWIGLU
    "MistralMLP": 0,
    "Qwen2MLP": 0,
    "Qwen3MLP": 0,
    "GraniteMLP": 0,
    "CohereMLP": 0,
    "Cohere2MLP": 0,
    "Phi3MLP": 0,
    "GemmaMLP": 1,  # ACTIVATION_GEGLU_EXACT
    "Gemma2MLP": 2,  # ACTIVATION_GEGLU_APPROX
}


def _extract_lora_params(lora_linear):
    """Extract (W, A, B, scaling) from a peft LoRA Linear module.

    Returns (W, A, B, scaling) or (W, None, None, 0.0) if no active adapter.
    """
    W = lora_linear.base_layer.weight

    if (
        lora_linear.disable_adapters
        or not lora_linear.active_adapters
        or lora_linear.active_adapters[0] not in lora_linear.lora_A
    ):
        return W, None, None, 0.0

    active = lora_linear.active_adapters[0]
    # PEFT stores lora_A as (rank, in_features), kernel expects (in_features, rank)
    A = lora_linear.lora_A[active].weight.T
    # PEFT stores lora_B as (out_features, rank), kernel expects (rank, out_features)
    B = lora_linear.lora_B[active].weight.T
    scaling = lora_linear.scaling[active]
    return W, A, B, scaling


def _has_lora(module, proj_name):
    """Check if a module's sub-module has active LoRA adapters."""
    proj = getattr(module, proj_name, None)
    if proj is None:
        return False
    return hasattr(proj, "lora_A") and len(getattr(proj, "lora_A", {})) > 0


def _no_bias(module, proj_name):
    """Check that a projection has no bias (required for fused QKV kernel)."""
    proj = getattr(module, proj_name, None)
    if proj is None:
        return False
    base = getattr(proj, "base_layer", proj)
    return base.bias is None


def _is_phi3_style_mlp(mlp):
    """Check if MLP uses combined gate_up_proj (Phi3 style)."""
    return hasattr(mlp, "gate_up_proj") and not hasattr(mlp, "gate_proj")


def _make_fused_lora_mlp_forward(original_forward, activation_type):
    """Create fused LoRA MLP forward using Opaque_LoRA_MLP kernel.

    Replaces separate gate_proj + up_proj + activation + down_proj
    with a single fused kernel call.

    Args:
        original_forward: Bound method of the MLP instance.
        activation_type: 0=SwiGLU, 1=GeGLU_exact, 2=GeGLU_approx.
    """
    def forward(self, x):
        if not x.is_cuda:
            return original_forward(x)
        from opaque.compat.kernels.lora import Opaque_LoRA_MLP

        dtype = x.dtype

        Wg, Ag, Bg, Sg = _extract_lora_params(self.gate_proj)
        Wu, Au, Bu, Su = _extract_lora_params(self.up_proj)
        Wd, Ad, Bd, Sd = _extract_lora_params(self.down_proj)

        # Cast LoRA weights to input dtype for mixed precision
        if Ag is not None:
            Ag, Bg = Ag.to(dtype), Bg.to(dtype)
        if Au is not None:
            Au, Bu = Au.to(dtype), Bu.to(dtype)
        if Ad is not None:
            Ad, Bd = Ad.to(dtype), Bd.to(dtype)

        out, _gate, _up, _h = Opaque_LoRA_MLP.apply(
            x,
            Wg,
            Ag,
            Bg,
            Sg,
            Wu,
            Au,
            Bu,
            Su,
            Wd,
            Ad,
            Bd,
            Sd,
            activation_type,
        )

        # Add biases if present (most models don't have MLP bias)
        if getattr(self.gate_proj, "base_layer", self.gate_proj).bias is not None:
            return original_forward(x)
        if getattr(self.down_proj, "base_layer", self.down_proj).bias is not None:
            return original_forward(x)

        return out

    return forward


# =============================================================================
# Fused LoRA QKV patching
# =============================================================================

# Attention classes with standard QKV pattern: q_proj(x).view().transpose(-3,-2)
# Excluded: Qwen3 (q_norm/k_norm), Phi3 (combined qkv_proj), Cohere (no transpose)
# Excluded: Qwen2 (has bias=True on Q/K/V)
_FUSEABLE_QKV_ATTENTION_CLASSES = {
    "LlamaAttention",
    "MistralAttention",
    "GemmaAttention",
    "Gemma2Attention",
    "GraniteAttention",
    "Cohere2Attention",
}


def _opaque_fused_lora_qkv(self, hidden_states):
    """Compute Q, K, V using fused Opaque_LoRA_QKV kernel.

    Replaces 3 separate q_proj/k_proj/v_proj LoRA calls with a single
    fused kernel call that shares X computation across all three projections.
    """
    from opaque.compat.kernels.lora import Opaque_LoRA_QKV

    dtype = hidden_states.dtype

    Wq, Aq, Bq, Sq = _extract_lora_params(self.q_proj)
    Wk, Ak, Bk, Sk = _extract_lora_params(self.k_proj)
    Wv, Av, Bv, Sv = _extract_lora_params(self.v_proj)

    # Cast LoRA weights to input dtype for mixed precision
    if Aq is not None:
        Aq, Bq = Aq.to(dtype), Bq.to(dtype)
    if Ak is not None:
        Ak, Bk = Ak.to(dtype), Bk.to(dtype)
    if Av is not None:
        Av, Bv = Av.to(dtype), Bv.to(dtype)

    return Opaque_LoRA_QKV.apply(
        hidden_states,
        Wq,
        Aq,
        Bq,
        Sq,
        Wk,
        Ak,
        Bk,
        Sk,
        Wv,
        Av,
        Bv,
        Sv,
    )


def _make_fused_qkv_attention_forward(original_forward):
    """Create attention forward with fused QKV LoRA projection.

    Replaces the standard attention forward when Q/K/V all have LoRA adapters.
    Uses Opaque_LoRA_QKV for the projection step, then continues with the
    standard RoPE + attention + o_proj pipeline.

    Uses negative indexing (transpose(-3, -2)) for vmap safety.

    Args:
        original_forward: Bound method of the attention instance.
    """
    def forward(
        self,
        hidden_states,
        position_embeddings,
        attention_mask=None,
        past_key_values=None,
        cache_position=None,
        **kwargs,
    ):
        if not hidden_states.is_cuda:
            return original_forward(
                hidden_states, position_embeddings,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                cache_position=cache_position,
                **kwargs,
            )
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        # Fused QKV projection via Opaque_LoRA_QKV
        Q, K, V = self._opaque_fused_qkv(hidden_states)
        query_states = Q.view(hidden_shape).transpose(-3, -2)
        key_states = K.view(hidden_shape).transpose(-3, -2)
        value_states = V.view(hidden_shape).transpose(-3, -2)

        # RoPE — resolve from the attention class's own module (already patched)
        model_module = sys.modules[type(self).__module__]
        apply_rotary_pos_emb = model_module.apply_rotary_pos_emb
        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        # KV cache (training: past_key_values is None)
        if past_key_values is not None:
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_values.update(
                key_states,
                value_states,
                self.layer_idx,
                cache_kwargs,
            )

        # Attention dispatch — resolve from the class's own module (already patched)
        eager_attention_forward = model_module.eager_attention_forward
        if self.config._attn_implementation != "eager":
            ALL_ATTENTION_FUNCTIONS = model_module.ALL_ATTENTION_FUNCTIONS
            attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]
        else:
            attention_interface = eager_attention_forward

        attn_output, attn_weights = attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
            **kwargs,
        )

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()

        # O projection (still uses individual LoRA_W via patched peft.Linear.forward)
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights

    return forward


def _find_decoder_layers(model):
    """Find decoder layers across different model architectures."""
    for path_parts in [
        ["model", "model", "layers"],
        ["base_model", "model", "model", "layers"],
    ]:
        obj = model
        for attr in path_parts:
            obj = getattr(obj, attr, None)
            if obj is None:
                break
        if obj is not None and hasattr(obj, "__iter__"):
            return list(obj)
    return []


def _auto_fuse_lora(model):
    """Auto-detect and fuse LoRA layers with Opaque fused kernels.

    Called automatically after get_peft_model(). Walks decoder layers and:
    1. Fuses Q/K/V projections when all three have LoRA (Opaque_LoRA_QKV)
    2. Fuses gate/up/down projections when all three have LoRA (Opaque_LoRA_MLP)
    """
    layers = _find_decoder_layers(model)
    if not layers:
        return

    qkv_count = 0
    mlp_count = 0

    for layer in layers:
        # --- QKV fusion ---
        attn = getattr(layer, "self_attn", None)
        if attn is not None:
            # Check attention class is fuseable (standard QKV pattern, no bias)
            attn_cls_name = type(attn).__name__
            if attn_cls_name not in _FUSEABLE_QKV_ATTENTION_CLASSES:
                # Try MRO for wrapped classes
                for parent_cls in type(attn).__mro__:
                    if parent_cls.__name__ in _FUSEABLE_QKV_ATTENTION_CLASSES:
                        attn_cls_name = parent_cls.__name__
                        break

            if (
                attn_cls_name in _FUSEABLE_QKV_ATTENTION_CLASSES
                and _has_lora(attn, "q_proj")
                and _has_lora(attn, "k_proj")
                and _has_lora(attn, "v_proj")
                and _no_bias(attn, "q_proj")
                and _no_bias(attn, "k_proj")
                and _no_bias(attn, "v_proj")
            ):
                attn._opaque_fused_qkv = types.MethodType(
                    _opaque_fused_lora_qkv,
                    attn,
                )
                fused_qkv_fwd = _make_fused_qkv_attention_forward(attn.forward)
                attn.forward = types.MethodType(fused_qkv_fwd, attn)
                qkv_count += 1

        # --- MLP fusion ---
        mlp = getattr(layer, "mlp", None)
        if mlp is None:
            continue

        # Skip Phi3-style combined gate_up_proj (not supported by fused kernel)
        if _is_phi3_style_mlp(mlp):
            continue

        # Check all three projections have LoRA
        if not (
            _has_lora(mlp, "gate_proj")
            and _has_lora(mlp, "up_proj")
            and _has_lora(mlp, "down_proj")
        ):
            continue

        # Determine activation type from MLP class name
        cls_name = type(mlp).__name__
        # The MLP might be wrapped by PEFT, check through base
        if cls_name not in _MLP_ACTIVATION_MAP:
            # Try unwrapping: PEFT wraps the module, check the class hierarchy
            for parent_cls in type(mlp).__mro__:
                if parent_cls.__name__ in _MLP_ACTIVATION_MAP:
                    cls_name = parent_cls.__name__
                    break
            else:
                continue

        activation_type = _MLP_ACTIVATION_MAP[cls_name]

        fused_mlp_fwd = _make_fused_lora_mlp_forward(mlp.forward, activation_type)
        mlp.forward = types.MethodType(fused_mlp_fwd, mlp)
        mlp_count += 1

    if qkv_count > 0 or mlp_count > 0:
        logger.debug(
            f"opaque: Fused LoRA applied to {qkv_count} QKV + {mlp_count} MLP layers"
        )


def patch_lora_model(model) -> None:
    """Manually apply fused LoRA patching (QKV + MLP) to a PEFT model.

    Use this when loading a pre-existing PEFT model (e.g., from checkpoint)
    without calling get_peft_model(). The auto-hook only fires on
    get_peft_model() calls.

    Detects and fuses:
    - Q/K/V projections with LoRA → Opaque_LoRA_QKV
    - gate/up/down projections with LoRA → Opaque_LoRA_MLP

    Args:
        model: A PEFT-wrapped model with LoRA adapters.
    """
    _auto_fuse_lora(model)


# =============================================================================
# Public API
# =============================================================================


def apply_kernel_patches() -> None:
    """Replace HF model components with Opaque Triton kernels.

    Patches at class/module level for:
    - MLP activations: SwiGLU and GeGLU variants
    - RoPE: apply_rotary_pos_emb for all supported models
    - Cross-entropy loss: ForCausalLM via LOSS_MAPPING
    - LoRA: peft.tuners.lora.Linear forward

    No-op when CUDA/Triton unavailable or OPAQUE_SKIP_TRANSFORMERS_KERNEL_PATCHES=all.
    """
    global _is_kernel_patched

    if _is_kernel_patched:
        return

    if not torch.cuda.is_available():
        _is_kernel_patched = True
        return

    try:
        import triton  # noqa: F401
    except ImportError:
        _is_kernel_patched = True
        return

    patched = []
    skip = parse_skip_env("OPAQUE_SKIP_TRANSFORMERS_KERNEL_PATCHES")
    if "all" in skip:
        _is_kernel_patched = True
        return

    # SwiGLU MLP
    if "swiglu" not in skip:
        for path, cls_name in _SWIGLU_MLP:
            if _patch_forward(path, cls_name, _make_swiglu_mlp_forward):
                patched.append(cls_name)

        # Phi3 MLP (combined gate_up_proj)
        for path, cls_name in _PHI3_MLP:
            if _patch_forward(path, cls_name, _make_phi3_mlp_forward):
                patched.append(cls_name)

        # GeGLU exact MLP (Gemma)
        for path, cls_name in _GEGLU_EXACT_MLP:
            if _patch_forward(path, cls_name, _make_geglu_exact_mlp_forward):
                patched.append(cls_name)

        # GeGLU approx MLP (Gemma2)
        for path, cls_name in _GEGLU_APPROX_MLP:
            if _patch_forward(path, cls_name, _make_geglu_approx_mlp_forward):
                patched.append(cls_name)

    # RoPE
    if "rope" not in skip:
        _patch_rope_functions(patched)

    # Cross-entropy loss (LOSS_MAPPING fallback for fp32 and non-fused models)
    if "ce" not in skip:
        _patch_cross_entropy_loss(patched)

    # Fused linear + cross-entropy (bf16/fp16: skips lm_head materialization)
    if "fused_ce" not in skip:
        _patch_fused_ce(patched)

    # LoRA
    if "lora" not in skip:
        _patch_lora_forward(patched)

    if patched:
        logger.debug(f"opaque: Applied Triton kernel patches to: {', '.join(patched)}")

    _is_kernel_patched = True


def is_kernel_patched() -> bool:
    """Check if Triton kernel patches have been applied."""
    return _is_kernel_patched
