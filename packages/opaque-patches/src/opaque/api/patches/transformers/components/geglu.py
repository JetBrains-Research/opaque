# Copyright (c) 2025 Opaque Authors
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import torch
import torch.nn as nn


def _make_geglu_exact_mlp_forward(original):
    """Gemma MLP forward using Opaque GeGLU exact kernel."""

    def forward(self, x):
        if not x.is_cuda:
            return original(self, x)
        from opaque.api.patches.kernels.geglu import Opaque_GeGLU_Exact

        return self.down_proj(
            Opaque_GeGLU_Exact.apply(self.gate_proj(x), self.up_proj(x))
        )

    return forward


def _make_geglu_approx_mlp_forward(original):
    """Gemma2 MLP forward using Opaque GeGLU approx kernel."""

    def forward(self, x):
        if not x.is_cuda:
            return original(self, x)
        from opaque.api.patches.kernels.geglu import Opaque_GeGLU_Approx

        return self.down_proj(
            Opaque_GeGLU_Approx.apply(self.gate_proj(x), self.up_proj(x))
        )

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

    from opaque.api.patches.kernels.rope_embedding import Opaque_RoPE_QK

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
            logits_flat,
            shift_labels_flat,
            ignore_index=ignore_index,
            reduction="sum",
        )
        if torch.is_tensor(num_items_in_batch):
            num_items_in_batch = num_items_in_batch.to(loss.device)
        return loss / num_items_in_batch
    else:
        return nn.functional.cross_entropy(
            logits_flat,
            shift_labels_flat,
            ignore_index=ignore_index,
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
            logits,
            labels,
            vocab_size,
            num_items_in_batch,
            ignore_index,
            shift_labels,
            **kwargs,
        )

    from opaque.api.patches.kernels.cross_entropy import Opaque_CrossEntropyLoss

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
    ("transformers.models.ministral.modeling_ministral", "MinistralForCausalLM"),
    ("transformers.models.qwen2.modeling_qwen2", "Qwen2ForCausalLM"),
    ("transformers.models.qwen3.modeling_qwen3", "Qwen3ForCausalLM"),
    ("transformers.models.smollm3.modeling_smollm3", "SmolLM3ForCausalLM"),
    ("transformers.models.gemma.modeling_gemma", "GemmaForCausalLM"),
    ("transformers.models.gemma2.modeling_gemma2", "Gemma2ForCausalLM"),
    ("transformers.models.granite.modeling_granite", "GraniteForCausalLM"),
    ("transformers.models.cohere.modeling_cohere", "CohereForCausalLM"),
    ("transformers.models.cohere2.modeling_cohere2", "Cohere2ForCausalLM"),
    ("transformers.models.olmo2.modeling_olmo2", "Olmo2ForCausalLM"),
    ("transformers.models.olmo3.modeling_olmo3", "Olmo3ForCausalLM"),
    ("transformers.models.glm4.modeling_glm4", "Glm4ForCausalLM"),
    ("transformers.models.gemma3.modeling_gemma3", "Gemma3ForCausalLM"),
    ("transformers.models.exaone4.modeling_exaone4", "Exaone4ForCausalLM"),
]


def _fused_linear_ce_loss_is_supported(
    logits_to_keep: int | torch.Tensor,
    kwargs: dict,
) -> bool:
    """Return True only when fused linear+CE can match HF ``loss_function`` behavior.

    Fused path uses full-sequence hidden states and fixed label shift. Non-default
    ``logits_to_keep`` needs sliced logits (and label alignment) — defer to the
    original forward.
    """
    if torch.is_tensor(logits_to_keep):
        return False
    if not isinstance(logits_to_keep, int):
        return False
    if logits_to_keep != 0:
        return False
    if kwargs.get("shift_labels") is not None:
        return False
    if kwargs.get("weight") is not None:
        return False
    ii = kwargs.get("ignore_index", -100)
    if torch.is_tensor(ii):
        return False
    return True


def _make_fused_ce_causal_lm_forward(original):
    """ForCausalLM forward with fused linear + cross-entropy loss.

    When labels are provided and hidden_states are bf16/fp16, skips ``lm_head``
    and computes loss from ``hidden_states @ lm_head.weight.T`` (CCE), unless
    ``loss_function`` would need unsupported options — then defers to the
    original forward (e.g. non-zero ``logits_to_keep``, ``shift_labels``, class
    ``weight``).
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
        logits_to_keep: int | torch.Tensor = 0,
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
                logits_to_keep=logits_to_keep,
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

        use_fused_ce = (
            hidden_states.is_cuda
            and hidden_states.dtype in (torch.bfloat16, torch.float16)
            and _fused_linear_ce_loss_is_supported(logits_to_keep, kwargs)
        )

        if use_fused_ce:
            from opaque.api.patches.kernels.linear_cross_entropy import (
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

            ignore_index = int(kwargs.get("ignore_index", -100))
            label_smoothing = float(kwargs.get("label_smoothing") or 0.0)

            nll_sum = Opaque_LinearCrossEntropyLoss.apply(
                hidden_states,
                weight,
                labels,
                ignore_index,
                softcap,
                label_smoothing,
            )

            num_items_in_batch = kwargs.get("num_items_in_batch")
            if num_items_in_batch is not None:
                if torch.is_tensor(num_items_in_batch):
                    num_items_in_batch = num_items_in_batch.to(nll_sum.device)
                loss = nll_sum / num_items_in_batch
            else:
                shifted_labels = labels[..., 1:].contiguous().flatten()
                n_valid = (shifted_labels != ignore_index).sum().float().clamp(min=1)
                loss = nll_sum / n_valid

            logits = None
        else:
            # Match HF: slice hidden states like ``lm_head`` in modeling code.
            slice_indices = (
                slice(-logits_to_keep, None)
                if isinstance(logits_to_keep, int)
                else logits_to_keep
            )
            logits = self.lm_head(hidden_states[:, slice_indices, :])
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
