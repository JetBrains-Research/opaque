# Copyright (c) 2025 Opaque Authors
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import torch
import torch.nn as nn

def _make_swiglu_mlp_forward(original):
    """SwiGLU MLP forward using Opaque Triton kernel."""

    def forward(self, x):
        if not x.is_cuda:
            return original(self, x)
        from opaque.patches.kernels.swiglu import Opaque_SwiGLU

        return self.down_proj(Opaque_SwiGLU.apply(self.gate_proj(x), self.up_proj(x)))

    return forward


def _rmsnorm_fac_llama(orig):
    return _make_rms_norm_forward(
        orig, casting_mode="llama", offset=0.0, in_place_bwd=True
    )


def _rmsnorm_fac_gemma(orig):
    return _make_rms_norm_forward(
        orig, casting_mode="gemma", offset=1.0, in_place_bwd=True
    )


def _rmsnorm_fac_gemma2(orig):
    return _make_rms_norm_forward(
        orig, casting_mode="gemma", offset=1.0, in_place_bwd=False
    )


def _rmsnorm_fac_olmo2(orig):
    # OLMo2 follows the same numeric formula as Llama-style RMSNorm but uses
    # non in-place backward in Liger's model-specific patching.
    return _make_rms_norm_forward(
        orig, casting_mode="llama", offset=0.0, in_place_bwd=False
    )


def _rmsnorm_fac_glm4(orig):
    return _make_rms_norm_forward(
        orig, casting_mode="llama", offset=0.0, in_place_bwd=False
    )


def _patch_rms_norm(patched: list) -> None:
    for path, cls_name in _RMSNORM_LLAMA_STYLE:
        if _patch_forward(path, cls_name, _rmsnorm_fac_llama):
            patched.append(f"{cls_name}(rmsnorm)")
    for path, cls_name in _RMSNORM_GEMMA:
        if _patch_forward(path, cls_name, _rmsnorm_fac_gemma):
            patched.append(f"{cls_name}(rmsnorm)")
    for path, cls_name in _RMSNORM_GEMMA2:
        if _patch_forward(path, cls_name, _rmsnorm_fac_gemma2):
            patched.append(f"{cls_name}(rmsnorm)")
    for path, cls_name in _RMSNORM_GEMMA3:
        if _patch_forward(path, cls_name, _rmsnorm_fac_gemma2):
            patched.append(f"{cls_name}(rmsnorm)")
    for path, cls_name in _RMSNORM_OLMO2:
        if _patch_forward(path, cls_name, _rmsnorm_fac_olmo2):
            patched.append(f"{cls_name}(rmsnorm)")
    for path, cls_name in _RMSNORM_GLM4:
        if _patch_forward(path, cls_name, _rmsnorm_fac_glm4):
            patched.append(f"{cls_name}(rmsnorm)")


# Fused residual add + post_attention_layernorm (Pre-LN block after attention;


def _post_attn_eps_and_weight(layer) -> tuple[torch.Tensor, float]:
    norm = layer.post_attention_layernorm
    w = norm.weight
    eps = float(getattr(norm, "variance_epsilon", None) or getattr(norm, "eps", 1e-6))
    return w, eps


def _fused_add_rms_fac_llama(orig):
    def forward(
        self,
        hidden_states,
        attention_mask=None,
        position_ids=None,
        past_key_values=None,
        use_cache=False,
        position_embeddings=None,
        **kwargs,
    ):
        if not hidden_states.is_cuda:
            return orig(
                self,
                hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                use_cache=use_cache,
                position_embeddings=position_embeddings,
                **kwargs,
            )
        from opaque.patches.kernels.fused_add_rms_norm import (
            Opaque_FusedAddRMSNorm,
        )

        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states, _ = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            position_embeddings=position_embeddings,
            **kwargs,
        )
        weight, eps = _post_attn_eps_and_weight(self)
        hidden_states, residual = Opaque_FusedAddRMSNorm.apply(
            hidden_states,
            residual,
            weight,
            eps,
            0.0,
            "llama",
            False,
        )
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states

    return forward


def _fused_add_rms_fac_gemma(orig):
    def forward(
        self,
        hidden_states,
        attention_mask=None,
        position_ids=None,
        past_key_values=None,
        use_cache=False,
        position_embeddings=None,
        **kwargs,
    ):
        if not hidden_states.is_cuda:
            return orig(
                self,
                hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                use_cache=use_cache,
                position_embeddings=position_embeddings,
                **kwargs,
            )
        from opaque.patches.kernels.fused_add_rms_norm import (
            Opaque_FusedAddRMSNorm,
        )

        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states, _ = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            position_embeddings=position_embeddings,
            **kwargs,
        )
        weight, eps = _post_attn_eps_and_weight(self)
        hidden_states, residual = Opaque_FusedAddRMSNorm.apply(
            hidden_states,
            residual,
            weight,
            eps,
            1.0,
            "gemma",
            False,
        )
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states

    return forward


def _fused_add_rms_fac_phi3(orig):
    def forward(
        self,
        hidden_states,
        attention_mask=None,
        position_ids=None,
        past_key_values=None,
        use_cache=False,
        position_embeddings=None,
        **kwargs,
    ):
        if not hidden_states.is_cuda:
            return orig(
                self,
                hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                use_cache=use_cache,
                position_embeddings=position_embeddings,
                **kwargs,
            )
        from opaque.patches.kernels.fused_add_rms_norm import (
            Opaque_FusedAddRMSNorm,
        )

        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states, _ = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            position_embeddings=position_embeddings,
            **kwargs,
        )
        xa = self.resid_attn_dropout(hidden_states)
        weight, eps = _post_attn_eps_and_weight(self)
        hidden_states, residual = Opaque_FusedAddRMSNorm.apply(
            xa,
            residual,
            weight,
            eps,
            0.0,
            "llama",
            False,
        )
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + self.resid_mlp_dropout(hidden_states)
        return hidden_states

    return forward


def _fused_add_rms_fac_granite(orig):
    def forward(
        self,
        hidden_states,
        attention_mask=None,
        position_ids=None,
        past_key_values=None,
        use_cache=False,
        position_embeddings=None,
        **kwargs,
    ):
        if not hidden_states.is_cuda:
            return orig(
                self,
                hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                use_cache=use_cache,
                position_embeddings=position_embeddings,
                **kwargs,
            )
        from opaque.patches.kernels.fused_add_rms_norm import (
            Opaque_FusedAddRMSNorm,
        )

        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states, _ = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            position_embeddings=position_embeddings,
            **kwargs,
        )
        xa = hidden_states * self.residual_multiplier
        weight, eps = _post_attn_eps_and_weight(self)
        hidden_states, residual = Opaque_FusedAddRMSNorm.apply(
            xa,
            residual,
            weight,
            eps,
            0.0,
            "llama",
            False,
        )
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states * self.residual_multiplier
        return hidden_states

    return forward


def _patch_fused_add_rms_norm_decoder(patched: list) -> None:
    for path, cls_name in _FUSED_ADD_RMS_DECODER_LLAMA:
        if _patch_forward(path, cls_name, _fused_add_rms_fac_llama):
            patched.append(f"{cls_name}(fused_add_rmsnorm)")
    for path, cls_name in _FUSED_ADD_RMS_DECODER_GEMMA:
        if _patch_forward(path, cls_name, _fused_add_rms_fac_gemma):
            patched.append(f"{cls_name}(fused_add_rmsnorm)")
    for path, cls_name in _FUSED_ADD_RMS_DECODER_PHI3:
        if _patch_forward(path, cls_name, _fused_add_rms_fac_phi3):
            patched.append(f"{cls_name}(fused_add_rmsnorm)")
    for path, cls_name in _FUSED_ADD_RMS_DECODER_GRANITE:
        if _patch_forward(path, cls_name, _fused_add_rms_fac_granite):
            patched.append(f"{cls_name}(fused_add_rmsnorm)")


def _make_phi3_mlp_forward(original):
    """Phi3 MLP forward (combined gate_up_proj) using Opaque Triton kernel."""

    def forward(self, hidden_states):
        if not hidden_states.is_cuda:
            return original(self, hidden_states)
        from opaque.patches.kernels.swiglu import Opaque_SwiGLU

        gate, up = self.gate_up_proj(hidden_states).chunk(2, dim=-1)
        return self.down_proj(Opaque_SwiGLU.apply(gate, up))

    return forward


