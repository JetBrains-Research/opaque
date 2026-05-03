# Copyright (c) 2025 Opaque Authors
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import torch
import torch.nn as nn

def _make_rms_norm_forward(
    original, *, casting_mode: str, offset: float, in_place_bwd: bool
):
    """RMSNorm forward using Opaque Triton kernel."""

    def forward(self, hidden_states):
        if not hidden_states.is_cuda:
            return original(self, hidden_states)
        from opaque.patches.kernels.rms_norm import Opaque_RMSNorm

        eps = getattr(self, "variance_epsilon", None) or getattr(self, "eps", 1e-6)
        return Opaque_RMSNorm.apply(
            hidden_states,
            self.weight,
            float(eps),
            float(offset),
            casting_mode,
            in_place_bwd,
            None,
        )

    return forward


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


