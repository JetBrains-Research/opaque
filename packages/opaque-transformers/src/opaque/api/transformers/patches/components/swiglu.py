# Copyright (c) 2025 Opaque Authors
# SPDX-License-Identifier: Apache-2.0
"""SwiGLU MLP replacements backed by Opaque's vmap-safe kernels."""

from __future__ import annotations


def _make_swiglu_mlp_forward(original):
    """SwiGLU MLP forward using Opaque Triton kernel."""

    def forward(self, x):
        if not x.is_cuda:
            return original(self, x)
        from opaque.api.patches.kernels.swiglu import Opaque_SwiGLU

        return self.down_proj(Opaque_SwiGLU.apply(self.gate_proj(x), self.up_proj(x)))

    return forward


def _make_phi3_mlp_forward(original):
    """Phi3 MLP forward (combined gate_up_proj) using Opaque Triton kernel."""

    def forward(self, hidden_states):
        if not hidden_states.is_cuda:
            return original(self, hidden_states)
        from opaque.api.patches.kernels.swiglu import Opaque_SwiGLU

        gate, up = self.gate_up_proj(hidden_states).chunk(2, dim=-1)
        return self.down_proj(Opaque_SwiGLU.apply(gate, up))

    return forward
