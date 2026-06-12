# Copyright (c) 2025 Opaque Authors
# SPDX-License-Identifier: Apache-2.0
"""Forward patch for HF v5 stacked-weight MoE experts (``*Experts`` modules).

Swaps only ``forward(hidden_states, top_k_index, top_k_weights)`` onto the
vmap/DP-safe :func:`opaque_moe`, leaving the router, aux loss, and
parameters untouched. This is a vmap-safety patch (DP-SGD needs it), not a CUDA
kernel — ``opaque_moe`` is pure torch and runs on CPU too, so there is no
device fallback to HF's (vmap-broken) forward.
"""

from __future__ import annotations


def _make_moe_experts_forward(original):
    """Build a stacked-weight MoE experts ``forward`` using the Opaque kernel."""

    def forward(self, hidden_states, top_k_index, top_k_weights):
        from opaque.api.patches.kernels.moe import opaque_moe

        # reshape (not view) keeps this vmap-traceable under an extra mapped dim.
        hidden_dim = self.gate_up_proj.shape[-1]
        orig_shape = hidden_states.shape
        x = hidden_states.reshape(-1, hidden_dim)
        idx = top_k_index.reshape(-1, top_k_index.shape[-1])
        weights = top_k_weights.reshape(-1, top_k_weights.shape[-1])

        out = opaque_moe(x, self.gate_up_proj, self.down_proj, idx, weights)
        return out.reshape(orig_shape)

    return forward
