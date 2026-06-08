# Copyright (c) 2025 Opaque Authors
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import torch
import logging

from ._utils import _active_lora_dtype

logger = logging.getLogger(__name__)


def _make_lora_linear_forward(original):
    """LoRA linear forward using Opaque kernel (vmap-compatible).

    Replaces peft.tuners.lora.Linear.forward. Uses Opaque_LoRA_W which
    computes base projection + LoRA delta in a single call.
    Falls back to PEFT's original forward on non-CUDA devices.
    """

    def forward(self, x, *args, **kwargs):
        if not x.is_cuda:
            return original(self, x, *args, **kwargs)

        from opaque.api.patches.kernels.lora import Opaque_LoRA_W

        if self.disable_adapters or not self.active_adapters:
            return self.base_layer(x)

        active = self.active_adapters[0]
        if active not in self.lora_A:
            return self.base_layer(x)

        dropout = self.lora_dropout[active]
        # Fused kernel passes x to both base linear and adapter. This is only
        # correct when dropout is a no-op; otherwise dropout would leak into
        # the base projection. Fall back to PEFT's original forward when
        # dropout is active (training with lora_dropout > 0).
        dropout_is_noop = (
            isinstance(dropout, torch.nn.Identity)
            or (isinstance(dropout, torch.nn.Dropout) and dropout.p == 0.0)
            or not self.training
        )
        if not dropout_is_noop:
            return original(self, x, *args, **kwargs)

        W = self.base_layer.weight
        # Conv1D stores weight as (in_features, out_features); F.linear expects
        # (out_features, in_features).  PEFT sets fan_in_fan_out=True for Conv1D.
        if getattr(self, "fan_in_fan_out", False):
            W = W.T
        # PEFT stores lora_A as (rank, in_features), kernel expects (in_features, rank).
        # Cast to the active dtype: under autocast that's the autocast dtype (the
        # kernel's interior matmuls are autocast-intercepted to that dtype, and a
        # still-fp32 B would mismatch at addmm_); otherwise follow x.dtype.
        target_dtype = _active_lora_dtype(x)
        A = self.lora_A[active].weight.T.to(target_dtype)
        # PEFT stores lora_B as (out_features, rank), kernel expects (rank, out_features)
        B = self.lora_B[active].weight.T.to(target_dtype)
        scaling = self.scaling[active]

        result = Opaque_LoRA_W.apply(x, W, A, B, scaling)

        # Add base layer bias if present (kernel does F.linear without bias)
        if self.base_layer.bias is not None:
            result = result + self.base_layer.bias

        # Handle additional active adapters (rare, but support it)
        for adapter in self.active_adapters[1:]:
            if adapter in self.lora_A:
                dropout_i = self.lora_dropout[adapter]
                x_i = dropout_i(x)
                A_i = self.lora_A[adapter].weight.T.to(target_dtype)
                B_i = self.lora_B[adapter].weight.T.to(target_dtype)
                scaling_i = self.scaling[adapter]
                lora_out = (x_i @ A_i) @ B_i * scaling_i
                result = result + lora_out

        return result

    return forward
