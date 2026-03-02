"""Kernel function API for compatibility patching.

This module powers ``opaque.compat.transformers`` kernel patching and is
primarily a compatibility implementation detail. Advanced users may call these
function wrappers directly via ``opaque.compat.kernels`` when needed.

Low-level ``Opaque_*`` autograd classes remain internal implementation details
and should be imported from concrete submodules only when patch internals need
them.
"""

import torch
import torch.nn.functional as F

try:
    # Loss functions
    from .cross_entropy import opaque_cross_entropy_loss
    from .linear_cross_entropy import opaque_linear_cross_entropy_loss

    # Activation functions
    from .swiglu import opaque_swiglu
    from .geglu import opaque_geglu_exact, opaque_geglu_approx

    # Position embeddings
    from .rope_embedding import opaque_rope, opaque_rope_qk, opaque_slow_rope

    # LoRA kernels
    from .lora import (
        opaque_lora_w,
        opaque_lora_qkv,
        opaque_lora_mlp,
        ACTIVATION_SWIGLU,
        ACTIVATION_GEGLU_EXACT,
        ACTIVATION_GEGLU_APPROX,
    )
except ModuleNotFoundError as import_error:
    if import_error.name != "triton":
        raise

    def _apply_logit_transforms(
        logits: torch.Tensor,
        logit_softcapping: float | int = 0,
        logit_scaling: float | int = 0,
    ) -> torch.Tensor:
        transformed = logits
        if logit_scaling != 0:
            transformed = transformed * logit_scaling
        if logit_softcapping != 0:
            transformed = logit_softcapping * torch.tanh(
                transformed / logit_softcapping
            )
        return transformed

    def _rotate_half(x: torch.Tensor) -> torch.Tensor:
        half = x.shape[-1] // 2
        x1 = x[..., :half]
        x2 = x[..., half:]
        return torch.cat((-x2, x1), dim=-1)

    def opaque_cross_entropy_loss(logits, labels, logit_softcapping=0, logit_scaling=0):
        logits = _apply_logit_transforms(logits, logit_softcapping, logit_scaling)
        flat_logits = logits.reshape(-1, logits.shape[-1])
        flat_labels = labels.reshape(-1)
        loss_flat = F.cross_entropy(flat_logits, flat_labels, reduction="none")
        return loss_flat.reshape(labels.shape)

    def opaque_linear_cross_entropy_loss(
        hidden_states,
        weight,
        labels,
        num_items_in_batch=None,
        ignore_index=-100,
        logit_softcapping=0,
    ):
        shifted_hidden = hidden_states[..., :-1, :].contiguous()
        shifted_labels = labels[..., 1:].contiguous()
        logits = shifted_hidden @ weight.transpose(-1, -2)
        logits = _apply_logit_transforms(logits, logit_softcapping, 0)
        loss_flat = F.cross_entropy(
            logits.reshape(-1, logits.shape[-1]),
            shifted_labels.reshape(-1),
            reduction="none",
            ignore_index=ignore_index,
        )
        nll_sum = loss_flat.sum()

        if num_items_in_batch is not None:
            if torch.is_tensor(num_items_in_batch):
                num_items_in_batch = num_items_in_batch.to(nll_sum.device)
            return nll_sum / num_items_in_batch

        n_valid = (
            (shifted_labels.reshape(-1) != ignore_index).sum().float().clamp(min=1)
        )
        return nll_sum / n_valid

    def opaque_swiglu(gate, up):
        return F.silu(gate) * up

    def opaque_geglu_exact(gate, up):
        return F.gelu(gate, approximate="none") * up

    def opaque_geglu_approx(gate, up):
        return F.gelu(gate, approximate="tanh") * up

    def opaque_rope(Q, cos, sin):
        return Q * cos + _rotate_half(Q) * sin

    def opaque_rope_qk(Q, K, cos, sin, rope_indices=None):
        return (Q * cos + _rotate_half(Q) * sin, K * cos + _rotate_half(K) * sin)

    def opaque_slow_rope(Q, cos, sin, position_ids=None):
        return opaque_rope(Q, cos, sin)

    def opaque_lora_w(X, W, A, B, scaling):
        return X @ W.transpose(-1, -2) + (X @ A @ B) * scaling

    def opaque_lora_qkv(X, Wq, Aq, Bq, Sq, Wk, Ak, Bk, Sk, Wv, Av, Bv, Sv):
        Q = X @ Wq.transpose(-1, -2) + (X @ Aq @ Bq) * Sq
        K = X @ Wk.transpose(-1, -2) + (X @ Ak @ Bk) * Sk
        V = X @ Wv.transpose(-1, -2) + (X @ Av @ Bv) * Sv
        return Q, K, V

    def opaque_lora_mlp(
        X,
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
        activation="swiglu",
    ):
        gate = X @ Wg.transpose(-1, -2) + (X @ Ag @ Bg) * Sg
        up = X @ Wu.transpose(-1, -2) + (X @ Au @ Bu) * Su

        if activation in (ACTIVATION_SWIGLU, "swiglu"):
            hidden = opaque_swiglu(gate, up)
        elif activation in (ACTIVATION_GEGLU_EXACT, "geglu_exact"):
            hidden = opaque_geglu_exact(gate, up)
        elif activation in (ACTIVATION_GEGLU_APPROX, "geglu_approx"):
            hidden = opaque_geglu_approx(gate, up)
        else:
            raise ValueError(f"Unknown activation: {activation}")

        return hidden @ Wd.transpose(-1, -2) + (hidden @ Ad @ Bd) * Sd

    ACTIVATION_SWIGLU = 0
    ACTIVATION_GEGLU_EXACT = 1
    ACTIVATION_GEGLU_APPROX = 2

__all__ = [
    # Loss
    "opaque_cross_entropy_loss",
    "opaque_linear_cross_entropy_loss",
    # Activations
    "opaque_swiglu",
    "opaque_geglu_exact",
    "opaque_geglu_approx",
    # Position embeddings
    "opaque_rope",
    "opaque_rope_qk",
    "opaque_slow_rope",
    # LoRA
    "opaque_lora_w",
    "opaque_lora_qkv",
    "opaque_lora_mlp",
    "ACTIVATION_SWIGLU",
    "ACTIVATION_GEGLU_EXACT",
    "ACTIVATION_GEGLU_APPROX",
]
