"""Fused Triton kernels (with PyTorch fallbacks) for Opaque.

Power the Hugging Face kernel-patching layer (``opaque.patches.transformers.components``)
and are usable directly as fast drop-in ops. When Triton is unavailable,
pure-PyTorch implementations are used as fallbacks so the public API works
everywhere.

Low-level ``Opaque_*`` autograd classes are internal implementation details;
import them from concrete submodules only when patch internals need them.
"""

import torch
import torch.nn.functional as F

try:
    # Loss functions
    from .cross_entropy import opaque_cross_entropy_loss, opaque_selective_log_softmax
    from .fused_add_rms_norm import opaque_fused_add_rms_norm
    from .geglu import opaque_geglu_approx, opaque_geglu_exact
    from .linear_cross_entropy import opaque_linear_cross_entropy_loss

    # LoRA kernels
    from .lora import (
        ACTIVATION_GEGLU_APPROX,
        ACTIVATION_GEGLU_EXACT,
        ACTIVATION_SWIGLU,
        opaque_lora_mlp,
        opaque_lora_qkv,
        opaque_lora_w,
    )

    # MoE expert FFN — single public op. ``opaque_moe`` transparently uses the
    # sparse grouped-GEMM Triton kernel on CUDA bf16/fp16 and the dense torch path
    # otherwise; fused-ness is an internal detail (like ``opaque_cross_entropy_loss``
    # chunking over large vocab), not a separate op, so it is not exported.
    from .moe import opaque_moe
    from .rms_norm import opaque_rms_norm

    # Position embeddings
    from .rope_embedding import opaque_rope, opaque_rope_qk, opaque_slow_rope

    # Activation functions
    from .swiglu import opaque_swiglu
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

    def opaque_cross_entropy_loss(
        logits,
        labels,
        logit_softcapping=0,
        logit_scaling=0,
        label_smoothing=0.0,
    ):
        """Compute unreduced cross-entropy with the PyTorch fallback.

        Args:
            logits: Logits with vocabulary as the final dimension.
            labels: Target indices with ``-100`` marking ignored positions.
            logit_softcapping: Optional symmetric logit softcap.
            logit_scaling: Optional multiplicative logit scale.
            label_smoothing: Cross-entropy label-smoothing weight.

        Returns:
            Per-token losses with the same shape as ``labels``.
        """
        logits = _apply_logit_transforms(logits, logit_softcapping, logit_scaling)
        flat_logits = logits.reshape(-1, logits.shape[-1])
        flat_labels = labels.reshape(-1)
        loss_flat = F.cross_entropy(
            flat_logits,
            flat_labels,
            reduction="none",
            label_smoothing=float(label_smoothing or 0.0),
        )
        return loss_flat.reshape(labels.shape)

    def opaque_selective_log_softmax(logits, indices):
        """Select log-softmax values, returning zero for ignored indices.

        Args:
            logits: Logits with vocabulary as the final dimension.
            indices: Vocabulary indices to select; ``-100`` is ignored.

        Returns:
            Selected log-probabilities with the same shape as ``indices``.
        """
        log_probs = torch.log_softmax(logits, dim=-1)
        # Match the Triton kernel's ignore convention: -100 returns 0.
        ignore = indices == -100
        safe = indices.masked_fill(ignore, 0)
        gathered = torch.gather(log_probs, dim=-1, index=safe.unsqueeze(-1)).squeeze(-1)
        return gathered.masked_fill(ignore, 0.0)

    # Chunked custom-autograd CE streams the log-sum-exp over vocab chunks
    # instead of materializing the full ``(tokens, vocab)`` logits, so
    # large-vocab CE fits in memory on MPS/CPU and stays ``vmap(grad)``-safe.
    from ._linear_ce_chunked import linear_cross_entropy_chunked

    def opaque_linear_cross_entropy_loss(
        hidden_states,
        weight,
        labels,
        ignore_index=-100,
        logit_softcapping=0,
        label_smoothing=0.0,
        use_token_scaling=False,
    ):
        """Compute per-token linear cross-entropy with the chunked fallback.

        Args:
            hidden_states: Token hidden states.
            weight: Output-projection weight matrix.
            labels: Target token indices.
            ignore_index: Target value excluded from the loss.
            logit_softcapping: Optional symmetric logit softcap.
            label_smoothing: Cross-entropy label-smoothing weight.
            use_token_scaling: Whether to apply detached token-confidence scaling.

        Returns:
            The mean loss over non-ignored target tokens.
        """
        return linear_cross_entropy_chunked(
            hidden_states,
            weight,
            labels,
            ignore_index,
            logit_softcapping,
            label_smoothing,
            use_token_scaling,
        )

    def opaque_swiglu(gate, up):
        """Apply the SiLU-gated linear-unit activation.

        Args:
            gate: Gate-projection activations.
            up: Up-projection activations matching ``gate``.

        Returns:
            The elementwise SwiGLU activation.
        """
        return F.silu(gate) * up

    def opaque_geglu_exact(gate, up):
        """Apply exact GeGLU using PyTorch GELU.

        Args:
            gate: Gate-projection activations.
            up: Up-projection activations matching ``gate``.

        Returns:
            The elementwise exact-GeGLU activation.
        """
        return F.gelu(gate, approximate="none") * up

    def opaque_geglu_approx(gate, up):
        """Apply tanh-approximated GeGLU using PyTorch GELU.

        Args:
            gate: Gate-projection activations.
            up: Up-projection activations matching ``gate``.

        Returns:
            The elementwise approximate-GeGLU activation.
        """
        return F.gelu(gate, approximate="tanh") * up

    def opaque_rope(Q, cos, sin):
        """Apply rotary position embeddings to a single tensor.

        Args:
            Q: Query or key tensor to rotate.
            cos: Cosine position-embedding coefficients.
            sin: Sine position-embedding coefficients.

        Returns:
            The rotated tensor.
        """
        return Q * cos + _rotate_half(Q) * sin

    def opaque_rope_qk(Q, K, cos, sin, rope_indices=None):
        """Apply rotary position embeddings to query and key tensors.

        Args:
            Q: Query tensor to rotate.
            K: Key tensor to rotate.
            cos: Cosine position-embedding coefficients.
            sin: Sine position-embedding coefficients.
            rope_indices: Unused compatibility argument.

        Returns:
            A tuple containing the rotated query and key tensors.
        """
        return (Q * cos + _rotate_half(Q) * sin, K * cos + _rotate_half(K) * sin)

    def opaque_slow_rope(Q, cos, sin, position_ids=None):
        """Apply the compatibility rotary-position-embedding fallback.

        Args:
            Q: Query or key tensor to rotate.
            cos: Cosine position-embedding coefficients.
            sin: Sine position-embedding coefficients.
            position_ids: Unused compatibility argument.

        Returns:
            The rotated tensor.
        """
        return opaque_rope(Q, cos, sin)

    def opaque_lora_w(X, W, A, B, scaling):
        """Apply a LoRA delta to one linear projection.

        Args:
            X: Input activations.
            W: Base projection weight.
            A: LoRA down-projection weight.
            B: LoRA up-projection weight.
            scaling: LoRA scaling factor.

        Returns:
            The projected activations with the LoRA delta applied.
        """
        return X @ W.transpose(-1, -2) + (X @ A @ B) * scaling

    def opaque_lora_qkv(X, Wq, Aq, Bq, Sq, Wk, Ak, Bk, Sk, Wv, Av, Bv, Sv):
        """Apply LoRA deltas to query, key, and value projections.

        Args:
            X: Input activations.
            Wq: Base query-projection weight.
            Aq: Query LoRA down-projection weight.
            Bq: Query LoRA up-projection weight.
            Sq: Query LoRA scaling factor.
            Wk: Base key-projection weight.
            Ak: Key LoRA down-projection weight.
            Bk: Key LoRA up-projection weight.
            Sk: Key LoRA scaling factor.
            Wv: Base value-projection weight.
            Av: Value LoRA down-projection weight.
            Bv: Value LoRA up-projection weight.
            Sv: Value LoRA scaling factor.

        Returns:
            A tuple containing projected query, key, and value tensors.
        """
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
        """Apply LoRA deltas to a gated MLP with the selected activation.

        Args:
            X: Input activations.
            Wg: Base gate-projection weight.
            Ag: Gate LoRA down-projection weight.
            Bg: Gate LoRA up-projection weight.
            Sg: Gate LoRA scaling factor.
            Wu: Base up-projection weight.
            Au: Up LoRA down-projection weight.
            Bu: Up LoRA up-projection weight.
            Su: Up LoRA scaling factor.
            Wd: Base down-projection weight.
            Ad: Down LoRA down-projection weight.
            Bd: Down LoRA up-projection weight.
            Sd: Down LoRA scaling factor.
            activation: GLU activation name or numeric kernel selector.

        Returns:
            The MLP output with all LoRA deltas applied.
        """
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

    def opaque_rms_norm(
        x,
        weight,
        eps=1e-6,
        offset=0.0,
        casting_mode="llama",
        *,
        in_place_backward=False,
        row_mode=None,
    ):
        """Apply RMS normalization with the PyTorch fallback.

        Args:
            x: Input tensor to normalize.
            weight: RMSNorm scale parameter.
            eps: Numerical-stability constant.
            offset: Optional additive offset to ``weight``.
            casting_mode: Precision behavior matching the model family.
            in_place_backward: Unused compatibility argument.
            row_mode: Unused compatibility argument.

        Returns:
            The RMS-normalized tensor.
        """
        del in_place_backward, row_mode
        orig = x.shape
        x2 = x.reshape(-1, x.shape[-1])
        if casting_mode in ("llama", "gemma"):
            xf = x2.float()
            ms = (xf * xf).mean(-1, keepdim=True)
            inv = torch.rsqrt(ms + eps)
            normed = (xf * inv).to(x.dtype)
        elif casting_mode == "none":
            ms = (x2 * x2).mean(-1, keepdim=True)
            eps_t = torch.tensor(
                eps, device=x2.device, dtype=x2.dtype, requires_grad=False
            )
            inv = torch.rsqrt(ms + eps_t)
            normed = x2 * inv
        else:
            raise ValueError(casting_mode)
        out = normed * (weight + offset)
        return out.view(orig)

    def opaque_fused_add_rms_norm(
        x,
        residual,
        weight,
        eps=1e-6,
        offset=0.0,
        casting_mode="llama",
        *,
        in_place_backward=False,
    ):
        """Add a residual and apply RMS normalization with the fallback.

        Args:
            x: Current layer activations.
            residual: Residual tensor to add to ``x``.
            weight: RMSNorm scale parameter.
            eps: Numerical-stability constant.
            offset: Optional additive offset to ``weight``.
            casting_mode: Precision behavior matching the model family.
            in_place_backward: Unused compatibility argument.

        Returns:
            A tuple containing normalized activations and the residual sum.
        """
        del in_place_backward
        S = x + residual
        y = opaque_rms_norm(
            S, weight, eps, offset, casting_mode, in_place_backward=False, row_mode=None
        )
        return y, S

    # MoE: the dense path is pure-torch (Triton-free) and is the public op.
    from .moe import opaque_moe


__all__ = [
    # Loss
    "opaque_cross_entropy_loss",
    "opaque_selective_log_softmax",
    "opaque_linear_cross_entropy_loss",
    # MoE expert FFN
    "opaque_moe",
    # Activations
    "opaque_swiglu",
    "opaque_geglu_exact",
    "opaque_geglu_approx",
    "opaque_rms_norm",
    "opaque_fused_add_rms_norm",
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
