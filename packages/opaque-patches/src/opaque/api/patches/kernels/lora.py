# Copyright (c) 2025 Opaque Authors
# SPDX-License-Identifier: Apache-2.0
#
# Triton LoRA kernels derive from the Unsloth project
# (Apache-2.0; https://github.com/unslothai/unsloth/blob/main/unsloth/kernels/fast_lora.py)
# and have been adapted to Opaque's vmap-friendly new-style autograd dispatch.
# See NOTICE in the repository root.
"""LoRA (Low-Rank Adaptation) kernels with vmap support for DP-SGD.

Implements the low-rank adapter structure from Hu et al., *LoRA: Low-Rank
Adaptation of Large Language Models* (https://arxiv.org/abs/2106.09685).

Ported from unsloth/kernels/fast_lora.py for vmap compatibility.
Uses new-style autograd API with setup_context for vmap support.

Three implementations:
1. LoRA_W: Generic LoRA for single projection (O-proj, etc.)
2. LoRA_QKV: Fused LoRA for Q, K, V projections
3. LoRA_MLP: Fused LoRA for MLP (gate, up, down) with SwiGLU / GeGLU

Optimizations (from Unsloth):
- Triton SwiGLU kernels for activation in LoRA_MLP (forward + backward)
- addmm_ for fused gradient accumulation (avoids temporary tensors)

For DP-SGD:
- Base weights (W) are frozen, only LoRA weights (A, B) are trained
- vmap computes per-example gradients for A and B
"""

import torch
import torch.nn.functional as F

from opaque.exceptions import ConfigurationError

from ._utils import (
    active_cuda_dtype,
    cast_to_dtype,
    ensure_cuda_tensors,
    follow_autocast,
)
from .geglu import (
    _triton_geglu_approx_backward_fused,
    _triton_geglu_approx_forward,
    _triton_geglu_exact_backward_fused,
    _triton_geglu_exact_forward,
)
from .swiglu import _triton_swiglu_backward_fused, _triton_swiglu_forward

# Activation types for LoRA_MLP
ACTIVATION_SWIGLU = 0
ACTIVATION_GEGLU_EXACT = 1
ACTIVATION_GEGLU_APPROX = 2

_ACTIVATION_FORWARD = {
    ACTIVATION_SWIGLU: _triton_swiglu_forward,
    ACTIVATION_GEGLU_EXACT: _triton_geglu_exact_forward,
    ACTIVATION_GEGLU_APPROX: _triton_geglu_approx_forward,
}

_ACTIVATION_BACKWARD_FUSED = {
    ACTIVATION_SWIGLU: _triton_swiglu_backward_fused,
    ACTIVATION_GEGLU_EXACT: _triton_geglu_exact_backward_fused,
    ACTIVATION_GEGLU_APPROX: _triton_geglu_approx_backward_fused,
}

_LORA_MLP_GRAD_CHUNK_ROWS = 128
_LORA_MLP_GRAD_CHUNK_EXAMPLES = 128

_ACTIVATION_NAMES = {
    "swiglu": ACTIVATION_SWIGLU,
    "geglu_exact": ACTIVATION_GEGLU_EXACT,
    "geglu_approx": ACTIVATION_GEGLU_APPROX,
}


def _validate_vmap_dims(in_dims, *, name, batched_indices):
    """Reject vmap layouts that cannot preserve the fused-kernel semantics."""
    for index, batch_dim in enumerate(in_dims):
        expected = 0 if index in batched_indices else None
        if batch_dim != expected:
            raise ConfigurationError(
                *(
                    f"{name} vmap requires inputs {sorted(batched_indices)} to be batched "
                    f"at dim 0 and all other inputs to be unbatched, got {in_dims}",
                )
            )


def _needs_lora_weight_grads(*adapter_pairs):
    return any(
        A is not None and B is not None and (A.requires_grad or B.requires_grad)
        for A, B in adapter_pairs
    )


def _require_frozen_qkv_biases(*biases):
    """Reject base biases whose gradients the fused kernel cannot compute."""
    if any(bias is not None and bias.requires_grad for bias in biases):
        raise ConfigurationError(
            *(
                "Opaque_LoRA_QKV requires frozen Q/K/V base biases; "
                "use the unfused projection path for trainable biases.",
            )
        )


def _adapters_are_packable(*adapter_pairs):
    if not adapter_pairs or any(A is None or B is None for A, B in adapter_pairs):
        return False
    first_A, _first_B = adapter_pairs[0]
    rank = first_A.shape[1]
    return all(
        A.shape == (first_A.shape[0], rank)
        and B.shape[0] == rank
        and A.device == first_A.device
        and B.device == first_A.device
        and A.dtype == first_A.dtype
        and B.dtype == first_A.dtype
        for A, B in adapter_pairs
    )


def _pack_adapters(*adapters):
    """Pack same-input adapters as concatenated A and scaled block-diagonal B."""
    pairs = tuple((A, B) for A, B, _ in adapters)
    if not _adapters_are_packable(*pairs):
        return None, None
    rank = adapters[0][0].shape[1]
    packed_A = torch.cat([A for A, _, _ in adapters], dim=1)
    packed_B = adapters[0][1].new_zeros(
        (rank * len(adapters), sum(B.shape[1] for _, B, _ in adapters))
    )
    row = column = 0
    for _A, B, scaling in adapters:
        packed_B[row : row + rank, column : column + B.shape[1]] = B * scaling
        row += rank
        column += B.shape[1]
    return packed_A, packed_B


def _pack_qkv_base(Wq, bq, Wk, bk, Wv, bv):
    packed_W = torch.cat((Wq, Wk, Wv), dim=0)
    packed_bias = None
    if any(bias is not None for bias in (bq, bk, bv)):
        packed_bias = torch.cat(
            tuple(
                bias if bias is not None else weight.new_zeros(weight.shape[0])
                for weight, bias in ((Wq, bq), (Wk, bk), (Wv, bv))
            )
        )
    return packed_W, packed_bias


def _lora_w_weight_backward_impl(grad_out, X, A, B, scaling):
    """Compute LoRA_W adapter gradients before releasing the saved input."""
    if A is None or B is None:
        return None, None

    hidden_dim = X.shape[-1]
    out_features = B.shape[-1]

    X_flat = X.reshape(-1, hidden_dim)
    grad_out_flat = grad_out.reshape(-1, out_features)
    grad_out_Bt = grad_out_flat @ B.t()

    dA = torch.empty_like(A)
    dA.addmm_(X_flat.t(), grad_out_Bt, alpha=scaling, beta=0)
    dB = torch.empty_like(B)
    dB.addmm_(A.t() @ X_flat.t(), grad_out_flat, alpha=scaling, beta=0)
    return dA, dB


def _lora_w_backward_lite(grad_out, W, A, B, scaling):
    """Lightweight LoRA_W backward: only computes dX (no weight grads).

    Used when LoRA weights are frozen (don't require grad), e.g. in vmap(grad())
    for DP-SGD. Avoids saving X in setup_context, reducing peak memory.
    """
    batch_shape = grad_out.shape[:-1]
    out_features = W.shape[0]
    grad_out_flat = grad_out.reshape(-1, out_features)

    dX = torch.mm(grad_out_flat, W)
    if A is not None and B is not None:
        dX.addmm_(grad_out_flat @ B.t(), A.t(), alpha=scaling, beta=1)
    return dX.reshape(*batch_shape, W.shape[1])


class _LoRAWBackward(torch.autograd.Function):
    """LoRA_W adapter-gradient stage wrapped for vmap(grad()) support."""

    @staticmethod
    def forward(grad_out, X, A, B, scaling):
        return _lora_w_weight_backward_impl(grad_out, X, A, B, scaling)

    @staticmethod
    def setup_context(ctx, inputs, output):
        pass

    @staticmethod
    def backward(ctx, *grad_outputs):
        raise NotImplementedError("Double backward not supported for LoRA_W")

    @staticmethod
    def vmap(info, in_dims, grad_out, X, A, B, scaling):
        _validate_vmap_dims(in_dims, name="_LoRAWBackward", batched_indices={0, 1})
        if A is None or B is None:
            return (None, None), (None, None)

        B_vmap = X.shape[0]
        hidden_dim = X.shape[-1]
        out_features = B.shape[-1]
        X_3d = X.reshape(B_vmap, -1, hidden_dim)
        grad_out_3d = grad_out.reshape(B_vmap, -1, out_features)

        grad_out_Bt = grad_out_3d @ B.t()
        dA = torch.bmm(X_3d.transpose(-2, -1), grad_out_Bt) * scaling
        XA = X_3d @ A
        dB = torch.bmm(XA.transpose(-2, -1), grad_out_3d) * scaling
        return (dA, dB), (0, 0)


class _LoRAWBackwardLite(torch.autograd.Function):
    """Lightweight backward: only computes dX (no weight grads, no X needed).

    Used when LoRA weights don't require grad (e.g. under vmap(grad()) where
    grad() detaches captured parameters). Avoids retaining X, reducing peak memory.
    """

    @staticmethod
    def forward(grad_out, W, A, B, scaling):
        return _lora_w_backward_lite(grad_out, W, A, B, scaling)

    @staticmethod
    def setup_context(ctx, inputs, output):
        pass

    @staticmethod
    def backward(ctx, *grad_outputs):
        raise NotImplementedError("Double backward not supported for LoRA_W")

    @staticmethod
    def vmap(info, in_dims, grad_out, W, A, B, scaling):
        _validate_vmap_dims(in_dims, name="_LoRAWBackwardLite", batched_indices={0})
        grad_out_bdim = in_dims[0]

        grad_out_merged = grad_out.reshape(-1, *grad_out.shape[2:])
        dX = _lora_w_backward_lite(grad_out_merged, W, A, B, scaling)
        dX = dX.reshape(*grad_out.shape[:-1], W.shape[1])
        return dX, grad_out_bdim


class Opaque_LoRA_W(torch.autograd.Function):
    """LoRA for single weight projection with vmap support.

    Computes: output = X @ W.T + X @ A @ B * scaling

    Used for O-projection and other single linear layers.
    """

    @staticmethod
    def forward(X, W, A, B, scaling):
        """Forward pass."""
        X, W, A, B = cast_to_dtype(active_cuda_dtype(X), X, W, A, B)
        out = F.linear(X, W)  # X @ W.T

        if A is not None and B is not None:
            # Out-of-place addmm: same fused BLAS call as addmm_ but doesn't
            # mutate `out`, avoiding autograd version-counter conflicts when
            # both X and A/B require grad (e.g. manual per-sample gradients).
            XA = X @ A
            out = torch.addmm(
                out.reshape(-1, out.shape[-1]),
                XA.reshape(-1, XA.shape[-1]),
                B,
                alpha=scaling,
                beta=1,
            ).reshape(out.shape)

        return out

    @staticmethod
    def setup_context(ctx, inputs, output):
        X, W, A, B, scaling = inputs
        # Under vmap(grad()), grad() detaches captured LoRA weights (requires_grad=False).
        # Skip saving X when weight grads aren't needed — reduces peak memory.
        needs_weight_grads = _needs_lora_weight_grads((A, B))
        if needs_weight_grads:
            ctx.save_for_backward(X, W, A, B)
        else:
            ctx.save_for_backward(W, A, B)
        ctx.needs_weight_grads = needs_weight_grads
        ctx.scaling = scaling
        ctx.compute_dtype = output.dtype

    @staticmethod
    def backward(ctx, grad_out):
        if ctx.needs_weight_grads:
            X, W, A, B = ctx.saved_tensors
            X, A, B = cast_to_dtype(ctx.compute_dtype, X, A, B)
            dA, dB = _LoRAWBackward.apply(grad_out, X, A, B, ctx.scaling)
            ctx.maybe_clear_saved_tensors()
            del X
            W, A, B = cast_to_dtype(ctx.compute_dtype, W, A, B)
            dX = _LoRAWBackwardLite.apply(grad_out, W, A, B, ctx.scaling)
        else:
            W, A, B = cast_to_dtype(ctx.compute_dtype, *ctx.saved_tensors)
            dX = _LoRAWBackwardLite.apply(grad_out, W, A, B, ctx.scaling)
            dA = dB = None
        return dX, None, dA, dB, None

    @staticmethod
    def vmap(info, in_dims, X, W, A, B, scaling):
        """Efficient vmap rule: merge vmap batch into regular batch.

        This is mathematically equivalent to vmap but avoids the overhead
        of custom backward rules. For linear operations, this approach
        gives identical gradients while being just as fast as non-vmapped code.
        """
        _validate_vmap_dims(in_dims, name="Opaque_LoRA_W", batched_indices={0})
        X_bdim = in_dims[0]
        X, W, A, B = cast_to_dtype(active_cuda_dtype(X), X, W, A, B)

        # Merge vmap batch into regular batch
        original_shape = X.shape
        X_merged = X.reshape(-1, *X.shape[2:])

        # Apply LoRA with addmm_ (same as non-vmap forward)
        out = F.linear(X_merged, W)
        if A is not None and B is not None:
            XA = X_merged @ A
            out.reshape(-1, out.shape[-1]).addmm_(
                XA.reshape(-1, XA.shape[-1]), B, alpha=scaling, beta=1
            )

        out = out.reshape(*original_shape[:-1], -1)
        return out, X_bdim


def _lora_qkv_weight_backward_impl(
    grad_Q, grad_K, grad_V, X, Aq, Bq, Sq, Ak, Bk, Sk, Av, Bv, Sv
):
    """Compute fused QKV adapter gradients before releasing the saved input."""
    X_flat = X.reshape(-1, X.shape[-1])
    grad_Q_flat = grad_Q.reshape(-1, grad_Q.shape[-1])
    grad_K_flat = grad_K.reshape(-1, grad_K.shape[-1])
    grad_V_flat = grad_V.reshape(-1, grad_V.shape[-1])

    def _adapter_grads(grad, A, B, scaling):
        if A is None or B is None:
            return None, None
        X_adapter = X_flat.to(A.dtype)
        grad_adapter = grad.to(A.dtype)
        grad_Bt = grad_adapter @ B.t()
        dA = torch.empty_like(A)
        dA.addmm_(X_adapter.t(), grad_Bt, alpha=scaling, beta=0)
        dB = torch.empty_like(B)
        dB.addmm_(A.t() @ X_adapter.t(), grad_adapter, alpha=scaling, beta=0)
        return dA, dB

    dAq, dBq = _adapter_grads(grad_Q_flat, Aq, Bq, Sq)
    dAk, dBk = _adapter_grads(grad_K_flat, Ak, Bk, Sk)
    dAv, dBv = _adapter_grads(grad_V_flat, Av, Bv, Sv)
    return dAq, dBq, dAk, dBk, dAv, dBv


class _LoRAQKVBackward(torch.autograd.Function):
    """Fused QKV adapter-gradient stage wrapped for vmap(grad()) support."""

    @staticmethod
    def forward(grad_Q, grad_K, grad_V, X, Aq, Bq, Sq, Ak, Bk, Sk, Av, Bv, Sv):
        return _lora_qkv_weight_backward_impl(
            grad_Q, grad_K, grad_V, X, Aq, Bq, Sq, Ak, Bk, Sk, Av, Bv, Sv
        )

    @staticmethod
    def setup_context(ctx, inputs, output):
        pass

    @staticmethod
    def backward(ctx, *grad_outputs):
        raise NotImplementedError("Double backward not supported for LoRA_QKV")

    @staticmethod
    def vmap(
        info,
        in_dims,
        grad_Q,
        grad_K,
        grad_V,
        X,
        Aq,
        Bq,
        Sq,
        Ak,
        Bk,
        Sk,
        Av,
        Bv,
        Sv,
    ):
        _validate_vmap_dims(
            in_dims, name="_LoRAQKVBackward", batched_indices={0, 1, 2, 3}
        )

        B_vmap = X.shape[0]
        X_3d = X.reshape(B_vmap, -1, X.shape[-1])
        grad_Q_3d = grad_Q.reshape(B_vmap, -1, grad_Q.shape[-1])
        grad_K_3d = grad_K.reshape(B_vmap, -1, grad_K.shape[-1])
        grad_V_3d = grad_V.reshape(B_vmap, -1, grad_V.shape[-1])

        def _per_sample_lora_grads(grad, A, B, scaling):
            if A is None or B is None:
                return None, None
            X_adapter = X_3d.to(A.dtype)
            grad_adapter = grad.to(A.dtype)
            grad_Bt = grad_adapter @ B.t()
            dA = torch.bmm(X_adapter.transpose(-2, -1), grad_Bt) * scaling
            XA = X_adapter @ A
            dB = torch.bmm(XA.transpose(-2, -1), grad_adapter) * scaling
            return dA, dB

        dAq, dBq = _per_sample_lora_grads(grad_Q_3d, Aq, Bq, Sq)
        dAk, dBk = _per_sample_lora_grads(grad_K_3d, Ak, Bk, Sk)
        dAv, dBv = _per_sample_lora_grads(grad_V_3d, Av, Bv, Sv)

        def _bdim(t):
            return 0 if t is not None else None

        return (
            (dAq, dBq, dAk, dBk, dAv, dBv),
            (_bdim(dAq), _bdim(dBq), _bdim(dAk), _bdim(dBk), _bdim(dAv), _bdim(dBv)),
        )


def _lora_qkv_backward_lite(
    grad_Q,
    grad_K,
    grad_V,
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
    packed_W=None,
    packed_A=None,
    packed_B=None,
):
    """Lightweight QKV backward: only computes dX (no weight grads, no X needed).

    Used when LoRA weights don't require grad (e.g. under vmap(grad()) where
    grad() detaches captured parameters). Avoids retaining X, reducing peak memory.
    """
    batch_shape = grad_Q.shape[:-1]
    grad_Q_flat = grad_Q.reshape(-1, grad_Q.shape[-1])
    grad_K_flat = grad_K.reshape(-1, grad_K.shape[-1])
    grad_V_flat = grad_V.reshape(-1, grad_V.shape[-1])

    grad_packed = torch.cat((grad_Q_flat, grad_K_flat, grad_V_flat), dim=1)
    if packed_W is None:
        packed_W = torch.cat((Wq, Wk, Wv), dim=0)
    dX = torch.mm(grad_packed, packed_W)

    # dX from LoRA contributions
    if packed_A is not None and packed_B is not None:
        dX.addmm_(grad_packed @ packed_B.t(), packed_A.t(), beta=1)
    else:
        if Aq is not None and Bq is not None:
            dX.addmm_(grad_Q_flat @ Bq.t(), Aq.t(), alpha=Sq, beta=1)
        if Ak is not None and Bk is not None:
            dX.addmm_(grad_K_flat @ Bk.t(), Ak.t(), alpha=Sk, beta=1)
        if Av is not None and Bv is not None:
            dX.addmm_(grad_V_flat @ Bv.t(), Av.t(), alpha=Sv, beta=1)

    return dX.reshape(*batch_shape, Wq.shape[1])


class _LoRAQKVBackwardLite(torch.autograd.Function):
    """Lightweight QKV backward: only dX, no weight grads, no X needed."""

    @staticmethod
    def forward(
        grad_Q,
        grad_K,
        grad_V,
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
        packed_W,
        packed_A,
        packed_B,
    ):
        return _lora_qkv_backward_lite(
            grad_Q,
            grad_K,
            grad_V,
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
            packed_W,
            packed_A,
            packed_B,
        )

    @staticmethod
    def setup_context(ctx, inputs, output):
        pass

    @staticmethod
    def backward(ctx, *grad_outputs):
        raise NotImplementedError("Double backward not supported for LoRA_QKV")

    @staticmethod
    def vmap(
        info,
        in_dims,
        grad_Q,
        grad_K,
        grad_V,
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
        packed_W,
        packed_A,
        packed_B,
    ):
        _validate_vmap_dims(
            in_dims, name="_LoRAQKVBackwardLite", batched_indices={0, 1, 2}
        )
        grad_Q_bdim = in_dims[0]

        grad_Q_merged = grad_Q.reshape(-1, *grad_Q.shape[2:])
        grad_K_merged = grad_K.reshape(-1, *grad_K.shape[2:])
        grad_V_merged = grad_V.reshape(-1, *grad_V.shape[2:])

        dX = _lora_qkv_backward_lite(
            grad_Q_merged,
            grad_K_merged,
            grad_V_merged,
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
            packed_W,
            packed_A,
            packed_B,
        )
        dX = dX.reshape(*grad_Q.shape[:-1], Wq.shape[1])
        return dX, grad_Q_bdim


class Opaque_LoRA_QKV(torch.autograd.Function):
    """Fused LoRA for Q, K, V projections with vmap support.

    Computes:
        Q = X @ Wq.T + bq + X @ Aq @ Bq * scaling_q
        K = X @ Wk.T + bk + X @ Ak @ Bk * scaling_k
        V = X @ Wv.T + bv + X @ Av @ Bv * scaling_v
    """

    @staticmethod
    def forward(
        X,
        Wq,
        Aq,
        Bq,
        Sq,
        bq,
        Wk,
        Ak,
        Bk,
        Sk,
        bk,
        Wv,
        Av,
        Bv,
        Sv,
        bv,
        packed_W=None,
        packed_bias=None,
        packed_A=None,
        packed_B=None,
    ):
        """Forward pass for Q, K, V projections."""
        _require_frozen_qkv_biases(bq, bk, bv)
        adapter_packable = (
            packed_A is not None and packed_B is not None
        ) or _adapters_are_packable((Aq, Bq), (Ak, Bk), (Av, Bv))
        (
            X,
            Wq,
            Aq,
            Bq,
            bq,
            Wk,
            Ak,
            Bk,
            bk,
            Wv,
            Av,
            Bv,
            bv,
            packed_W,
            packed_bias,
            packed_A,
            packed_B,
        ) = cast_to_dtype(
            active_cuda_dtype(X),
            X,
            Wq,
            Aq,
            Bq,
            bq,
            Wk,
            Ak,
            Bk,
            bk,
            Wv,
            Av,
            Bv,
            bv,
            packed_W,
            packed_bias,
            packed_A,
            packed_B,
        )
        X_flat = X.reshape(-1, X.shape[-1])
        if packed_W is None:
            packed_W, packed_bias = _pack_qkv_base(Wq, bq, Wk, bk, Wv, bv)
        packed = F.linear(X, packed_W, packed_bias)
        if adapter_packable and (packed_A is None or packed_B is None):
            packed_A, packed_B = _pack_adapters(
                (Aq, Bq, Sq), (Ak, Bk, Sk), (Av, Bv, Sv)
            )
        if packed_A is not None and packed_B is not None:
            packed.reshape(-1, packed.shape[-1]).addmm_(
                X_flat @ packed_A, packed_B, beta=1
            )
        else:
            starts = (0, Wq.shape[0], Wq.shape[0] + Wk.shape[0])
            for start, A, B, scaling in zip(
                starts,
                (Aq, Ak, Av),
                (Bq, Bk, Bv),
                (Sq, Sk, Sv),
                strict=True,
            ):
                if A is not None and B is not None:
                    projection = packed[..., start : start + B.shape[1]]
                    delta = (X_flat @ A @ B).reshape(projection.shape)
                    projection.add_(delta, alpha=scaling)
        return packed.split((Wq.shape[0], Wk.shape[0], Wv.shape[0]), dim=-1)

    @staticmethod
    def setup_context(ctx, inputs, output):
        (
            X,
            Wq,
            Aq,
            Bq,
            Sq,
            bq,
            Wk,
            Ak,
            Bk,
            Sk,
            bk,
            Wv,
            Av,
            Bv,
            Sv,
            bv,
            *packed_inputs,
        ) = inputs
        packed_W, _packed_bias, packed_A, packed_B = (
            *packed_inputs,
            None,
            None,
            None,
            None,
        )[:4]
        if packed_W is None:
            packed_W, _ = _pack_qkv_base(Wq, bq, Wk, bk, Wv, bv)
        if (
            packed_A is None
            and packed_B is None
            and _adapters_are_packable((Aq, Bq), (Ak, Bk), (Av, Bv))
        ):
            packed_A, packed_B = _pack_adapters(
                (Aq, Bq, Sq), (Ak, Bk, Sk), (Av, Bv, Sv)
            )
        # Under vmap(grad()), grad() detaches captured LoRA weights (requires_grad=False).
        # Skip saving X when weight grads aren't needed — reduces peak memory.
        needs_weight_grads = _needs_lora_weight_grads((Aq, Bq), (Ak, Bk), (Av, Bv))
        if needs_weight_grads:
            ctx.save_for_backward(
                X, Wq, Aq, Bq, Wk, Ak, Bk, Wv, Av, Bv, packed_W, packed_A, packed_B
            )
        else:
            ctx.save_for_backward(
                Wq, Aq, Bq, Wk, Ak, Bk, Wv, Av, Bv, packed_W, packed_A, packed_B
            )
        ctx.extra_inputs = len(packed_inputs)
        ctx.needs_weight_grads = needs_weight_grads
        ctx.Sq = Sq
        ctx.Sk = Sk
        ctx.Sv = Sv
        ctx.compute_dtype = output[0].dtype

    @staticmethod
    def backward(ctx, grad_Q, grad_K, grad_V):
        Sq, Sk, Sv = ctx.Sq, ctx.Sk, ctx.Sv

        if ctx.needs_weight_grads:
            X, Wq, Aq, Bq, Wk, Ak, Bk, Wv, Av, Bv, packed_W, packed_A, packed_B = (
                ctx.saved_tensors
            )
            dAq, dBq, dAk, dBk, dAv, dBv = _LoRAQKVBackward.apply(
                grad_Q,
                grad_K,
                grad_V,
                X,
                Aq,
                Bq,
                Sq,
                Ak,
                Bk,
                Sk,
                Av,
                Bv,
                Sv,
            )
            ctx.maybe_clear_saved_tensors()
            del X
        else:
            Wq, Aq, Bq, Wk, Ak, Bk, Wv, Av, Bv, packed_W, packed_A, packed_B = (
                ctx.saved_tensors
            )
            dAq = dBq = dAk = dBk = dAv = dBv = None

        Wq, Aq, Bq, Wk, Ak, Bk, Wv, Av, Bv, packed_W, packed_A, packed_B = (
            cast_to_dtype(
                ctx.compute_dtype,
                Wq,
                Aq,
                Bq,
                Wk,
                Ak,
                Bk,
                Wv,
                Av,
                Bv,
                packed_W,
                packed_A,
                packed_B,
            )
        )
        dX = _LoRAQKVBackwardLite.apply(
            grad_Q,
            grad_K,
            grad_V,
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
            packed_W,
            packed_A,
            packed_B,
        )

        gradients = (
            dX,
            None,
            dAq,
            dBq,
            None,
            None,  # Q: Wq, Aq, Bq, Sq, bq
            None,
            dAk,
            dBk,
            None,
            None,  # K: Wk, Ak, Bk, Sk, bk
            None,
            dAv,
            dBv,
            None,
            None,  # V: Wv, Av, Bv, Sv, bv
        )
        return gradients + (None,) * ctx.extra_inputs

    @staticmethod
    def vmap(
        info,
        in_dims,
        X,
        Wq,
        Aq,
        Bq,
        Sq,
        bq,
        Wk,
        Ak,
        Bk,
        Sk,
        bk,
        Wv,
        Av,
        Bv,
        Sv,
        bv,
        packed_W=None,
        packed_bias=None,
        packed_A=None,
        packed_B=None,
    ):
        """Efficient vmap rule: merge vmap batch into regular batch."""
        _require_frozen_qkv_biases(bq, bk, bv)
        _validate_vmap_dims(in_dims, name="Opaque_LoRA_QKV", batched_indices={0})
        X_bdim = in_dims[0]
        adapter_packable = (
            packed_A is not None and packed_B is not None
        ) or _adapters_are_packable((Aq, Bq), (Ak, Bk), (Av, Bv))
        (
            X,
            Wq,
            Aq,
            Bq,
            bq,
            Wk,
            Ak,
            Bk,
            bk,
            Wv,
            Av,
            Bv,
            bv,
            packed_W,
            packed_bias,
            packed_A,
            packed_B,
        ) = cast_to_dtype(
            active_cuda_dtype(X),
            X,
            Wq,
            Aq,
            Bq,
            bq,
            Wk,
            Ak,
            Bk,
            bk,
            Wv,
            Av,
            Bv,
            bv,
            packed_W,
            packed_bias,
            packed_A,
            packed_B,
        )

        # Merge vmap batch into regular batch
        original_shape = X.shape
        X_merged = X.reshape(-1, *X.shape[2:])

        X_flat = X_merged.reshape(-1, X_merged.shape[-1])
        if packed_W is None:
            packed_W, packed_bias = _pack_qkv_base(Wq, bq, Wk, bk, Wv, bv)
        packed = F.linear(X_merged, packed_W, packed_bias)
        if adapter_packable and (packed_A is None or packed_B is None):
            packed_A, packed_B = _pack_adapters(
                (Aq, Bq, Sq), (Ak, Bk, Sk), (Av, Bv, Sv)
            )
        if packed_A is not None and packed_B is not None:
            packed.reshape(-1, packed.shape[-1]).addmm_(
                X_flat @ packed_A, packed_B, beta=1
            )
        else:
            starts = (0, Wq.shape[0], Wq.shape[0] + Wk.shape[0])
            for start, A, B, scaling in zip(
                starts,
                (Aq, Ak, Av),
                (Bq, Bk, Bv),
                (Sq, Sk, Sv),
                strict=True,
            ):
                if A is not None and B is not None:
                    projection = packed[..., start : start + B.shape[1]]
                    delta = (X_flat @ A @ B).reshape(projection.shape)
                    projection.add_(delta, alpha=scaling)
        Q, K, V = packed.split((Wq.shape[0], Wk.shape[0], Wv.shape[0]), dim=-1)
        Q = Q.reshape(*original_shape[:-1], Wq.shape[0])
        K = K.reshape(*original_shape[:-1], Wk.shape[0])
        V = V.reshape(*original_shape[:-1], Wv.shape[0])

        return (Q, K, V), (X_bdim, X_bdim, X_bdim)


def _lora_adapter_grad_accumulator(A, B):
    if A is None or B is None:
        return None
    dtype = torch.promote_types(A.dtype, B.dtype)
    if dtype in (torch.float16, torch.bfloat16):
        dtype = torch.float32
    A_acc = A.to(dtype)
    B_acc = B.to(dtype)
    return (
        A_acc,
        B_acc,
        torch.zeros_like(A, dtype=dtype),
        torch.zeros_like(B, dtype=dtype),
    )


def _accumulate_lora_adapter_grads(accumulator, input_, grad, scaling):
    if accumulator is None:
        return
    A, B, dA, dB = accumulator
    input_acc = input_.to(A.dtype)
    grad_acc = grad.to(A.dtype)
    grad_Bt = grad_acc @ B.t()
    dA.addmm_(input_acc.t(), grad_Bt, alpha=scaling, beta=1)
    dB.addmm_((input_acc @ A).t(), grad_acc, alpha=scaling, beta=1)


def _finish_lora_adapter_grads(accumulator):
    if accumulator is None:
        return None, None
    _, _, dA, dB = accumulator
    return dA, dB


def _per_sample_lora_adapter_grads(input_, grad, A, B, scaling):
    if A is None or B is None:
        return None, None
    dA = input_.new_zeros((input_.shape[0], input_.shape[-1], A.shape[-1]))
    dB = input_.new_zeros((input_.shape[0], A.shape[-1], grad.shape[-1]))
    for start in range(0, input_.shape[0], _LORA_MLP_GRAD_CHUNK_EXAMPLES):
        stop = min(start + _LORA_MLP_GRAD_CHUNK_EXAMPLES, input_.shape[0])
        input_chunk = input_[start:stop]
        grad_chunk = grad[start:stop]
        grad_Bt = grad_chunk @ B.t()
        dA[start:stop] = torch.bmm(input_chunk.transpose(-2, -1), grad_Bt) * scaling
        dB[start:stop] = (
            torch.bmm((input_chunk @ A).transpose(-2, -1), grad_chunk) * scaling
        )
    return dA, dB


def _lora_mlp_weight_backward_impl(
    grad_out,
    X,
    Ag,
    Bg,
    Sg,
    Au,
    Bu,
    Su,
    Wd,
    Ad,
    Bd,
    Sd,
    gate,
    up,
    activation_type,
):
    """Compute fused MLP adapter gradients before releasing the saved input."""
    X_flat = X.reshape(-1, X.shape[-1])
    grad_out_flat = grad_out.reshape(-1, grad_out.shape[-1])
    gate_flat = gate.reshape(-1, gate.shape[-1])
    up_flat = up.reshape(-1, up.shape[-1])
    act_backward_fused = _ACTIVATION_BACKWARD_FUSED[activation_type]
    gate_accumulator = _lora_adapter_grad_accumulator(Ag, Bg)
    up_accumulator = _lora_adapter_grad_accumulator(Au, Bu)
    down_accumulator = _lora_adapter_grad_accumulator(Ad, Bd)

    for start in range(0, X_flat.shape[0], _LORA_MLP_GRAD_CHUNK_ROWS):
        stop = min(start + _LORA_MLP_GRAD_CHUNK_ROWS, X_flat.shape[0])
        X_chunk = X_flat[start:stop]
        grad_out_chunk = grad_out_flat[start:stop]
        dh = grad_out_chunk @ Wd
        if Ad is not None and Bd is not None:
            dh.addmm_(grad_out_chunk @ Bd.t(), Ad.t(), alpha=Sd, beta=1)
        h, dgate, dup = act_backward_fused(
            dh, gate_flat[start:stop], up_flat[start:stop]
        )
        _accumulate_lora_adapter_grads(down_accumulator, h, grad_out_chunk, Sd)
        _accumulate_lora_adapter_grads(gate_accumulator, X_chunk, dgate, Sg)
        _accumulate_lora_adapter_grads(up_accumulator, X_chunk, dup, Su)

    dAg, dBg = _finish_lora_adapter_grads(gate_accumulator)
    dAu, dBu = _finish_lora_adapter_grads(up_accumulator)
    dAd, dBd = _finish_lora_adapter_grads(down_accumulator)
    return (
        gate_flat.reshape(gate.shape),
        up_flat.reshape(up.shape),
        dAg,
        dBg,
        dAu,
        dBu,
        dAd,
        dBd,
    )


def _adjacent_mlp_grad_pack(dgate, dup):
    """Recover the packed gate/up gradient when both tensors are adjacent views."""
    dgate_flat = dgate.reshape(-1, dgate.shape[-1])
    dup_flat = dup.reshape(-1, dup.shape[-1])
    intermediate = dgate_flat.shape[1]
    if (
        dgate_flat.shape == dup_flat.shape
        and dgate_flat.stride() == (2 * intermediate, 1)
        and dup_flat.stride() == dgate_flat.stride()
        and dup_flat.storage_offset() == dgate_flat.storage_offset() + intermediate
        and dgate_flat.untyped_storage().data_ptr()
        == dup_flat.untyped_storage().data_ptr()
    ):
        return dgate_flat.as_strided(
            (dgate_flat.shape[0], 2 * intermediate),
            (2 * intermediate, 1),
            dgate_flat.storage_offset(),
        )
    return torch.cat((dgate_flat, dup_flat), dim=1)


def _lora_mlp_input_backward_impl(
    dgate,
    dup,
    Wg,
    Ag,
    Bg,
    Sg,
    Wu,
    Au,
    Bu,
    Su,
    packed_W=None,
    packed_A=None,
    packed_B=None,
):
    """Compute the MLP input gradient without retaining the forward input."""
    batch_shape = dgate.shape[:-1]
    dgate_flat = dgate.reshape(-1, dgate.shape[-1])
    dup_flat = dup.reshape(-1, dup.shape[-1])

    grad_packed = _adjacent_mlp_grad_pack(dgate_flat, dup_flat)
    if packed_W is None:
        packed_W = torch.cat((Wg, Wu), dim=0)
    dX = torch.mm(grad_packed, packed_W)
    if packed_A is not None and packed_B is not None:
        dX.addmm_(grad_packed @ packed_B.t(), packed_A.t(), beta=1)
    else:
        if Ag is not None and Bg is not None:
            dX.addmm_(dgate_flat @ Bg.t(), Ag.t(), alpha=Sg, beta=1)
        if Au is not None and Bu is not None:
            dX.addmm_(dup_flat @ Bu.t(), Au.t(), alpha=Su, beta=1)
    return dX.reshape(*batch_shape, Wg.shape[1])


class _LoRAMLPBackward(torch.autograd.Function):
    """Fused MLP adapter-gradient stage wrapped for vmap(grad()) support."""

    @staticmethod
    def forward(
        grad_out,
        X,
        Ag,
        Bg,
        Sg,
        Au,
        Bu,
        Su,
        Wd,
        Ad,
        Bd,
        Sd,
        gate,
        up,
        activation_type,
    ):
        return _lora_mlp_weight_backward_impl(
            grad_out,
            X,
            Ag,
            Bg,
            Sg,
            Au,
            Bu,
            Su,
            Wd,
            Ad,
            Bd,
            Sd,
            gate,
            up,
            activation_type,
        )

    @staticmethod
    def setup_context(ctx, inputs, output):
        pass

    @staticmethod
    def backward(ctx, *grad_outputs):
        raise NotImplementedError("Double backward not supported for LoRA_MLP")

    @staticmethod
    def vmap(
        info,
        in_dims,
        grad_out,
        X,
        Ag,
        Bg,
        Sg,
        Au,
        Bu,
        Su,
        Wd,
        Ad,
        Bd,
        Sd,
        gate,
        up,
        activation_type,
    ):
        _validate_vmap_dims(
            in_dims, name="_LoRAMLPBackward", batched_indices={0, 1, 12, 13}
        )

        B_vmap = X.shape[0]
        hidden_dim = X.shape[-1]
        inter_dim = gate.shape[-1]
        grad_out_merged = grad_out.reshape(-1, *grad_out.shape[2:])
        gate_merged = gate.reshape(-1, *gate.shape[2:])
        up_merged = up.reshape(-1, *up.shape[2:])
        grad_out_flat = grad_out_merged.reshape(-1, grad_out.shape[-1])
        gate_flat = gate_merged.reshape(-1, inter_dim)
        up_flat = up_merged.reshape(-1, inter_dim)

        dh = grad_out_flat @ Wd
        if Ad is not None and Bd is not None:
            dh.addmm_(grad_out_flat @ Bd.t(), Ad.t(), alpha=Sd, beta=1)

        act_backward_fused = _ACTIVATION_BACKWARD_FUSED[activation_type]
        h, dgate, dup = act_backward_fused(dh, gate_flat, up_flat)

        X_3d = X.reshape(B_vmap, -1, hidden_dim)
        grad_out_3d = grad_out.reshape(B_vmap, -1, grad_out.shape[-1])
        h_3d = h.reshape(B_vmap, -1, inter_dim)
        dgate_3d = dgate.reshape(B_vmap, -1, inter_dim)
        dup_3d = dup.reshape(B_vmap, -1, inter_dim)

        dAd, dBd = _per_sample_lora_adapter_grads(h_3d, grad_out_3d, Ad, Bd, Sd)
        dAg, dBg = _per_sample_lora_adapter_grads(X_3d, dgate_3d, Ag, Bg, Sg)
        dAu, dBu = _per_sample_lora_adapter_grads(X_3d, dup_3d, Au, Bu, Su)

        def _bdim(t):
            return 0 if t is not None else None

        return (
            (
                dgate.reshape(gate.shape),
                dup.reshape(up.shape),
                dAg,
                dBg,
                dAu,
                dBu,
                dAd,
                dBd,
            ),
            (
                0,
                0,
                _bdim(dAg),
                _bdim(dBg),
                _bdim(dAu),
                _bdim(dBu),
                _bdim(dAd),
                _bdim(dBd),
            ),
        )


class _LoRAMLPInputBackward(torch.autograd.Function):
    """MLP input-gradient stage wrapped for vmap(grad()) support."""

    @staticmethod
    def forward(
        dgate,
        dup,
        Wg,
        Ag,
        Bg,
        Sg,
        Wu,
        Au,
        Bu,
        Su,
        packed_W,
        packed_A,
        packed_B,
    ):
        return _lora_mlp_input_backward_impl(
            dgate,
            dup,
            Wg,
            Ag,
            Bg,
            Sg,
            Wu,
            Au,
            Bu,
            Su,
            packed_W,
            packed_A,
            packed_B,
        )

    @staticmethod
    def setup_context(ctx, inputs, output):
        pass

    @staticmethod
    def backward(ctx, *grad_outputs):
        raise NotImplementedError("Double backward not supported for LoRA_MLP")

    @staticmethod
    def vmap(
        info,
        in_dims,
        dgate,
        dup,
        Wg,
        Ag,
        Bg,
        Sg,
        Wu,
        Au,
        Bu,
        Su,
        packed_W,
        packed_A,
        packed_B,
    ):
        _validate_vmap_dims(
            in_dims, name="_LoRAMLPInputBackward", batched_indices={0, 1}
        )
        dgate_merged = dgate.reshape(-1, *dgate.shape[2:])
        dup_merged = dup.reshape(-1, *dup.shape[2:])
        dX = _lora_mlp_input_backward_impl(
            dgate_merged,
            dup_merged,
            Wg,
            Ag,
            Bg,
            Sg,
            Wu,
            Au,
            Bu,
            Su,
            packed_W,
            packed_A,
            packed_B,
        )
        return dX.reshape(*dgate.shape[:-1], Wg.shape[1]), 0


def _lora_mlp_backward_lite(
    grad_out,
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
    gate,
    up,
    activation_type,
    packed_W=None,
    packed_A=None,
    packed_B=None,
):
    """Lightweight MLP backward: only computes dX (no weight grads, no X needed).

    Still needs gate/up for activation backward (recompute h, compute dgate/dup).
    """
    batch_shape = grad_out.shape[:-1]
    grad_out_flat = grad_out.reshape(-1, grad_out.shape[-1])
    gate_flat = gate.reshape(-1, gate.shape[-1])
    up_flat = up.reshape(-1, up.shape[-1])

    # Backward through down projection
    dh = grad_out_flat @ Wd
    if Ad is not None and Bd is not None:
        dh.addmm_(grad_out_flat @ Bd.t(), Ad.t(), alpha=Sd, beta=1)

    # Fused backward: recompute h, overwrite gate→dgate, up→dup
    act_backward_fused = _ACTIVATION_BACKWARD_FUSED[activation_type]
    _h, dgate, dup = act_backward_fused(dh, gate_flat, up_flat)

    # dX: fresh allocation (no X buffer to reuse)
    dX = _lora_mlp_input_backward_impl(
        dgate,
        dup,
        Wg,
        Ag,
        Bg,
        Sg,
        Wu,
        Au,
        Bu,
        Su,
        packed_W,
        packed_A,
        packed_B,
    )

    return dX.reshape(*batch_shape, Wg.shape[1])


class _LoRAMLPBackwardLite(torch.autograd.Function):
    """Lightweight MLP backward: only dX, no weight grads, no X needed."""

    @staticmethod
    def forward(
        grad_out,
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
        gate,
        up,
        activation_type,
        packed_W,
        packed_A,
        packed_B,
    ):
        return _lora_mlp_backward_lite(
            grad_out,
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
            gate,
            up,
            activation_type,
            packed_W,
            packed_A,
            packed_B,
        )

    @staticmethod
    def setup_context(ctx, inputs, output):
        pass

    @staticmethod
    def backward(ctx, *grad_outputs):
        raise NotImplementedError("Double backward not supported for LoRA_MLP")

    @staticmethod
    def vmap(
        info,
        in_dims,
        grad_out,
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
        gate,
        up,
        activation_type,
        packed_W,
        packed_A,
        packed_B,
    ):
        if in_dims[13] is None and in_dims[14] is None:
            raise NotImplementedError("Double backward not supported for LoRA_MLP")
        _validate_vmap_dims(
            in_dims, name="_LoRAMLPBackwardLite", batched_indices={0, 13, 14}
        )
        grad_out_bdim = in_dims[0]

        grad_out_merged = grad_out.reshape(-1, *grad_out.shape[2:])
        gate_merged = gate.reshape(-1, *gate.shape[2:])
        up_merged = up.reshape(-1, *up.shape[2:])

        dX = _lora_mlp_backward_lite(
            grad_out_merged,
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
            gate_merged,
            up_merged,
            activation_type,
            packed_W,
            packed_A,
            packed_B,
        )
        dX = dX.reshape(*grad_out.shape[:-1], Wg.shape[1])
        return dX, grad_out_bdim


class Opaque_LoRA_MLP(torch.autograd.Function):
    """Fused LoRA for MLP (gate, up, down) with configurable GLU activation.

    Uses Triton kernels for the activation (matching Unsloth's callback pattern).
    Supports SwiGLU (default), GeGLU exact, and GeGLU approx via activation_type.

    Computes:
        gate = X @ Wg.T + X @ Ag @ Bg * Sg
        up = X @ Wu.T + X @ Au @ Bu * Su
        h = activation(gate) * up  # via Triton kernel
        out = h @ Wd.T + h @ Ad @ Bd * Sd
    """

    @staticmethod
    def forward(
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
        activation_type,
        packed_W=None,
        packed_A=None,
        packed_B=None,
    ):
        """Forward pass for MLP with configurable GLU activation."""
        adapter_packable = (
            packed_A is not None and packed_B is not None
        ) or _adapters_are_packable((Ag, Bg), (Au, Bu))
        X, Wg, Ag, Bg, Wu, Au, Bu, Wd, Ad, Bd, packed_W, packed_A, packed_B = (
            cast_to_dtype(
                active_cuda_dtype(X),
                X,
                Wg,
                Ag,
                Bg,
                Wu,
                Au,
                Bu,
                Wd,
                Ad,
                Bd,
                packed_W,
                packed_A,
                packed_B,
            )
        )
        X_flat = X.reshape(-1, X.shape[-1])
        if packed_W is None:
            packed_W = torch.cat((Wg, Wu), dim=0)
        gate_up = F.linear(X, packed_W)
        if adapter_packable and (packed_A is None or packed_B is None):
            packed_A, packed_B = _pack_adapters((Ag, Bg, Sg), (Au, Bu, Su))
        if packed_A is not None and packed_B is not None:
            gate_up.reshape(-1, gate_up.shape[-1]).addmm_(
                X_flat @ packed_A, packed_B, beta=1
            )
        else:
            intermediate = Wg.shape[0]
            if Ag is not None and Bg is not None:
                gate = gate_up[..., :intermediate]
                gate.add_((X_flat @ Ag @ Bg).reshape(gate.shape), alpha=Sg)
            if Au is not None and Bu is not None:
                up = gate_up[..., intermediate:]
                up.add_((X_flat @ Au @ Bu).reshape(up.shape), alpha=Su)
        gate, up = gate_up.split((Wg.shape[0], Wu.shape[0]), dim=-1)

        # Use Triton kernel for activation (Unsloth callback pattern)
        act_forward = _ACTIVATION_FORWARD[activation_type]
        h = act_forward(gate, up)

        out = F.linear(h, Wd)
        if Ad is not None and Bd is not None:
            h_flat = h.reshape(-1, h.shape[-1])
            out.reshape(-1, out.shape[-1]).addmm_(h_flat @ Ad, Bd, alpha=Sd, beta=1)

        return out, gate, up, h

    @staticmethod
    def setup_context(ctx, inputs, output):
        (
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
            activation_type,
            *packed_inputs,
        ) = inputs
        packed_W, packed_A, packed_B = (*packed_inputs, None, None, None)[:3]
        if packed_W is None:
            packed_W = torch.cat((Wg, Wu), dim=0)
        if (
            packed_A is None
            and packed_B is None
            and _adapters_are_packable((Ag, Bg), (Au, Bu))
        ):
            packed_A, packed_B = _pack_adapters((Ag, Bg, Sg), (Au, Bu, Su))
        # Recompute the intermediate-width gate/up tensors in backward. Saving
        # the much narrower input is essential for long-sequence DP vmap batches.
        # Under vmap(grad()), grad() detaches captured LoRA weights (requires_grad=False).
        needs_weight_grads = _needs_lora_weight_grads((Ag, Bg), (Au, Bu), (Ad, Bd))
        ctx.save_for_backward(
            X, Wg, Ag, Bg, Wu, Au, Bu, Wd, Ad, Bd, packed_W, packed_A, packed_B
        )
        ctx.extra_inputs = len(packed_inputs)
        ctx.needs_weight_grads = needs_weight_grads
        ctx.Sg = Sg
        ctx.Su = Su
        ctx.Sd = Sd
        ctx.activation_type = activation_type
        ctx.compute_dtype = output[0].dtype

    @staticmethod
    def backward(ctx, grad_out, grad_gate, grad_up, grad_h):
        if getattr(ctx, "_opaque_backward_done", False):
            raise NotImplementedError("Repeated backward not supported for LoRA_MLP")
        ctx._opaque_backward_done = True
        Sg, Su, Sd = ctx.Sg, ctx.Su, ctx.Sd
        saved_tensors = ctx.saved_tensors
        adapter_dtypes = tuple(
            tensor.dtype if tensor is not None else None
            for tensor in (
                saved_tensors[2],
                saved_tensors[3],
                saved_tensors[5],
                saved_tensors[6],
                saved_tensors[8],
                saved_tensors[9],
            )
        )
        X, Wg, Ag, Bg, Wu, Au, Bu, Wd, Ad, Bd, packed_W, packed_A, packed_B = (
            cast_to_dtype(ctx.compute_dtype, *saved_tensors)
        )
        X_flat = X.reshape(-1, X.shape[-1])
        gate_up = F.linear(X, packed_W)
        if packed_A is not None and packed_B is not None:
            gate_up.reshape(-1, gate_up.shape[-1]).addmm_(
                X_flat @ packed_A, packed_B, beta=1
            )
        else:
            intermediate = Wg.shape[0]
            if Ag is not None and Bg is not None:
                gate = gate_up[..., :intermediate]
                gate.add_((X_flat @ Ag @ Bg).reshape(gate.shape), alpha=Sg)
            if Au is not None and Bu is not None:
                up = gate_up[..., intermediate:]
                up.add_((X_flat @ Au @ Bu).reshape(up.shape), alpha=Su)
        gate, up = gate_up.split((Wg.shape[0], Wu.shape[0]), dim=-1)

        if ctx.needs_weight_grads:
            dgate, dup, dAg, dBg, dAu, dBu, dAd, dBd = _LoRAMLPBackward.apply(
                grad_out,
                X,
                Ag,
                Bg,
                Sg,
                Au,
                Bu,
                Su,
                Wd,
                Ad,
                Bd,
                Sd,
                gate,
                up,
                ctx.activation_type,
            )
            dAg, dBg, dAu, dBu, dAd, dBd = (
                grad.to(dtype) if grad is not None and grad.dtype != dtype else grad
                for grad, dtype in zip(
                    (dAg, dBg, dAu, dBu, dAd, dBd), adapter_dtypes, strict=True
                )
            )
            ctx.maybe_clear_saved_tensors()
            del X, X_flat, Wd, Ad, Bd, gate, up
            Wg, Ag, Bg, Wu, Au, Bu, packed_W, packed_A, packed_B = cast_to_dtype(
                ctx.compute_dtype,
                Wg,
                Ag,
                Bg,
                Wu,
                Au,
                Bu,
                packed_W,
                packed_A,
                packed_B,
            )
            dX = _LoRAMLPInputBackward.apply(
                dgate,
                dup,
                Wg,
                Ag,
                Bg,
                Sg,
                Wu,
                Au,
                Bu,
                Su,
                packed_W,
                packed_A,
                packed_B,
            )
        else:
            ctx.maybe_clear_saved_tensors()
            del X, X_flat
            dX = _LoRAMLPBackwardLite.apply(
                grad_out,
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
                gate,
                up,
                ctx.activation_type,
                packed_W,
                packed_A,
                packed_B,
            )
            dAg = dBg = dAu = dBu = dAd = dBd = None

        gradients = (
            dX,
            None,
            dAg,
            dBg,
            None,  # gate
            None,
            dAu,
            dBu,
            None,  # up
            None,
            dAd,
            dBd,
            None,  # down
            None,  # activation_type
        )
        return gradients + (None,) * ctx.extra_inputs

    @staticmethod
    def vmap(
        info,
        in_dims,
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
        activation_type,
        packed_W=None,
        packed_A=None,
        packed_B=None,
    ):
        """Efficient vmap rule: merge vmap batch into regular batch.

        The staged backward Functions are dispatched from Opaque_LoRA_MLP.backward().
        No autograd graph is needed here, so addmm_ and direct Triton activation
        kernels remain safe.
        """
        _validate_vmap_dims(in_dims, name="Opaque_LoRA_MLP", batched_indices={0})
        X_bdim = in_dims[0]
        adapter_packable = (
            packed_A is not None and packed_B is not None
        ) or _adapters_are_packable((Ag, Bg), (Au, Bu))
        X, Wg, Ag, Bg, Wu, Au, Bu, Wd, Ad, Bd, packed_W, packed_A, packed_B = (
            cast_to_dtype(
                active_cuda_dtype(X),
                X,
                Wg,
                Ag,
                Bg,
                Wu,
                Au,
                Bu,
                Wd,
                Ad,
                Bd,
                packed_W,
                packed_A,
                packed_B,
            )
        )

        # Merge vmap batch into regular batch
        original_shape = X.shape
        X_merged = X.reshape(-1, *X.shape[2:])
        X_flat = X_merged.reshape(-1, X_merged.shape[-1])

        # Apply MLP with addmm_ + direct Triton activation (no autograd needed)
        if packed_W is None:
            packed_W = torch.cat((Wg, Wu), dim=0)
        gate_up = F.linear(X_merged, packed_W)
        if adapter_packable and (packed_A is None or packed_B is None):
            packed_A, packed_B = _pack_adapters((Ag, Bg, Sg), (Au, Bu, Su))
        if packed_A is not None and packed_B is not None:
            gate_up.reshape(-1, gate_up.shape[-1]).addmm_(
                X_flat @ packed_A, packed_B, beta=1
            )
        else:
            intermediate = Wg.shape[0]
            if Ag is not None and Bg is not None:
                gate = gate_up[..., :intermediate]
                gate.add_((X_flat @ Ag @ Bg).reshape(gate.shape), alpha=Sg)
            if Au is not None and Bu is not None:
                up = gate_up[..., intermediate:]
                up.add_((X_flat @ Au @ Bu).reshape(up.shape), alpha=Su)
        gate, up = gate_up.split((Wg.shape[0], Wu.shape[0]), dim=-1)

        # Direct Triton activation kernel (safe: backward uses the staged Functions)
        act_forward = _ACTIVATION_FORWARD[activation_type]
        h = act_forward(gate, up)

        out = F.linear(h, Wd)
        if Ad is not None and Bd is not None:
            h_flat = h.reshape(-1, h.shape[-1])
            out.reshape(-1, out.shape[-1]).addmm_(h_flat @ Ad, Bd, alpha=Sd, beta=1)

        # Reshape back
        out = out.reshape(*original_shape[:-1], -1)
        gate = gate.reshape(*original_shape[:-1], -1)
        up = up.reshape(*original_shape[:-1], -1)
        h = h.reshape(*original_shape[:-1], -1)

        return (out, gate, up, h), (X_bdim, X_bdim, X_bdim, X_bdim)


# ============================================================================
# Convenience wrappers
# ============================================================================


def opaque_lora_w(X, W, A, B, scaling):
    """Apply LoRA linear projection with vmap support.

    Args:
        X: Input tensor (batch, seq_len, hidden_dim)
        W: Base weight (out_features, in_features), frozen
        A: LoRA A weight (in_features, rank)
        B: LoRA B weight (rank, out_features)
        scaling: LoRA scaling (lora_alpha / lora_r)

    Returns:
        Output tensor (batch, seq_len, out_features)
    """
    ensure_cuda_tensors(X, W, A, B, fn_name="opaque_lora_w")
    (X,) = follow_autocast(X)
    return Opaque_LoRA_W.apply(X, W, A, B, scaling)


def opaque_lora_qkv(
    X, Wq, Aq, Bq, Sq, Wk, Ak, Bk, Sk, Wv, Av, Bv, Sv, bq=None, bk=None, bv=None
):
    """Apply LoRA to Q, K, V projections with vmap support.

    Args:
        X: Input activations.
        Wq: Frozen Q-projection weight.
        Aq: Q-projection LoRA A weight.
        Bq: Q-projection LoRA B weight.
        Sq: Q-projection LoRA scaling factor.
        Wk: Frozen K-projection weight.
        Ak: K-projection LoRA A weight.
        Bk: K-projection LoRA B weight.
        Sk: K-projection LoRA scaling factor.
        Wv: Frozen V-projection weight.
        Av: V-projection LoRA A weight.
        Bv: V-projection LoRA B weight.
        Sv: V-projection LoRA scaling factor.
        bq: Optional frozen Q-projection base bias.
        bk: Optional frozen K-projection base bias.
        bv: Optional frozen V-projection base bias.

    Returns:
        Tuple of (Q, K, V)
    """
    ensure_cuda_tensors(
        X,
        Wq,
        Aq,
        Bq,
        bq,
        Wk,
        Ak,
        Bk,
        bk,
        Wv,
        Av,
        Bv,
        bv,
        fn_name="opaque_lora_qkv",
    )
    (X,) = follow_autocast(X)
    return Opaque_LoRA_QKV.apply(
        X, Wq, Aq, Bq, Sq, bq, Wk, Ak, Bk, Sk, bk, Wv, Av, Bv, Sv, bv
    )


def opaque_lora_mlp(
    X, Wg, Ag, Bg, Sg, Wu, Au, Bu, Su, Wd, Ad, Bd, Sd, activation="swiglu"
):
    """Apply LoRA MLP with configurable GLU activation and vmap support.

    Args:
        X: MLP input activations.
        Wg: Base gate-projection weight.
        Ag: Gate-projection LoRA down-projection weight.
        Bg: Gate-projection LoRA up-projection weight.
        Sg: Gate-projection LoRA scaling factor.
        Wu: Base up-projection weight.
        Au: Up-projection LoRA down-projection weight.
        Bu: Up-projection LoRA up-projection weight.
        Su: Up-projection LoRA scaling factor.
        Wd: Base down-projection weight.
        Ad: Down-projection LoRA down-projection weight.
        Bd: Down-projection LoRA up-projection weight.
        Sd: Down-projection LoRA scaling factor.
        activation: Activation type - "swiglu" (default), "geglu_exact", or "geglu_approx".

    Returns:
        Output tensor (batch, seq_len, hidden_dim)
    """
    ensure_cuda_tensors(
        X,
        Wg,
        Ag,
        Bg,
        Wu,
        Au,
        Bu,
        Wd,
        Ad,
        Bd,
        fn_name="opaque_lora_mlp",
    )
    (X,) = follow_autocast(X)
    if isinstance(activation, str):
        activation_type = _ACTIVATION_NAMES[activation]
    else:
        activation_type = activation
    result = Opaque_LoRA_MLP.apply(
        X, Wg, Ag, Bg, Sg, Wu, Au, Bu, Su, Wd, Ad, Bd, Sd, activation_type
    )
    return result[0]  # Return only the output, not intermediate tensors
