# Copyright (c) 2025 Opaque Authors
# Copyright (c) 2024 LinkedIn Corporation (Liger Kernel)
# SPDX-License-Identifier: Apache-2.0 AND BSD-2-Clause
#
# Triton RMSNorm kernels derive from the Liger Kernel project (BSD-2-Clause,
# Copyright LinkedIn Corporation), which incorporated prior Unsloth Apache-2.0
# RMSNorm code. See:
# https://github.com/linkedin/Liger-Kernel/blob/main/src/liger_kernel/ops/rms_norm.py
# See ./../../../../../../NOTICE in the repository root.
"""RMSNorm Triton kernel with vmap(grad(...)) support (DP-SGD)."""

from __future__ import annotations

import math

import torch
import triton
import triton.language as tl

from ._utils import calculate_settings, follow_autocast, torch_gpu_device, triton_cast

try:
    _tv = tuple(int(p) for p in triton.__version__.split(".")[:3] if p.isdigit())
    if _tv >= (3, 0, 0):
        try:
            from triton.language.extra.libdevice import rsqrt
        except ModuleNotFoundError:
            from triton.language.extra.cuda.libdevice import rsqrt
    else:
        raise ImportError
except (ImportError, ValueError):
    rsqrt = tl.math.rsqrt

_STR_TO_CASTING = {
    "llama": 0,
    "gemma": 1,
    "none": -1,
}

_TORCH_TO_TRITON_DTYPES = {
    torch.float16: tl.float16,
    torch.bfloat16: tl.bfloat16,
    torch.float32: tl.float32,
}


def _casting_mode_int(casting_mode: str | int) -> int:
    if isinstance(casting_mode, int):
        if casting_mode not in _STR_TO_CASTING.values():
            raise ValueError(f"Invalid casting_mode int: {casting_mode}")
        return casting_mode
    if casting_mode not in _STR_TO_CASTING:
        raise ValueError(f"Invalid casting_mode: {casting_mode}")
    return _STR_TO_CASTING[casting_mode]


@triton.jit
def _rms_norm_forward_kernel(
    Y_ptr,
    Y_row_stride,
    X_ptr,
    X_row_stride,
    W_ptr,
    W_row_stride,
    RSTD_ptr,
    RSTD_row_stride,
    n_cols,
    eps,
    offset,
    casting_mode: tl.constexpr,
    elementwise_affine: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    row_idx = tl.program_id(0)
    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < n_cols

    # int64 stride math: at vmap-mb=1024, seq=1024, hidden=4096 the row offset
    # reaches 1M * 4096 ≈ 4.3e9 elements and overflows int32 → CUDA IMA. Mirrors
    # the ``cross_entropy.py`` pattern for the same reason.
    y_base = Y_ptr + row_idx * triton_cast(Y_row_stride, tl.int64)
    x_base = X_ptr + row_idx * triton_cast(X_row_stride, tl.int64)
    rstd_base = RSTD_ptr + row_idx * triton_cast(RSTD_row_stride, tl.int64)

    X_row = tl.load(x_base + col_offsets, mask=mask, other=0)
    X_row_dtype = X_row.dtype
    if elementwise_affine:
        W_row = tl.load(W_ptr + col_offsets, mask=mask, other=0)

    if casting_mode == 0:
        X_row = X_row.to(tl.float32)

    if casting_mode == 1:
        if elementwise_affine:
            W_row = W_row.to(tl.float32)
        X_row = X_row.to(tl.float32)

    if casting_mode == -1:
        eps = eps.to(X_row_dtype)
        offset = offset.to(X_row_dtype)

    mean_square = tl.sum(X_row * X_row, axis=0) / n_cols
    row_rstd = rsqrt(mean_square + eps)
    tl.store(rstd_base, row_rstd)

    X_row = X_row * row_rstd

    if casting_mode == 0:
        X_row = X_row.to(X_row_dtype)

    if elementwise_affine:
        Y_row = X_row * (offset + W_row)
    else:
        Y_row = X_row

    if casting_mode == 1:
        Y_row = Y_row.to(X_row_dtype)

    tl.store(y_base + col_offsets, Y_row, mask=mask)


@triton.jit
def _rms_norm_forward_block_kernel(
    Y_ptr,
    Y_row_stride,
    X_ptr,
    X_row_stride,
    W_ptr,
    W_row_stride,
    RSTD_ptr,
    RSTD_row_stride,
    n_rows,
    n_cols,
    eps,
    offset,
    rows_per_program,
    casting_mode: tl.constexpr,
    elementwise_affine: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """Block variant of the forward: one program per SM, each looping
    ``rows_per_program`` rows — mirrors :func:`_rms_norm_backward_kernel`.

    Used for the small-hidden-dim + many-rows regime (e.g. Mellum-2.0 q_norm /
    k_norm at head_dim=128 under vmapped DP-SGD) where the per-row kernel would
    launch one tiny program per row.  The per-row body is identical to
    :func:`_rms_norm_forward_kernel`; only the grid/loop differs.  The shared
    weight is loaded once (loop-invariant)."""
    row_block_id = tl.program_id(0)
    row_start = row_block_id * rows_per_program
    row_end = tl.minimum((row_block_id + 1) * rows_per_program, n_rows)
    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < n_cols

    if elementwise_affine:
        W_row = tl.load(W_ptr + col_offsets, mask=mask, other=0)
        if casting_mode == 1:
            W_row = W_row.to(tl.float32)

    # int64 stride math — same int32-overflow guard as the per-row kernel.
    y_stride64 = triton_cast(Y_row_stride, tl.int64)
    x_stride64 = triton_cast(X_row_stride, tl.int64)
    rstd_stride64 = triton_cast(RSTD_row_stride, tl.int64)

    for row_idx in range(row_start, row_end):
        y_base = Y_ptr + row_idx * y_stride64
        x_base = X_ptr + row_idx * x_stride64
        rstd_base = RSTD_ptr + row_idx * rstd_stride64

        X_row = tl.load(x_base + col_offsets, mask=mask, other=0)
        X_row_dtype = X_row.dtype

        if casting_mode == 0:
            X_row = X_row.to(tl.float32)
        if casting_mode == 1:
            X_row = X_row.to(tl.float32)
        if casting_mode == -1:
            eps_r = eps.to(X_row_dtype)
            offset_r = offset.to(X_row_dtype)
        else:
            eps_r = eps
            offset_r = offset

        mean_square = tl.sum(X_row * X_row, axis=0) / n_cols
        row_rstd = rsqrt(mean_square + eps_r)
        tl.store(rstd_base, row_rstd)

        X_row = X_row * row_rstd

        if casting_mode == 0:
            X_row = X_row.to(X_row_dtype)

        if elementwise_affine:
            Y_row = X_row * (offset_r + W_row)
        else:
            Y_row = X_row

        if casting_mode == 1:
            Y_row = Y_row.to(X_row_dtype)

        tl.store(y_base + col_offsets, Y_row, mask=mask)


@triton.jit
def _rms_norm_backward_kernel(
    dY_ptr,
    dY_row_stride,
    dX_ptr,
    dX_row_stride,
    X_ptr,
    X_row_stride,
    X_dtype: tl.constexpr,
    W_ptr,
    W_row_stride,
    RSTD_ptr,
    RSTD_row_stride,
    dW_ptr,
    dW_row_stride,
    n_rows,
    n_cols,
    offset,
    rows_per_program,
    casting_mode: tl.constexpr,
    elementwise_affine: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    row_block_id = tl.program_id(0)
    row_start = row_block_id * rows_per_program
    row_end = tl.minimum((row_block_id + 1) * rows_per_program, n_rows)
    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < n_cols

    if elementwise_affine:
        dW_row = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
        W_row = tl.load(W_ptr + col_offsets, mask=mask, other=0.0)
        W_row = W_row + offset

    # int64 stride math — same int32-overflow guard as the forward kernel.
    dy_stride64 = triton_cast(dY_row_stride, tl.int64)
    dx_stride64 = triton_cast(dX_row_stride, tl.int64)
    x_stride64 = triton_cast(X_row_stride, tl.int64)
    rstd_stride64 = triton_cast(RSTD_row_stride, tl.int64)

    for row_idx in range(row_start, row_end):
        dy_base = dY_ptr + row_idx * dy_stride64
        dx_base = dX_ptr + row_idx * dx_stride64
        x_base = X_ptr + row_idx * x_stride64
        rstd_base = RSTD_ptr + row_idx * rstd_stride64

        dY_row = tl.load(dy_base + col_offsets, mask=mask, other=0.0)
        X_row = tl.load(x_base + col_offsets, mask=mask, other=0.0)
        rstd_row = tl.load(rstd_base)

        X_row = X_row.to(tl.float32)

        if casting_mode == 0:
            if elementwise_affine:
                m = (dY_row * W_row).to(tl.float32)
            else:
                m = dY_row.to(tl.float32)
        elif casting_mode == 1:
            dY_row = dY_row.to(tl.float32)
            if elementwise_affine:
                m = dY_row * W_row
            else:
                m = dY_row
        else:
            if elementwise_affine:
                m = dY_row * W_row
            else:
                m = dY_row

        dX_row = rstd_row * m
        dX_row += rstd_row * (
            -(1 / n_cols) * rstd_row * rstd_row * tl.sum(m * X_row, axis=0) * X_row
        )

        if elementwise_affine:
            if casting_mode == 0:
                dW_row += dY_row * (X_row * rstd_row).to(X_dtype)
            else:
                dW_row += dY_row * (X_row * rstd_row)

        tl.store(dx_base + col_offsets, dX_row.to(X_dtype), mask=mask)

    if elementwise_affine:
        tl.store(dW_ptr + row_block_id * dW_row_stride + col_offsets, dW_row, mask=mask)


def _rms_norm_forward_triton(
    X: torch.Tensor,
    W: torch.Tensor | None,
    eps: float,
    offset: float,
    casting_mode: int,
    row_mode: bool | None,
):
    """Returns (Y, X_2d, RSTD, BLOCK_SIZE, num_warps)."""
    shape = X.shape
    dim = shape[-1]
    X = X.contiguous().view(-1, dim)
    n_rows, n_cols = X.shape
    BLOCK_SIZE, num_warps = calculate_settings(n_cols)

    Y = torch.empty((n_rows, n_cols), dtype=X.dtype, device=X.device)
    rstd_dtype = torch.float32 if casting_mode in (0, 1) else X.dtype
    RSTD = torch.empty(n_rows, dtype=rstd_dtype, device=X.device)

    elementwise_affine = W is not None
    W_contig = W.contiguous() if elementwise_affine else None

    def grid(meta):
        return (n_rows,)

    # The per-row forward (grid ``(n_rows,)``, one program per row) is
    # launch-bound when the hidden dim is small (``BLOCK_SIZE <= 256``, e.g.
    # Mellum-2.0 q_norm / k_norm at head_dim=128) AND ``n_rows >= 32k`` (vmapped
    # microbatch * seq_len * num_heads under DP-SGD): one tiny program per row.
    # In that regime use the block kernel — one program per SM, each looping
    # ``rows_per_program`` rows (like the backward) — to amortize launch
    # overhead. Identical math; ``row_mode`` forces the per-row path.
    use_block = not (BLOCK_SIZE > 256 or n_rows < 4096 * 8 or row_mode)

    with torch_gpu_device(X.device):
        if use_block:
            sm_count = (
                torch.cuda.get_device_properties(X.device).multi_processor_count
                if X.device.type == "cuda"
                else 1
            )
            rows_per_program = math.ceil(n_rows / sm_count)
            _rms_norm_forward_block_kernel[(sm_count,)](
                Y,
                Y.stride(0),
                X,
                X.stride(0),
                W_contig,
                W_contig.stride(0) if elementwise_affine else 0,
                RSTD,
                RSTD.stride(0),
                n_rows,
                n_cols,
                eps,
                offset,
                rows_per_program,
                casting_mode,
                elementwise_affine=elementwise_affine,
                BLOCK_SIZE=BLOCK_SIZE,
                num_warps=num_warps,
            )
        else:
            _rms_norm_forward_kernel[grid](
                Y,
                Y.stride(0),
                X,
                X.stride(0),
                W_contig,
                W_contig.stride(0) if elementwise_affine else 0,
                RSTD,
                RSTD.stride(0),
                n_cols,
                eps,
                offset,
                casting_mode,
                elementwise_affine=elementwise_affine,
                BLOCK_SIZE=BLOCK_SIZE,
                num_warps=num_warps,
            )

    return Y.view(*shape), X, RSTD, BLOCK_SIZE, num_warps


def _rms_norm_backward_triton(
    dY: torch.Tensor,
    X: torch.Tensor,
    W: torch.Tensor | None,
    RSTD: torch.Tensor,
    offset: float,
    casting_mode: int,
    BLOCK_SIZE: int,
    num_warps: int,
    in_place: bool,
    row_mode: bool | None,
):
    del row_mode  # reserved for future block-kernel parity
    shape = dY.shape
    dim = shape[-1]
    dY = dY.contiguous().view(-1, dim)
    n_rows, n_cols = dY.shape

    if n_cols > BLOCK_SIZE:
        raise RuntimeError(
            f"RMSNorm hidden dim {n_cols} exceeds fused block limit {BLOCK_SIZE}."
        )

    elementwise_affine = W is not None
    if X.device.type == "cuda":
        sm_count = torch.cuda.get_device_properties(X.device).multi_processor_count
    else:
        sm_count = 1

    if elementwise_affine:
        _dW = torch.empty((sm_count, n_cols), dtype=torch.float32, device=W.device)
    else:
        _dW = None

    rows_per_program = math.ceil(n_rows / sm_count)
    grid = (sm_count,)

    if in_place:
        dX = dY
    else:
        dX = torch.zeros_like(dY)

    W_contig = W.contiguous() if elementwise_affine else None

    x_dtype_triton = _TORCH_TO_TRITON_DTYPES[X.dtype]

    with torch_gpu_device(X.device):
        _rms_norm_backward_kernel[grid](
            dY,
            dY.stride(0),
            dX,
            dX.stride(0),
            X,
            X.stride(0),
            x_dtype_triton,
            W_contig,
            W_contig.stride(0) if elementwise_affine else 0,
            RSTD,
            RSTD.stride(0),
            _dW,
            _dW.stride(0) if elementwise_affine else 0,
            n_rows,
            n_cols,
            offset,
            rows_per_program,
            casting_mode,
            elementwise_affine=elementwise_affine,
            BLOCK_SIZE=BLOCK_SIZE,
            num_warps=num_warps,
        )

    dX = dX.view(*shape)
    if elementwise_affine:
        dW = _dW.sum(dim=0).to(W.dtype)
    else:
        dW = dY.new_zeros(0)
    return dX, dW


def _empty_weight(device, dtype) -> torch.Tensor:
    """Placeholder when elementwise_affine is False (autograd needs a tensor)."""
    return torch.empty(0, device=device, dtype=dtype)


class _RMSNormBackward(torch.autograd.Function):
    """Backward as separate Function for vmap(grad(...))."""

    @staticmethod
    def forward(dY, X, W, RSTD, meta_i, offset_tensor):
        casting_mode = int(meta_i[0].item())
        BLOCK_SIZE = int(meta_i[1].item())
        num_warps = int(meta_i[2].item())
        in_place = bool(meta_i[3].item())
        elementwise_affine = bool(meta_i[4].item())
        offset = float(offset_tensor.item())
        W_real = W if elementwise_affine and W.numel() > 0 else None
        return _rms_norm_backward_triton(
            dY,
            X,
            W_real,
            RSTD,
            offset,
            casting_mode,
            BLOCK_SIZE,
            num_warps,
            in_place,
            None,
        )

    @staticmethod
    def setup_context(ctx, inputs, output):
        del ctx  # noqa: ARG001

    @staticmethod
    def backward(ctx, *grad_outputs):
        raise NotImplementedError("Double backward not supported for RMSNorm")

    @staticmethod
    def vmap(info, in_dims, dY, X, W, RSTD, meta_i, offset_tensor):
        del info
        dy_b, x_b, w_b, r_b, m_b, o_b = in_dims
        if m_b is not None or o_b is not None:
            raise ValueError("meta_i and offset_tensor must not be vmapped")
        if w_b is not None:
            raise ValueError("W must not be vmapped")
        if dy_b != 0 or x_b != 0 or r_b != 0:
            raise ValueError("dY, X, RSTD must be vmapped at dim 0")
        H = X.shape[-1]
        head = dY.shape[:-1]
        B = dY.shape[0]
        dY_m = dY.reshape(-1, H)
        X_m = X.reshape(-1, H)
        R_m = RSTD.reshape(-1)
        W_use = W if (W is not None and W.numel() > 0) else None
        casting_mode = int(meta_i[0].item())

        # Missing flag defaults to trainable: a spurious per-example dW only
        # costs memory; a spurious batch-sum dW leaks across examples.
        w_trainable = bool(meta_i[5].item()) if meta_i.numel() > 5 else True
        dW_out = None
        if W_use is not None and w_trainable:
            # Per-example dW: the Triton call sums dW over the merged (B*T, H)
            # batch, which would hand every example the batch-sum gradient.
            # Must run before the Triton call — in_place overwrites dY with dX.
            T_flat = dY_m.shape[0] // B
            dY_3d = dY_m.view(B, T_flat, H)
            X_3d = X_m.view(B, T_flat, H)
            R_3d = R_m.view(B, T_flat)
            x_normed = X_3d.float() * R_3d.unsqueeze(-1)  # (B, T_flat, H)
            if casting_mode == 0:  # llama: cast normed X back to X dtype
                dW_per = (dY_3d * x_normed.to(X_3d.dtype)).float().sum(dim=1)
            else:  # gemma / none: accumulate in float32
                dW_per = (dY_3d.float() * x_normed).sum(dim=1)
            dW_out = dW_per.to(W.dtype)  # (B, H)

        dX, dW = _rms_norm_backward_triton(
            dY_m,
            X_m,
            W_use,
            R_m,
            float(offset_tensor.item()),
            casting_mode,
            int(meta_i[1].item()),
            int(meta_i[2].item()),
            bool(meta_i[3].item()),
            None,
        )
        dX_out = dX.view(*head, H)
        if dW_out is not None:
            return (dX_out, dW_out), (dy_b, 0)
        if W_use is not None:
            # W frozen: zeros, not the kernel's batch-sum — if ever consumed,
            # zeros stall training visibly instead of leaking across examples.
            return (dX_out, torch.zeros_like(dW)), (dy_b, None)
        return (dX_out, dW), (dy_b, None)


class Opaque_RMSNorm(torch.autograd.Function):
    """RMSNorm with Llama / Gemma casting modes (HuggingFace-aligned)."""

    @staticmethod
    def forward(X, W, eps, offset, casting_mode, in_place, row_mode):
        cm = _casting_mode_int(casting_mode)
        orig_shape = X.shape
        Y, _, _, _, _ = _rms_norm_forward_triton(
            X.contiguous(), W, float(eps), float(offset), cm, row_mode
        )
        return Y.view(orig_shape)

    @staticmethod
    def setup_context(ctx, inputs, output):
        X, W, eps, offset, casting_mode, in_place, row_mode = inputs
        cm = _casting_mode_int(casting_mode)
        orig_shape = X.shape
        X_saved = X.detach().clone().contiguous()
        dim = X_saved.shape[-1]
        X2d = X_saved.view(-1, dim)

        xf = X2d.float()
        ms = (xf * xf).mean(dim=-1)
        RSTD = torch.rsqrt(ms + float(eps))
        if cm == -1:
            ms0 = (X2d * X2d).mean(dim=-1)
            RSTD = torch.rsqrt(
                ms0 + torch.tensor(float(eps), device=X2d.device, dtype=X2d.dtype)
            )

        bs, nw = calculate_settings(dim)
        ctx.original_shape = orig_shape
        # meta_i[5]: W trainability — under vmap(grad()) frozen weights arrive
        # detached, so this tells the vmap rule whether per-example dW is needed.
        ctx.meta_i = torch.tensor(
            [
                cm,
                bs,
                nw,
                int(in_place),
                int(W is not None),
                int(W is not None and W.requires_grad),
            ],
            device=X_saved.device,
            dtype=torch.int64,
        )
        ctx.offset_tensor = torch.tensor(
            float(offset), device=X_saved.device, dtype=torch.float32
        )

        if W is not None:
            ctx.save_for_backward(X2d, W.contiguous(), RSTD)
        else:
            ctx.save_for_backward(X2d, RSTD)

    @staticmethod
    def backward(ctx, grad_output):
        go = grad_output.contiguous()
        dim = go.shape[-1]
        go2 = go.reshape(-1, dim)
        saved = ctx.saved_tensors
        if len(saved) == 3:
            X_s, W, RSTD = saved
            W_arg = W
        else:
            X_s, RSTD = saved
            W_arg = _empty_weight(X_s.device, X_s.dtype)

        dX, dW = _RMSNormBackward.apply(
            go2, X_s, W_arg, RSTD, ctx.meta_i, ctx.offset_tensor
        )
        dX = dX.view(ctx.original_shape)
        if ctx.meta_i[4].item():
            return dX, dW, None, None, None, None, None
        return dX, None, None, None, None, None, None

    @staticmethod
    def vmap(info, in_dims, X, W, eps, offset, casting_mode, in_place, row_mode):
        del info
        x_b, w_b = in_dims[0], in_dims[1]
        if w_b is not None:
            raise ValueError("Opaque_RMSNorm vmap: weight must not be batched")
        if x_b != 0:
            raise ValueError("Opaque_RMSNorm vmap: X must be vmapped at dim 0")
        cm = _casting_mode_int(casting_mode)
        shape = X.shape
        H = shape[-1]
        Xf = X.reshape(-1, H).contiguous()
        Yf, _, _, _, _ = _rms_norm_forward_triton(
            Xf,
            W,
            float(eps),
            float(offset),
            cm,
            row_mode,
        )
        return Yf.view(shape), 0


def opaque_rms_norm(
    x: torch.Tensor,
    weight: torch.Tensor,
    eps: float = 1e-6,
    offset: float = 0.0,
    casting_mode: str = "llama",
    *,
    in_place_backward: bool = False,
    row_mode: bool | None = None,
) -> torch.Tensor:
    """Public API: fused RMSNorm (CUDA only when Triton path is used)."""
    if not x.is_cuda:
        raise RuntimeError("opaque_rms_norm Triton path requires CUDA")
    x, weight = follow_autocast(x, weight)
    return Opaque_RMSNorm.apply(
        x,
        weight,
        eps,
        offset,
        casting_mode,
        in_place_backward,
        row_mode,
    )
