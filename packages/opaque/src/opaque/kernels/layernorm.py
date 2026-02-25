"""LayerNorm kernel with vmap support for DP-SGD."""
import triton
import triton.language as tl
import torch
from .utils import calculate_settings, torch_gpu_device


@triton.jit
def layernorm_forward(
    Y,
    Y_row_stride,
    X,
    X_row_stride,
    W,
    b,
    r,
    mu,
    n_cols: tl.constexpr,
    eps: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    row_idx = tl.program_id(0)
    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < n_cols

    Y += row_idx * Y_row_stride
    X += row_idx * X_row_stride
    r += row_idx
    mu += row_idx

    X_row = tl.load(X + col_offsets, mask=mask, other=0).to(tl.float32)
    W_row = tl.load(W + col_offsets, mask=mask, other=0).to(tl.float32)
    b_row = tl.load(b + col_offsets, mask=mask, other=0).to(tl.float32)

    mean_X = tl.sum(X_row, axis=0) / n_cols
    XX = tl.where(mask, X_row - mean_X, 0)
    row_var = tl.sum(XX * XX, axis=0) / n_cols
    inv_var = tl.math.rsqrt(row_var + eps)
    tl.store(r, inv_var)
    tl.store(mu, mean_X)
    output = (XX * inv_var) * W_row + b_row
    tl.store(Y + col_offsets, output, mask=mask)


@triton.jit
def layernorm_backward(
    dY,
    dY_row_stride,
    X,
    X_row_stride,
    W,
    b,
    r,
    mu,
    n_cols: tl.constexpr,
    eps: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    row_idx = tl.program_id(0)
    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < n_cols

    dY += row_idx * dY_row_stride
    X += row_idx * X_row_stride
    r += row_idx
    mu += row_idx

    dY_row = tl.load(dY + col_offsets, mask=mask, other=0).to(tl.float32)
    X_row = tl.load(X + col_offsets, mask=mask, other=0).to(tl.float32)
    W_row = tl.load(W + col_offsets, mask=mask, other=0).to(tl.float32)

    inv_var = tl.load(r).to(tl.float32)
    mean = tl.load(mu).to(tl.float32)
    normed = (X_row - mean) * inv_var
    dY_W = dY_row * W_row
    dX_row = (
        dY_W
        - tl.sum(dY_W, axis=0) / n_cols
        - normed * tl.sum(dY_W * normed, axis=0) / n_cols
    )
    dX_row = dX_row * inv_var
    tl.store(dY + col_offsets, dX_row, mask=mask)


class NewStyleLayerNorm(torch.autograd.Function):
    @staticmethod
    def forward(X, W, b, eps):
        """New-style API forward without ctx parameter."""
        original_shape = X.shape
        dim = original_shape[-1]
        X_flat = X.reshape(-1, dim)
        n_rows, n_cols = X_flat.shape

        BLOCK_SIZE, num_warps = calculate_settings(n_cols)
        device = X.device

        Y = torch.empty((n_rows, n_cols), dtype=X.dtype, device=device)
        r = torch.empty(n_rows, dtype=torch.float32, device=device)
        mu = torch.empty(n_rows, dtype=torch.float32, device=device)

        with torch_gpu_device(device):
            layernorm_forward[(n_rows,)](
                Y, Y.stride(0),
                X_flat, X_flat.stride(0),
                W, b, r, mu,
                n_cols, eps,
                BLOCK_SIZE=BLOCK_SIZE,
                num_warps=num_warps,
            )

        Y = Y.reshape(original_shape)
        return Y, r, mu, BLOCK_SIZE, num_warps

    @staticmethod
    def setup_context(ctx, inputs, output):
        X, W, b, eps = inputs
        Y, r, mu, BLOCK_SIZE, num_warps = output
        ctx.save_for_backward(X, W, b, r, mu)
        ctx.eps = eps
        ctx.BLOCK_SIZE = BLOCK_SIZE
        ctx.num_warps = num_warps

    @staticmethod
    def backward(ctx, grad_Y, grad_r, grad_mu, grad_block, grad_warps):
        X, W, b, r, mu = ctx.saved_tensors

        original_shape = grad_Y.shape
        dim = original_shape[-1]
        grad_Y_flat = grad_Y.reshape(-1, dim).contiguous()
        X_flat = X.reshape(-1, dim)
        n_rows, n_cols = grad_Y_flat.shape

        with torch_gpu_device(grad_Y.device):
            layernorm_backward[(n_rows,)](
                grad_Y_flat, grad_Y_flat.stride(0),
                X_flat, X_flat.stride(0),
                W, b, r, mu,
                n_cols, ctx.eps,
                BLOCK_SIZE=ctx.BLOCK_SIZE,
                num_warps=ctx.num_warps,
            )

        grad_X = grad_Y_flat.reshape(original_shape)
        return grad_X, None, None, None

    @staticmethod
    def vmap(info, in_dims, X, W, b, eps):
        """Custom vmap rule for DP-SGD."""
        X_bdim, W_bdim, b_bdim, eps_bdim = in_dims

        if X_bdim != 0:
            raise ValueError(f"X should be batched at dim 0, got {X_bdim}")
        if W_bdim is not None or b_bdim is not None:
            raise ValueError("W and b should not be batched")

        output = NewStyleLayerNorm.apply(X, W, b, eps)
        # Only Y is batched, intermediates are per-row
        return output, (X_bdim, X_bdim, X_bdim, None, None)


def layernorm_vmap(X, W, b, eps=1e-5):
    """Convenience wrapper."""
    Y, _, _, _, _ = NewStyleLayerNorm.apply(X, W, b, eps)
    return Y
