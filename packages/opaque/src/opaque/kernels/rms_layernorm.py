"""RMS LayerNorm kernel adapted from Unsloth with vmap support.

Based on Unsloth's implementation:
https://github.com/unslothai/unsloth/blob/main/unsloth/kernels/rms_layernorm.py

Modifications:
- Added custom vmap rule for DP-SGD compatibility
- Simplified to remove Gemma-specific logic
- Added comprehensive documentation
"""

import torch
import triton
import triton.language as tl
from .utils import calculate_settings, torch_gpu_device


@triton.jit
def _rms_layernorm_forward(
    output_ptr,
    output_row_stride: tl.constexpr,
    input_ptr,
    input_row_stride: tl.constexpr,
    weight_ptr,
    weight_row_stride: tl.constexpr,
    inv_var_ptr,
    inv_var_row_stride: tl.constexpr,
    n_cols: tl.constexpr,
    eps: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """
    RMS LayerNorm forward kernel.

    Computes: output = (input / sqrt(mean(input^2) + eps)) * weight

    Each program processes one row (one sequence position).
    """
    row_idx = tl.program_id(0)
    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < n_cols

    # Pointer arithmetic for this row
    output_ptr += row_idx * output_row_stride
    input_ptr += row_idx * input_row_stride
    inv_var_ptr += row_idx * inv_var_row_stride

    # Load input and weight
    input_row = tl.load(input_ptr + col_offsets, mask=mask, other=0).to(tl.float32)
    weight_row = tl.load(weight_ptr + col_offsets, mask=mask, other=0)

    # Compute RMS: sqrt(mean(x^2))
    row_var = tl.sum(input_row * input_row, axis=0) / n_cols
    inv_var = tl.math.rsqrt(row_var + eps)

    # Save inverse variance for backward
    tl.store(inv_var_ptr, inv_var)

    # Normalize and scale
    normed = input_row * inv_var
    normed = normed.to(weight_row.dtype)
    output = normed * weight_row

    # Store output
    tl.store(output_ptr + col_offsets, output, mask=mask)


@triton.jit
def _rms_layernorm_backward(
    grad_output_ptr,
    grad_output_row_stride: tl.constexpr,
    grad_input_ptr,
    grad_input_row_stride: tl.constexpr,
    input_ptr,
    input_row_stride: tl.constexpr,
    weight_ptr,
    weight_row_stride: tl.constexpr,
    inv_var_ptr,
    inv_var_row_stride: tl.constexpr,
    n_cols: tl.constexpr,
    eps: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """
    RMS LayerNorm backward kernel.

    Computes gradients with respect to input and weight.
    Each program processes one row.

    Following Unsloth's formula for numerical stability:
    output = inv_var / n_cols * (n_cols * dY_W - normed * rowsum_dY_normed)
    """
    row_idx = tl.program_id(0)
    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < n_cols

    # Pointer arithmetic
    grad_output_ptr += row_idx * grad_output_row_stride
    grad_input_ptr += row_idx * grad_input_row_stride
    input_ptr += row_idx * input_row_stride
    inv_var_ptr += row_idx * inv_var_row_stride

    # Load saved values in float32 for precision
    grad_output_row = tl.load(grad_output_ptr + col_offsets, mask=mask, other=0).to(tl.float32)
    input_row = tl.load(input_ptr + col_offsets, mask=mask, other=0).to(tl.float32)
    weight_row = tl.load(weight_ptr + col_offsets, mask=mask, other=0).to(tl.float32)
    inv_var = tl.load(inv_var_ptr).to(tl.float32)

    # Compute normalized input
    normed = input_row * inv_var

    # dY_W = grad_output * weight
    dY_W = grad_output_row * weight_row

    # Unsloth's numerically stable formula:
    # output = inv_var / n_cols * (n_cols * dY_W - normed * rowsum_dY_normed)
    rowsum_dY_normed = tl.sum(dY_W * normed, axis=0)
    grad_input_row = inv_var / n_cols * (n_cols * dY_W - normed * rowsum_dY_normed)

    # Store gradient
    tl.store(grad_input_ptr + col_offsets, grad_input_row, mask=mask)


class _RMSNormBackward(torch.autograd.Function):
    """Backward pass wrapped as autograd.Function for vmap(grad()) support."""

    @staticmethod
    def forward(grad_output, x, weight, inv_var, eps):
        original_shape = x.shape
        n_cols = x.shape[-1]
        n_rows = x.numel() // n_cols

        x_flat = x.reshape(n_rows, n_cols).contiguous()
        grad_output_flat = grad_output.reshape(n_rows, n_cols).contiguous()

        # Compute grad_weight before in-place backward overwrites grad_output
        normed = x_flat * inv_var.unsqueeze(-1)
        grad_weight = (grad_output_flat * normed).sum(dim=0)

        BLOCK_SIZE, num_warps = calculate_settings(n_cols)
        grid = (n_rows,)

        with torch_gpu_device(x.device):
            # In-place: write grad_input into grad_output_flat buffer
            _rms_layernorm_backward[grid](
                grad_output_flat, grad_output_flat.stride(0),
                grad_output_flat, grad_output_flat.stride(0),
                x_flat, x_flat.stride(0),
                weight, weight.stride(0),
                inv_var, inv_var.stride(0),
                n_cols, eps,
                BLOCK_SIZE=BLOCK_SIZE,
                num_warps=num_warps,
            )

        return grad_output_flat.reshape(original_shape), grad_weight

    @staticmethod
    def setup_context(ctx, inputs, output):
        pass

    @staticmethod
    def backward(ctx, *grad_outputs):
        raise NotImplementedError("Double backward not supported for RMSNorm")

    @staticmethod
    def vmap(info, in_dims, grad_output, x, weight, inv_var, eps):
        grad_output_bdim, x_bdim, weight_bdim, inv_var_bdim, eps_bdim = in_dims

        assert weight_bdim is None, "weight should not be batched in vmap"
        assert eps_bdim is None, "eps should not be batched in vmap"

        n_cols = x.shape[-1]
        vmap_batch = x.shape[0]

        # Merge vmap batch into rows for Triton kernel
        x_flat = x.reshape(-1, n_cols).contiguous()
        grad_output_flat = grad_output.reshape(-1, n_cols).contiguous()
        inv_var_flat = inv_var.reshape(-1).contiguous()
        n_rows = x_flat.shape[0]

        # Compute grad_weight before in-place backward overwrites grad_output
        normed = x_flat * inv_var_flat.unsqueeze(-1)
        rows_per_example = n_rows // vmap_batch
        normed_reshaped = normed.reshape(vmap_batch, rows_per_example, n_cols)
        grad_output_reshaped = grad_output_flat.reshape(vmap_batch, rows_per_example, n_cols)
        grad_weight = (grad_output_reshaped * normed_reshaped).sum(dim=1)  # (vmap_batch, n_cols)

        BLOCK_SIZE, num_warps = calculate_settings(n_cols)
        grid = (n_rows,)

        with torch_gpu_device(x.device):
            # In-place: write grad_input into grad_output_flat buffer
            _rms_layernorm_backward[grid](
                grad_output_flat, grad_output_flat.stride(0),
                grad_output_flat, grad_output_flat.stride(0),
                x_flat, x_flat.stride(0),
                weight, weight.stride(0),
                inv_var_flat, inv_var_flat.stride(0),
                n_cols, eps,
                BLOCK_SIZE=BLOCK_SIZE,
                num_warps=num_warps,
            )

        return (grad_output_flat.reshape(x.shape), grad_weight), (x_bdim, 0)


class Opaque_RMSNorm(torch.autograd.Function):
    """
    RMS LayerNorm with custom vmap support.

    Implements: output = (x / sqrt(mean(x^2) + eps)) * weight

    This is the normalization used in LLaMA and other modern LLMs.
    """

    @staticmethod
    def forward(x, weight, eps):
        """
        Args:
            x: Input tensor (..., hidden_size)
            weight: Scale parameters (hidden_size,)
            eps: Small constant for numerical stability

        Returns:
            Tuple of (output, inv_var) for setup_context
        """
        # Flatten to 2D for kernel
        original_shape = x.shape
        n_rows = x.numel() // x.shape[-1]
        n_cols = x.shape[-1]

        x_flat = x.reshape(n_rows, n_cols).contiguous()

        # Allocate output and inverse variance
        output = torch.empty_like(x_flat)
        inv_var = torch.empty(n_rows, dtype=torch.float32, device=x.device)

        # Launch kernel
        BLOCK_SIZE, num_warps = calculate_settings(n_cols)
        grid = (n_rows,)

        with torch_gpu_device(x.device):
            _rms_layernorm_forward[grid](
                output,
                output.stride(0),
                x_flat,
                x_flat.stride(0),
                weight,
                weight.stride(0),
                inv_var,
                inv_var.stride(0),
                n_cols,
                eps,
                BLOCK_SIZE=BLOCK_SIZE,
                num_warps=num_warps,
            )

        # Return only output and inv_var
        # PyTorch will keep inputs (x, weight) available for setup_context
        return output.reshape(original_shape), inv_var

    @staticmethod
    def setup_context(ctx, inputs, output):
        """Setup context for new-style API (required for vmap)."""
        x, weight, eps = inputs
        output_reshaped, inv_var = output

        # Save tensors from inputs - PyTorch optimizes to avoid duplication
        ctx.save_for_backward(x, weight, inv_var)
        ctx.eps = eps
        ctx.original_shape = x.shape

    @staticmethod
    def backward(ctx, grad_output, grad_inv_var):
        x, weight, inv_var = ctx.saved_tensors
        grad_input, grad_weight = _RMSNormBackward.apply(
            grad_output, x, weight, inv_var, ctx.eps
        )
        return grad_input, grad_weight, None

    @staticmethod
    def vmap(info, in_dims, x, weight, eps):
        """Custom vmap rule for DP-SGD.

        Calls Triton forward kernel directly with merged batch dims.
        """
        x_bdim, weight_bdim, eps_bdim = in_dims

        if weight_bdim is not None:
            raise ValueError("weight should not be batched in vmap")
        if eps_bdim is not None:
            raise ValueError("eps should not be batched in vmap")

        batched_shape = x.shape
        n_cols = x.shape[-1]
        x_flat = x.reshape(-1, n_cols).contiguous()
        n_rows = x_flat.shape[0]

        output = torch.empty_like(x_flat)
        inv_var = torch.empty(n_rows, dtype=torch.float32, device=x.device)

        BLOCK_SIZE, num_warps = calculate_settings(n_cols)
        grid = (n_rows,)

        with torch_gpu_device(x.device):
            _rms_layernorm_forward[grid](
                output, output.stride(0),
                x_flat, x_flat.stride(0),
                weight, weight.stride(0),
                inv_var, inv_var.stride(0),
                n_cols, eps,
                BLOCK_SIZE=BLOCK_SIZE,
                num_warps=num_warps,
            )

        output = output.reshape(batched_shape)
        # Reshape inv_var so vmap batch dim is explicit
        vmap_batch = batched_shape[0]
        inv_var = inv_var.reshape(vmap_batch, -1)

        return (output, inv_var), (x_bdim, x_bdim)


def opaque_rms_norm(x, weight, eps=1e-6):
    """
    Apply RMS LayerNorm using Triton kernel.

    Args:
        x: Input tensor (..., hidden_size)
        weight: Scale parameters (hidden_size,)
        eps: Small constant for numerical stability (default: 1e-6)

    Returns:
        Normalized output (..., hidden_size)

    Example:
        >>> x = torch.randn(4, 128, 4096, device='cuda')
        >>> weight = torch.ones(4096, device='cuda')
        >>> out = opaque_rms_norm(x, weight)
    """
    result = Opaque_RMSNorm.apply(x, weight, eps)
    # Forward returns tuple (output, inv_var, x_flat, original_shape)
    # but .apply() should only return first element when setup_context is defined
    # If it returns tuple, unwrap it
    if isinstance(result, tuple):
        return result[0]
    return result
