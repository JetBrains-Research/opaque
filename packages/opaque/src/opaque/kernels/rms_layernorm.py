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
    """
    row_idx = tl.program_id(0)
    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < n_cols

    # Pointer arithmetic
    grad_output_ptr += row_idx * grad_output_row_stride
    grad_input_ptr += row_idx * grad_input_row_stride
    input_ptr += row_idx * input_row_stride
    inv_var_ptr += row_idx * inv_var_row_stride

    # Load saved values
    grad_output_row = tl.load(grad_output_ptr + col_offsets, mask=mask, other=0).to(tl.float32)
    input_row = tl.load(input_ptr + col_offsets, mask=mask, other=0).to(tl.float32)
    weight_row = tl.load(weight_ptr + col_offsets, mask=mask, other=0).to(tl.float32)
    inv_var = tl.load(inv_var_ptr).to(tl.float32)

    # Backward pass
    # d_output/d_input involves chain rule through normalization
    normed = input_row * inv_var

    # Gradient w.r.t normalized = grad_output * weight
    grad_normed = grad_output_row * weight_row

    # Gradient w.r.t input (complex formula from chain rule)
    # d_loss/d_x = (d_loss/d_norm) * d_norm/d_x
    # where d_norm/d_x = (1/rms) * (I - (x*x.T)/(x.T*x))
    grad_norm_mean = tl.sum(grad_normed * normed, axis=0) / n_cols
    grad_input_row = (grad_normed - normed * grad_norm_mean) * inv_var
    grad_input_row = grad_input_row.to(grad_output_row.dtype)

    # Store gradient
    tl.store(grad_input_ptr + col_offsets, grad_input_row, mask=mask)


class RMSLayerNorm(torch.autograd.Function):
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
        BLOCK_SIZE = triton.next_power_of_2(n_cols)
        grid = (n_rows,)

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
        # Only grad_output matters - grad_inv_var is for the auxiliary output
        x, weight, inv_var = ctx.saved_tensors
        eps = ctx.eps
        original_shape = ctx.original_shape

        # Reshape x to flat for kernel
        n_cols = x.shape[-1]
        n_rows = x.numel() // n_cols
        x_flat = x.reshape(n_rows, n_cols).contiguous()

        grad_output_flat = grad_output.reshape(n_rows, n_cols).contiguous()

        # Allocate gradient
        grad_input = torch.empty_like(x_flat)

        # Launch backward kernel
        BLOCK_SIZE = triton.next_power_of_2(n_cols)
        grid = (n_rows,)

        _rms_layernorm_backward[grid](
            grad_output_flat,
            grad_output_flat.stride(0),
            grad_input,
            grad_input.stride(0),
            x_flat,
            x_flat.stride(0),
            weight,
            weight.stride(0),
            inv_var,
            inv_var.stride(0),
            n_cols,
            eps,
            BLOCK_SIZE=BLOCK_SIZE,
        )

        # Compute grad_weight (accumulated across all rows)
        if ctx.needs_input_grad[1]:
            # grad_weight = sum over batch of (grad_output * normed)
            normed = x_flat * inv_var.unsqueeze(-1)
            grad_weight = (grad_output_flat * normed).sum(dim=0)
        else:
            grad_weight = None

        # Reshape back
        grad_input = grad_input.reshape(original_shape)

        return grad_input, grad_weight, None

    @staticmethod
    def vmap(info, in_dims, x, weight, eps):
        """
        Custom vmap rule for DP-SGD.

        Args:
            info: VmapInfo with batch_size
            in_dims: Tuple of (x_bdim, weight_bdim, eps_bdim)
            x, weight, eps: Input tensors

        Returns:
            (output, output_bdim)
        """
        x_bdim, weight_bdim, eps_bdim = in_dims

        # weight and eps should not be batched
        if weight_bdim is not None:
            raise ValueError("weight should not be batched in vmap")
        if eps_bdim is not None:
            raise ValueError("eps should not be batched in vmap")

        # Apply to batched x
        output = RMSLayerNorm.apply(x, weight, eps)

        # Output has same batch dimension as input
        return output, x_bdim


def rms_layernorm(x, weight, eps=1e-6):
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
        >>> out = rms_layernorm(x, weight)
    """
    result = RMSLayerNorm.apply(x, weight, eps)
    # Forward returns tuple (output, inv_var, x_flat, original_shape)
    # but .apply() should only return first element when setup_context is defined
    # If it returns tuple, unwrap it
    if isinstance(result, tuple):
        return result[0]
    return result
