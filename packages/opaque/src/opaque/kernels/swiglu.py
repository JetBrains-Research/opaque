"""SwiGLU kernel with vmap support for DP-SGD."""
import triton
import triton.language as tl
import torch
from .utils import torch_gpu_device, INT32_SAFETY_BUFFER

NUM_INT32_ELEMENTS = 2**31
SAFE_INT32_BUFFER_MULTIPLIER = 4
BLOCK_SIZE = 1024
INT32_SAFETY_BUFFER = NUM_INT32_ELEMENTS - BLOCK_SIZE * SAFE_INT32_BUFFER_MULTIPLIER


@triton.jit
def _fg_kernel(
    e,
    g,
    h,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
    LONG_INDEXING: tl.constexpr,
):
    block_idx = tl.program_id(0)
    if LONG_INDEXING:
        offsets = block_idx.to(tl.int64) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE).to(tl.int64)
        n_elements = tl.cast(n_elements, tl.int64)
    else:
        offsets = block_idx * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    e_row = tl.load(e + offsets, mask=mask, other=0)
    g_row = tl.load(g + offsets, mask=mask, other=0)

    # f = e * sigmoid(e)
    f_row = e_row * tl.sigmoid(e_row)
    # h = f * g
    h_row = f_row * g_row

    tl.store(h + offsets, h_row, mask=mask)


@triton.jit
def _DWf_DW_dfg_kernel(
    DW,
    e,
    g,
    de_out,
    dg_out,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
    LONG_INDEXING: tl.constexpr,
):
    """
    SwiGLU backward pass: h = silu(e) * g = (e * sigmoid(e)) * g

    Gradients:
    - grad_g = grad_out * silu(e) = grad_out * (e * sigmoid(e))
    - grad_e = grad_out * g * sigmoid(e) * (1 + e * (1 - sigmoid(e)))
    """
    block_idx = tl.program_id(0)
    if LONG_INDEXING:
        offsets = block_idx.to(tl.int64) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE).to(tl.int64)
        n_elements = tl.cast(n_elements, tl.int64)
    else:
        offsets = block_idx * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    grad_out = tl.load(DW + offsets, mask=mask, other=0)
    e_row = tl.load(e + offsets, mask=mask, other=0)
    g_row = tl.load(g + offsets, mask=mask, other=0)

    se_row = tl.sigmoid(e_row)
    f_row = se_row * e_row  # silu(e) = e * sigmoid(e)

    # grad_g = grad_out * silu(e)
    dg_row = grad_out * f_row

    # grad_e = grad_out * g * sigmoid(e) * (1 + e * (1 - sigmoid(e)))
    de_row = grad_out * g_row * se_row * (1.0 + e_row * (1.0 - se_row))

    # Store derivatives to output buffers
    tl.store(de_out + offsets, de_row, mask=mask)
    tl.store(dg_out + offsets, dg_row, mask=mask)


class NewStyleSwiGLU(torch.autograd.Function):
    @staticmethod
    def forward(gate, up):
        """New-style API forward without ctx parameter."""
        original_shape = gate.shape
        gate_flat = gate.reshape(-1)
        up_flat = up.reshape(-1)
        n_elements = gate_flat.numel()

        h = torch.empty_like(gate_flat)
        grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)

        with torch_gpu_device(gate.device):
            _fg_kernel[grid](
                gate_flat, up_flat, h,
                n_elements,
                BLOCK_SIZE=BLOCK_SIZE,
                LONG_INDEXING=0 if n_elements <= INT32_SAFETY_BUFFER else 1,
            )

        return h.reshape(original_shape), gate_flat, up_flat

    @staticmethod
    def setup_context(ctx, inputs, output):
        gate, up = inputs
        h, gate_flat, up_flat = output
        ctx.save_for_backward(gate_flat, up_flat)
        ctx.original_shape = gate.shape
        ctx.n_elements = gate_flat.numel()

    @staticmethod
    def backward(ctx, grad_h, grad_gate_flat, grad_up_flat):
        gate_flat, up_flat = ctx.saved_tensors
        grad_h_flat = grad_h.reshape(-1).contiguous()
        n_elements = ctx.n_elements

        # Allocate output buffers for gradients
        grad_gate_out = torch.empty_like(gate_flat)
        grad_up_out = torch.empty_like(up_flat)

        grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
        with torch_gpu_device(grad_h.device):
            _DWf_DW_dfg_kernel[grid](
                grad_h_flat, gate_flat, up_flat,
                grad_gate_out, grad_up_out,
                n_elements,
                BLOCK_SIZE=BLOCK_SIZE,
                LONG_INDEXING=0 if n_elements <= INT32_SAFETY_BUFFER else 1,
            )

        return grad_gate_out.reshape(ctx.original_shape), grad_up_out.reshape(ctx.original_shape)

    @staticmethod
    def vmap(info, in_dims, gate, up):
        """Custom vmap rule for DP-SGD."""
        gate_bdim, up_bdim = in_dims

        if gate_bdim != 0 or up_bdim != 0:
            raise ValueError("Both gate and up should be batched at dim 0")

        h, gate_flat, up_flat = NewStyleSwiGLU.apply(gate, up)
        # Return only the output h with batch dimension 0
        # gate_flat and up_flat are not exposed outside the Function
        return h, gate_bdim


def swiglu_vmap(gate, up):
    """Convenience wrapper."""
    h, _, _ = NewStyleSwiGLU.apply(gate, up)
    return h
