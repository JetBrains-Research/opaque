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

    # Compute in float32 for precision (matching Unsloth)
    e_row = tl.load(e + offsets, mask=mask, other=0).to(tl.float32)
    g_row = tl.load(g + offsets, mask=mask, other=0)

    # f = e * sigmoid(e)
    f_row = e_row * tl.sigmoid(e_row)
    f_row = f_row.to(g_row.dtype)  # Cast back to input dtype
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

    Following Unsloth's exact formula:
    - e = e.float()
    - se = 1.0 / (1.0 + torch.exp(-e))
    - f = (se * e).to(dtype)
    - h = f * g
    - df = DW * f
    - dg = DW * g
    - de = (dg.float() * se * (1.0 + e * (1.0 - se))).to(dtype)
    """
    block_idx = tl.program_id(0)
    if LONG_INDEXING:
        offsets = block_idx.to(tl.int64) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE).to(tl.int64)
        n_elements = tl.cast(n_elements, tl.int64)
    else:
        offsets = block_idx * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    DW_row = tl.load(DW + offsets, mask=mask, other=0)
    # Compute in float32 for precision (matching Unsloth)
    e_row = tl.load(e + offsets, mask=mask, other=0).to(tl.float32)
    g_row = tl.load(g + offsets, mask=mask, other=0)

    # se = sigmoid(e) in float32
    se_row = tl.sigmoid(e_row)
    # f = (se * e).to(dtype)
    f_row = se_row * e_row
    f_row = f_row.to(DW_row.dtype)

    # df = DW * f (this is grad_up in our convention)
    df_row = DW_row * f_row

    # dg = DW * g (intermediate for computing de)
    dg_row = DW_row * g_row

    # de = (dg.float() * se * (1.0 + e * (1.0 - se))).to(dtype)
    de_row = dg_row.to(tl.float32) * se_row * (1.0 + e_row * (1.0 - se_row))
    de_row = de_row.to(DW_row.dtype)

    # Store derivatives to output buffers
    # de_out = grad_gate, dg_out = grad_up
    tl.store(de_out + offsets, de_row, mask=mask)
    tl.store(dg_out + offsets, df_row, mask=mask)


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

        return h.reshape(original_shape)

    @staticmethod
    def setup_context(ctx, inputs, output):
        gate, up = inputs
        ctx.save_for_backward(gate.reshape(-1), up.reshape(-1))
        ctx.original_shape = gate.shape
        ctx.n_elements = gate.numel()

    @staticmethod
    def backward(ctx, grad_h):
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

        h = NewStyleSwiGLU.apply(gate, up)
        return h, gate_bdim


def swiglu_vmap(gate, up):
    """Convenience wrapper."""
    return NewStyleSwiGLU.apply(gate, up)


def triton_swiglu_forward(gate, up):
    """Direct Triton SwiGLU forward: h = silu(gate) * up.

    Calls the Triton kernel directly without autograd wrapper.
    For use as a callback in LoRA_MLP.
    """
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

    return h.reshape(original_shape)


def triton_swiglu_backward(grad_h, gate, up):
    """Direct Triton SwiGLU backward.

    Calls the Triton backward kernel directly without autograd wrapper.
    For use as a callback in LoRA_MLP.

    Args:
        grad_h: Gradient of output (same shape as gate/up)
        gate: Gate input from forward pass
        up: Up input from forward pass

    Returns:
        (grad_gate, grad_up) tuple
    """
    original_shape = gate.shape
    grad_h_flat = grad_h.reshape(-1).contiguous()
    gate_flat = gate.reshape(-1).contiguous()
    up_flat = up.reshape(-1).contiguous()
    n_elements = gate_flat.numel()

    grad_gate = torch.empty_like(gate_flat)
    grad_up = torch.empty_like(up_flat)

    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
    with torch_gpu_device(gate.device):
        _DWf_DW_dfg_kernel[grid](
            grad_h_flat, gate_flat, up_flat,
            grad_gate, grad_up,
            n_elements,
            BLOCK_SIZE=BLOCK_SIZE,
            LONG_INDEXING=0 if n_elements <= INT32_SAFETY_BUFFER else 1,
        )

    return grad_gate.reshape(original_shape), grad_up.reshape(original_shape)
