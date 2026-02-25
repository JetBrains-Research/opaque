"""GeGLU (exact and approx) kernels with vmap support for DP-SGD."""
import triton
import triton.language as tl
import torch
from .utils import triton_tanh, torch_gpu_device, INT32_SAFETY_BUFFER

NUM_INT32_ELEMENTS = 2**31
SAFE_INT32_BUFFER_MULTIPLIER = 4
BLOCK_SIZE = 1024
INT32_SAFETY_BUFFER = NUM_INT32_ELEMENTS - BLOCK_SIZE * SAFE_INT32_BUFFER_MULTIPLIER


# ===== GeGLU Exact (erf-based) =====

@triton.jit
def _exact_forward_kernel(
    e, g, h,
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

    e_row = tl.load(e + offsets, mask=mask, other=0).to(tl.float32)
    g_row = tl.load(g + offsets, mask=mask, other=0)

    # f = 1/2 * e * (1 + erf(1/sqrt(2) * e))
    f_row = 0.5 * e_row * (tl.math.erf(tl.math.rsqrt(2.0) * e_row) + 1.0)
    f_row = f_row.to(g_row.dtype)
    h_row = f_row * g_row

    tl.store(h + offsets, h_row, mask=mask)


@triton.jit
def _exact_backward_kernel(
    DW, e, g, de_out, dg_out,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
    LONG_INDEXING: tl.constexpr,
):
    """
    GeGLU (exact) backward pass: h = gelu(e) * g

    f = gelu(e) = 1/2 * e * (1 + erf(e/sqrt(2)))

    Gradients:
    - grad_g = grad_out * gelu(e)
    - grad_e = grad_out * g * gelu'(e)
      where gelu'(e) = 1/2 * (1 + erf(e/sqrt(2))) + e/sqrt(2*pi) * exp(-e^2/2)
    """
    block_idx = tl.program_id(0)
    if LONG_INDEXING:
        offsets = block_idx.to(tl.int64) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE).to(tl.int64)
        n_elements = tl.cast(n_elements, tl.int64)
    else:
        offsets = block_idx * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    grad_out = tl.load(DW + offsets, mask=mask, other=0).to(tl.float32)
    e_row = tl.load(e + offsets, mask=mask, other=0).to(tl.float32)
    g_row = tl.load(g + offsets, mask=mask, other=0).to(tl.float32)

    # Compute gelu(e)
    cdf = 0.5 * (tl.math.erf(tl.math.rsqrt(2.0) * e_row) + 1.0)
    f_row = cdf * e_row  # gelu(e)

    # grad_g = grad_out * gelu(e)
    dg_row = grad_out * f_row

    # grad_e = grad_out * g * gelu'(e)
    # gelu'(e) = cdf + e/sqrt(2*pi) * exp(-e^2/2)
    t = 0.3989422804014327  # 1/sqrt(2*pi)
    gelu_prime = cdf + t * e_row * tl.exp(-0.5 * e_row * e_row)
    de_row = grad_out * g_row * gelu_prime

    tl.store(de_out + offsets, de_row.to(de_out.dtype.element_ty), mask=mask)
    tl.store(dg_out + offsets, dg_row.to(dg_out.dtype.element_ty), mask=mask)


class NewStyleGeGLUExact(torch.autograd.Function):
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
            _exact_forward_kernel[grid](
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

        # Allocate output buffers
        grad_gate_out = torch.empty_like(gate_flat)
        grad_up_out = torch.empty_like(up_flat)

        grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
        with torch_gpu_device(grad_h.device):
            _exact_backward_kernel[grid](
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
        h, gate_flat, up_flat = NewStyleGeGLUExact.apply(gate, up)
        # Return only the output h with batch dimension 0
        return h, gate_bdim


# ===== GeGLU Approx (tanh-based) =====

@triton.jit
def _approx_forward_kernel(
    e, g, h,
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

    s = 0.7978845608028654  # math.sqrt(2 / math.pi)
    e_row = tl.load(e + offsets, mask=mask, other=0).to(tl.float32)
    g_row = tl.load(g + offsets, mask=mask, other=0)

    f_row = 0.5 * e_row * (triton_tanh(s * e_row * (1.0 + 0.044715 * e_row * e_row)) + 1.0)
    f_row = f_row.to(g_row.dtype)
    h_row = f_row * g_row

    tl.store(h + offsets, h_row, mask=mask)


@triton.jit
def _approx_backward_kernel(
    DW, e, g, de_out, dg_out,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
    LONG_INDEXING: tl.constexpr,
):
    """
    GeGLU (approx tanh) backward pass: h = gelu_tanh(e) * g

    f = gelu_tanh(e) = 0.5 * e * (1 + tanh(sqrt(2/pi) * (e + 0.044715 * e^3)))

    Gradients:
    - grad_g = grad_out * gelu_tanh(e)
    - grad_e = grad_out * g * gelu_tanh'(e)
    """
    block_idx = tl.program_id(0)
    if LONG_INDEXING:
        offsets = block_idx.to(tl.int64) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE).to(tl.int64)
        n_elements = tl.cast(n_elements, tl.int64)
    else:
        offsets = block_idx * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    grad_out = tl.load(DW + offsets, mask=mask, other=0).to(tl.float32)
    e_row = tl.load(e + offsets, mask=mask, other=0).to(tl.float32)
    g_row = tl.load(g + offsets, mask=mask, other=0).to(tl.float32)

    s = 0.7978845608028654  # sqrt(2/pi)
    inner = s * (e_row + 0.044715 * e_row * e_row * e_row)
    tanh_inner = triton_tanh(inner)
    T = 1.0 + tanh_inner
    T2 = 0.5 * T

    # f = gelu_tanh(e) = 0.5 * e * T
    f_row = T2 * e_row

    # grad_g = grad_out * gelu_tanh(e)
    dg_row = grad_out * f_row

    # gelu_tanh'(e) = 0.5 * T + 0.5 * e * sech^2(inner) * s * (1 + 3*0.044715*e^2)
    sech2 = 1.0 - tanh_inner * tanh_inner
    gelu_prime = T2 + 0.5 * e_row * sech2 * s * (1.0 + 3.0 * 0.044715 * e_row * e_row)

    # grad_e = grad_out * g * gelu_tanh'(e)
    de_row = grad_out * g_row * gelu_prime

    tl.store(de_out + offsets, de_row.to(de_out.dtype.element_ty), mask=mask)
    tl.store(dg_out + offsets, dg_row.to(dg_out.dtype.element_ty), mask=mask)


class NewStyleGeGLUApprox(torch.autograd.Function):
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
            _approx_forward_kernel[grid](
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

        # Allocate output buffers
        grad_gate_out = torch.empty_like(gate_flat)
        grad_up_out = torch.empty_like(up_flat)

        grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
        with torch_gpu_device(grad_h.device):
            _approx_backward_kernel[grid](
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
        h, gate_flat, up_flat = NewStyleGeGLUApprox.apply(gate, up)
        # Return only the output h with batch dimension 0
        return h, gate_bdim


# Convenience wrappers
def geglu_exact_vmap(gate, up):
    h, _, _ = NewStyleGeGLUExact.apply(gate, up)
    return h


def geglu_approx_vmap(gate, up):
    h, _, _ = NewStyleGeGLUApprox.apply(gate, up)
    return h
