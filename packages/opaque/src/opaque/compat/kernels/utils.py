"""Kernel utilities - self-contained, no unsloth runtime dependency.

Extracted from unsloth/kernels/utils.py for standalone use in opaque kernels.
"""

import triton
import triton.language as tl
from contextlib import nullcontext
import torch

# Constants
MAX_FUSED_SIZE: int = 65536
next_power_of_2 = triton.next_power_of_2


def calculate_settings(n: int) -> tuple[int, int]:
    """Calculate block size and num_warps for Triton kernel launch.

    Args:
        n: Problem size (e.g., hidden dimension)

    Returns:
        Tuple of (BLOCK_SIZE, num_warps)
    """
    BLOCK_SIZE: int = next_power_of_2(n)
    if BLOCK_SIZE > MAX_FUSED_SIZE:
        raise RuntimeError(
            f"Cannot launch Triton kernel since n = {n} exceeds "
            f"the maximum CUDA blocksize = {MAX_FUSED_SIZE}."
        )
    num_warps: int = 4
    if BLOCK_SIZE >= 32768:
        num_warps = 32
    elif BLOCK_SIZE >= 8192:
        num_warps = 16
    elif BLOCK_SIZE >= 2048:
        num_warps = 8
    return BLOCK_SIZE, num_warps


# Device context manager
DEVICE_COUNT = torch.cuda.device_count() if torch.cuda.is_available() else 0

if DEVICE_COUNT > 1:

    def torch_gpu_device(device):
        """Context manager for multi-GPU."""
        return torch.cuda.device(device)
else:

    def torch_gpu_device(device):
        """No-op context manager for single GPU."""
        return nullcontext()


# Triton version compatibility
def _get_triton_version():
    """Get triton version as tuple for comparison."""
    version_str = triton.__version__
    parts = version_str.split(".")
    return tuple(int(p) for p in parts[:3] if p.isdigit())


_TRITON_VERSION = _get_triton_version()

if _TRITON_VERSION >= (3, 0, 0):
    try:
        from triton.language.extra import libdevice

        triton_tanh = libdevice.tanh
    except ImportError:
        triton_tanh = tl.math.tanh
    triton_cast = tl.cast
else:
    triton_tanh = tl.math.tanh

    @triton.jit
    def triton_cast(x, dtype):
        return x.to(dtype)


# Int32 safety limits for large tensor indexing
NUM_INT32_ELEMENTS = 2**31
SAFE_INT32_BUFFER_MULTIPLIER = 4
BLOCK_SIZE_DEFAULT = 1024
INT32_SAFETY_BUFFER = (
    NUM_INT32_ELEMENTS - BLOCK_SIZE_DEFAULT * SAFE_INT32_BUFFER_MULTIPLIER
)


def needs_long_indexing(n_elements: int) -> bool:
    """Check if tensor requires int64 indexing."""
    return n_elements > INT32_SAFETY_BUFFER


# =============================================================================
# Linear cross-entropy utilities (ported from cut_cross_entropy)
# =============================================================================


def _build_flat_valids(
    targets: torch.Tensor,
    ignore_index: int,
) -> torch.Tensor | None:
    """Build flat index tensor of valid (non-ignored) token positions.

    Assumes targets are already pre-shifted and flattened.
    Returns None if all tokens are valid (optimization).
    """
    valids = (targets != ignore_index).nonzero().to(torch.int32)
    assert valids.size(1) == 1
    return valids.squeeze(1) if valids.numel() != targets.numel() else None


def b_bin_fn(b: int) -> int:
    """Batch size binning for autotune key stability."""
    if b >= 1024:
        return 1024
    elif b <= 128:
        return 128
    else:
        return 512


def ensure_cuda_tensors(*tensors: torch.Tensor, fn_name: str) -> None:
    """Validate that all tensors are CUDA tensors for Triton kernels.

    Args:
        *tensors: Input tensors to validate
        fn_name: Public API function name for error reporting

    Raises:
        RuntimeError: If any tensor is not on CUDA
    """
    for tensor in tensors:
        if not torch.is_tensor(tensor):
            continue
        if tensor.device.type != "cuda":
            raise RuntimeError(
                f"{fn_name} requires CUDA tensors (Triton kernel backend); "
                f"got device={tensor.device.type}. "
                "Use non-kernel PyTorch path on MPS/CPU."
            )


# =============================================================================
# Triton JIT utilities for fused linear cross-entropy kernels
# =============================================================================

if _TRITON_VERSION >= (3, 0, 0):
    try:
        from triton.language.extra.libdevice import log1p as _tl_log1p
    except ImportError:
        _tl_log1p = tl.math.log1p
else:
    _tl_log1p = tl.math.log1p


@triton.jit
def tl_softcapping(v, softcap):
    return triton_tanh(v / softcap) * softcap


@triton.jit
def tl_softcapping_grad(dv, v, softcap):
    v = v / softcap
    return dv * (1 - v * v)


@triton.jit
def tl_logaddexp(a, b):
    minx = tl.minimum(a, b)
    mx = tl.maximum(a, b)
    return _tl_log1p(tl.exp(minx - mx)) + mx


@triton.jit
def tl_lock_add(ptrs, v, mask, lock_ptr):
    while tl.atomic_cas(lock_ptr, 0, 1) == 1:
        pass

    cur_v = tl.load(ptrs, mask=mask, other=0.0, eviction_policy="evict_last")
    new_v = v + cur_v
    tl.store(ptrs, new_v, mask=mask, eviction_policy="evict_last")

    tl.debug_barrier()
    tl.atomic_xchg(lock_ptr, 0)
