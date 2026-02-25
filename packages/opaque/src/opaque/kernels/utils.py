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
    parts = version_str.split('.')
    return tuple(int(p) for p in parts[:3] if p.isdigit())


_TRITON_VERSION = _get_triton_version()

if _TRITON_VERSION >= (3, 0, 0):
    try:
        from triton.language.extra import libdevice
        triton_tanh = libdevice.tanh
    except ImportError:
        triton_tanh = tl.math.tanh
else:
    triton_tanh = tl.math.tanh


# Int32 safety limits for large tensor indexing
NUM_INT32_ELEMENTS = 2**31
SAFE_INT32_BUFFER_MULTIPLIER = 4
BLOCK_SIZE_DEFAULT = 1024
INT32_SAFETY_BUFFER = NUM_INT32_ELEMENTS - BLOCK_SIZE_DEFAULT * SAFE_INT32_BUFFER_MULTIPLIER


def needs_long_indexing(n_elements: int) -> bool:
    """Check if tensor requires int64 indexing."""
    return n_elements > INT32_SAFETY_BUFFER
