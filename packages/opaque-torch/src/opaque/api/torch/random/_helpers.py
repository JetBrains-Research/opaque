"""Torch-specific bridges for Opaque's immutable RNG keys."""

from __future__ import annotations

import contextlib
import os
from typing import TYPE_CHECKING

import numpy as np

import torch
from opaque.random import split

if TYPE_CHECKING:
    from opaque.random.types import RngKey

_MAX_TORCH_SEED = 2**63 - 1


def generator_from_key(rng_key: RngKey) -> torch.Generator:
    """Create a deterministic ``torch.Generator`` from an immutable key."""
    return torch.Generator().manual_seed(rng_key.seed % _MAX_TORCH_SEED)


def set_reproducible_pytorch_seed(key_val: RngKey) -> None:
    """Configure Torch, NumPy, cuDNN, and cuBLAS reproducibility."""
    seeds = split(key_val, num=3)
    torch_cpu_seed = seeds[0].seed % (2**31 - 1)
    torch_gpu_seed = seeds[1].seed % (2**31 - 1)
    numpy_seed = seeds[2].seed % (2**32)

    torch.manual_seed(int(torch_cpu_seed))
    torch.cuda.manual_seed_all(int(torch_gpu_seed))
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(int(torch_gpu_seed))
    with contextlib.suppress(ImportError, RuntimeError):
        np.random.seed(int(numpy_seed))

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    with contextlib.suppress(AttributeError, RuntimeError):
        torch.use_deterministic_algorithms(True)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"


__all__ = ["generator_from_key", "set_reproducible_pytorch_seed"]
