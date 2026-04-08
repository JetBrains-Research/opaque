"""Opaque: Functional DP-SGD for PyTorch.

Composable primitives for differentially private model training: per-example
gradient clipping, calibrated noise injection, privacy accounting, and Poisson
sampling. Built on ``torch.func`` with explicit state.

Modules:

- ``opaque.clipping``: Per-example gradient clipping (clipped_grad, clipped_fun, clip_pytree)
- ``opaque.noise``: Gaussian noise, truncated Gaussian, matrix-factorization correlated noise
- ``opaque.accounting``: PLD-based privacy accounting with composition, calibration, and metrics
- ``opaque.sampling``: Poisson, truncated Poisson, and cyclic Poisson samplers
- ``opaque.auditing``: Empirical privacy auditing via membership inference
- ``opaque.distributed``: DDP utilities (gradient aggregation, state sync)
- ``opaque.compat``: HuggingFace auto-patching for vmap compatibility
"""

# Lazy imports for optional dependencies
try:
    from opaque import accounting, auditing, distributed, sampling
except ImportError:
    # Optional dependencies not available - kernels will still work
    accounting = None
    auditing = None
    distributed = None
    sampling = None

from opaque.clipping import (
    AdaptiveClipState,
    ClipState,
    FixedClipState,
    adaptive_clipped_grad,
    clip_pytree,
    clipped_fun,
    clipped_grad,
)
from opaque.noise import (
    band_mf_noise,
    blt_mf_noise,
    custom_mf_noise,
    dense_mf_noise,
    gaussian_noise,
    identity_mf_noise,
    truncated_gaussian_noise,
)
from opaque.sampling import (
    CyclicPoissonSampler,
    PoissonSampler,
    TruncatedPoissonSampler,
    poisson_collate,
)
from opaque._env import parse_skip_env
from opaque.utils import make_functional, per_group, with_batch_dim
from opaque.utils.per_group import PerGroup

# =============================================================================
# Auto-patching for compatible libraries
# =============================================================================
# Environment variables (each accepts "all" or comma-separated names):
#   OPAQUE_SKIP_COMPAT_PATCHES=all (or transformers)
#   OPAQUE_SKIP_TRANSFORMERS_PATCHES=all (or vmap,kernels)
#   OPAQUE_SKIP_TRANSFORMERS_VMAP_PATCHES=all (or shared,standard,gemma2,phi3)
#   OPAQUE_SKIP_TRANSFORMERS_KERNEL_PATCHES=all (or swiglu,rope,ce,fused_ce,lora)

_opaque_skip_compat = parse_skip_env("OPAQUE_SKIP_COMPAT_PATCHES")

if "all" not in _opaque_skip_compat:
    from opaque.compat import apply_compat_patches

    apply_compat_patches()

__all__ = [
    # Clipping
    "clip_pytree",
    "clipped_fun",
    "clipped_grad",
    "adaptive_clipped_grad",
    "ClipState",
    "FixedClipState",
    "AdaptiveClipState",
    # Sampling
    "sampling",
    "PoissonSampler",
    "TruncatedPoissonSampler",
    "CyclicPoissonSampler",
    "poisson_collate",
    # Noise
    "gaussian_noise",
    "truncated_gaussian_noise",
    "band_mf_noise",
    "blt_mf_noise",
    "custom_mf_noise",
    "dense_mf_noise",
    "identity_mf_noise",
    # Accounting
    "accounting",
    # Auditing
    "auditing",
    # Distributed
    "distributed",
    # Utils
    "make_functional",
    "with_batch_dim",
    "PerGroup",
    "per_group",
]
