# Copyright (c) 2025 Opaque Authors
# SPDX-License-Identifier: Apache-2.0
"""Single source of truth for per-device hardware capabilities.

Opaque historically scattered device feature checks across the codebase: the
kernel router asks "CUDA + Triton?", the trainer config asks HuggingFace's
``is_torch_bf16_gpu_available()``, and the examples hard-code per-device dtype
and ``torch.compile`` rules.  That drift is how macOS/MPS functionality became
"hiddenly unavailable" — bf16 silently downgraded to fp32, fused kernels
silently disabled, ``torch.compile`` silently mis-targeted.

:func:`device_capabilities` answers all of those questions in one place so call
sites query a capability instead of re-deriving it.  The underlying probes are
deliberately empirical (allocate-and-run) rather than version-sniffing — so they
track whatever the installed PyTorch actually supports on the host hardware — and
each is cached per device *type* (``functools.lru_cache``), so a repeated query
re-runs no probe even though :func:`device_capabilities` itself rebuilds the
lightweight record each call.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from opaque.api.engine import runtime

__all__ = [
    "DeviceCapabilities",
    "device_capabilities",
    "fused_kernels_available",
    "sdpa_autocast_under_vmap_broken",
]


def fused_kernels_available() -> bool:
    """Host-level check: a CUDA device *and* an importable Triton.

    This is the runtime the Triton fused kernels (rope / rms_norm / swiglu /
    geglu / cross-entropy / fused-linear-CE) need.  On MPS / CPU (or CUDA
    without Triton) it returns ``False`` and callers fall back to the
    pure-PyTorch eager implementations.
    """
    return runtime.device_fused_kernels_available()


def sdpa_autocast_under_vmap_broken(device_type: str) -> bool:
    """Whether ``torch.autocast`` fails to cast SDPA under ``vmap(grad)`` here.

    PyTorch bug (MPS): wrapped functorch tensors lose the MPS dispatch key, so
    ``AutocastMPS`` is never re-derived and ``torch.autocast`` silently no-ops
    under ``vmap``/``grad``.  A bf16 DP step (RoPE leaves q/k fp32 while v is the
    bf16 v_proj output) then crashes in ``scaled_dot_product_attention`` with a
    query/value dtype mismatch.  It casts fine on CPU/CUDA, and on MPS *without*
    functorch.  Tracked upstream as pytorch/pytorch#187265 (fix #187282).

    Empirical (run the failing op once, cached) and matched to the live torch, so
    the eager-attention workaround auto-drops the moment a release fixes it.
    Returns ``False`` on non-MPS devices and when the op runs cleanly.
    """
    return runtime.device_sdpa_autocast_under_vmap_broken(device_type)


@dataclass(frozen=True)
class DeviceCapabilities:
    """What a given device can do, resolved once at the call site.

    Attributes:
        device_type: ``"cuda"``, ``"mps"``, or ``"cpu"``.
        supports_bf16: bf16 tensors/ops run on this device.
        supports_compile: ``torch.compile`` targets this device.
        recommended_compile_backend: the backend to default to
            (``"inductor"``) or ``None`` when compile isn't recommended.
        supports_fused_kernels: the Triton fused-kernel runtime is available
            (CUDA + Triton); otherwise eager fallbacks are used.
        peak_memory_trackable: the device exposes a cheap, resettable
            peak-memory counter that ``step_perf`` can zero each step (CUDA).
            False on MPS — its reserved high-water is itself an exact peak
            figure, but resetting it costs an ``empty_cache``, too heavy to run
            per step — and on CPU, which has no peak counter at all.
        supports_pin_memory: pinned host memory accelerates H2D copies here
            (CUDA only; a no-op on CPU, a noisy warning on MPS).
    """

    device_type: str
    supports_bf16: bool
    supports_compile: bool
    recommended_compile_backend: str | None
    supports_fused_kernels: bool
    peak_memory_trackable: bool
    supports_pin_memory: bool

    @property
    def is_accelerator(self) -> bool:
        """True for GPU-class devices (CUDA, MPS); False for CPU."""
        return self.device_type in ("cuda", "mps")


def device_capabilities(device: Any) -> DeviceCapabilities:
    """Resolve the :class:`DeviceCapabilities` for ``device``.

    Args:
        device: a ``torch.device`` or device string (``"cuda"``, ``"mps"``,
            ``"cpu"``, ``"cuda:1"`` …); only the device *type* matters.

    Returns:
        The capability record for that device type.

    Example:
        >>> caps = device_capabilities("mps")
        >>> dtype = torch.bfloat16 if caps.supports_bf16 else torch.float32
    """
    return runtime.device_capabilities(device)
