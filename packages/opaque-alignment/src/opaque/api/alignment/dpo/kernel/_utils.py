# Copyright (c) 2025 Opaque Authors
# SPDX-License-Identifier: Apache-2.0
"""Shared kernel helpers for the alignment fused-linear kernels.

These helpers are **intentionally duplicated** from
``opaque-patches/src/opaque/api/patches/kernels/_utils.py``. The
duplication is deliberate: making ``opaque-patches`` a hard dependency of
``opaque-alignment`` purely for the autocast / dtype-promotion shims is worse
than copying the ~30 lines we actually use. Only the helpers consumed by the
fused-linear-preference kernel are reproduced here — the Triton-specific
utilities are not.

``follow_autocast`` is the autocast-aware entry shim: public ``opaque_*``
wrappers call it so a kernel used inside ``torch.autocast(...)`` runs in the
active autocast dtype end-to-end rather than producing a dtype-passthrough
hybrid graph. On CPU (and whenever autocast is inactive) it is a no-op.
"""

from __future__ import annotations

import torch

__all__ = ["follow_autocast", "promote_dtypes"]


def follow_autocast(*tensors: object) -> tuple[object, ...]:
    """Cast floating-point tensors to the active autocast dtype, if any.

    Public ``opaque_*`` wrappers call this at the entry point so that a kernel
    used inside ``torch.autocast(device_type=..., dtype=...)`` runs in the
    autocast dtype end-to-end. Without this, the kernel would be
    dtype-passthrough — not autocast-aware — producing a hybrid graph that
    defeats the user's autocast intent.

    The active device type is read from the first floating-point CUDA tensor;
    if none is present the CPU autocast state is consulted. Non-tensor
    arguments, integer tensors, and ``None`` are passed through unchanged. The
    cast is a no-op when autocast is inactive (the common CPU-test path) or
    when a tensor already has the target dtype.

    Args:
        *tensors: Mixed tensors / non-tensors in call order.

    Returns:
        Tuple of the same length and order, with floating-point tensors cast
        to the active autocast dtype where applicable.
    """
    # Determine the relevant device type from the tensors themselves so this
    # works for both CUDA and CPU autocast regions.
    device_type: str | None = None
    for t in tensors:
        if isinstance(t, torch.Tensor) and t.is_floating_point():
            device_type = t.device.type
            break

    if device_type is None or not torch.is_autocast_enabled(device_type):
        return tensors

    target = torch.get_autocast_dtype(device_type)
    out: list[object] = []
    for t in tensors:
        if (
            isinstance(t, torch.Tensor)
            and t.is_floating_point()
            and t.device.type == device_type
            and t.dtype != target
        ):
            out.append(t.to(target))
        else:
            out.append(t)
    return tuple(out)


def promote_dtypes(*tensors: torch.Tensor) -> tuple[torch.Tensor, ...]:
    """Promote floating-point tensors to their common ``torch.result_type``.

    Mirrors the dtype-promotion shim used by the ``opaque-patches`` kernels:
    when a low-precision activation (bf16/fp16) is matmul-ed against a
    higher-precision weight, accumulating in the promoted dtype keeps the
    reduction numerically faithful. Non-floating tensors are returned
    unchanged.

    Args:
        *tensors: Tensors to align to a common dtype.

    Returns:
        Tuple of the same tensors, with the floating-point ones cast to the
        common promoted dtype.
    """
    float_tensors = [t for t in tensors if t.is_floating_point()]
    if not float_tensors:
        return tensors
    promoted = float_tensors[0].dtype
    for t in float_tensors[1:]:
        promoted = torch.promote_types(promoted, t.dtype)
    return tuple(
        t.to(promoted) if t.is_floating_point() and t.dtype != promoted else t
        for t in tensors
    )
