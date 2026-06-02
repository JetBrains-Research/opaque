# Copyright (c) 2025 Opaque Authors
# SPDX-License-Identifier: Apache-2.0
"""Bridge to the opaque-patches fused linear cross-entropy kernel.

The fused alignment primitives — :func:`~opaque.api.alignment.sft.loss.fused_nll_loss`,
:func:`~opaque.api.alignment.sft.loss.fused_dft_loss`, and
:func:`~opaque.api.alignment.logprob.fused_sequence_logp` — share one accelerated
backend: the patches Triton ``Opaque_LinearCrossEntropyLoss`` kernel. It computes
the per-token cross-entropy ``Σ CE`` against ``hidden @ weight.T`` without
materialising the ``(T, V)`` logits and recomputes the LSE in its backward.
``opaque-patches`` is the optional ``opaque-alignment[patches]`` dependency, so
each primitive falls back to its eager (logits-materialising) counterpart when
the kernel is unavailable (e.g. CPU CI).

These are **per-example** helpers: call them on a single example ``(T, H)`` and
drive them with ``vmap(grad(...))`` (the ``clipped_grad`` DP-SGD path); the
kernel's merged forward/backward vmap rules then make the whole microbatch one
forward + one backward kernel launch. They call ``.apply`` directly — *not*
``torch.vmap`` — because the kernel's manual vmap rule recomputes the forward
with the raw (non-autograd) ``_forward_impl``, so ``grad(vmap(...))`` silently
yields *zero* gradients; the outer ``vmap(grad)`` instead dispatches through the
kernel's merged vmap rules.
"""

from __future__ import annotations

import torch


def lce_available(hidden: torch.Tensor) -> bool:
    """True when the patches fused linear-CE kernel can run for ``hidden``.

    The Triton kernel needs CUDA + half precision (bf16/fp16), and
    ``opaque-patches`` must be importable (the ``[patches]`` extra). Otherwise
    the caller uses its eager fallback, so CPU / no-Triton CI stays green.
    """
    if not hidden.is_cuda or hidden.dtype not in (torch.float16, torch.bfloat16):
        return False
    try:
        import opaque.api.patches.kernels.linear_cross_entropy  # noqa: F401
    except Exception:
        return False
    return True


def linear_nll_sum(
    hidden: torch.Tensor,
    weight: torch.Tensor,
    labels: torch.Tensor,
    *,
    use_token_scaling: bool = False,
) -> torch.Tensor:
    """Per-example ``Σ CE`` (optionally DFT-token-scaled) via the patches kernel.

    Returns the unreduced sum of per-token cross-entropy over this example's
    non-ignored (``!= -100``) shifted tokens, computed from ``hidden @ weight.T``
    without materialising logits. ``use_token_scaling=True`` weights each token's
    CE by its detached model confidence ``softmax(logits)[target]`` (DFT).

    Call per example and drive with ``vmap(grad)``; see the module docstring for
    why this uses ``.apply`` directly rather than ``torch.vmap``.
    """
    from opaque.api.patches.kernels.linear_cross_entropy import (
        Opaque_LinearCrossEntropyLoss,
    )

    return Opaque_LinearCrossEntropyLoss.apply(
        hidden, weight, labels, -100, 0, 0.0, use_token_scaling
    )
