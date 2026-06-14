# Copyright (c) 2025 Opaque Authors
# SPDX-License-Identifier: Apache-2.0
"""Non-Triton sparse MoE (``torch._grouped_mm``) parity tests (MPS/CPU).

The sparse grouped path must match the dense ``Opaque_MoE`` oracle — forward and,
crucially, the DP-SGD ``vmap(grad)`` **per-sample** expert-weight gradients
(``dW1`` ``(B,E,2I,H)`` / ``dW2`` ``(B,E,H,I)``, never summed across the batch).
"""

# ``I`` is the per-expert intermediate dim (paired with ``2I``), as in the kernel.
# ruff: noqa: E741

from __future__ import annotations

import pytest
import torch
from torch.func import grad, vmap

from opaque.api.patches.kernels._grouped_moe import (
    Opaque_GroupedMoE,
    grouped_mm_available,
)
from opaque.api.patches.kernels.moe import Opaque_MoE, opaque_moe

_TOL = 1e-3  # fp32 grouped_mm vs the dense expert loop: accumulation-order roundoff

pytestmark = pytest.mark.skipif(
    not grouped_mm_available(), reason="torch._grouped_mm unavailable"
)


def _inputs(device, E, I=32, H=16, K=2, B=3, T=6):
    torch.manual_seed(0)
    gu = torch.randn(E, 2 * I, H, device=device)
    dn = torch.randn(E, H, I, device=device)
    x3 = torch.randn(B, T, H, device=device)
    idx3 = torch.randint(0, E, (B, T, K), device=device)
    w3 = torch.rand(B, T, K, device=device)
    return gu, dn, x3, idx3, w3, B, T, H


def _check_parity(device: str) -> None:
    gu, dn, x3, idx3, w3, B, T, H = _inputs(device, E=4)
    N = B * T
    x2, idx2, w2 = (
        x3.reshape(N, H),
        idx3.reshape(N, -1),
        w3.reshape(N, -1),
    )

    # forward: sparse Function vs dense Function
    o_s = Opaque_GroupedMoE.apply(x2, gu, dn, idx2, w2)
    o_d = Opaque_MoE.apply(x2, gu, dn, idx2, w2)
    assert (o_s - o_d).abs().max().item() < _TOL, "forward"

    # DP vmap(grad): per-example grads wrt hidden + both expert weights + router.
    def f_s(xx, gg, dd, ii, ww):
        return Opaque_GroupedMoE.apply(xx, gg, dd, ii, ww).sum()

    def f_d(xx, gg, dd, ii, ww):
        return Opaque_MoE.apply(xx, gg, dd, ii, ww).sum()

    in_dims = (0, None, None, 0, 0)
    gs = vmap(grad(f_s, argnums=(0, 1, 2, 4)), in_dims=in_dims)(x3, gu, dn, idx3, w3)
    gd = vmap(grad(f_d, argnums=(0, 1, 2, 4)), in_dims=in_dims)(x3, gu, dn, idx3, w3)
    for name, a, b in zip(("dx", "d_gate_up", "d_down", "d_router"), gs, gd):
        assert (a - b).abs().max().item() < _TOL, f"vmap(grad) {name}"


def test_grouped_moe_parity_cpu():
    _check_parity("cpu")


@pytest.mark.mps
def test_grouped_moe_parity_mps():
    _check_parity("mps")


def _frob(a, b):
    a, b = a.float(), b.float()
    return (a - b).norm().item() / b.norm().clamp(min=1e-6).item()


def _check_bf16_within_floor(device: str) -> None:
    """bf16 sparse vs dense within the bf16 floor (relative Frobenius — the
    DP-relevant metric; per-element max-relative is meaningless on the near-zero
    grad entries). The dense oracle is itself bf16 (matmuls) + fp32 expert-sum, so
    the bar is bf16 precision, not fp32."""
    gu, dn, x3, idx3, w3, B, T, H = _inputs(device, E=16)
    gu, dn, x3, w3 = (t.to(torch.bfloat16) for t in (gu, dn, x3, w3))
    N = B * T
    x2, idx2, w2 = x3.reshape(N, H), idx3.reshape(N, -1), w3.reshape(N, -1)
    assert (
        _frob(
            Opaque_GroupedMoE.apply(x2, gu, dn, idx2, w2),
            Opaque_MoE.apply(x2, gu, dn, idx2, w2),
        )
        < 1e-2
    ), "forward"

    def f_s(xx, gg, dd, ii, ww):
        return Opaque_GroupedMoE.apply(xx, gg, dd, ii, ww).float().sum()

    def f_d(xx, gg, dd, ii, ww):
        return Opaque_MoE.apply(xx, gg, dd, ii, ww).float().sum()

    in_dims = (0, None, None, 0, 0)
    gs = vmap(grad(f_s, argnums=(0, 1, 2)), in_dims=in_dims)(x3, gu, dn, idx3, w3)
    gd = vmap(grad(f_d, argnums=(0, 1, 2)), in_dims=in_dims)(x3, gu, dn, idx3, w3)
    for name, a, b in zip(("dx", "d_gate_up", "d_down"), gs, gd):
        assert _frob(a, b) < 1e-2, f"vmap(grad) {name}"


def test_grouped_moe_bf16_cpu():
    _check_bf16_within_floor("cpu")


@pytest.mark.mps
def test_grouped_moe_bf16_mps():
    _check_bf16_within_floor("mps")


def _check_dispatch(device: str) -> None:
    # opaque_moe routes large-E to the sparse path and small-E to dense; both must
    # match the dense oracle. (Only correctness is asserted; the E threshold is a
    # perf heuristic.)
    for E in (4, 16):
        gu, dn, x3, idx3, w3, B, T, H = _inputs(device, E=E)
        x2, idx2, w2 = (
            x3.reshape(B * T, H),
            idx3.reshape(B * T, -1),
            w3.reshape(B * T, -1),
        )
        out = opaque_moe(x2, gu, dn, idx2, w2)
        ref = Opaque_MoE.apply(x2, gu, dn, idx2, w2)
        assert (out - ref).abs().max().item() < _TOL, f"dispatch E={E}"


def test_grouped_moe_dispatch_cpu():
    _check_dispatch("cpu")


@pytest.mark.mps
def test_grouped_moe_dispatch_mps():
    _check_dispatch("mps")


def _check_fused_gate(device: str) -> None:
    # ``fused=False`` forces the dense ``Opaque_MoE`` compat path even for large E
    # where the grouped-GEMM path is eligible: the output is bit-identical to the
    # dense oracle (a clean proof of the route), while ``fused=True`` takes the
    # grouped path (equal to the oracle only within the accumulation-roundoff
    # floor). This is the gate the ``fused_moe`` patch flag rides on.
    gu, dn, x3, idx3, w3, B, T, H = _inputs(device, E=16)
    x2, idx2, w2 = (
        x3.reshape(B * T, H),
        idx3.reshape(B * T, -1),
        w3.reshape(B * T, -1),
    )
    ref = Opaque_MoE.apply(x2, gu, dn, idx2, w2)
    dense = opaque_moe(x2, gu, dn, idx2, w2, fused=False)
    grouped = opaque_moe(x2, gu, dn, idx2, w2, fused=True)
    assert (dense - ref).abs().max().item() == 0.0, "fused=False must take dense path"
    assert (grouped - ref).abs().max().item() < _TOL, "fused=True grouped parity"


def test_grouped_moe_fused_gate_cpu():
    _check_fused_gate("cpu")


@pytest.mark.mps
def test_grouped_moe_fused_gate_mps():
    _check_fused_gate("mps")
