# Copyright (c) 2025 Opaque Authors
# SPDX-License-Identifier: Apache-2.0
"""Portable chunked linear-CE (non-Triton MPS/CPU path) parity tests.

The complement of ``test_linear_cross_entropy.py`` (CUDA + Triton): this pins
the pure-PyTorch chunked kernel that runs where Triton is unavailable. It must
match the eager ``matmul + cross_entropy`` reference bit-for-bit (same math,
streamed over vocab chunks) for every feature, on the direct call and under
``vmap(grad)`` (the DP-SGD path), with frozen and trainable lm-head weight.

Tensors are deliberately tiny — this is a correctness contract, not a memory
benchmark (the streaming memory win is measured out of band; asserting it in CI
would risk OOMing the small MPS runner).
"""

from __future__ import annotations

import pytest
import torch
from torch.func import grad, vmap

import opaque.api.patches.kernels._linear_ce_chunked as mod
from opaque.api.patches.kernels._linear_ce_chunked import (
    linear_cross_entropy_chunked,
)

_TOL = 5e-4  # fp32 streaming vs materialized: roundoff only


def _softcap(x, sc):
    return sc * torch.tanh(x / sc) if sc else x


def _eager_mean(
    hidden,
    weight,
    labels,
    ignore_index=-100,
    logit_softcapping=0,
    label_smoothing=0.0,
    use_token_scaling=False,
):
    """Materialized, vmap-safe reference matching the kernel's math + reduction."""
    sc = logit_softcapping or None
    flat = _softcap(hidden[..., :-1, :] @ weight.transpose(-1, -2), sc)
    flat = flat.reshape(-1, weight.shape[0])
    t = labels[..., 1:].reshape(-1)
    valid = t != ignore_index
    lse = torch.logsumexp(flat, -1)
    lt = flat.gather(1, t.clamp(min=0)[:, None]).squeeze(1)
    nll = lse - lt
    if label_smoothing:
        e = label_smoothing
        loss = (1 - e) * nll + e * (lse - flat.mean(-1))
    else:
        loss = nll
    if use_token_scaling:
        loss = torch.exp(lt - lse).detach() * loss
    loss = torch.where(valid, loss, loss.new_zeros(()))
    return loss.sum() / valid.sum().clamp(min=1).to(loss.dtype)


_FEATURES = [
    pytest.param({}, id="plain"),
    pytest.param({"_ignore": True}, id="ignore_index"),
    pytest.param({"logit_softcapping": 30.0}, id="softcap"),
    pytest.param({"label_smoothing": 0.1}, id="label_smoothing"),
    pytest.param({"use_token_scaling": True}, id="token_scaling"),
]


def _check_parity(device: str) -> None:
    torch.manual_seed(0)
    b, t, d = 3, 7, 16
    for feat_param in (p.values[0] for p in _FEATURES):
        feat = dict(feat_param)
        ignore = feat.pop("_ignore", False)
        # vocab 50 -> single chunk; 256 with _CHUNK_VOCAB=64 -> 4 chunks
        for vocab, chunk_vocab in ((50, 16384), (256, 64)):
            old = mod._CHUNK_VOCAB
            mod._CHUNK_VOCAB = chunk_vocab
            try:
                h = (torch.randn(b, t, d, device=device)).requires_grad_(True)
                w = (torch.randn(vocab, d, device=device) * 0.1).requires_grad_(True)
                lab = torch.randint(0, vocab, (b, t), device=device)
                if ignore:
                    lab[:, 1] = -100

                lk = linear_cross_entropy_chunked(h, w, lab, **feat)
                lr = _eager_mean(h, w, lab, **feat)
                assert (lk - lr).abs().item() < _TOL, f"forward {feat} v{vocab}"

                gkh, gkw = torch.autograd.grad(lk, (h, w))
                grh, grw = torch.autograd.grad(lr, (h, w))
                assert (gkh - grh).abs().max().item() < _TOL, f"d_hidden {feat}"
                assert (gkw - grw).abs().max().item() < _TOL, f"d_weight {feat}"

                # vmap(grad) — the DP-SGD per-example path.
                wf = w.detach()
                hd = h.detach()

                def fk(hh, ww, ll, f=feat):
                    return linear_cross_entropy_chunked(hh, ww, ll, **f)

                def fr(hh, ww, ll, f=feat):
                    return _eager_mean(hh, ww, ll, **f)

                # frozen head: grad wrt hidden
                gk = vmap(grad(fk, 0), in_dims=(0, None, 0))(hd, wf, lab)
                gr = vmap(grad(fr, 0), in_dims=(0, None, 0))(hd, wf, lab)
                assert (gk - gr).abs().max().item() < _TOL, f"vmap d_hidden {feat}"
                # trainable head: per-example grad wrt weight [B, V, D]
                gk2 = vmap(grad(fk, (0, 1)), in_dims=(0, None, 0))(hd, wf, lab)
                gr2 = vmap(grad(fr, (0, 1)), in_dims=(0, None, 0))(hd, wf, lab)
                assert (gk2[1] - gr2[1]).abs().max().item() < _TOL, (
                    f"vmap d_weight {feat}"
                )
            finally:
                mod._CHUNK_VOCAB = old


def test_chunked_linear_ce_parity_cpu():
    _check_parity("cpu")


@pytest.mark.mps
@pytest.mark.slow
def test_chunked_linear_ce_parity_mps():
    _check_parity("mps")


def _check_bf16_streams_fp32(device: str) -> None:
    """bf16 inputs are streamed in fp32 (matmul accumulation + LSE), so the loss
    is fp32 and matches the fp32-accurate reference — not the coarse bf16 value
    a naive bf16 matmul would give. Mirrors HF / Triton-CCE fp32 accumulation."""
    torch.manual_seed(0)
    h = torch.randn(2, 8, 16, device=device, dtype=torch.bfloat16)
    w = (torch.randn(4096, 16, device=device) * 0.1).bfloat16()
    lab = torch.randint(0, 4096, (2, 8), device=device)
    loss = linear_cross_entropy_chunked(h, w, lab)
    assert loss.dtype == torch.float32
    ref = _eager_mean(h.float(), w.float(), lab)
    # < 1e-3: a bf16-accumulated matmul would be off by ~1e-2 at this magnitude.
    assert (loss - ref).abs().item() < 1e-3, (loss.item(), ref.item())


def test_chunked_linear_ce_bf16_streams_fp32_cpu():
    _check_bf16_streams_fp32("cpu")


@pytest.mark.mps
def test_chunked_linear_ce_bf16_streams_fp32_mps():
    _check_bf16_streams_fp32("mps")
