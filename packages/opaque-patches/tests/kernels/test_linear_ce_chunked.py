# Copyright (c) 2025 Opaque Authors
# SPDX-License-Identifier: Apache-2.0
"""Portable chunked linear-CE (non-Triton MPS/CPU path) parity tests.

The complement of ``test_linear_cross_entropy.py`` (CUDA + Triton): this pins
the pure-PyTorch chunked kernel that runs where Triton is unavailable. It must
match eager ``matmul + cross_entropy`` precision staging within streaming
roundoff for every feature, on the direct call and under ``vmap(grad)`` (the
DP-SGD path), with frozen and trainable lm-head weight.

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
    logit_scale=1.0,
):
    """Materialized, vmap-safe reference matching the kernel's math + reduction."""
    sc = logit_softcapping or None
    flat = _softcap((hidden[..., :-1, :] @ weight.transpose(-1, -2)) * logit_scale, sc)
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
    pytest.param({"logit_scale": 0.25, "logit_softcapping": 30.0}, id="scaled_softcap"),
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


def test_chunked_linear_ce_backward_does_not_allocate_dense_onehot(monkeypatch):
    torch.manual_seed(0)
    h = torch.randn(2, 8, 16, requires_grad=True)
    w = torch.randn(256, 16, requires_grad=True)
    labels = torch.randint(0, 256, (2, 8))
    allocated_shapes = []
    original_zeros_like = torch.zeros_like

    def record_zeros_like(tensor, *args, **kwargs):
        allocated_shapes.append(tuple(tensor.shape))
        return original_zeros_like(tensor, *args, **kwargs)

    old = mod._CHUNK_VOCAB
    mod._CHUNK_VOCAB = 64
    try:
        loss = linear_cross_entropy_chunked(h, w, labels)
        monkeypatch.setattr(torch, "zeros_like", record_zeros_like)
        torch.autograd.grad(loss, (h, w))
    finally:
        mod._CHUNK_VOCAB = old

    assert (14, 64) not in allocated_shapes


def test_chunked_linear_ce_bounds_probability_tile_width(monkeypatch):
    torch.manual_seed(0)
    h = torch.randn(1, 3, 4, requires_grad=True)
    w = torch.randn(20_000, 4)
    labels = torch.randint(0, w.shape[0], (1, 3))
    exp_widths = []
    original_exp = torch.exp

    def record_exp(tensor, *args, **kwargs):
        if tensor.ndim == 2:
            exp_widths.append(tensor.shape[-1])
        return original_exp(tensor, *args, **kwargs)

    monkeypatch.setattr(torch, "exp", record_exp)
    loss = linear_cross_entropy_chunked(h, w, labels, chunk_vocab=2048)
    torch.autograd.grad(loss, h)

    assert exp_widths
    assert max(exp_widths) <= 2048


@pytest.mark.parametrize("vmapped", [False, True], ids=["direct", "vmap"])
@pytest.mark.parametrize("use_token_scaling", [False, True], ids=["plain", "scaled"])
@pytest.mark.parametrize("has_ignored", [False, True], ids=["all-valid", "ignored"])
def test_chunked_backward_reuses_forward_statistics(
    monkeypatch, vmapped, use_token_scaling, has_ignored
):
    torch.manual_seed(1)
    b, t, d, vocab = 2, 6, 8, 32
    hidden = torch.randn(b, t, d)
    weight = torch.randn(vocab, d)
    labels = torch.randint(0, vocab, (b, t))
    if has_ignored:
        labels[:, 2] = -100

    calls = 0
    original = mod._stream_lse

    def recording_stream_lse(*args, **kwargs):
        nonlocal calls
        calls += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(mod, "_stream_lse", recording_stream_lse)

    def loss(h, w, lab):
        return linear_cross_entropy_chunked(
            h, w, lab, use_token_scaling=use_token_scaling
        )

    if vmapped:
        vmap(grad(loss, (0, 1)), in_dims=(0, None, 0))(hidden, weight, labels)
    else:
        hidden.requires_grad_(True)
        weight.requires_grad_(True)
        torch.autograd.grad(loss(hidden, weight, labels), (hidden, weight))

    assert calls == 1


@pytest.mark.mps
def test_chunked_linear_ce_parity_mps():
    _check_parity("mps")


def _check_bf16_streams_fp32(device: str) -> None:
    """BF16 linear projections are promoted before FP32 CE statistics."""
    torch.manual_seed(0)
    h = torch.randn(2, 8, 16, device=device, dtype=torch.bfloat16)
    w = (torch.randn(4096, 16, device=device) * 0.1).bfloat16()
    lab = torch.randint(0, 4096, (2, 8), device=device)
    loss = linear_cross_entropy_chunked(h, w, lab)
    assert loss.dtype == torch.float32
    logits = (h[..., :-1, :] @ w.t()).float()
    targets = lab[..., 1:]
    ref = torch.nn.functional.cross_entropy(
        logits.reshape(-1, logits.shape[-1]), targets.reshape(-1)
    )
    assert (loss - ref).abs().item() < 1e-3, (loss.item(), ref.item())


def test_chunked_linear_ce_bf16_streams_fp32_cpu():
    _check_bf16_streams_fp32("cpu")


@pytest.mark.mps
def test_chunked_linear_ce_bf16_streams_fp32_mps():
    _check_bf16_streams_fp32("mps")
