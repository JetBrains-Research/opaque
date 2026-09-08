# Copyright (c) 2025 Opaque Authors
# SPDX-License-Identifier: Apache-2.0
"""Portable memory-efficient linear cross-entropy (no Triton).

A pure-PyTorch, chunked, custom-autograd replacement for the Triton
``Opaque_LinearCrossEntropyLoss``. It never materializes the full
``(tokens, vocab)`` logit matrix — the forward streams an online log-sum-exp
over vocab chunks and the backward recomputes each chunk — so peak memory on
MPS/CPU (where Triton is unavailable) stays bounded by one ``(tokens, chunk)``
tile instead of the whole vocab. Linear projections retain the input precision
used by eager ``matmul`` while LSE and probability arithmetic run in FP32.

Composes with ``vmap(grad(...))`` via ``generate_vmap_rule`` so the DP-SGD
per-example path works identically to the Triton kernel, and supports the same
feature surface: ``logit_softcapping`` (Gemma2), HF-style ``label_smoothing``,
and ``use_token_scaling`` (DFT). Grad-w.r.t.-weight is emitted only when the
weight requires grad (``needs_input_grad[1]``), so a frozen LoRA head skips it.

The autograd.Function operates on the already-shifted, flattened ``e`` /
``targets`` and returns the per-token loss; the label shift, ignore masking, and
reduction live in the wrappers (regular autograd). Keeping them outside the
Function is what preserves the memory win under ``functorch`` ``vmap(grad)`` —
folding the shift/sum into ``forward`` makes the transform retain the per-chunk
forward activations and defeats the streaming.
"""

from __future__ import annotations

import torch

# Vocab columns materialized per chunk by default. Callers can lower this for
# large DP-vmap microbatches, where the physical tile also includes examples.
# Backward recomputes one chunk at a time to form gradients.
_CHUNK_VOCAB = 16384


def _num_chunks(vocab: int, chunk_vocab: int | None = None) -> int:
    width = _CHUNK_VOCAB if chunk_vocab is None else chunk_vocab
    return max(1, (vocab + width - 1) // width)


def _softcap(logits: torch.Tensor, softcap: float | None) -> torch.Tensor:
    if softcap is None:
        return logits
    return softcap * torch.tanh(logits / softcap)


def _compute_dtype(e: torch.Tensor, weight: torch.Tensor) -> torch.dtype:
    """Stream in at least fp32: upcast bf16/fp16 inputs, preserve fp32/fp64."""
    return torch.promote_types(
        torch.promote_types(e.dtype, weight.dtype), torch.float32
    )


def _linear_chunk(e: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """Match eager linear precision, then promote logits for CE statistics."""
    compute_dtype = _compute_dtype(e, weight)
    if e.dtype != weight.dtype:
        return e.to(compute_dtype) @ weight.to(compute_dtype).t()
    return (e @ weight.t()).to(compute_dtype)


def _stream_lse(
    e,
    weight,
    targets,
    softcap,
    chunks,
    logit_scale=1.0,
    need_logit_target=True,
    need_sum_logits=False,
):
    """One streamed pass over vocab chunks.

    Returns ``(lse, logit_target, sum_logits)`` each shaped ``(N,)`` — the
    log-sum-exp, the (softcapped) logit at each row's target, and the row sum of
    all (softcapped) logits. ``logit_target`` / ``sum_logits`` are ``None``
    unless requested: each is a per-chunk accumulator, and computing one that
    isn't needed still pins every chunk's logits in the functorch ``vmap(grad)``
    graph. No ``(N, V)`` tensor lives past a single chunk.
    """
    N = e.shape[0]
    V = weight.shape[0]
    Vc = (V + chunks - 1) // chunks
    # Match eager linear precision, then stream CE statistics in >= fp32.
    cdt = _compute_dtype(e, weight)
    m = torch.full((N,), float("-inf"), dtype=cdt, device=e.device)
    s = torch.zeros(N, dtype=cdt, device=e.device)
    logit_target = (
        torch.zeros(N, dtype=cdt, device=e.device) if need_logit_target else None
    )
    sum_logits = torch.zeros(N, dtype=cdt, device=e.device) if need_sum_logits else None
    zero = torch.zeros((), dtype=cdt, device=e.device)
    for c in range(chunks):
        lo, hi = c * Vc, min((c + 1) * Vc, V)
        if lo >= hi:
            break
        lc = _softcap(
            _linear_chunk(e, weight[lo:hi]) * logit_scale, softcap
        )  # (N, hi-lo)
        cmax = torch.maximum(m, lc.max(-1).values)
        s = s * torch.exp(m - cmax) + torch.exp(lc - cmax[:, None]).sum(-1)
        m = cmax
        if need_sum_logits:
            sum_logits = sum_logits + lc.sum(-1)
        if need_logit_target:
            sel = (targets >= lo) & (targets < hi)
            idx = (targets - lo).clamp(0, hi - lo - 1)
            logit_target = logit_target + torch.where(
                sel, lc.gather(1, idx[:, None]).squeeze(1), zero
            )
    lse = m + torch.log(s)
    return lse, logit_target, sum_logits


def _per_token_loss(lse, logit_target, sum_logits, vocab, label_smoothing):
    """Per-token CE from the streamed stats (HF ``cross_entropy`` semantics)."""
    nll = lse - logit_target
    if label_smoothing:
        eps = label_smoothing
        # (1-eps)*nll + eps*(lse - mean_v logit_v)
        return (1.0 - eps) * nll + eps * (lse - sum_logits / vocab)
    return nll


class _ChunkedLinearCE(torch.autograd.Function):
    """Chunked linear + cross-entropy, per token.

    Operates on the pre-shifted, flattened ``e`` (N, D) and ``targets`` (N,) and
    returns the per-token loss (N,). The shift / ignore-mask / reduction live in
    the wrappers.
    """

    generate_vmap_rule = True

    @classmethod
    def apply(cls, *args):
        """Return per-token loss while retaining internal statistics outputs."""
        return super().apply(*args)[0]

    @staticmethod
    def forward(
        e,
        weight,
        targets,
        logit_softcapping=0,
        label_smoothing=0.0,
        use_token_scaling=False,
        logit_scale=1.0,
        chunk_vocab=None,
    ):
        softcap = logit_softcapping if logit_softcapping != 0 else None
        lse, logit_target, sum_logits = _stream_lse(
            e,
            weight,
            targets,
            softcap,
            _num_chunks(weight.shape[0], chunk_vocab),
            logit_scale,
            need_sum_logits=float(label_smoothing) != 0.0,
        )
        loss = _per_token_loss(
            lse, logit_target, sum_logits, weight.shape[0], float(label_smoothing)
        )
        if use_token_scaling:
            # Detached confidence p_t = softmax(logits)[target] (DFT).
            token_weight = torch.exp(logit_target - lse).detach()
            loss = token_weight * loss
        else:
            token_weight = lse.new_empty(0)
        return loss, lse, token_weight

    @staticmethod
    def setup_context(ctx, inputs, output):
        (
            e,
            weight,
            targets,
            logit_softcapping,
            label_smoothing,
            use_token_scaling,
            logit_scale,
            chunk_vocab,
        ) = inputs
        _, lse, token_weight = output
        ctx.mark_non_differentiable(lse, token_weight)
        ctx.save_for_backward(e, weight, targets, lse, token_weight)
        ctx.softcap = logit_softcapping if logit_softcapping != 0 else None
        ctx.label_smoothing = float(label_smoothing)
        ctx.use_token_scaling = bool(use_token_scaling)
        ctx.logit_scale = float(logit_scale)
        ctx.chunk_vocab = chunk_vocab

    @staticmethod
    def backward(ctx, grad_loss, _grad_lse, _grad_token_weight):
        e, weight, targets, lse, token_weight = ctx.saved_tensors
        softcap = ctx.softcap
        eps = ctx.label_smoothing
        compute_dc = ctx.needs_input_grad[1]
        V = weight.shape[0]
        chunks = _num_chunks(V, ctx.chunk_vocab)
        Vc = (V + chunks - 1) // chunks

        cdt = _compute_dtype(e, weight)
        row = grad_loss.to(cdt)
        if ctx.use_token_scaling:
            row = row * token_weight

        # CE derivatives are FP32, then cross the same cast boundary as eager
        # BF16 logits before each linear backward matmul.
        low_precision_linear = (
            e.dtype in {torch.float16, torch.bfloat16} and weight.dtype == e.dtype
        )
        ef = e if low_precision_linear else e.to(cdt)
        grad_e = torch.zeros_like(e, dtype=cdt)
        # Accumulate per-vocab-chunk weight grads out-of-place and concat: under
        # vmap the weight is shared (unbatched) while gl@e is per-example, so an
        # in-place slice-assign into a (V, D) buffer is illegal — cat over the
        # vocab dim keeps each piece batched correctly.
        w_chunks: list[torch.Tensor] | None = [] if compute_dc else None
        for c in range(chunks):
            lo, hi = c * Vc, min((c + 1) * Vc, V)
            if lo >= hi:
                break
            wc = weight[lo:hi] if low_precision_linear else weight[lo:hi].to(cdt)
            lc = _softcap(_linear_chunk(ef, wc) * ctx.logit_scale, softcap)
            p = torch.exp(lc - lse[:, None])  # softmax chunk
            sel = (targets >= lo) & (targets < hi)
            idx = (targets - lo).clamp(0, hi - lo - 1)
            # q_v = (1-eps)*onehot + eps/V ; dloss/dlogit = p - q
            if eps:
                p.sub_(eps / V)
            target_mass = sel[:, None].to(p.dtype) * (1.0 - eps)
            p.scatter_add_(1, idx[:, None], -target_mass)
            gl = p
            if softcap is not None:
                gl.mul_(1.0 - (lc / softcap) ** 2)  # tanh-cap chain rule
            gl.mul_(ctx.logit_scale)
            gl.mul_(row[:, None])
            linear_grad = gl.to(e.dtype) if low_precision_linear else gl
            grad_e = grad_e + (linear_grad @ wc).to(cdt)
            if compute_dc:
                w_chunks.append(linear_grad.t() @ ef)
        grad_e = grad_e.to(e.dtype)
        grad_w = torch.cat(w_chunks, dim=0).to(weight.dtype) if compute_dc else None
        return grad_e, grad_w, None, None, None, None, None, None


def linear_nll_sum_chunked(
    hidden_states,
    weight,
    labels,
    ignore_index=-100,
    logit_softcapping=0,
    label_smoothing=0.0,
    use_token_scaling=False,
    logit_scale=1.0,
    chunk_vocab=None,
):
    """Unreduced NLL sum over non-ignored tokens.

    Drop-in for ``Opaque_LinearCrossEntropyLoss.apply`` (same positional args,
    same return): HF-style label shift (position ``i`` predicts ``labels[i+1]``);
    the caller handles the mean reduction. Ignored positions contribute zero.
    """
    e = hidden_states[..., :-1, :].contiguous().flatten(0, -2)  # (N, D)
    targets = labels[..., 1:].contiguous().flatten()  # (N,)
    nll = _ChunkedLinearCE.apply(
        e,
        weight,
        targets,
        logit_softcapping,
        label_smoothing,
        use_token_scaling,
        logit_scale,
        chunk_vocab,
    )
    valid = targets != ignore_index
    return torch.where(valid, nll, nll.new_zeros(())).sum()


def linear_cross_entropy_chunked(
    hidden_states,
    weight,
    labels,
    ignore_index=-100,
    logit_softcapping=0,
    label_smoothing=0.0,
    use_token_scaling=False,
    logit_scale=1.0,
    chunk_vocab=None,
):
    """Mean-reduced chunked linear CE — matches ``opaque_linear_cross_entropy_loss``.

    Divides the NLL sum by the count of non-ignored tokens.
    """
    nll_sum = linear_nll_sum_chunked(
        hidden_states,
        weight,
        labels,
        ignore_index,
        logit_softcapping,
        label_smoothing,
        use_token_scaling,
        logit_scale,
        chunk_vocab,
    )
    targets = labels[..., 1:].contiguous().flatten()
    n_valid = (targets != ignore_index).sum().clamp(min=1).to(nll_sum.dtype)
    return nll_sum / n_valid
