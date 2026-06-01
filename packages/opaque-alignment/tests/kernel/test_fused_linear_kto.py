# Copyright (c) 2025 Opaque Authors
# SPDX-License-Identifier: Apache-2.0
"""CPU tests for the chunked fused-linear KTO kernel (plan §7.10, units η.2+η.4).

These tests validate the *numeric* and *composability* contracts of the
pure-PyTorch chunked unpaired-preference (KTO) kernel on CPU:

- **Eager-vs-fused parity** (within ``1e-4``): an all-at-once reference that
  materialises the full ``(B, T, V)`` completion logits → per-sequence logp →
  per-example log-ratio → :func:`kto_loss`, compared against
  :func:`opaque_fused_linear_kto_loss` with ``chunk_size`` ∈ {1, 2}. Chunking
  must not change the result. Uses a mixed boolean ``label`` (some desirable,
  some undesirable) and a fixed scalar ``kl``.
- **chunk_size invariance** (within ``1e-5``): the result is identical across
  ``chunk_size`` ∈ {1, 2, 4} — chunking is a pure partition of the batch axis,
  and the scalar ``kl`` is the *same constant* in every chunk.
- **grad composability** (within ``1e-4``): ``torch.func.grad`` w.r.t.
  ``hidden_states`` of ``opaque_fused_linear_kto_loss(...).sum()`` is finite and
  matches the eager-reference gradient. The gradient must flow through the chunk
  boundaries identically.

The GPU peak-memory win (``< (B, T, V)``) is validated separately via a Cadence
preset (plan §7.10); these CPU tests cover parity + composability only, so no
``@pytest.mark.cuda`` is needed (the kernel is pure torch).

Imports target the concrete implementation paths because the public façade
wiring is handled by unit η.W.
"""

from __future__ import annotations

import pytest
import torch
from torch.func import grad

from opaque.api.alignment.kernel._fused_linear_unpaired import (
    fused_linear_unpaired_preference,
)
from opaque.api.alignment.kernel._kto_dispatch import opaque_fused_linear_kto_loss
from opaque.api.alignment.logprob._sequence import sequence_logp
from opaque.api.alignment.kto.loss._kto import kto_loss

# Small, fixed dims per plan §7.10.
_B = 4  # number of completions (examples)
_T = 6  # sequence length
_H = 8  # hidden size
_V = 16  # vocab size

# Fixed scalar KL term (z_0); the caller-computed detached batch-mean. Held
# constant across chunks/grads so we can assert it is broadcast unchanged.
_KL = 0.3

_PARITY_ATOL = 1e-4
_INVARIANCE_ATOL = 1e-5


def _make_inputs(
    *,
    seed: int = 0,
    requires_grad: bool = False,
    dtype: torch.dtype = torch.float64,
) -> dict[str, torch.Tensor]:
    """Build a deterministic (B, ...) unpaired-preference (KTO) input bundle.

    Each row is a single completion with a boolean ``label`` (desirable vs
    undesirable). ``completion_labels`` carries ``-100`` on the prompt span (a
    couple of leading positions per example) so the completion mask has a clear
    prompt/completion boundary. The ``label`` is *mixed* (not all-True /
    all-False) so both branches of :func:`kto_loss` are exercised. float64 is
    used so parity / gradient tolerances are not dominated by float32 noise.
    """
    gen = torch.Generator().manual_seed(seed)
    hidden = torch.randn(
        _B, _T, _H, generator=gen, dtype=dtype, requires_grad=requires_grad
    )
    lm_head = torch.randn(_V, _H, generator=gen, dtype=dtype)
    target_ids = torch.randint(0, _V, (_B, _T), generator=gen)

    # completion_labels: -100 on the prompt span, target ids elsewhere. Vary the
    # prompt length per example so the mask is not uniform across the batch.
    completion_labels = target_ids.clone()
    prompt_lens = torch.tensor([1, 2, 1, 3])[:_B] % _T
    for i in range(_B):
        completion_labels[i, : int(prompt_lens[i].item())] = -100

    # Mixed labels: alternate desirable / undesirable so both kto_loss branches
    # are hit within a single batch (and across chunk boundaries).
    label = torch.tensor([True, False, True, False])[:_B]
    ref_logp = torch.randn(_B, generator=gen, dtype=dtype)
    return {
        "hidden_states": hidden,
        "lm_head_weight": lm_head,
        "target_ids": target_ids,
        "completion_labels": completion_labels,
        "label": label,
        "ref_logp": ref_logp,
    }


def _eager_reference(
    hidden_states: torch.Tensor,
    lm_head_weight: torch.Tensor,
    target_ids: torch.Tensor,
    completion_labels: torch.Tensor,
    label: torch.Tensor,
    ref_logp: torch.Tensor,
    *,
    beta: float,
    kl: torch.Tensor,
) -> torch.Tensor:
    """All-at-once reference: materialise the full (B, T, V) completion logits.

    This is intentionally the *un-chunked* path — it computes every example's
    logits in a single matmul, reduces to per-sequence logp, forms the
    per-example log-ratio, splits it into the chosen/rejected slots via the
    boolean ``label``, and evaluates :func:`kto_loss` with the scalar ``kl``
    broadcast across the batch. The fused kernel must match it.
    """
    logits = hidden_states @ lm_head_weight.transpose(-2, -1)  # (B, T, V)
    completion_mask = (completion_labels != -100).to(logits.dtype)
    seq_logp = sequence_logp(logits, target_ids, completion_mask)  # (B,)
    logratio = seq_logp - ref_logp
    label_f = label.bool().to(logratio.dtype)
    chosen_logratio = logratio * label_f
    rejected_logratio = logratio * (1.0 - label_f)
    return kto_loss(
        chosen_logratio,
        rejected_logratio,
        label.bool(),
        beta=beta,
        kl=kl,
    )


def _kl_scalar(dtype: torch.dtype = torch.float64) -> torch.Tensor:
    """A fixed, detached scalar ``kl`` (z_0) tensor in the given dtype."""
    return torch.tensor(_KL, dtype=dtype)


# ---------------------------------------------------------------------------
# Eager-vs-fused parity (within 1e-4)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("chunk_size", [1, 2])
def test_eager_vs_fused_parity(chunk_size: int) -> None:
    """Fused (chunked) KTO loss matches the all-at-once eager reference."""
    beta = 0.1
    inputs = _make_inputs(seed=1)
    kl = _kl_scalar()

    expected = _eager_reference(**inputs, beta=beta, kl=kl)
    actual = opaque_fused_linear_kto_loss(
        **inputs, beta=beta, kl=kl, chunk_size=chunk_size
    )

    assert actual.shape == (_B,)
    assert torch.allclose(actual, expected, atol=_PARITY_ATOL, rtol=0.0)


def test_eager_vs_fused_parity_all_desirable() -> None:
    """Parity holds when every example is desirable (label all-True)."""
    beta = 0.15
    inputs = _make_inputs(seed=10)
    inputs["label"] = torch.ones(_B, dtype=torch.bool)
    kl = _kl_scalar()

    expected = _eager_reference(**inputs, beta=beta, kl=kl)
    actual = opaque_fused_linear_kto_loss(**inputs, beta=beta, kl=kl, chunk_size=2)
    assert torch.allclose(actual, expected, atol=_PARITY_ATOL, rtol=0.0)


def test_eager_vs_fused_parity_all_undesirable() -> None:
    """Parity holds when every example is undesirable (label all-False)."""
    beta = 0.15
    inputs = _make_inputs(seed=11)
    inputs["label"] = torch.zeros(_B, dtype=torch.bool)
    kl = _kl_scalar()

    expected = _eager_reference(**inputs, beta=beta, kl=kl)
    actual = opaque_fused_linear_kto_loss(**inputs, beta=beta, kl=kl, chunk_size=2)
    assert torch.allclose(actual, expected, atol=_PARITY_ATOL, rtol=0.0)


# ---------------------------------------------------------------------------
# chunk_size invariance (within 1e-5)
# ---------------------------------------------------------------------------


def test_chunk_size_invariance() -> None:
    """Result is identical across chunk_size ∈ {1, 2, 4}.

    The scalar ``kl`` is the *same constant* in every chunk; chunking is a pure
    partition of the batch axis, so the per-example losses do not interact.
    """
    beta = 0.1
    inputs = _make_inputs(seed=2)
    kl = _kl_scalar()

    base = opaque_fused_linear_kto_loss(**inputs, beta=beta, kl=kl, chunk_size=1)
    for chunk_size in (2, 4):
        other = opaque_fused_linear_kto_loss(
            **inputs, beta=beta, kl=kl, chunk_size=chunk_size
        )
        assert torch.allclose(base, other, atol=_INVARIANCE_ATOL, rtol=0.0), (
            f"chunk_size={chunk_size} diverged from chunk_size=1"
        )


def test_chunk_size_covers_uneven_partition() -> None:
    """chunk_size=3 with B=4 (an uneven final chunk) still matches chunk_size=1."""
    beta = 0.2
    inputs = _make_inputs(seed=7)
    kl = _kl_scalar()
    base = opaque_fused_linear_kto_loss(**inputs, beta=beta, kl=kl, chunk_size=1)
    uneven = opaque_fused_linear_kto_loss(**inputs, beta=beta, kl=kl, chunk_size=3)
    assert torch.allclose(base, uneven, atol=_INVARIANCE_ATOL, rtol=0.0)


def test_kl_is_broadcast_as_constant_across_chunks() -> None:
    """The scalar ``kl`` enters every chunk unchanged (not recomputed per chunk).

    Adversarial check: if ``kl`` were (incorrectly) recomputed or rescaled per
    chunk, the loss would shift between ``chunk_size=1`` and ``chunk_size=4``.
    We additionally confirm that changing ``kl`` moves the loss (so it is a live
    input, not silently dropped) while the chunk-invariance still holds for the
    new value.
    """
    beta = 0.1
    inputs = _make_inputs(seed=12)

    for kl_value in (0.0, 0.3, 1.5):
        kl = torch.tensor(kl_value, dtype=torch.float64)
        loss_c1 = opaque_fused_linear_kto_loss(**inputs, beta=beta, kl=kl, chunk_size=1)
        loss_c4 = opaque_fused_linear_kto_loss(**inputs, beta=beta, kl=kl, chunk_size=4)
        assert torch.allclose(loss_c1, loss_c4, atol=_INVARIANCE_ATOL, rtol=0.0)

    # kl is a live input: different kl ⇒ different loss.
    loss_lo = opaque_fused_linear_kto_loss(
        **inputs, beta=beta, kl=torch.tensor(0.0, dtype=torch.float64)
    )
    loss_hi = opaque_fused_linear_kto_loss(
        **inputs, beta=beta, kl=torch.tensor(1.5, dtype=torch.float64)
    )
    assert not torch.allclose(loss_lo, loss_hi, atol=_INVARIANCE_ATOL, rtol=0.0)


# ---------------------------------------------------------------------------
# grad composability (within 1e-4)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("chunk_size", [1, 2])
def test_grad_composability(chunk_size: int) -> None:
    """torch.func.grad w.r.t. hidden_states is finite and matches the eager grad.

    The gradient must flow through the chunk boundaries identically to the
    all-at-once form (chunking is a pure partition; ``torch.cat`` of per-chunk
    losses has the same backward).
    """
    beta = 0.1
    inputs = _make_inputs(seed=3)
    kl = _kl_scalar()
    static = {k: v for k, v in inputs.items() if k != "hidden_states"}
    hidden = inputs["hidden_states"]

    def fused_sum(h: torch.Tensor) -> torch.Tensor:
        return opaque_fused_linear_kto_loss(
            h, **static, beta=beta, kl=kl, chunk_size=chunk_size
        ).sum()

    def eager_sum(h: torch.Tensor) -> torch.Tensor:
        return _eager_reference(h, **static, beta=beta, kl=kl).sum()

    fused_grad = grad(fused_sum)(hidden)
    eager_grad = grad(eager_sum)(hidden)

    assert fused_grad.shape == hidden.shape
    assert torch.isfinite(fused_grad).all()
    assert torch.allclose(fused_grad, eager_grad, atol=_PARITY_ATOL, rtol=0.0)


def test_grad_chunk_size_invariance() -> None:
    """The gradient is itself invariant to chunk_size (within 1e-5)."""
    beta = 0.1
    inputs = _make_inputs(seed=4)
    kl = _kl_scalar()
    static = {k: v for k, v in inputs.items() if k != "hidden_states"}
    hidden = inputs["hidden_states"]

    def fused_sum(h: torch.Tensor, chunk_size: int) -> torch.Tensor:
        return opaque_fused_linear_kto_loss(
            h, **static, beta=beta, kl=kl, chunk_size=chunk_size
        ).sum()

    grad_1 = grad(lambda h: fused_sum(h, 1))(hidden)
    grad_4 = grad(lambda h: fused_sum(h, 4))(hidden)
    assert torch.allclose(grad_1, grad_4, atol=_INVARIANCE_ATOL, rtol=0.0)


def test_detached_kl_has_no_grad_path_to_model() -> None:
    """A *detached* ``kl`` (the Tier-2 contract) leaves no grad path to the loss.

    KTO's ``z_0`` is a stop-gradient batch aggregate: the *caller* must
    ``.detach()`` it before passing it in (plan §8.1; ``kto_loss`` documents that
    detachment is the caller's responsibility and does not re-detach internally).
    This test verifies the contract end-to-end: a graph-bearing source value, fed
    through ``.detach()`` into the kernel, contributes *no* gradient back to its
    source — i.e. the detach severs the path exactly as the privacy ledger
    requires.
    """
    beta = 0.1
    # hidden_states carries the live graph (as the policy forward would), so the
    # backward has a real path to differentiate; kl is fed in detached.
    inputs = _make_inputs(seed=9, requires_grad=True)
    hidden = inputs["hidden_states"]
    # Source value with a live graph, as a trainer's online-KL estimate would be.
    kl_source = torch.tensor(_KL, dtype=torch.float64, requires_grad=True)
    kl = (kl_source * 2.0).detach()  # caller-side detach (the contract)
    assert not kl.requires_grad

    loss = opaque_fused_linear_kto_loss(**inputs, beta=beta, kl=kl).sum()
    hidden_grad, src_grad = torch.autograd.grad(
        loss, (hidden, kl_source), allow_unused=True
    )
    # The policy path is differentiable...
    assert hidden_grad is not None
    assert torch.isfinite(hidden_grad).all()
    # ...but the detached kl severs any path back to its source.
    assert src_grad is None


def test_undetached_kl_grad_is_documented_caller_responsibility() -> None:
    """If the caller forgets to detach ``kl``, the gradient *does* flow through it.

    Adversarial documentation test: the kernel/``kto_loss`` deliberately do NOT
    re-detach ``kl`` internally — detachment is the caller's Tier-2 responsibility
    (plan §8.1). So passing a ``requires_grad`` ``kl`` yields a *non-zero*
    gradient. This pins the contract: the stop-gradient lives at the call site,
    not inside the kernel, and the kernel does not silently mask a caller bug.
    """
    beta = 0.1
    inputs = _make_inputs(seed=9)
    kl = torch.tensor(_KL, dtype=torch.float64, requires_grad=True)

    loss = opaque_fused_linear_kto_loss(**inputs, beta=beta, kl=kl).sum()
    (kl_grad,) = torch.autograd.grad(loss, kl, allow_unused=True)
    assert kl_grad is not None
    assert torch.isfinite(kl_grad).all()


# ---------------------------------------------------------------------------
# Dispatch ⇄ core consistency + autocast entry safety on CPU
# ---------------------------------------------------------------------------


def test_dispatch_matches_core_directly() -> None:
    """The public dispatcher matches calling the chunked core directly.

    The dispatcher only adds the autocast-follow shim (a no-op on CPU), so on
    CPU it must be bit-for-bit identical to the core.
    """
    beta = 0.1
    inputs = _make_inputs(seed=13)
    kl = _kl_scalar()
    via_dispatch = opaque_fused_linear_kto_loss(
        **inputs, beta=beta, kl=kl, chunk_size=2
    )
    via_core = fused_linear_unpaired_preference(
        **inputs, beta=beta, kl=kl, chunk_size=2
    )
    assert torch.equal(via_dispatch, via_core)


def test_autocast_entry_is_noop_when_inactive_on_cpu() -> None:
    """follow_autocast is a true no-op when no autocast region is active.

    The float32 result computed with autocast *inactive* must be bit-for-bit
    unchanged (same dtype, same values) — the entry shim does not perturb the
    common CPU-test path.
    """
    beta = 0.1
    inputs = _make_inputs(seed=5, dtype=torch.float32)
    kl = _kl_scalar(dtype=torch.float32)
    out = opaque_fused_linear_kto_loss(**inputs, beta=beta, kl=kl)
    assert out.dtype == torch.float32
    with torch.autocast(device_type="cpu", enabled=False):
        out_disabled = opaque_fused_linear_kto_loss(**inputs, beta=beta, kl=kl)
    assert out_disabled.dtype == torch.float32
    assert torch.equal(out, out_disabled)


def test_autocast_entry_follows_active_cpu_autocast() -> None:
    """Inside an active bf16 CPU autocast region the kernel runs in bf16, safely.

    ``_follow_autocast`` is autocast-*aware*: when a region is active it casts
    the float inputs to the autocast dtype (here bf16), so the kernel produces a
    finite bf16 result that tracks the float32 reference within bf16 tolerance.
    """
    beta = 0.1
    inputs = _make_inputs(seed=5, dtype=torch.float32)
    kl = _kl_scalar(dtype=torch.float32)
    plain = opaque_fused_linear_kto_loss(**inputs, beta=beta, kl=kl)
    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        wrapped = opaque_fused_linear_kto_loss(**inputs, beta=beta, kl=kl)
    assert wrapped.dtype == torch.bfloat16
    assert torch.isfinite(wrapped).all()
    # bf16 matmul on the (T, V) logits is coarse; only assert the autocast path
    # stays numerically sane.
    assert torch.allclose(wrapped.float(), plain, atol=1e-1, rtol=0.0)


def test_invalid_chunk_size_raises() -> None:
    """A non-positive chunk_size is rejected."""
    inputs = _make_inputs(seed=6)
    kl = _kl_scalar()
    with pytest.raises(ValueError, match="chunk_size"):
        opaque_fused_linear_kto_loss(**inputs, beta=0.1, kl=kl, chunk_size=0)
