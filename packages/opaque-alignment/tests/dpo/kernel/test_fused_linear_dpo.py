# Copyright (c) 2025 Opaque Authors
# SPDX-License-Identifier: Apache-2.0
"""CPU tests for the chunked fused-linear DPO kernel (plan §7.10, units η.1+η.3).

These tests validate the *numeric* and *composability* contracts of the
pure-PyTorch chunked-preference kernel on CPU:

- **Eager-vs-fused parity** (within ``1e-4``): an all-at-once reference that
  materialises the full ``(2B, T, V)`` logits → per-sequence logp → DPO loss,
  compared against :func:`fused_linear_dpo_loss` with ``chunk_size`` ∈
  {1, 2}. Chunking must not change the result.
- **chunk_size invariance** (within ``1e-5``): the result is identical across
  ``chunk_size`` ∈ {1, 2, 4} — chunking is a pure partition of the pairs axis.
- **grad composability** (within ``1e-4``): ``torch.func.grad`` w.r.t.
  ``hidden_states`` of ``fused_linear_dpo_loss(...).sum()`` is finite and
  matches the eager-reference gradient. The gradient must flow through the chunk
  boundaries identically.
- A couple of ``per_pair_loss_fn`` values (:func:`sigmoid_loss`, :func:`hinge_loss`).

The kernel takes an eager per-pair loss *callable* directly (no string registry;
mirrors ``fused_linear_sft_loss(loss_fn=...)``).

The GPU peak-memory win (``< (B,T,V)``) is validated separately via a Cadence
preset (plan §7.10); these CPU tests cover parity + composability only, so no
``@pytest.mark.cuda`` is needed (the kernel is pure torch).

Imports target the concrete implementation paths because the public façade
wiring is handled by unit η.W.
"""

from __future__ import annotations

import pytest
import torch
from torch.func import grad

from opaque.api.alignment.dpo.kernel._dpo_dispatch import fused_linear_dpo_loss
from opaque.api.alignment.dpo.loss import hinge_loss, sigmoid_loss
from opaque.api.alignment.logprob._sequence import sequence_logp

# Small, fixed dims per plan §7.10.
_B = 4  # number of preference pairs
_T = 6  # sequence length
_H = 8  # hidden size
_V = 16  # vocab size

_PARITY_ATOL = 1e-4
_INVARIANCE_ATOL = 1e-5


def _make_inputs(
    *,
    seed: int = 0,
    requires_grad: bool = False,
    dtype: torch.dtype = torch.float64,
) -> dict[str, torch.Tensor]:
    """Build a deterministic concatenated (2B, ...) DPO input bundle.

    Rows ``[0:B]`` are the chosen sequences, rows ``[B:2B]`` are the rejected
    sequences (the layout documented in ``_dpo_dispatch``). float64 is used so
    parity / gradient tolerances are not dominated by float32 noise.
    """
    gen = torch.Generator().manual_seed(seed)
    two_b = 2 * _B
    hidden = torch.randn(
        two_b, _T, _H, generator=gen, dtype=dtype, requires_grad=requires_grad
    )
    lm_head = torch.randn(_V, _H, generator=gen, dtype=dtype)
    target_ids = torch.randint(0, _V, (two_b, _T), generator=gen)
    # Mask out (at least) the first position so the causal shift has a clear
    # prompt/completion boundary; keep a non-trivial completion span.
    completion_mask = torch.ones(two_b, _T, dtype=dtype)
    completion_mask[:, 0] = 0.0
    ref_chosen = torch.randn(_B, generator=gen, dtype=dtype)
    ref_rejected = torch.randn(_B, generator=gen, dtype=dtype)
    return {
        "hidden_states": hidden,
        "lm_head_weight": lm_head,
        "target_ids": target_ids,
        "completion_mask": completion_mask,
        "ref_chosen_logp": ref_chosen,
        "ref_rejected_logp": ref_rejected,
    }


def _eager_reference(
    hidden_states: torch.Tensor,
    lm_head_weight: torch.Tensor,
    target_ids: torch.Tensor,
    completion_mask: torch.Tensor,
    ref_chosen_logp: torch.Tensor,
    ref_rejected_logp: torch.Tensor,
    *,
    beta: float,
    per_pair_loss_fn,
) -> torch.Tensor:
    """All-at-once reference: materialise the full (2B, T, V) logits.

    This is intentionally the *un-chunked* path — it computes every pair's
    logits in a single matmul, reduces to per-sequence logp, forms log-ratios,
    and evaluates the DPO variant. The fused kernel must match it.
    """
    batch = hidden_states.shape[0] // 2
    logits = hidden_states @ lm_head_weight.transpose(-2, -1)  # (2B, T, V)
    seq_logp = sequence_logp(logits, target_ids, completion_mask)  # (2B,)
    chosen_logp = seq_logp[:batch]
    rejected_logp = seq_logp[batch:]
    chosen_logratio = chosen_logp - ref_chosen_logp
    rejected_logratio = rejected_logp - ref_rejected_logp
    return per_pair_loss_fn(chosen_logratio, rejected_logratio, beta=beta)


# ---------------------------------------------------------------------------
# Eager-vs-fused parity (within 1e-4)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("per_pair_loss_fn", [sigmoid_loss, hinge_loss])
@pytest.mark.parametrize("chunk_size", [1, 2])
def test_eager_vs_fused_parity(per_pair_loss_fn, chunk_size: int) -> None:
    """Fused (chunked) loss matches the all-at-once eager reference."""
    beta = 0.1
    inputs = _make_inputs(seed=1)

    expected = _eager_reference(**inputs, beta=beta, per_pair_loss_fn=per_pair_loss_fn)
    actual = fused_linear_dpo_loss(
        **inputs, beta=beta, per_pair_loss_fn=per_pair_loss_fn, chunk_size=chunk_size
    )

    assert actual.shape == (_B,)
    assert torch.allclose(actual, expected, atol=_PARITY_ATOL, rtol=0.0)


# ---------------------------------------------------------------------------
# chunk_size invariance (within 1e-5)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("per_pair_loss_fn", [sigmoid_loss, hinge_loss])
def test_chunk_size_invariance(per_pair_loss_fn) -> None:
    """Result is identical across chunk_size ∈ {1, 2, 4}."""
    beta = 0.1
    inputs = _make_inputs(seed=2)

    base = fused_linear_dpo_loss(
        **inputs, beta=beta, per_pair_loss_fn=per_pair_loss_fn, chunk_size=1
    )
    for chunk_size in (2, 4):
        other = fused_linear_dpo_loss(
            **inputs,
            beta=beta,
            per_pair_loss_fn=per_pair_loss_fn,
            chunk_size=chunk_size,
        )
        assert torch.allclose(base, other, atol=_INVARIANCE_ATOL, rtol=0.0), (
            f"chunk_size={chunk_size} diverged from chunk_size=1"
        )


def test_chunk_size_covers_uneven_partition() -> None:
    """chunk_size=3 with B=4 (an uneven final chunk) still matches chunk_size=1."""
    beta = 0.2
    inputs = _make_inputs(seed=7)
    base = fused_linear_dpo_loss(
        **inputs, beta=beta, per_pair_loss_fn=sigmoid_loss, chunk_size=1
    )
    uneven = fused_linear_dpo_loss(
        **inputs, beta=beta, per_pair_loss_fn=sigmoid_loss, chunk_size=3
    )
    assert torch.allclose(base, uneven, atol=_INVARIANCE_ATOL, rtol=0.0)


# ---------------------------------------------------------------------------
# grad composability (within 1e-4)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("per_pair_loss_fn", [sigmoid_loss, hinge_loss])
@pytest.mark.parametrize("chunk_size", [1, 2])
def test_grad_composability(per_pair_loss_fn, chunk_size: int) -> None:
    """torch.func.grad w.r.t. hidden_states is finite and matches the eager grad.

    The gradient must flow through the chunk boundaries identically to the
    all-at-once form (chunking is a pure partition; ``torch.cat`` of per-chunk
    losses has the same backward).
    """
    beta = 0.1
    inputs = _make_inputs(seed=3)
    static = {k: v for k, v in inputs.items() if k != "hidden_states"}
    hidden = inputs["hidden_states"]

    def fused_sum(h: torch.Tensor) -> torch.Tensor:
        return fused_linear_dpo_loss(
            h,
            **static,
            beta=beta,
            per_pair_loss_fn=per_pair_loss_fn,
            chunk_size=chunk_size,
        ).sum()

    def eager_sum(h: torch.Tensor) -> torch.Tensor:
        return _eager_reference(
            h, **static, beta=beta, per_pair_loss_fn=per_pair_loss_fn
        ).sum()

    fused_grad = grad(fused_sum)(hidden)
    eager_grad = grad(eager_sum)(hidden)

    assert fused_grad.shape == hidden.shape
    assert torch.isfinite(fused_grad).all()
    assert torch.allclose(fused_grad, eager_grad, atol=_PARITY_ATOL, rtol=0.0)


def test_grad_chunk_size_invariance() -> None:
    """The gradient is itself invariant to chunk_size (within 1e-5)."""
    beta = 0.1
    inputs = _make_inputs(seed=4)
    static = {k: v for k, v in inputs.items() if k != "hidden_states"}
    hidden = inputs["hidden_states"]

    def fused_sum(h: torch.Tensor, chunk_size: int) -> torch.Tensor:
        return fused_linear_dpo_loss(
            h, **static, beta=beta, per_pair_loss_fn=sigmoid_loss, chunk_size=chunk_size
        ).sum()

    grad_1 = grad(lambda h: fused_sum(h, 1))(hidden)
    grad_4 = grad(lambda h: fused_sum(h, 4))(hidden)
    assert torch.allclose(grad_1, grad_4, atol=_INVARIANCE_ATOL, rtol=0.0)


# ---------------------------------------------------------------------------
# Autocast entry safety on CPU + dispatch error paths
# ---------------------------------------------------------------------------


def test_autocast_entry_is_noop_when_inactive_on_cpu() -> None:
    """follow_autocast is a true no-op when no autocast region is active.

    The float32 result computed with autocast *inactive* must be bit-for-bit
    unchanged (same dtype, same values) — i.e. the entry shim does not perturb
    the common CPU-test path.
    """
    beta = 0.1
    inputs = _make_inputs(seed=5, dtype=torch.float32)
    out = fused_linear_dpo_loss(
        **inputs, beta=beta, per_pair_loss_fn=sigmoid_loss
    )
    assert out.dtype == torch.float32
    # Recompute under a no-op (disabled) autocast region: still float32, equal.
    with torch.autocast(device_type="cpu", enabled=False):
        out_disabled = fused_linear_dpo_loss(
            **inputs, beta=beta, per_pair_loss_fn=sigmoid_loss
        )
    assert out_disabled.dtype == torch.float32
    assert torch.equal(out, out_disabled)


def test_autocast_entry_follows_active_cpu_autocast() -> None:
    """Inside an active bf16 CPU autocast region the kernel runs in bf16, safely.

    ``follow_autocast`` is autocast-*aware*: when a region is active it casts the
    float inputs to the autocast dtype (here bf16), so the kernel produces a
    finite bf16 result that tracks the float32 reference within bf16 tolerance.
    The point is that the autocast entry is safe (no crash, no NaN) and honours
    the user's autocast intent rather than silently staying float32.
    """
    beta = 0.1
    inputs = _make_inputs(seed=5, dtype=torch.float32)
    plain = fused_linear_dpo_loss(
        **inputs, beta=beta, per_pair_loss_fn=sigmoid_loss
    )
    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        wrapped = fused_linear_dpo_loss(
            **inputs, beta=beta, per_pair_loss_fn=sigmoid_loss
        )
    assert wrapped.dtype == torch.bfloat16
    assert torch.isfinite(wrapped).all()
    # Compare in float32; bf16 matmul on the (T, V) logits is coarse, so allow a
    # loose tolerance — we only assert the autocast path stays numerically sane.
    assert torch.allclose(wrapped.float(), plain, atol=1e-1, rtol=0.0)


def test_odd_batch_dim_raises() -> None:
    """A non-even leading (2B) dimension is rejected."""
    inputs = _make_inputs(seed=6)
    inputs["hidden_states"] = inputs["hidden_states"][:-1]  # 2B-1 rows
    inputs["target_ids"] = inputs["target_ids"][:-1]
    inputs["completion_mask"] = inputs["completion_mask"][:-1]
    with pytest.raises(ValueError, match="2B"):
        fused_linear_dpo_loss(**inputs, beta=0.1, per_pair_loss_fn=sigmoid_loss)
