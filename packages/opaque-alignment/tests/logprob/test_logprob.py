# Copyright (c) 2025 Opaque Authors
# SPDX-License-Identifier: Apache-2.0
"""Unit + vmap-safety tests for the logprob primitives (plan §7.5, §11.2/§11.3).

Covers the three pure logprob helpers:

- :func:`selective_log_softmax` — gather of ``log_softmax`` at indices.
- :func:`sequence_logp` — DPO/causal-LM per-sequence completion logp.

Each function has >=3 hand-computed reference cases (tiny tensors where the
expected logp is computed by hand or via a reference ``F.log_softmax`` loop),
plus a ``torch.func.vmap(torch.func.grad(...))`` finite-gradient contract test
(mirrors ``opaque-engine/tests/functional/test_functional.py``). Imports target
the concrete impl paths because the public façade is not wired yet (unit β.W).
"""

from __future__ import annotations

import math

import pytest
import torch
import torch.nn.functional as F  # noqa: N812
from torch.func import grad, vmap

from opaque.api.alignment.logprob._gather import selective_log_softmax
from opaque.api.alignment.logprob._sequence import sequence_logp

# ---------------------------------------------------------------------------
# selective_log_softmax
# ---------------------------------------------------------------------------


def test_selective_log_softmax_hand_uniform() -> None:
    """Uniform logits → log p = log(1/V) for every selected token."""
    v = 4
    logits = torch.zeros(3, v)  # all-equal logits → uniform distribution
    indices = torch.tensor([0, 1, 3])
    out = selective_log_softmax(logits, indices)
    expected = torch.full((3,), math.log(1.0 / v))
    assert out.shape == (3,)
    assert torch.allclose(out, expected, atol=1e-6)


def test_selective_log_softmax_hand_two_class() -> None:
    """Two-logit case computed by hand: logp = z_i - logsumexp(z)."""
    logits = torch.tensor([[1.0, 3.0]])
    lse = math.log(math.exp(1.0) + math.exp(3.0))
    # pick index 1 (the value 3.0)
    out = selective_log_softmax(logits, torch.tensor([1]))
    assert torch.allclose(out, torch.tensor([3.0 - lse]), atol=1e-6)
    # pick index 0 (the value 1.0)
    out0 = selective_log_softmax(logits, torch.tensor([0]))
    assert torch.allclose(out0, torch.tensor([1.0 - lse]), atol=1e-6)


def test_selective_log_softmax_hand_2d_grid() -> None:
    """A (T,) input (no batch axis) returns a scalar-per-position vector."""
    logits = torch.tensor([2.0, 0.0, -1.0])
    lse = torch.logsumexp(logits, dim=-1)
    out = selective_log_softmax(logits, torch.tensor(0))  # 0-dim index
    assert out.shape == ()
    assert torch.allclose(out, logits[0] - lse, atol=1e-6)


def test_selective_log_softmax_matches_reference() -> None:
    """Matches the explicit ``F.log_softmax(...).gather(...).squeeze(-1)``."""
    torch.manual_seed(0)
    logits = torch.randn(2, 5, 7)
    indices = torch.randint(0, 7, (2, 5))
    out = selective_log_softmax(logits, indices)
    ref = F.log_softmax(logits, dim=-1).gather(-1, indices.unsqueeze(-1)).squeeze(-1)
    assert out.shape == (2, 5)
    assert torch.allclose(out, ref, atol=1e-6)


def test_selective_log_softmax_large_logits_stable() -> None:
    """Large logits must not overflow (log-sum-exp trick, never exp(logits))."""
    logits = torch.tensor([[1e4, 1e4 + 1.0, 1e4 - 2.0]])
    out = selective_log_softmax(logits, torch.tensor([1]))
    assert torch.isfinite(out).all()
    # Shift-invariance: subtracting a constant leaves log-softmax unchanged.
    shifted = selective_log_softmax(logits - 1e4, torch.tensor([1]))
    assert torch.allclose(out, shifted, atol=1e-4)


# ---------------------------------------------------------------------------
# sequence_logp
# ---------------------------------------------------------------------------


def _reference_sequence_logp(
    logits: torch.Tensor, input_ids: torch.Tensor, mask: torch.Tensor
) -> torch.Tensor:
    """Independent reference: shift, per-token logp loop, mask, sum."""
    shifted = logits[..., :-1, :]
    targets = input_ids[..., 1:]
    m = mask[..., 1:].to(logits.dtype)
    logp = F.log_softmax(shifted, dim=-1).gather(-1, targets.unsqueeze(-1))
    logp = logp.squeeze(-1)
    return (logp * m).sum(dim=-1)


def test_sequence_logp_hand_single_completion_token() -> None:
    """One completion token: logp equals that single token's logp."""
    # T=2, V=3. Only the prediction for position 0 (target = input_ids[1]) counts.
    logits = torch.tensor([[[0.0, 0.0, 0.0], [5.0, 5.0, 5.0]]])  # (1, 2, 3)
    input_ids = torch.tensor([[2, 1]])  # target at shift is input_ids[1] = 1
    # mask shifted is completion_mask[1:] = [1] → the single prediction counts.
    completion_mask = torch.tensor([[0, 1]])
    out = sequence_logp(logits, input_ids, completion_mask)
    expected = torch.tensor([math.log(1.0 / 3)])  # uniform logits row at pos 0
    assert out.shape == (1,)
    assert torch.allclose(out, expected, atol=1e-6)


def test_sequence_logp_hand_masked_out_is_zero() -> None:
    """An all-zero (shifted) mask yields a zero summed logp."""
    torch.manual_seed(1)
    logits = torch.randn(2, 4, 6)
    input_ids = torch.randint(0, 6, (2, 4))
    completion_mask = torch.zeros(2, 4, dtype=torch.long)
    out = sequence_logp(logits, input_ids, completion_mask)
    assert torch.allclose(out, torch.zeros(2), atol=1e-7)


def test_sequence_logp_hand_two_tokens_sum() -> None:
    """Two completion tokens: result is the sum of their per-token logps."""
    # T=3, V=2. shifts produce 2 predictions; both masked in.
    logits = torch.tensor(
        [[[1.0, 0.0], [0.0, 2.0], [3.0, 0.0]]]  # (1, 3, 2)
    )
    input_ids = torch.tensor([[0, 1, 0]])  # targets after shift: [1, 0]
    completion_mask = torch.tensor([[1, 1, 1]])  # shifted → [1, 1]
    # pos0 predicts token 1: logp = 0 - logsumexp([1,0])
    lse0 = math.log(math.exp(1.0) + math.exp(0.0))
    # pos1 predicts token 0: logp = 0 - logsumexp([0,2])
    lse1 = math.log(math.exp(0.0) + math.exp(2.0))
    expected = torch.tensor([(0.0 - lse0) + (0.0 - lse1)])
    out = sequence_logp(logits, input_ids, completion_mask)
    assert torch.allclose(out, expected, atol=1e-6)


def test_sequence_logp_matches_reference_batched() -> None:
    """Batched (B, T, V) matches the independent reference implementation."""
    torch.manual_seed(2)
    logits = torch.randn(3, 5, 8)
    input_ids = torch.randint(0, 8, (3, 5))
    completion_mask = torch.randint(0, 2, (3, 5))
    out = sequence_logp(logits, input_ids, completion_mask)
    ref = _reference_sequence_logp(logits, input_ids, completion_mask)
    assert out.shape == (3,)
    assert torch.allclose(out, ref, atol=1e-6)


def test_sequence_logp_per_example_matches_batched() -> None:
    """Per-example (T, V) call equals the corresponding batched row.

    Guards against silently dropping the batch axis: a per-example call must
    return a 0-dim scalar, and stacking these must equal the batched result.
    """
    torch.manual_seed(3)
    logits = torch.randn(4, 6, 5)
    input_ids = torch.randint(0, 5, (4, 6))
    completion_mask = torch.randint(0, 2, (4, 6))

    batched = sequence_logp(logits, input_ids, completion_mask)
    per_example = torch.stack(
        [sequence_logp(logits[i], input_ids[i], completion_mask[i]) for i in range(4)]
    )
    # Each per-example result must be a true scalar (no leftover batch axis).
    single = sequence_logp(logits[0], input_ids[0], completion_mask[0])
    assert single.shape == ()
    assert torch.allclose(batched, per_example, atol=1e-6)


def test_sequence_logp_ld_alpha_not_implemented() -> None:
    """ld_alpha is a documented stub: any non-None value raises."""
    logits = torch.randn(1, 3, 4)
    input_ids = torch.randint(0, 4, (1, 3))
    completion_mask = torch.ones(1, 3, dtype=torch.long)
    with pytest.raises(NotImplementedError, match="ld_alpha"):
        sequence_logp(logits, input_ids, completion_mask, ld_alpha=1.0)


def test_sequence_logp_shared_prefix_len_ignored_when_alpha_none() -> None:
    """shared_prefix_len is accepted and ignored when ld_alpha is None."""
    torch.manual_seed(4)
    logits = torch.randn(2, 4, 6)
    input_ids = torch.randint(0, 6, (2, 4))
    completion_mask = torch.ones(2, 4, dtype=torch.long)
    out_with = sequence_logp(
        logits,
        input_ids,
        completion_mask,
        ld_alpha=None,
        shared_prefix_len=torch.tensor([1, 2]),
    )
    out_without = sequence_logp(logits, input_ids, completion_mask)
    assert torch.allclose(out_with, out_without, atol=1e-7)


# ---------------------------------------------------------------------------
# vmap(grad(...)) contract tests (plan §11.2)
# ---------------------------------------------------------------------------


def test_sequence_logp_vmap_grad_finite() -> None:
    """vmap(grad(sequence_logp-sum)) over a (B, T, V) batch yields finite grads.

    Mirrors the ``torch.func.vmap`` / ``torch.func.grad`` style in
    opaque-engine's functional tests. grad is taken over the float ``logits``
    argument (argnum 0); ``input_ids`` and ``completion_mask`` ride along.
    """
    b, t, v = 4, 6, 8
    torch.manual_seed(8)
    logits = torch.randn(b, t, v)
    input_ids = torch.randint(0, v, (b, t))
    completion_mask = torch.randint(0, 2, (b, t))

    def per_example(lg: torch.Tensor, ids: torch.Tensor, m: torch.Tensor):
        return sequence_logp(lg, ids, m).sum()

    grads = vmap(grad(per_example))(logits, input_ids, completion_mask)
    assert grads.shape == (b, t, v)
    assert torch.isfinite(grads).all()


def test_selective_log_softmax_vmap_grad_finite() -> None:
    """vmap(grad(selective_log_softmax-sum)) over a (B, T, V) batch is finite."""
    b, t, v = 4, 6, 8
    torch.manual_seed(10)
    logits = torch.randn(b, t, v)
    indices = torch.randint(0, v, (b, t))

    def per_example(lg: torch.Tensor, idx: torch.Tensor):
        return selective_log_softmax(lg, idx).sum()

    grads = vmap(grad(per_example))(logits, indices)
    assert grads.shape == (b, t, v)
    assert torch.isfinite(grads).all()
