"""Unit tests for the alignment metrics (§7.9).

Hand-computed reference cases for each metric primitive:

- :func:`reward_metrics` — chosen/rejected means, accuracy fraction, margin
  formula on tiny per-example tensors.
- :func:`kl_estimator` — ``policy - ref`` with ``clamp_min`` and the
  detach flag.
- :func:`entropy_from_logits` — uniform logits over ``V`` classes give
  entropy ``log(V)``; masking selects the contributing positions.
- :func:`mean_token_accuracy` — a hand-built shifted next-token case.

Every metric is telemetry (§3.3 telemetry rule), so each test also asserts
the returned tensor is detached (``requires_grad is False``) and that masked
reductions do not divide by zero on an all-masked input.
"""

from __future__ import annotations

import math

import pytest
import torch

from opaque.api.alignment.metric._kl import kl_estimator
from opaque.api.alignment.metric._reward import reward_metrics
from opaque.api.alignment.metric._token import (
    entropy_from_logits,
    mean_token_accuracy,
)


# --------------------------------------------------------------------------- #
# reward_metrics
# --------------------------------------------------------------------------- #
def test_reward_metrics_reference() -> None:
    chosen = torch.tensor([1.0, 2.0, 0.5])
    rejected = torch.tensor([0.0, 3.0, 0.5])
    beta = 0.5
    out = reward_metrics(chosen, rejected, beta=beta)

    # rewards/chosen = beta * mean(chosen) = 0.5 * (3.5 / 3)
    assert out["rewards/chosen"].item() == pytest.approx(0.5 * (3.5 / 3))
    # rewards/rejected = beta * mean(rejected) = 0.5 * (3.5 / 3)
    assert out["rewards/rejected"].item() == pytest.approx(0.5 * (3.5 / 3))
    # accuracies = fraction where chosen > rejected: [T, F, F] -> 1/3
    assert out["rewards/accuracies"].item() == pytest.approx(1.0 / 3.0)
    # margins = beta * mean(chosen - rejected) = 0.5 * mean([1, -1, 0]) = 0
    assert out["rewards/margins"].item() == pytest.approx(0.0)


def test_reward_metrics_all_chosen_win() -> None:
    chosen = torch.tensor([2.0, 5.0])
    rejected = torch.tensor([1.0, 4.0])
    out = reward_metrics(chosen, rejected, beta=1.0)
    assert out["rewards/accuracies"].item() == pytest.approx(1.0)
    # margins = 1.0 * mean([1, 1]) = 1.0
    assert out["rewards/margins"].item() == pytest.approx(1.0)


def test_reward_metrics_detached_and_scalar() -> None:
    chosen = torch.tensor([1.0, 2.0], requires_grad=True)
    rejected = torch.tensor([0.0, 1.0], requires_grad=True)
    out = reward_metrics(chosen, rejected, beta=0.3)
    for key, value in out.items():
        assert value.requires_grad is False, key
        assert value.ndim == 0, key


# --------------------------------------------------------------------------- #
# kl_estimator
# --------------------------------------------------------------------------- #
def test_kl_estimator_reference() -> None:
    policy = torch.tensor([1.0, -2.0, 0.5])
    ref = torch.tensor([0.0, 0.0, 0.25])
    # raw kl = [1.0, -2.0, 0.25]; clamp at 0 -> [1.0, 0.0, 0.25]
    kl = kl_estimator(policy, ref)
    assert torch.allclose(kl, torch.tensor([1.0, 0.0, 0.25]))


def test_kl_estimator_custom_clamp() -> None:
    policy = torch.tensor([0.1, 0.2])
    ref = torch.tensor([1.0, 0.0])
    # raw kl = [-0.9, 0.2]; clamp at -0.5 -> [-0.5, 0.2]
    kl = kl_estimator(policy, ref, clamp_min=-0.5)
    assert torch.allclose(kl, torch.tensor([-0.5, 0.2]))


def test_kl_estimator_detach_default() -> None:
    policy = torch.tensor([1.0, 2.0], requires_grad=True)
    ref = torch.tensor([0.0, 0.0], requires_grad=True)
    kl = kl_estimator(policy, ref)
    assert kl.requires_grad is False


def test_kl_estimator_no_detach_keeps_graph() -> None:
    policy = torch.tensor([1.0, 2.0], requires_grad=True)
    ref = torch.tensor([0.0, 0.0])
    kl = kl_estimator(policy, ref, detach=False)
    assert kl.requires_grad is True


# --------------------------------------------------------------------------- #
# entropy_from_logits
# --------------------------------------------------------------------------- #
def test_entropy_uniform_logits_equals_log_v() -> None:
    vocab = 7
    # Equal logits over V classes -> uniform softmax -> entropy = log(V).
    logits = torch.zeros(4, vocab)
    mask = torch.ones(4)
    ent = entropy_from_logits(logits, mask)
    assert ent.item() == pytest.approx(math.log(vocab))


def test_entropy_masked_positions_only() -> None:
    vocab = 5
    logits = torch.zeros(3, vocab)
    # One-hot-ish very peaked position should have ~0 entropy, but it is masked
    # out, so the mean must equal log(V) from the two uniform positions.
    logits[2] = torch.tensor([100.0, 0.0, 0.0, 0.0, 0.0])
    mask = torch.tensor([1.0, 1.0, 0.0])
    ent = entropy_from_logits(logits, mask)
    assert ent.item() == pytest.approx(math.log(vocab))


def test_entropy_all_masked_no_divide_by_zero() -> None:
    logits = torch.zeros(3, 6)
    mask = torch.zeros(3)
    ent = entropy_from_logits(logits, mask)
    assert ent.item() == pytest.approx(0.0)
    assert torch.isfinite(ent).item()


def test_entropy_detached() -> None:
    logits = torch.zeros(2, 4, requires_grad=True)
    mask = torch.ones(2)
    ent = entropy_from_logits(logits, mask)
    assert ent.requires_grad is False
    assert ent.ndim == 0


# --------------------------------------------------------------------------- #
# mean_token_accuracy
# --------------------------------------------------------------------------- #
def test_mean_token_accuracy_reference() -> None:
    # seq length 4, vocab 3. After shift, predictions at positions [0,1,2]
    # are compared against labels[1:] = labels[1], labels[2], labels[3].
    # Build logits so argmax along vocab is deterministic.
    logits = torch.tensor(
        [
            [
                [9.0, 0.0, 0.0],  # pos 0 -> argmax 0
                [0.0, 9.0, 0.0],  # pos 1 -> argmax 1
                [0.0, 0.0, 9.0],  # pos 2 -> argmax 2
                [9.0, 0.0, 0.0],  # pos 3 -> dropped by shift
            ]
        ]
    )
    labels = torch.tensor([[5, 0, 2, 2]])  # labels[1:] = [0, 2, 2]
    mask = torch.tensor([[1, 1, 1, 1]])
    # shifted preds = [0, 1, 2]; shifted labels = [0, 2, 2]; shifted mask = [1,1,1].
    # correct = [True, False, True] -> 2 / 3.
    acc = mean_token_accuracy(logits, labels, mask)
    assert acc.item() == pytest.approx(2.0 / 3.0)


def test_mean_token_accuracy_respects_mask() -> None:
    logits = torch.tensor(
        [
            [
                [9.0, 0.0],  # pos 0 -> argmax 0
                [9.0, 0.0],  # pos 1 -> argmax 0
                [0.0, 9.0],  # pos 2 -> argmax 1
            ]
        ]
    )
    labels = torch.tensor([[0, 0, 1]])  # labels[1:] = [0, 1]
    # shifted mask = mask[1:] = [1, 0]: only the first shifted position counts.
    mask = torch.tensor([[1, 1, 0]])
    # preds[0:2] = [0, 0]; labels[1:] = [0, 1]; only position 0 is unmasked and
    # correct -> 1 / 1 = 1.0.
    acc = mean_token_accuracy(logits, labels, mask)
    assert acc.item() == pytest.approx(1.0)


def test_mean_token_accuracy_all_masked_no_divide_by_zero() -> None:
    logits = torch.zeros(1, 4, 3)
    labels = torch.zeros(1, 4, dtype=torch.long)
    mask = torch.zeros(1, 4, dtype=torch.long)
    acc = mean_token_accuracy(logits, labels, mask)
    assert acc.item() == pytest.approx(0.0)
    assert torch.isfinite(acc).item()


def test_mean_token_accuracy_detached() -> None:
    logits = torch.zeros(1, 4, 3, requires_grad=True)
    labels = torch.zeros(1, 4, dtype=torch.long)
    mask = torch.ones(1, 4, dtype=torch.long)
    acc = mean_token_accuracy(logits, labels, mask)
    assert acc.requires_grad is False
    assert acc.ndim == 0
