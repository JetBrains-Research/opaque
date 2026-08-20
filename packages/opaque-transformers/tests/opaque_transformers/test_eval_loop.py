"""Unit tests for opaque.api.transformers.trainer._eval helpers.

Covers ``_PredictionAccumulator`` and the reporting helpers without
instantiating a model.  Evaluation cadence comes from HF's
``DefaultFlowCallback`` and is covered end-to-end in
``tests/validation/test_dp_trainer.py``.
"""

from __future__ import annotations

import pytest
import torch

from opaque.api.transformers.trainer._eval import (
    EvalPrediction,
    EvaluationResult,
    _PredictionAccumulator,
    with_metric_prefix,
)

# ---------------------------------------------------------------------------
# _PredictionAccumulator
# ---------------------------------------------------------------------------


def _b(loss: float, logits_shape=(2, 4), labels_shape=(2, 4)) -> tuple:
    """Build a fake batch's tensors.

    Inputs is the bare main-input tensor (HF parity: the accumulator
    collects ``inputs[main_input_name]`` as a single tensor, not a
    dict).
    """
    return (
        torch.tensor(loss),
        torch.randn(*logits_shape),
        torch.randint(0, 10, labels_shape),
        torch.zeros(logits_shape[0], dtype=torch.long),
    )


def _add_batch(acc: _PredictionAccumulator, loss: float, **kw) -> None:
    """Helper that derives ``batch_size`` from the synthetic batch shape."""
    args = _b(loss, **kw)
    acc.add(*args, batch_size=int(args[1].shape[0]))


class TestPredictionAccumulator:
    def test_default_concatenates_at_finalize(self):
        acc = _PredictionAccumulator()
        for _ in range(3):
            _add_batch(acc, 0.5)
        preds, labels, inputs, losses = acc.finalize()
        assert preds.shape == (6, 4)
        assert labels.shape == (6, 4)
        assert inputs is None  # include_inputs=False
        assert losses is None  # include_losses=False

    def test_loss_only_short_circuits(self):
        acc = _PredictionAccumulator(prediction_loss_only=True, include_losses=True)
        # ``include_losses`` now stores only **real** 1-D per-example
        # losses (produced by the vmap'd eval closure).  Pass per-example
        # tensors of length ``batch_size`` directly.
        for vals in ([0.1, 0.2], [0.3, 0.4], [0.5, 0.6]):
            acc.add(
                loss=torch.tensor(vals),
                logits=torch.randn(2, 4),
                labels=torch.randint(0, 10, (2, 4)),
                inputs=torch.zeros(2, dtype=torch.long),
                batch_size=2,
            )
        preds, labels, _inputs, losses = acc.finalize()
        assert preds is None
        assert labels is None
        assert losses is not None
        assert losses.shape == (6,)
        # Real per-example: each value distinct, not replicated.
        assert losses[0].item() == pytest.approx(0.1)
        assert losses[1].item() == pytest.approx(0.2)
        assert losses[5].item() == pytest.approx(0.6)

    def test_do_concat_false_returns_lists(self):
        acc = _PredictionAccumulator(eval_do_concat_batches=False)
        for _ in range(3):
            _add_batch(acc, 0.5)
        preds, labels, _, _ = acc.finalize()
        assert isinstance(preds, list)
        assert len(preds) == 3
        assert isinstance(labels, list)
        assert len(labels) == 3

    def test_accumulation_steps_flush_cadence(self):
        # 5 batches with eval_accumulation_steps=2 → flushes after batch 2
        # and batch 4 (2 cold chunks).  ``finalize`` performs a third flush
        # for the trailing batch (5th), producing 3 cold chunks total.
        acc = _PredictionAccumulator(eval_accumulation_steps=2)
        for _ in range(5):
            _add_batch(acc, 0.5)
        # Two flushes before finalize (the trailing batch is still hot).
        assert len(acc._cold_logits) == 2
        acc.finalize()
        assert len(acc._cold_logits) == 3

    def test_accumulation_steps_none_no_intermediate_flushes(self):
        acc = _PredictionAccumulator(eval_accumulation_steps=None)
        for _ in range(4):
            _add_batch(acc, 0.5)
        # No flushes happen during ``add`` when accumulation_steps is None.
        assert acc._cold_logits == []
        acc.finalize()
        # ``finalize`` performs the single trailing flush.
        assert len(acc._cold_logits) == 1

    def test_include_inputs_populates_finalize(self):
        # HF parity: ``EvalPrediction.inputs`` is a bare tensor (the
        # main input column collected via
        # ``inputs_decode = inputs[main_input_name]``), not a dict.
        acc = _PredictionAccumulator(include_inputs=True)
        for _ in range(3):
            _add_batch(acc, 0.5)
        _, _, inputs, _ = acc.finalize()
        assert inputs is not None
        assert inputs.shape == (6,)

    def test_include_losses_is_per_example(self):
        # Real per-example losses: each entry distinct.
        acc = _PredictionAccumulator(include_losses=True)
        for batch_vals in ([0.1, 0.2], [0.3, 0.4], [0.5, 0.6], [0.7, 0.8]):
            acc.add(
                loss=torch.tensor(batch_vals),
                logits=torch.randn(2, 4),
                labels=torch.randint(0, 10, (2, 4)),
                inputs=torch.zeros(2, dtype=torch.long),
                batch_size=2,
            )
        _, _, _, losses = acc.finalize()
        assert losses is not None
        assert losses.shape == (8,)
        # Per-example values are stored as-is, not replicated batch-mean.
        for idx, expected in enumerate([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]):
            assert losses[idx].item() == pytest.approx(expected)

    def test_include_losses_drops_scalar_batch_mean(self):
        # Scalar (batch-mean) losses carry no per-example information so
        # the accumulator silently drops them — populating
        # ``EvalPrediction.losses`` from a replicated mean is fake by
        # construction.  Users wanting real per-example losses must set
        # ``include_for_metrics=['loss']`` so ``prediction_step`` takes
        # the vmap'd eval path.
        acc = _PredictionAccumulator(include_losses=True)
        for v in (0.1, 0.2, 0.3):
            _add_batch(acc, v)
        _, _, _, losses = acc.finalize()
        assert losses is None

    def test_missing_loss_still_collects_predictions(self):
        acc = _PredictionAccumulator(include_losses=True)
        logits = torch.randn(2, 3)
        labels = torch.arange(2)
        acc.add(
            loss=None,
            logits=logits,
            labels=labels,
            inputs=None,
            batch_size=2,
        )
        preds, label_ids, _, losses = acc.finalize()
        assert preds.shape == (2, 3)
        assert label_ids.shape == (2,)
        assert losses is None

    def test_tuple_predictions_and_labels_round_trip(self):
        acc = _PredictionAccumulator()
        acc.add(
            loss=torch.tensor(0.5),
            logits=(torch.randn(2, 3), torch.randn(2, 4)),
            labels=(torch.arange(2), torch.arange(2) + 10),
            inputs=None,
            batch_size=2,
        )
        acc.add(
            loss=torch.tensor(0.6),
            logits=(torch.randn(1, 3), torch.randn(1, 4)),
            labels=(torch.arange(1), torch.arange(1) + 10),
            inputs=None,
            batch_size=1,
        )
        preds, label_ids, _, _ = acc.finalize()
        assert isinstance(preds, tuple)
        assert isinstance(label_ids, tuple)
        assert preds[0].shape == (3, 3)
        assert preds[1].shape == (3, 4)
        assert label_ids[0].shape == (3,)
        assert label_ids[1].shape == (3,)


# ---------------------------------------------------------------------------
# with_metric_prefix
# ---------------------------------------------------------------------------


class TestWithMetricPrefix:
    def test_adds_missing_prefix(self):
        out = with_metric_prefix({"acc": 0.9, "f1": 0.8}, "eval")
        assert out == {"eval_acc": 0.9, "eval_f1": 0.8}

    def test_preserves_existing_prefix(self):
        out = with_metric_prefix({"eval_acc": 0.9, "f1": 0.8}, "eval")
        assert out == {"eval_acc": 0.9, "eval_f1": 0.8}

    def test_empty_prefix_is_passthrough(self):
        out = with_metric_prefix({"acc": 0.9}, "")
        assert out == {"acc": 0.9}


# ---------------------------------------------------------------------------
# Re-exports
# ---------------------------------------------------------------------------


class TestReExports:
    def test_eval_prediction_accepts_all_fields(self):
        ep = EvalPrediction(
            predictions=torch.zeros(2, 3),
            label_ids=torch.zeros(2, 3, dtype=torch.long),
            inputs={"input_ids": torch.zeros(2, dtype=torch.long)},
            losses=torch.zeros(1),
        )
        assert ep.predictions is not None
        assert ep.label_ids is not None
        # ``inputs`` and ``losses`` are positional/keyword fields across HF
        # versions; accessing them must not raise.
        assert getattr(ep, "inputs", None) is not None
        assert getattr(ep, "losses", None) is not None

    def test_evaluation_result_constructible(self):
        out = EvaluationResult(
            predictions=None,
            label_ids=None,
            metrics={"eval_loss": 0.0},
            num_samples=0,
        )
        assert out.metrics == {"eval_loss": 0.0}
