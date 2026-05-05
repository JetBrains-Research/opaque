"""Unit tests for opaque.transformers.trainer._eval helpers.

Covers ``should_run_eval_at_step``, ``_PredictionAccumulator``, and
``validate_eval_args`` without instantiating a model — every parity rule
for the eval loop is exercised in milliseconds.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pytest
import torch

from opaque.transformers.trainer._eval import (
    EvalLoopOutput,
    EvalPrediction,
    _PredictionAccumulator,
    should_run_eval_at_step,
    validate_eval_args,
    with_metric_prefix,
)


# ---------------------------------------------------------------------------
# Stand-in args object
# ---------------------------------------------------------------------------


@dataclass
class _Args:
    """Minimal stand-in for DPTrainingArguments — only the fields helpers read."""

    eval_strategy: str = "no"
    eval_delay: float = 0.0
    eval_steps: int | None = None
    eval_accumulation_steps: int | None = None
    eval_do_concat_batches: bool = True
    prediction_loss_only: bool = False
    batch_eval_metrics: bool = False
    include_for_metrics: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# should_run_eval_at_step
# ---------------------------------------------------------------------------


class TestShouldRunEvalAtStep:
    def test_strategy_no_never_fires(self):
        a = _Args(eval_strategy="no")
        for step in (0, 1, 5, 100):
            assert should_run_eval_at_step(a, step, 0.0, 1) is False

    def test_strategy_steps_fires_at_cadence(self):
        a = _Args(eval_strategy="steps")
        # eval_steps_resolved=4 → fires at 4, 8, 12, ...
        assert should_run_eval_at_step(a, 4, 0.0, 4) is True
        assert should_run_eval_at_step(a, 8, 0.0, 4) is True
        assert should_run_eval_at_step(a, 5, 0.0, 4) is False

    def test_strategy_steps_never_at_zero(self):
        a = _Args(eval_strategy="steps")
        assert should_run_eval_at_step(a, 0, 0.0, 4) is False

    def test_strategy_steps_with_delay(self):
        # eval_delay=10, eval_steps=2, max_steps=12 → fires at 10 and 12
        a = _Args(eval_strategy="steps", eval_delay=10)
        assert should_run_eval_at_step(a, 2, 0.0, 2) is False
        assert should_run_eval_at_step(a, 4, 0.0, 2) is False
        assert should_run_eval_at_step(a, 8, 0.0, 2) is False
        assert should_run_eval_at_step(a, 10, 0.0, 2) is True
        assert should_run_eval_at_step(a, 12, 0.0, 2) is True

    def test_strategy_epoch_fires_only_at_integer_epoch(self):
        a = _Args(eval_strategy="epoch")
        # Step-based gating: should NOT fire mid-epoch even if step matches.
        assert should_run_eval_at_step(a, 5, 0.5, 5) is False
        # Epoch boundary: fires.
        assert should_run_eval_at_step(a, 0, 1.0, 1) is True
        assert should_run_eval_at_step(a, 0, 2.0, 1) is True

    def test_strategy_epoch_never_at_epoch_zero(self):
        a = _Args(eval_strategy="epoch")
        assert should_run_eval_at_step(a, 0, 0.0, 1) is False

    def test_strategy_epoch_with_delay(self):
        # eval_delay=1.0 with epoch strategy: skip epoch 1, fire at epoch 2.
        a = _Args(eval_strategy="epoch", eval_delay=2.0)
        assert should_run_eval_at_step(a, 0, 1.0, 1) is False
        assert should_run_eval_at_step(a, 0, 2.0, 1) is True


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
        for v in (0.1, 0.2, 0.3):
            _add_batch(acc, v)
        preds, labels, inputs, losses = acc.finalize()
        assert preds is None
        assert labels is None
        # Per-example losses (HF parity): each batch's reduced loss is
        # repeated by ``batch_size``; with bs=2 over 3 batches that's 6.
        assert losses is not None
        assert losses.shape == (6,)
        assert losses[0].item() == pytest.approx(0.1)
        assert losses[1].item() == pytest.approx(0.1)
        assert losses[5].item() == pytest.approx(0.3)

    def test_do_concat_false_returns_lists(self):
        acc = _PredictionAccumulator(eval_do_concat_batches=False)
        for _ in range(3):
            _add_batch(acc, 0.5)
        preds, labels, _, _ = acc.finalize()
        assert isinstance(preds, list) and len(preds) == 3
        assert isinstance(labels, list) and len(labels) == 3

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
        # HF parity: ``EvalPrediction.losses`` is per-example, length =
        # total samples (sum of batch sizes), not per-batch.
        acc = _PredictionAccumulator(include_losses=True)
        for v in (0.1, 0.2, 0.3, 0.4):
            _add_batch(acc, v)
        _, _, _, losses = acc.finalize()
        assert losses is not None
        # 4 batches × bs=2 = 8 per-example losses.
        assert losses.shape == (8,)
        # Each batch's value is repeated by its batch size.
        assert losses[0].item() == pytest.approx(0.1)
        assert losses[1].item() == pytest.approx(0.1)
        assert losses[6].item() == pytest.approx(0.4)
        assert losses[7].item() == pytest.approx(0.4)

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
# validate_eval_args
# ---------------------------------------------------------------------------


class TestValidateEvalArgs:
    def test_no_op_at_defaults(self):
        validate_eval_args(_Args(), compute_metrics=None)  # does not raise

    def test_batch_eval_metrics_without_compute_metrics_raises(self):
        a = _Args(batch_eval_metrics=True)
        with pytest.raises(ValueError, match="batch_eval_metrics=True"):
            validate_eval_args(a, compute_metrics=None)

    def test_batch_eval_metrics_with_compute_metrics_passes(self):
        a = _Args(batch_eval_metrics=True)
        validate_eval_args(
            a,
            compute_metrics=lambda ep, compute_result=False: {"x": 0.0},
        )

    def test_batch_eval_metrics_requires_compute_result_parameter(self):
        a = _Args(batch_eval_metrics=True)
        with pytest.raises(ValueError, match="compute_result"):
            validate_eval_args(a, compute_metrics=lambda ep, **kw: {"x": 0.0})

    def test_unknown_include_for_metrics_key_raises(self):
        a = _Args(include_for_metrics=["foo"])
        with pytest.raises(ValueError, match="include_for_metrics"):
            validate_eval_args(a, compute_metrics=None)

    def test_known_include_for_metrics_keys_pass(self):
        validate_eval_args(_Args(include_for_metrics=["inputs"]), None)
        validate_eval_args(_Args(include_for_metrics=["loss"]), None)
        validate_eval_args(_Args(include_for_metrics=["inputs", "loss"]), None)


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

    def test_eval_loop_output_constructible(self):
        out = EvalLoopOutput(
            predictions=None,
            label_ids=None,
            metrics={"eval_loss": 0.0},
            num_samples=0,
        )
        assert out.metrics == {"eval_loss": 0.0}
