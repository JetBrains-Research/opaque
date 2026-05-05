"""Phase 4c tests: eval-only dtype context manager.

Covers:
- ``bf16_full_eval=True``: model is cast to bf16 inside the eval context
  and restored to the training dtype on exit.
- ``bf16_full_eval=True`` + ``bf16=True``: no-op (model already bf16);
  the model stays bf16 after the context exits.
- ``fp16_full_eval=True``: raises ``NotImplementedError`` at use time.
- Mutual exclusivity between ``bf16_full_eval`` and ``fp16_full_eval``
  is rejected at args construction (already-validated regression test).
"""

from __future__ import annotations

import pytest
import torch

from opaque.transformers.trainer import DPTrainingArguments
from opaque.transformers.trainer._precision import eval_dtype


def _args(**overrides) -> DPTrainingArguments:
    defaults = dict(
        output_dir=None,
        save_strategy="no",
        dp_target_epsilon=10.0,
        dp_noise_multiplier=1.0,
        use_cpu=True,
    )
    defaults.update(overrides)
    return DPTrainingArguments(**defaults)


# ----------------------------------------------------------------------------
# bf16_full_eval cast + restore
# ----------------------------------------------------------------------------


def test_bf16_full_eval_casts_inside_and_restores_on_exit():
    model = torch.nn.Linear(4, 2)
    train_dtype = torch.float32
    args = _args(bf16_full_eval=True)
    assert next(model.parameters()).dtype == torch.float32
    with eval_dtype(model, args, train_dtype):
        assert next(model.parameters()).dtype == torch.bfloat16
    # Restored after context exit.
    assert next(model.parameters()).dtype == torch.float32


def test_bf16_full_eval_with_bf16_training_is_noop():
    """If the model is already bf16, the context is a no-op."""
    model = torch.nn.Linear(4, 2).to(dtype=torch.bfloat16)
    train_dtype = torch.bfloat16
    args = _args(bf16=True, bf16_full_eval=True)
    with eval_dtype(model, args, train_dtype):
        assert next(model.parameters()).dtype == torch.bfloat16
    assert next(model.parameters()).dtype == torch.bfloat16


def test_bf16_full_eval_false_does_nothing():
    model = torch.nn.Linear(4, 2)
    train_dtype = torch.float32
    args = _args()  # bf16_full_eval default False
    with eval_dtype(model, args, train_dtype):
        assert next(model.parameters()).dtype == torch.float32
    assert next(model.parameters()).dtype == torch.float32


def test_bf16_full_eval_restores_even_on_exception():
    """Training dtype is restored even when the eval body raises."""
    model = torch.nn.Linear(4, 2)
    train_dtype = torch.float32
    args = _args(bf16_full_eval=True)
    with pytest.raises(RuntimeError, match="boom"):
        with eval_dtype(model, args, train_dtype):
            assert next(model.parameters()).dtype == torch.bfloat16
            raise RuntimeError("boom")
    # The finally clause must have restored fp32.
    assert next(model.parameters()).dtype == torch.float32


# ----------------------------------------------------------------------------
# fp16_full_eval — full-cast for the eval scope (HF parity)
# ----------------------------------------------------------------------------


def test_fp16_full_eval_casts_model_for_eval_scope():
    """fp16_full_eval=True casts the model to fp16 inside the context and
    restores the train_dtype on exit (HF parity: trainer.py:2661)."""
    model = torch.nn.Linear(4, 2)
    args = _args(fp16_full_eval=True)
    assert next(model.parameters()).dtype == torch.float32
    with eval_dtype(model, args, torch.float32):
        assert next(model.parameters()).dtype == torch.float16
    assert next(model.parameters()).dtype == torch.float32


def test_fp16_full_eval_no_op_when_already_fp16():
    """If the caller pre-cast to fp16, eval_dtype is a no-op; train_dtype
    drives the (no-op) restore on exit."""
    model = torch.nn.Linear(4, 2).to(torch.float16)
    args = _args(fp16_full_eval=True)
    with eval_dtype(model, args, torch.float16):
        assert next(model.parameters()).dtype == torch.float16
    assert next(model.parameters()).dtype == torch.float16


# ----------------------------------------------------------------------------
# Mutual exclusivity (already validated at args construction)
# ----------------------------------------------------------------------------


def test_bf16_and_fp16_full_eval_mutually_exclusive():
    with pytest.raises(ValueError, match="full eval"):
        _args(bf16_full_eval=True, fp16_full_eval=True)
