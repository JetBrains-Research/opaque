"""Precision helpers for DPTrainer.

Eval-time full-cast: HF parity for ``bf16_full_eval=True`` /
``fp16_full_eval=True`` casts the model in place for the eval scope and
restores it on exit.  This is distinct from training-time precision,
where ``bf16=True`` / ``fp16=True`` enable autocast (no model cast) —
see :meth:`opaque.transformers.trainer.DPTrainer._setup_precision`.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import TYPE_CHECKING, Iterator

import torch

if TYPE_CHECKING:
    from ._config import DPTrainingArguments


@contextmanager
def eval_dtype(
    model: torch.nn.Module,
    args: "DPTrainingArguments",
    train_dtype: torch.dtype,
) -> Iterator[None]:
    """Cast ``model`` to the eval dtype for the duration of the context.

    On entry, casts the model in place if ``args.bf16_full_eval`` or
    ``args.fp16_full_eval`` is set *and* the model isn't already at the
    target dtype.  On exit, restores the model to ``train_dtype``
    (captured at trainer ``__init__``).

    Args:
        model: The model to cast.  Mutated in place (and restored on exit).
        args: Training arguments; reads ``bf16_full_eval`` /
            ``fp16_full_eval``.
        train_dtype: The dtype the model should be in *outside* this
            context — captured by ``DPTrainer.__init__`` so nested calls
            and the no-op case behave correctly.

    Yields:
        Nothing.  The context manager is purely for the side effect.

    HF-parity note: HF's ``bf16_full_eval`` / ``fp16_full_eval`` cast the
    model rather than using autocast (see ``transformers.trainer.py`` line
    ~2660), and we mirror that.
    """
    if args.fp16_full_eval:
        target = torch.float16
    elif args.bf16_full_eval:
        target = torch.bfloat16
    else:
        yield
        return

    current = next(model.parameters()).dtype
    if current == target:
        # Already there (e.g. ``bf16=True`` was also set) — no-op.
        yield
        return

    model.to(dtype=target)
    try:
        yield
    finally:
        model.to(dtype=train_dtype)
