"""Trainer implementation package (``opaque.api.transformers.trainer``).

Import the stable façade from :mod:`opaque.transformers.trainer` or the
symbols re-exported from :mod:`opaque.transformers`.
"""

from __future__ import annotations

from ._config import TrainingArguments
from ._dp_trainer import DPTrainer, EvaluationResult, TrainOutput

__all__ = [
    "DPTrainer",
    "EvaluationResult",
    "TrainingArguments",
    "TrainOutput",
]
