"""DP Trainer public façade.

Implementation lives in :mod:`opaque.api.transformers.trainer`.
"""

from __future__ import annotations

from opaque.api.transformers.trainer import (
    DPTrainer,
    EvaluationResult,
    TrainingArguments,
    TrainOutput,
)

__all__ = [
    "DPTrainer",
    "EvaluationResult",
    "TrainingArguments",
    "TrainOutput",
]
