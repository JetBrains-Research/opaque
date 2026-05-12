"""DP Trainer public façade.

Implementation lives in :mod:`opaque.api.transformers.trainer`.
"""

from __future__ import annotations

from opaque.api.transformers.trainer import (
    DPTrainer,
    DPTrainingArguments,
    PredictionOutput,
    TrainOutput,
    default_dp_hp_backend,
)

__all__ = [
    "DPTrainer",
    "DPTrainingArguments",
    "PredictionOutput",
    "TrainOutput",
    "default_dp_hp_backend",
]
