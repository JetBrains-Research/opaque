"""Trainer implementation package (``opaque.api.transformers.trainer``).

Import the stable façade from :mod:`opaque.transformers.trainer` or the
symbols re-exported from :mod:`opaque.transformers`.
"""

from __future__ import annotations

from transformers.trainer_utils import PredictionOutput

from ._config import DPTrainingArguments
from ._dp_trainer import DPTrainer, TrainOutput, default_dp_hp_backend

__all__ = [
    "DPTrainer",
    "DPTrainingArguments",
    "PredictionOutput",
    "TrainOutput",
    "default_dp_hp_backend",
]
