"""Trainer implementation package (``opaque.api.transformers.trainer``).

Import the stable façade from :mod:`opaque.transformers.trainer` or the
symbols re-exported from :mod:`opaque.transformers`.
"""

from __future__ import annotations

from . import types
from ._dp_trainer import DPTrainer
from ._training_arguments import TrainingArguments

__all__ = [
    "DPTrainer",
    "TrainingArguments",
    "types",
]
