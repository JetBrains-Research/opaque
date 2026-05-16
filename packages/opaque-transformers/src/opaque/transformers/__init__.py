"""Hugging Face Trainer integration for Opaque.

This module is a thin façade over the trainer implementation package shipped
in the same wheel. Importing it does **not** mutate Hugging Face globals:
:class:`~opaque.transformers.trainer.DPTrainer` applies runtime compat patches
and ``apply_model_patches(..., compat=use_compat_patches, performance=True, kernels=use_performance_kernels)``
during construction. For scripts that use HF primitives without
``DPTrainer``, call :func:`patch_all` once for global runtime shims.
"""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version as _pkg_version

from opaque.api.transformers._runtime_bootstrap import (
    is_patched,
    is_vmap_patched,
    patch_all,
)
from opaque.api.transformers.trainer import (
    DPTrainer,
    TrainingArguments,
    PredictionOutput,
    TrainOutput,
)

try:
    __version__ = _pkg_version("opaque-transformers")
except PackageNotFoundError:
    __version__ = "0.0.0"

__all__ = [
    "__version__",
    "DPTrainer",
    "TrainingArguments",
    "PredictionOutput",
    "TrainOutput",
    "is_patched",
    "is_vmap_patched",
    "patch_all",
]
