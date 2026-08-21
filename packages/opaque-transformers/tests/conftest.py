"""Conftest for opaque-transformers tests.

We install the global HF runtime compat shims (masking, collator, checkpoint
hooks) at module load so they match what
:class:`~opaque.transformers.trainer.DPTrainer` applies during ``__init__``.
Guards: missing sub-packages must not break collection.

Shared LoRA/DP-SGD helpers and ``MODEL_CONFIGS`` live in
``opaque-transformers/tests/_shared.py`` — tests import them directly from that
module. Session-scoped fixture wrappers are re-exported here for convenience.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

# Make sibling `_shared.py` importable without relying on package-relative
# imports (pytest runs under --import-mode=importlib; adding a `tests/__init__.py`
# at multiple package roots collides, so tests/* are treated as plain dirs).
sys.path.append(str(Path(__file__).parent))

from _hf_shared import MODEL_CONFIGS, STANDARD_LORA_CONFIG

# Apply global runtime compat patches (same env semantics as
# DPTrainer.__init__) so test collection matches the trainer's runtime.
try:
    from opaque.transformers.patches import apply_runtime_patches

    apply_runtime_patches(compat=True)
except ImportError:
    pass


@pytest.fixture(scope="session")
def model_configs():
    """Provide model configurations to tests."""
    return MODEL_CONFIGS


@pytest.fixture(scope="session")
def standard_lora_config():
    """Provide standard LoRA configuration."""
    return STANDARD_LORA_CONFIG
