"""Conftest for opaque-transformers tests.

Importing ``opaque.performance`` and ``opaque.transformers`` auto-applies
their respective patches. We import them here at module load so the patches
are live for the whole ``packages/opaque-transformers/tests`` tree (validation,
distributed, huggingface subpackages).

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

from _hf_shared import MODEL_CONFIGS, STANDARD_LORA_CONFIG  # noqa: E402

# Touch both sub-packages so their on-import patching runs before any test
# module is collected. Guarded: missing sub-packages must not break collection.
try:
    import opaque.performance  # noqa: F401
except ImportError:
    pass
try:
    import opaque.transformers  # noqa: F401
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
