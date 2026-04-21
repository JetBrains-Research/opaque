"""Conftest for opaque-huggingface tests.

Applies ``opaque.patch_all()`` once per session so HF/vmap/kernel patches are
in effect for the whole ``packages/opaque-huggingface/tests`` tree (validation,
distributed, huggingface subpackages). Patching is opt-in in production; tests
opt in via this conftest.

Shared LoRA/DP-SGD helpers and ``MODEL_CONFIGS`` live in
``opaque-huggingface/tests/_shared.py`` — tests import them directly from that
module. Session-scoped fixture wrappers are re-exported here for convenience.
"""

from __future__ import annotations

import pytest

from ._shared import MODEL_CONFIGS, STANDARD_LORA_CONFIG


@pytest.fixture(scope="session", autouse=True)
def _apply_opaque_patches():
    """Opt into performance + HF patches for HF-backed tests."""
    import opaque

    try:
        opaque.patch_all()
    except Exception:
        # If a sub-package is missing, patch_all still works (hook is None);
        # any unexpected error must not break test collection.
        pass
    yield


@pytest.fixture(scope="session")
def model_configs():
    """Provide model configurations to tests."""
    return MODEL_CONFIGS


@pytest.fixture(scope="session")
def standard_lora_config():
    """Provide standard LoRA configuration."""
    return STANDARD_LORA_CONFIG
