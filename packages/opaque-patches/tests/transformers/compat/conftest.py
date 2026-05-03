"""Conftest for HuggingFace compatibility tests.

Re-exports fixtures from _helpers (which also provides ``prepare_lora_model``
and ``run_clipped_grad_test`` as plain functions that tests import directly).

Applies ``opaque.patches.apply_runtime_patches()`` once at collection time so that
the runtime patches are in effect for the whole test session. Patching
is opt-in in production; tests opt in via this conftest.
"""

import pytest

from ._helpers import qwen2_config, qwen2_tokenizer  # noqa: F401


@pytest.fixture(scope="session", autouse=True)
def _apply_opaque_hf_patches():
    transformers = pytest.importorskip("transformers")  # noqa: F841
    from opaque.patches import apply_runtime_patches

    apply_runtime_patches()
    yield
