"""DP-SGD ↔ patches integration conftest.

Re-exports the patches fixtures (``qwen2_config``, ``qwen2_tokenizer``)
and applies ``opaque.patches.apply_runtime_patches()`` once at session
scope so the runtime patches are in effect throughout the test
session. Patching is opt-in in production; integration tests opt in
via this conftest.
"""

import pytest

from integration_tests.dpsgd_patches._helpers import (
    qwen2_config,  # noqa: F401  (pytest fixture re-export)
    qwen2_tokenizer,  # noqa: F401  (pytest fixture re-export)
)


@pytest.fixture(scope="session", autouse=True)
def _apply_opaque_hf_patches():
    transformers = pytest.importorskip("transformers")  # noqa: F841
    from opaque.patches import apply_runtime_patches

    apply_runtime_patches()
    yield
