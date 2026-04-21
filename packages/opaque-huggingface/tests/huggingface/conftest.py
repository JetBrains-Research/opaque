"""Conftest for HuggingFace compatibility tests.

Re-exports fixtures from _helpers (which also provides ``prepare_lora_model``
and ``run_clipped_grad_test`` as plain functions that tests import directly).
"""

from ._helpers import qwen2_config, qwen2_tokenizer  # noqa: F401
