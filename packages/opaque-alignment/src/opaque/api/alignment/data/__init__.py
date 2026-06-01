"""Shared dataset transforms impl — chat-template helpers.

General, method-agnostic formatting helpers (shared). Preference prompt
extraction (``extract_prompt``) is DPO-specific and lives in
:mod:`opaque.api.alignment.dpo.data`.
"""

from opaque.api.alignment.data._chat_template import (
    clone_chat_template,
    get_training_chat_template,
)

__all__ = ["clone_chat_template", "get_training_chat_template"]
