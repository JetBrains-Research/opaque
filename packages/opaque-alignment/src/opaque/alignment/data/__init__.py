"""opaque.alignment.data façade — prompt extraction + chat-template helpers.

The KTO completion rotation lives under :mod:`opaque.alignment.kto.data`.
"""

from opaque.api.alignment.data import (
    clone_chat_template,
    extract_prompt,
    get_training_chat_template,
)

__all__ = ["extract_prompt", "clone_chat_template", "get_training_chat_template"]
