"""opaque.alignment.data façade — prompt extraction + chat-template helpers."""

from opaque.api.alignment.data import (
    clone_chat_template,
    extract_prompt,
    get_training_chat_template,
)

__all__ = ["extract_prompt", "clone_chat_template", "get_training_chat_template"]
