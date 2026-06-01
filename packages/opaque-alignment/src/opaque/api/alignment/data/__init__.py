"""Dataset transforms impl — prompt extraction and chat-template helpers.

The KTO completion rotation lives under :mod:`opaque.api.alignment.kto.data`.
"""

from opaque.api.alignment.data._chat_template import (
    clone_chat_template,
    get_training_chat_template,
)
from opaque.api.alignment.data._prompt import extract_prompt

__all__ = ["extract_prompt", "clone_chat_template", "get_training_chat_template"]
