"""Dataset transforms impl — prompt extraction, KTO rotation, sequence
packing (BFD / wrapped), and chat-template helpers.
"""

from opaque.api.alignment.data._chat_template import (
    clone_chat_template,
    get_training_chat_template,
)
from opaque.api.alignment.data._kto_rotation import rotate_kto_completions
from opaque.api.alignment.data._packing import pack_bfd, pack_bfd_split, pack_wrapped
from opaque.api.alignment.data._prompt import extract_prompt

__all__ = [
    "extract_prompt",
    "rotate_kto_completions",
    "pack_bfd",
    "pack_bfd_split",
    "pack_wrapped",
    "clone_chat_template",
    "get_training_chat_template",
]
