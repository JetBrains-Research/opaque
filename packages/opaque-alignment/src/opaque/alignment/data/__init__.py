"""opaque.alignment.data façade — re-exports dataset transforms."""

from opaque.api.alignment.data import (
    clone_chat_template,
    extract_prompt,
    get_training_chat_template,
    pack_bfd,
    pack_bfd_split,
    pack_wrapped,
    rotate_kto_completions,
)

__all__ = [
    "extract_prompt",
    "rotate_kto_completions",
    "pack_bfd",
    "pack_bfd_split",
    "pack_wrapped",
    "clone_chat_template",
    "get_training_chat_template",
]
