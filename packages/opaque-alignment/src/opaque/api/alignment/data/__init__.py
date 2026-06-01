"""Dataset transforms impl — prompt extraction + KTO rotation (packing /
chat-template arrive in Phase θ).
"""

from opaque.api.alignment.data._kto_rotation import rotate_kto_completions
from opaque.api.alignment.data._prompt import extract_prompt

__all__ = ["extract_prompt", "rotate_kto_completions"]
