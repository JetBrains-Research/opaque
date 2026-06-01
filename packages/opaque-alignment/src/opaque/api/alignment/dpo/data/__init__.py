"""DPO dataset transforms impl — preference prompt extraction.

``extract_prompt`` is DPO-specific (it derives the shared prompt prefix of a
chosen/rejected pair). Method-agnostic chat-template helpers live in the shared
:mod:`opaque.api.alignment.data`.
"""

from opaque.api.alignment.dpo.data._prompt import extract_prompt

__all__ = ["extract_prompt"]
