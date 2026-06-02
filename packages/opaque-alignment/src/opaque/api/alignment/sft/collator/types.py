"""Output schema for the language-modeling (SFT) collator.

:class:`LMBatch` is the return shape of
:func:`~opaque.api.alignment.sft.collator._language_modeling.language_modeling_collator`,
re-exported publicly at ``opaque.alignment.sft.collator.types`` so callers can
type-hint the collator's output. It is a ``TypedDict`` (not a dataclass) so
downstream code consumes the returned dict directly, with no unpacking step. All
tensor shapes use ``(B, L)`` for micro-batch size ``B`` and padded length ``L``.
"""

from __future__ import annotations

from typing import TypedDict

import torch

__all__ = ["LMBatch"]


class LMBatch(TypedDict, total=False):
    """Output schema for :func:`language_modeling_collator`.

    All present fields are :class:`torch.Tensor` instances with ``dtype=torch.long``.

    Required keys (always present):

    ``input_ids``
        Token ids, shape ``(B, L)``.  Right-padded to ``L`` with the
        ``pad_token_id`` supplied to the factory; each example is first
        truncated (keep-start) to ``max_length`` before padding.
        ``L = min(max_length, max_example_length_in_batch)``, rounded up to
        the nearest multiple of ``pad_to_multiple_of`` when that argument is
        set.

    ``attention_mask``
        Binary mask, shape ``(B, L)``.  ``1`` on real tokens, ``0`` on pad
        positions.

    ``labels``
        Token ids with pad positions replaced by ``-100`` (the
        ``ignore_index`` recognised by :class:`torch.nn.CrossEntropyLoss`),
        shape ``(B, L)``.  When the collator was created with
        ``completion_only_loss=True`` the non-completion positions (where the
        ``completion_mask`` field of the raw example is ``0``) are also set to
        ``-100``.

    Optional keys (present only when applicable):

    ``completion_mask``
        Binary mask, shape ``(B, L)``.  Included when **at least one** input
        example in the batch carries a ``"completion_mask"`` field.  ``1`` on
        completion tokens, ``0`` elsewhere (including pad positions).
    """

    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    labels: torch.Tensor
    completion_mask: torch.Tensor
