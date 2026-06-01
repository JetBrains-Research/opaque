"""KTO unpaired-preference collator (plan §7.6, §7.2).

Factory function returning a callable that collates a list of per-example dicts
into a batched ``dict[str, torch.Tensor]`` suitable for KTO training.

Each input example carries:

- ``completion_input_ids`` (``list[int]``) — full prompt + completion token ids.
- ``completion_labels`` (``list[int]``) — same length; prompt positions are
  ``-100`` (ignored), completion positions carry the target token id.
- ``label`` (``bool``) — ``True`` = desirable, ``False`` = undesirable.

Optionally (e.g. after :func:`~opaque.alignment.data.rotate_kto_completions`):

- ``KL_completion_input_ids`` / ``KL_completion_labels`` — the rotated KL
  completion sequence used to estimate the reference KL baseline.

Optionally (when reference logprobs have been pre-computed):

- ``reference_logps`` (``float``) — reference model log-prob for the
  completion.
- ``reference_KL_logps`` (``float``) — reference model log-prob for the KL
  completion.

The factory returns a single callable that closes over ``pad_token_id``,
``max_length``, and ``calculate_KL``.  No user-instantiated class is exposed —
this follows AGENTS.md rule 9 and the factory pattern described in §3.1.

Output tensors
--------------
Always present:

- ``completion_input_ids``     ``(B, L)`` long, right-padded.
- ``completion_attention_mask````(B, L)`` long (1 = real, 0 = pad).
- ``completion_labels``        ``(B, L)`` long, pad positions → ``-100``.
- ``label``                    ``(B,)`` bool.

When ``calculate_KL=True`` **and** all examples carry ``KL_completion_input_ids``:

- ``KL_completion_input_ids``      ``(B, Lk)`` long, right-padded.
- ``KL_completion_attention_mask`` ``(B, Lk)`` long.
- ``KL_completion_labels``         ``(B, Lk)`` long, pad → ``-100``.

When all examples carry ``reference_logps``:

- ``reference_logps`` ``(B,)`` float32.

When all examples carry ``reference_KL_logps``:

- ``reference_KL_logps`` ``(B,)`` float32.

Truncation strategy: keep the **start** (prefix) up to ``max_length`` tokens.
Padding: right-pad with ``pad_token_id`` for input ids / attention mask;
right-pad with ``-100`` for labels.  ``pad_to_multiple_of`` is intentionally
not a parameter here — skip per spec.
"""

from __future__ import annotations

from typing import Callable

import torch

__all__ = ["unpaired_preference_collator"]

_LABEL_PAD_ID: int = -100


def _pad_sequence_right(
    sequences: list[list[int]],
    pad_value: int,
    max_length: int,
) -> torch.Tensor:
    """Right-pad and truncate (keep start) a list of token-id lists.

    Parameters
    ----------
    sequences:
        Variable-length token-id lists (one per example in the batch).
    pad_value:
        Value to use for pad positions.
    max_length:
        Sequences longer than this are truncated to ``max_length`` from the
        start; shorter sequences are right-padded to ``max_length``.

    Returns
    -------
    torch.Tensor
        Shape ``(B, max_length)`` with dtype ``torch.long``.
    """
    b = len(sequences)
    out = torch.full((b, max_length), fill_value=pad_value, dtype=torch.long)
    for i, seq in enumerate(sequences):
        # Truncate keep-start, then write.
        trunc = seq[:max_length]
        out[i, : len(trunc)] = torch.tensor(trunc, dtype=torch.long)
    return out


def _build_attention_mask(input_ids: torch.Tensor, pad_token_id: int) -> torch.Tensor:
    """Return a ``(B, L)`` long mask: 1 where token != pad, 0 where pad."""
    return (input_ids != pad_token_id).long()


def unpaired_preference_collator(
    pad_token_id: int,
    max_length: int,
    *,
    calculate_KL: bool = True,
) -> Callable[[list[dict]], dict[str, torch.Tensor]]:
    """Return a KTO collator callable.

    Parameters
    ----------
    pad_token_id:
        Token id used to right-pad ``completion_input_ids`` and
        ``KL_completion_input_ids``.  Pad positions in the ``*_labels``
        tensors always receive ``-100`` regardless.
    max_length:
        Maximum sequence length.  Sequences are truncated from the **end**
        (keep-start strategy) to this length before padding.
    calculate_KL:
        When ``True`` (default), emit ``KL_completion_*`` keys if — and only
        if — the input batch carries ``KL_completion_input_ids``.  When
        ``False``, the ``KL_completion_*`` keys are never emitted even when the
        inputs contain them.

    Returns
    -------
    Callable[[list[dict]], dict[str, torch.Tensor]]
        A pure function (no hidden state) that accepts a list of per-example
        dicts and returns a batched tensor dict suitable for KTO training.
    """

    def _collate(examples: list[dict]) -> dict[str, torch.Tensor]:
        # ------------------------------------------------------------------
        # 1. Completion sequences (always required)
        # ------------------------------------------------------------------
        completion_input_ids_list: list[list[int]] = [
            ex["completion_input_ids"] for ex in examples
        ]
        completion_labels_list: list[list[int]] = [
            ex["completion_labels"] for ex in examples
        ]

        completion_input_ids = _pad_sequence_right(
            completion_input_ids_list, pad_value=pad_token_id, max_length=max_length
        )
        completion_labels_padded = _pad_sequence_right(
            completion_labels_list, pad_value=_LABEL_PAD_ID, max_length=max_length
        )
        completion_attention_mask = _build_attention_mask(
            completion_input_ids, pad_token_id
        )

        # ------------------------------------------------------------------
        # 2. Label (bool) tensor
        # ------------------------------------------------------------------
        label_tensor = torch.tensor(
            [bool(ex["label"]) for ex in examples], dtype=torch.bool
        )

        out: dict[str, torch.Tensor] = {
            "completion_input_ids": completion_input_ids,
            "completion_attention_mask": completion_attention_mask,
            "completion_labels": completion_labels_padded,
            "label": label_tensor,
        }

        # ------------------------------------------------------------------
        # 3. KL completion sequences (conditional)
        # ------------------------------------------------------------------
        has_kl = all("KL_completion_input_ids" in ex for ex in examples)
        if calculate_KL and has_kl:
            kl_input_ids_list: list[list[int]] = [
                ex["KL_completion_input_ids"] for ex in examples
            ]
            kl_labels_list: list[list[int]] = [
                ex["KL_completion_labels"] for ex in examples
            ]

            kl_input_ids = _pad_sequence_right(
                kl_input_ids_list, pad_value=pad_token_id, max_length=max_length
            )
            kl_labels_padded = _pad_sequence_right(
                kl_labels_list, pad_value=_LABEL_PAD_ID, max_length=max_length
            )
            kl_attention_mask = _build_attention_mask(kl_input_ids, pad_token_id)

            out["KL_completion_input_ids"] = kl_input_ids
            out["KL_completion_attention_mask"] = kl_attention_mask
            out["KL_completion_labels"] = kl_labels_padded

        # ------------------------------------------------------------------
        # 4. Optional pre-computed reference log-probs
        # ------------------------------------------------------------------
        if all("reference_logps" in ex for ex in examples):
            out["reference_logps"] = torch.tensor(
                [float(ex["reference_logps"]) for ex in examples],
                dtype=torch.float32,
            )

        if all("reference_KL_logps" in ex for ex in examples):
            out["reference_KL_logps"] = torch.tensor(
                [float(ex["reference_KL_logps"]) for ex in examples],
                dtype=torch.float32,
            )

        return out

    return _collate
