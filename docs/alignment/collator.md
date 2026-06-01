# Alignment collators

Collators are **factory functions returning a callable** (AGENTS.md rule 9) —
not user-instantiated classes. Each returned `collate(examples)` produces a
dict of batched tensors.

## `language_modeling_collator`

SFT / causal-LM collation. Output: `input_ids` `(B, L)`, `attention_mask`,
`labels` (pad → `-100`; `completion_only_loss` also masks non-completion
tokens), optional `completion_mask`. Supports `pad_to_multiple_of`. Sequences
longer than `max_length` are truncated keep-start (matching `trl`'s
`SFTTrainer`); no example is dropped.

## `preference_collator`

DPO collation in a **`(B, ...)` layout** — chosen and rejected stay as separate
`(B, Lc)` / `(B, Lr)` tensors rather than TRL's concatenated `(2B, L)`, because
per-example DP-SGD clips per *pair* (the chosen and rejected forwards share one
per-example gradient). Output: `chosen_input_ids`/`chosen_attention_mask`/
`chosen_completion_mask`, the `rejected_*` trio, and optional `ref_chosen_logps`
/ `ref_rejected_logps` `(B,)`.

## `unpaired_preference_collator`

KTO collation. Output: `completion_input_ids`/`completion_attention_mask`/
`completion_labels`, a bool `label` `(B,)`, and — when `calculate_KL` and the
rotated `KL_*` fields are present — `KL_completion_*`; optional
`reference_logps` / `reference_KL_logps`.

::: opaque.alignment.collator

::: opaque.alignment.collator.types
