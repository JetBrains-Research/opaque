"""Chat-template helpers for alignment training.

- :func:`clone_chat_template` copies a chat template (and associated special
  tokens, e.g. ``<|im_start|>``) from a source tokenizer onto a destination
  tokenizer, then resizes the model's input-embedding matrix to the new vocab.
- :func:`get_training_chat_template` returns a Jinja2 chat template that wraps
  the assistant turn with ``{% generation %}`` / ``{% endgeneration %}`` so
  ``apply_chat_template(..., return_assistant_tokens_mask=True)`` yields a
  boolean mask usable for ``assistant_only_loss``.

Ordering invariant: :func:`clone_chat_template` mutates the embedding matrix
via ``resize_token_embeddings``. A functional snapshot
(``make_functional`` / ``torch.func.functional_call``) captures the embedding
tensor at snapshot time, so clone the template **before** snapshotting;
otherwise the snapshot keeps the old, smaller matrix and the functional forward
errors at runtime (index-out-of-range in the embedding lookup) the first time a
newly-added token id is seen.
"""

from __future__ import annotations

__all__ = ["clone_chat_template", "get_training_chat_template"]

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from transformers import PreTrainedModel, PreTrainedTokenizerBase

logger = logging.getLogger(__name__)

# Sentinel strings used by Jinja2 chat-template generation markers.
_GEN_START = "{% generation %}"
_GEN_END = "{% endgeneration %}"


def clone_chat_template(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    source_tokenizer_or_path: PreTrainedTokenizerBase | str,
) -> tuple[PreTrainedModel, PreTrainedTokenizerBase, list[int]]:
    """Copy a chat template and special tokens from *source* onto *tokenizer*.

    The function:

    1. Loads the *source* tokenizer (if a path/name string is given, it is
       loaded via ``AutoTokenizer.from_pretrained`` with
       ``trust_remote_code=False``).
    2. Copies ``source.chat_template`` onto ``tokenizer.chat_template``.
    3. Collects every token that is marked ``special=True`` in the source
       tokenizer's ``added_tokens_decoder`` and adds any that are **missing**
       from *tokenizer* via ``tokenizer.add_special_tokens``.
    4. Calls ``model.resize_token_embeddings(len(tokenizer))`` to extend (or
       confirm the size of) the model's input-embedding matrix.

    Ordering invariant: this mutates the model's embedding matrix and must be
    called before any ``make_functional`` / ``torch.func.functional_call``
    snapshot, otherwise the snapshot keeps the old (smaller) matrix and the
    functional forward errors at runtime (embedding index-out-of-range) when a
    newly-added token id is first seen.

    Args:
        model: The :class:`~transformers.PreTrainedModel` whose embedding
            matrix will be resized to accommodate the cloned special tokens.
        tokenizer: The destination :class:`~transformers.PreTrainedTokenizerBase`
            that will receive the chat template and any new special tokens.
        source_tokenizer_or_path: Either a loaded
            :class:`~transformers.PreTrainedTokenizerBase` whose ``chat_template``
            and special tokens are to be copied, **or** a string (local path or
            Hugging Face Hub model ID) that will be passed to
            ``AutoTokenizer.from_pretrained``.

    Returns:
        A ``(model, tokenizer, added_token_ids)`` tuple.  ``model`` and
        ``tokenizer`` are mutated in-place and also returned for convenient
        chaining.  ``added_token_ids`` is the list of vocabulary indices of the
        special tokens newly added to *tokenizer* (empty when none were added).
        Callers fine-tuning with PEFT use it to mark the new embedding rows
        trainable, since a frozen base would never learn an embedding for a
        token that did not exist at pre-training time.

    Raises:
        ValueError: If *source_tokenizer_or_path* does not have a
            ``chat_template`` attribute set (i.e. the attribute is ``None``).
        TypeError: If *source_tokenizer_or_path* is neither a string nor a
            :class:`~transformers.PreTrainedTokenizerBase`.
    """
    # Resolve source tokenizer.
    if isinstance(source_tokenizer_or_path, str):
        from transformers import AutoTokenizer  # lazy import

        source_tokenizer = AutoTokenizer.from_pretrained(
            source_tokenizer_or_path,
            trust_remote_code=False,
        )
    else:
        # Duck-type check: must look like a tokenizer.
        if not hasattr(source_tokenizer_or_path, "chat_template"):
            raise TypeError(
                "source_tokenizer_or_path must be a string path or a "
                "PreTrainedTokenizerBase instance; "
                f"got {type(source_tokenizer_or_path)!r}"
            )
        source_tokenizer = source_tokenizer_or_path

    # Copy chat template.
    chat_template = getattr(source_tokenizer, "chat_template", None)
    if chat_template is None:
        raise ValueError(
            "The source tokenizer does not have a chat_template set "
            "(chat_template is None).  Set it before calling "
            "clone_chat_template."
        )
    tokenizer.chat_template = chat_template

    # Collect new special tokens from source. Only tokens flagged special=True
    # (e.g. <|im_start|>, <|im_end|>, <bos>, <eos>) are propagated; regular
    # sub-word tokens are not copied.
    tokens_to_add: list[str] = []

    added_tokens_decoder: dict = {}
    if hasattr(source_tokenizer, "added_tokens_decoder"):
        added_tokens_decoder = source_tokenizer.added_tokens_decoder  # {id: AddedToken}

    existing_vocab: set[str] = set(tokenizer.get_vocab().keys())

    for added_token in added_tokens_decoder.values():
        token_str = str(added_token)
        is_special = getattr(added_token, "special", False)
        if is_special and token_str not in existing_vocab:
            tokens_to_add.append(token_str)

    added_token_ids: list[int] = []
    if tokens_to_add:
        n_added = tokenizer.add_special_tokens(
            {"additional_special_tokens": tokens_to_add}
        )
        # Resolve the vocabulary indices of the tokens we just added so PEFT
        # callers can mark exactly those embedding rows trainable. Filter out
        # any ``unk``-mapped id defensively (a well-behaved add never yields it).
        unk_id = getattr(tokenizer, "unk_token_id", None)
        for token_str in tokens_to_add:
            tid = tokenizer.convert_tokens_to_ids(token_str)
            if tid is not None and tid != unk_id:
                added_token_ids.append(int(tid))
        logger.info(
            "clone_chat_template: added %d new special token(s) to tokenizer: %s",
            n_added,
            tokens_to_add,
        )
    else:
        logger.debug("clone_chat_template: no new special tokens to add.")

    # Resize model embeddings. Called unconditionally so the invariant
    # ``get_input_embeddings().weight.shape[0] == len(tokenizer)`` holds on
    # exit even when no new tokens were added (resize is a no-op then).
    if hasattr(model, "resize_token_embeddings"):
        model.resize_token_embeddings(len(tokenizer))
    else:
        logger.warning(
            "clone_chat_template: model has no resize_token_embeddings method; "
            "skipping embedding resize.  The embedding size may not match the "
            "tokenizer vocabulary size."
        )

    return model, tokenizer, added_token_ids


def get_training_chat_template(tokenizer: PreTrainedTokenizerBase) -> str:
    """Return a chat-template string with assistant-turn generation markers.

    Wraps the assistant-turn content of *tokenizer*'s chat template with
    ``{% generation %}`` / ``{% endgeneration %}`` Jinja2 markers so that
    ``tokenizer.apply_chat_template(..., return_assistant_tokens_mask=True)``
    returns a per-token boolean mask suitable for ``assistant_only_loss``.

    The function is **idempotent**: if the template already contains
    ``{% generation %}``, it is returned unchanged.

    Strategy (close to TRL ``chat_template_utils.py``):

    - If the tokenizer's current template already contains the generation
      marker block, return it unchanged.
    - Otherwise, locate the *last* occurrence of the Jinja2 content expression
      ``{{ generation_token }}`` (the TRL v2 canonical form) or, as a fallback,
      the last ``{{ message['content'] }}`` expression that appears inside the
      final ``if … 'assistant' …`` branch of the template.  Wrap the matched
      expression with the generation-marker tags.
    - If neither pattern is found (non-standard template), wrap the entire
      template body — between the outer loop end and the final closing brace —
      with a best-effort marker.  A warning is logged when this fallback fires.

    Args:
        tokenizer: A tokenizer whose ``chat_template`` attribute must be a
            non-empty Jinja2 string.

    Returns:
        A Jinja2 template string guaranteed to contain
        ``{% generation %}`` ... ``{% endgeneration %}``.

    Raises:
        ValueError: If ``tokenizer.chat_template`` is ``None`` or empty.
    """
    template: str | None = getattr(tokenizer, "chat_template", None)
    if not template:
        raise ValueError(
            "tokenizer.chat_template is not set.  Assign a Jinja2 template "
            "string to tokenizer.chat_template before calling "
            "get_training_chat_template."
        )

    # Idempotency: already has the marker.
    if _GEN_START in template:
        return template

    # Strategy 1: wrap {{ generation_token }} — TRL v2 canonical placeholder.
    _GENERATION_TOKEN_EXPR = "{{ generation_token }}"
    if _GENERATION_TOKEN_EXPR in template:
        candidate = template.replace(
            _GENERATION_TOKEN_EXPR,
            f"{_GEN_START}{_GENERATION_TOKEN_EXPR}{_GEN_END}",
            1,
        )
        if _generation_block_renders(candidate):
            return candidate

    # Strategy 2: wrap the assistant content expression inside the assistant
    # conditional block. Handles ChatML, Llama-3, Phi-3, etc.
    template_out = _wrap_assistant_content(template)
    if template_out is not None and _generation_block_renders(template_out):
        return template_out

    # Strategy 2b: shared OR-clause rendering (Qwen2.5-Instruct pattern). When
    # user/system/assistant turns share one expression inside a multi-condition
    # ``or`` clause, the last assistant mention is the tool-call ``elif`` and
    # Strategy 2 would wrap an unreachable expression (empty
    # ``generation_indices``). Rewrite the shared expression with an inner
    # role-guard so only the assistant case renders inside the mask.
    template_out = _wrap_shared_or_clause(template)
    if template_out is not None and _generation_block_renders(template_out):
        return template_out

    # Strategy 3: best-effort fallback. Wrap the whole template body; unlikely
    # to yield a correct assistant-only mask but stays syntactically valid.
    logger.warning(
        "get_training_chat_template: could not locate the assistant-turn "
        "content expression in the chat template.  Falling back to wrapping "
        "the entire template body.  The resulting assistant_tokens_mask may "
        "be incorrect.  Consider setting the template manually."
    )
    return f"{_GEN_START}{template}{_GEN_END}"


# Private helpers


def _wrap_assistant_content(template: str) -> str | None:
    """Attempt to insert generation markers around assistant content.

    Searches for the last Jinja2 ``{{ ... content ... }}`` expression that
    occurs inside an ``assistant`` conditional branch (identified by the
    presence of ``'assistant'`` in the preceding ``if`` or ``elif`` tag).

    Returns the modified template string on success, or ``None`` if no
    suitable expression is found.
    """
    import re

    # Find all assistant branch tags ({% if/elif ... 'assistant' ... %}).
    assistant_block_pat = re.compile(
        r"\{%-?\s*(?:if|elif)\b[^%]*['\"]assistant['\"][^%]*-?%\}"
    )
    matches = list(assistant_block_pat.finditer(template))
    if not matches:
        # No explicit assistant branch found.  Try a simpler heuristic:
        # wrap the last {{ ... content ... }} expression in the template.
        content_pat = re.compile(r"\{\{-?\s*[^}]*\bcontent\b[^}]*-?\}\}")
        content_matches = list(content_pat.finditer(template))
        if not content_matches:
            return None
        last = content_matches[-1]
        return (
            template[: last.start()]
            + _GEN_START
            + last.group()
            + _GEN_END
            + template[last.end() :]
        )

    # Take the last assistant branch start.
    last_branch_start = matches[-1].end()

    # Find the first content expression after the branch tag — the one that
    # renders the assistant reply (typically the next {{ … }}).
    substring = template[last_branch_start:]
    content_pat = re.compile(r"\{\{-?\s*[^}]*\bcontent\b[^}]*-?\}\}")
    m = content_pat.search(substring)
    if m is None:
        return None

    abs_start = last_branch_start + m.start()
    abs_end = last_branch_start + m.end()
    return (
        template[:abs_start]
        + _GEN_START
        + template[abs_start:abs_end]
        + _GEN_END
        + template[abs_end:]
    )


def _wrap_shared_or_clause(template: str) -> str | None:
    """Handle templates that render user/system/assistant via one shared expression.

    Some chat templates (notably Qwen2.5-Instruct) place an OR clause like::

        {%- if (message.role == "user") or (message.role == "system" and not loop.first)
              or (message.role == "assistant" and not message.tool_calls) %}
            {{- '<|im_start|>' + message.role + '\\n' + message.content + '<|im_end|>' + '\\n' }}

    inside the message loop.  Wrapping the shared expression with
    ``{% generation %}`` markers would (incorrectly) flag user and system
    tokens as assistant tokens.  This helper rewrites the shared expression
    with an inner ``{%- if message.role == 'assistant' %}`` guard so the
    assistant case renders through a ``{% generation %}``-wrapped branch and
    user/system tokens fall through unflagged.

    Returns ``None`` when the template doesn't match the shared-OR-clause
    pattern (the caller falls back to Strategy 3).
    """
    import re

    or_clause_pat = re.compile(
        r"\{%-?\s*(?:if|elif)\b[^%]*['\"]assistant['\"][^%]*\bor\b[^%]*-?%\}"
        r"|\{%-?\s*(?:if|elif)\b[^%]*\bor\b[^%]*['\"]assistant['\"][^%]*-?%\}"
    )
    match = or_clause_pat.search(template)
    if match is None:
        return None

    # Find the FIRST content expression after the OR clause and before the
    # next Jinja control tag — that's the shared expression we need to split.
    after = template[match.end() :]
    next_ctrl = re.search(r"\{%-?\s*(?:elif|else|endif|endfor)\b", after)
    bound = next_ctrl.start() if next_ctrl else len(after)
    inner = after[:bound]

    content_pat = re.compile(r"\{\{-?\s*[^}]*\bcontent\b[^}]*-?\}\}")
    expr_match = content_pat.search(inner)
    if expr_match is None:
        return None

    expr = expr_match.group()
    abs_start = match.end() + expr_match.start()
    abs_end = match.end() + expr_match.end()

    # Role-guarded variant: wrap the whole expression for the assistant case so
    # every token it emits (content + im_end + newline) is flagged. Flagging
    # im_end is desirable — the model learns to stop.
    guarded = (
        "{%- if message.role == 'assistant' %}"
        + _GEN_START
        + expr
        + _GEN_END
        + "{%- else %}"
        + expr
        + "{%- endif %}"
    )
    return template[:abs_start] + guarded + template[abs_end:]


def _generation_block_renders(template: str) -> bool:
    """Return True iff the candidate template emits non-empty assistant indices.

    Compiles *template* and renders a single ``[user, assistant]`` turn to
    confirm the ``{% generation %}`` block fires at least once.  This is the
    final correctness check after each wrap strategy: a wrap that lands
    inside an unreachable branch (e.g. Qwen2.5-Instruct's tool-call elif)
    produces empty ``generation_indices`` at render time.
    """
    try:
        from transformers.utils.chat_template_utils import (
            _compile_jinja_template,
            _render_with_assistant_indices,
        )
    except ImportError:
        # No transformers installed (e.g. minimal CI env) — skip validation.
        return True

    try:
        compiled = _compile_jinja_template(template)
        msgs = [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "hi"},
        ]
        _, indices = _render_with_assistant_indices(compiled, msgs, None, None, False)
    except Exception:
        # Template won't compile or render — treat as failure of this strategy.
        return False

    # ``indices`` is a list of ``(start, end)`` byte-offset spans; a non-empty,
    # non-degenerate span means the generation block fired during the render.
    return bool(indices) and any(end > start for start, end in indices)
