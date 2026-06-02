"""Chat-template helpers for alignment training.

Ported from ``trl/chat_template_utils.py`` (TRL ≥ 0.12, lines 28-119).
The two public functions here support:

1. :func:`clone_chat_template` — copy a chat template (and any associated
   special tokens, e.g. ``<|im_start|>``) from a source tokenizer onto a
   destination tokenizer, then resize the model's input-embedding matrix to
   match the new vocabulary size.

2. :func:`get_training_chat_template` — return a Jinja2 chat-template string
   that wraps the assistant-turn content with ``{% generation %}`` /
   ``{% endgeneration %}`` markers so that
   ``tokenizer.apply_chat_template(..., return_assistant_tokens_mask=True)``
   yields a boolean mask usable for ``assistant_only_loss``.

**Risk α10 — ordering invariant (MUST READ before use):**
:func:`clone_chat_template` calls ``model.resize_token_embeddings(len(tokenizer))``
which *mutates* the model's embedding matrix (adds new rows initialised from a
multivariate-normal distribution fitted to the existing rows). Any functional
snapshot taken with ``opaque.engine.functional.make_functional`` (or
``torch.func.functional_call``) captures the embedding-weight tensor *at
snapshot time*. If you call :func:`clone_chat_template` **after** taking a
snapshot the snapshot will contain the *old*, smaller embedding matrix and the
vocabulary expansion will be invisible to the functional forward pass.

**Correct order:**

.. code-block:: python

    model, tokenizer = clone_chat_template(model, tokenizer, source_path)
    # ONLY NOW snapshot the model:
    fmodel, params, buffers = make_functional(model)

Calling ``clone_chat_template`` after ``make_functional`` will silently
produce a shape mismatch when ``fmodel`` is applied to token IDs that require
the expanded vocabulary.
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

# The marker block that encloses the content expression of an assistant turn.
# We wrap *the entire generation span* (content plus surrounding whitespace
# captured by the template) with these markers so that
# ``apply_chat_template(..., return_assistant_tokens_mask=True)`` produces a
# correct per-token boolean mask.


def clone_chat_template(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    source_tokenizer_or_path: PreTrainedTokenizerBase | str,
) -> tuple[PreTrainedModel, PreTrainedTokenizerBase]:
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

    **Risk α10 — CRITICAL ordering invariant:**
    This function mutates the model's embedding matrix.  It **MUST** be called
    **before** any ``make_functional`` / ``torch.func.functional_call``
    snapshot is taken.  If you call it after snapshotting, the snapshot will
    contain the old (smaller) embedding matrix and the expanded vocabulary will
    be invisible to the functional forward pass, producing silent shape
    mismatches at runtime.

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
        A ``(model, tokenizer)`` tuple.  Both objects are mutated in-place and
        also returned for convenient chaining.

    Raises:
        ValueError: If *source_tokenizer_or_path* does not have a
            ``chat_template`` attribute set (i.e. the attribute is ``None``).
        TypeError: If *source_tokenizer_or_path* is neither a string nor a
            :class:`~transformers.PreTrainedTokenizerBase`.
    """
    # ------------------------------------------------------------------
    # 1. Resolve source tokenizer.
    # ------------------------------------------------------------------
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

    # ------------------------------------------------------------------
    # 2. Copy chat template.
    # ------------------------------------------------------------------
    chat_template = getattr(source_tokenizer, "chat_template", None)
    if chat_template is None:
        raise ValueError(
            "The source tokenizer does not have a chat_template set "
            "(chat_template is None).  Set it before calling "
            "clone_chat_template."
        )
    tokenizer.chat_template = chat_template

    # ------------------------------------------------------------------
    # 3. Collect new special tokens from source.
    #
    # We use added_tokens_decoder (available since transformers ≥ 4.30) to
    # enumerate tokens that the source tokenizer added to its vocabulary.
    # We only propagate tokens flagged as special=True (e.g. <|im_start|>,
    # <|im_end|>, <bos>, <eos>); regular sub-word tokens are not copied.
    # ------------------------------------------------------------------
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

    if tokens_to_add:
        n_added = tokenizer.add_special_tokens(
            {"additional_special_tokens": tokens_to_add}
        )
        logger.info(
            "clone_chat_template: added %d new special token(s) to tokenizer: %s",
            n_added,
            tokens_to_add,
        )
    else:
        logger.debug("clone_chat_template: no new special tokens to add.")

    # ------------------------------------------------------------------
    # 4. Resize model embeddings.
    #
    # resize_token_embeddings is a no-op when len(tokenizer) matches the
    # current embedding size; it extends the matrix otherwise.
    # We call it unconditionally so the invariant
    #   model.get_input_embeddings().weight.shape[0] == len(tokenizer)
    # holds on exit even if no new tokens were added.
    # ------------------------------------------------------------------
    if hasattr(model, "resize_token_embeddings"):
        model.resize_token_embeddings(len(tokenizer))
    else:
        logger.warning(
            "clone_chat_template: model has no resize_token_embeddings method; "
            "skipping embedding resize.  The embedding size may not match the "
            "tokenizer vocabulary size."
        )

    return model, tokenizer


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

    # ------------------------------------------------------------------
    # Idempotency: already has the marker.
    # ------------------------------------------------------------------
    if _GEN_START in template:
        return template

    # ------------------------------------------------------------------
    # Strategy 1: wrap {{ generation_token }} — TRL v2 canonical placeholder.
    # ------------------------------------------------------------------
    _GENERATION_TOKEN_EXPR = "{{ generation_token }}"
    if _GENERATION_TOKEN_EXPR in template:
        return template.replace(
            _GENERATION_TOKEN_EXPR,
            f"{_GEN_START}{_GENERATION_TOKEN_EXPR}{_GEN_END}",
            1,
        )

    # ------------------------------------------------------------------
    # Strategy 2: find the assistant content expression inside the assistant
    # conditional block.
    #
    # We look for the *last* occurrence of '{{ message[' (or '{{message[')
    # inside an assistant branch.  This handles ChatML, Llama-3, Phi-3, etc.
    # ------------------------------------------------------------------
    template_out = _wrap_assistant_content(template)
    if template_out is not None:
        return template_out

    # ------------------------------------------------------------------
    # Strategy 3: best-effort fallback.  Wrap everything between the final
    # loop body and the end of the template with generation markers.  This
    # is unlikely to yield a correct assistant-only mask but at least
    # produces a syntactically valid template.
    # ------------------------------------------------------------------
    logger.warning(
        "get_training_chat_template: could not locate the assistant-turn "
        "content expression in the chat template.  Falling back to wrapping "
        "the entire template body.  The resulting assistant_tokens_mask may "
        "be incorrect.  Consider setting the template manually."
    )
    return f"{_GEN_START}{template}{_GEN_END}"


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _wrap_assistant_content(template: str) -> str | None:
    """Attempt to insert generation markers around assistant content.

    Searches for the last Jinja2 ``{{ ... content ... }}`` expression that
    occurs inside an ``assistant`` conditional branch (identified by the
    presence of ``'assistant'`` in the preceding ``if`` or ``elif`` tag).

    Returns the modified template string on success, or ``None`` if no
    suitable expression is found.
    """
    import re

    # Locate the last occurrence of an assistant branch.
    # We look for something like: {% if ... 'assistant' ... %} or
    #                             {% elif ... 'assistant' ... %}
    # then find the content expression within it.

    # Find all positions where 'assistant' appears inside Jinja tags.
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

    # Within the substring from last_branch_start onward, find the first
    # content expression.  (Using the first, not the last, because we want
    # the expression that renders the assistant reply — typically the very
    # next {{ … }} after the branch tag.)
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
