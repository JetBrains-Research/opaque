"""Tests for :mod:`opaque.api.alignment.data._chat_template`.

Plan §7.7 + risk α10 (resize_token_embeddings ordering).

Coverage:

- :func:`get_training_chat_template`

  * Returns a template string containing ``{% generation %}``.
  * Idempotent when the template already contains ``{% generation %}``.
  * Raises ``ValueError`` on an unset (``None``) template.
  * Works on ChatML-style and simple role-conditional templates.

- :func:`clone_chat_template`

  * Chat template is copied from source onto destination tokenizer.
  * New special tokens from source are added to destination.
  * ``model.get_input_embeddings().weight.shape[0] == len(tokenizer)``
    (the resize invariant — risk α10).
  * No-op when destination tokenizer already has all of source's special
    tokens; vocab length and embedding size are unchanged.
  * Works with a pre-loaded source tokenizer object (not just a path string).

All tests are CPU-only, network-free, and use the smallest possible tokenizer
and model configs.  ``transformers`` is required; tests are skipped when it is
not installed via ``pytest.importorskip``.
"""

from __future__ import annotations

import pytest

transformers = pytest.importorskip("transformers")

from transformers import LlamaConfig, LlamaForCausalLM, PreTrainedTokenizerFast  # noqa: E402
from tokenizers import Tokenizer, AddedToken  # noqa: E402
from tokenizers.models import BPE  # noqa: E402
from tokenizers.pre_tokenizers import Whitespace  # noqa: E402

from opaque.api.alignment.data._chat_template import (  # noqa: E402
    clone_chat_template,
    get_training_chat_template,
)

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

_GEN_START = "{% generation %}"
_GEN_END = "{% endgeneration %}"

# A minimal ChatML-style template with an explicit assistant branch.
_CHATML_TEMPLATE = (
    "{% for message in messages %}"
    "{% if message['role'] == 'user' %}"
    "user: {{ message['content'] }}\n"
    "{% elif message['role'] == 'assistant' %}"
    "assistant: {{ message['content'] }}\n"
    "{% endif %}"
    "{% endfor %}"
)

# Same template but already containing the generation markers.
_CHATML_WITH_GEN = (
    "{% for message in messages %}"
    "{% if message['role'] == 'user' %}"
    "user: {{ message['content'] }}\n"
    "{% elif message['role'] == 'assistant' %}"
    "assistant: {% generation %}{{ message['content'] }}{% endgeneration %}\n"
    "{% endif %}"
    "{% endfor %}"
)


def _make_fast_tokenizer(
    extra_specials: list[str] | None = None,
    n_regular: int = 3,
) -> PreTrainedTokenizerFast:
    """Return a :class:`PreTrainedTokenizerFast` with a trivial BPE vocab.

    The vocabulary always contains ``<unk>`` (special) plus *n_regular* normal
    tokens ``w0, w1, …``.  Optionally, additional special tokens are added.
    No network access is performed.
    """
    tok = Tokenizer(BPE(unk_token="<unk>"))
    tok.pre_tokenizer = Whitespace()
    specials = [AddedToken("<unk>", special=True)]
    if extra_specials:
        specials += [AddedToken(s, special=True) for s in extra_specials]
    tok.add_special_tokens(specials)
    tok.add_tokens([f"w{i}" for i in range(n_regular)])
    return PreTrainedTokenizerFast(tokenizer_object=tok, unk_token="<unk>")


def _make_tiny_model(vocab_size: int) -> LlamaForCausalLM:
    """Return the smallest possible LlamaForCausalLM on CPU."""
    cfg = LlamaConfig(
        vocab_size=vocab_size,
        hidden_size=16,
        num_hidden_layers=2,
        num_attention_heads=4,
        intermediate_size=32,
    )
    return LlamaForCausalLM(cfg)


# ---------------------------------------------------------------------------
# Tests: get_training_chat_template
# ---------------------------------------------------------------------------


class TestGetTrainingChatTemplate:
    def test_generation_marker_inserted(self) -> None:
        """Template without markers should gain {% generation %} / {% endgeneration %}."""
        tokenizer = _make_fast_tokenizer()
        tokenizer.chat_template = _CHATML_TEMPLATE
        result = get_training_chat_template(tokenizer)
        assert _GEN_START in result
        assert _GEN_END in result

    def test_idempotent_when_markers_present(self) -> None:
        """Template that already has {% generation %} must be returned unchanged."""
        tokenizer = _make_fast_tokenizer()
        tokenizer.chat_template = _CHATML_WITH_GEN
        result = get_training_chat_template(tokenizer)
        assert result == _CHATML_WITH_GEN

    def test_idempotent_double_call(self) -> None:
        """Calling the function twice on a plain template yields the same output both times."""
        tokenizer = _make_fast_tokenizer()
        tokenizer.chat_template = _CHATML_TEMPLATE
        first = get_training_chat_template(tokenizer)
        # Simulate a second call by setting chat_template to the result.
        tokenizer2 = _make_fast_tokenizer()
        tokenizer2.chat_template = first
        second = get_training_chat_template(tokenizer2)
        assert first == second

    def test_simple_template_no_role_branch(self) -> None:
        """Template with a simple content expression but no role branch still gets markers."""
        tokenizer = _make_fast_tokenizer()
        # Minimal template: just render message content.
        tokenizer.chat_template = (
            "{% for m in messages %}{{ m['content'] }}{% endfor %}"
        )
        result = get_training_chat_template(tokenizer)
        assert _GEN_START in result
        assert _GEN_END in result

    def test_raises_on_none_template(self) -> None:
        """ValueError is raised when chat_template is None."""
        tokenizer = _make_fast_tokenizer()
        # PreTrainedTokenizerFast defaults chat_template to None.
        tokenizer.chat_template = None
        with pytest.raises(ValueError, match="chat_template"):
            get_training_chat_template(tokenizer)

    def test_raises_on_empty_template(self) -> None:
        """ValueError is raised when chat_template is an empty string."""
        tokenizer = _make_fast_tokenizer()
        tokenizer.chat_template = ""
        with pytest.raises(ValueError, match="chat_template"):
            get_training_chat_template(tokenizer)

    def test_original_template_unchanged(self) -> None:
        """The tokenizer's chat_template attribute must not be mutated in-place."""
        tokenizer = _make_fast_tokenizer()
        original = _CHATML_TEMPLATE
        tokenizer.chat_template = original
        _ = get_training_chat_template(tokenizer)
        # The tokenizer should still hold the original (unmodified) template.
        assert tokenizer.chat_template == original

    def test_markers_placed_around_content_not_header(self) -> None:
        """The generation markers must wrap the *content* expression, not surrounding text."""
        tokenizer = _make_fast_tokenizer()
        tokenizer.chat_template = _CHATML_TEMPLATE
        result = get_training_chat_template(tokenizer)
        # The user-turn content expression must NOT be wrapped.
        # Find the assistant branch in the result and confirm markers are there.
        assert "assistant: " in result
        # The generation marker must appear inside the assistant branch, not around user branch.
        assistant_idx = result.find("assistant:")
        gen_idx = result.find(_GEN_START)
        assert gen_idx > assistant_idx, (
            "{% generation %} should appear after the 'assistant:' literal"
        )

    def test_shared_or_clause_qwen_pattern(self) -> None:
        """OR-clause shared rendering (Qwen2.5-Instruct) wraps assistant only.

        Templates that emit user/system/assistant turns through one shared
        ``{{ ... message.role ... message.content ... }}`` inside a
        multi-condition ``{%- if (role == "user") or ... or (role ==
        "assistant" and not tool_calls) %}`` must NOT have user/system tokens
        flagged by ``return_assistant_tokens_mask=True``.  Strategy 2b in
        ``get_training_chat_template`` rewrites the shared expression with an
        inner role-guard so only the assistant render path goes through the
        ``{% generation %}`` block.
        """
        # Minimal Qwen-style template that triggers Strategy 2b: the OR clause
        # combines user/system/assistant-without-tool-calls through one shared
        # expression, while a second elif handles the assistant-with-tool-calls
        # case (which Strategy 2's "last assistant branch" heuristic would pick
        # by default, producing an unreachable generation block).
        qwen_style = (
            "{% for message in messages %}"
            '{%- if (message.role == "user") or (message.role == "system" and not loop.first) '
            'or (message.role == "assistant" and not message.tool_calls) %}'
            "{{- message.role + ': ' + message.content + '\\n' }}"
            '{%- elif message.role == "assistant" %}'
            "{{- message.role + ': ' + message.content + ' (tool-call)\\n' }}"
            "{%- endif %}"
            "{% endfor %}"
        )
        tokenizer = _make_fast_tokenizer()
        tokenizer.chat_template = qwen_style
        result = get_training_chat_template(tokenizer)
        assert _GEN_START in result
        assert _GEN_END in result

        # The wrap must include a role-guard so user/system don't enter the
        # generation block.
        assert "message.role == 'assistant'" in result, (
            "Strategy 2b should add an inner role-guard around the shared expression"
        )

        # Validate the wrapped template actually flags assistant content via
        # HF's tracker when rendered.
        from transformers.utils.chat_template_utils import (
            _compile_jinja_template,
            _render_with_assistant_indices,
        )

        compiled = _compile_jinja_template(result)
        msgs = [
            {"role": "user", "content": "first question"},
            {"role": "assistant", "content": "first answer"},
            {"role": "user", "content": "second question"},
            {"role": "assistant", "content": "second answer"},
        ]
        rendered, indices = _render_with_assistant_indices(
            compiled, msgs, None, None, False
        )
        assert indices, "generation block must fire for assistant turns"
        # Every span must be non-degenerate.
        assert all(end > start for start, end in indices)
        # And the spans must cover assistant text, NOT user text.
        for start, end in indices:
            span = rendered[start:end]
            assert "first question" not in span and "second question" not in span, (
                f"user content leaked into generation span: {span!r}"
            )
        joined = "".join(rendered[s:e] for s, e in indices)
        assert "first answer" in joined and "second answer" in joined


# ---------------------------------------------------------------------------
# Tests: clone_chat_template
# ---------------------------------------------------------------------------


class TestCloneChatTemplate:
    def test_chat_template_copied(self) -> None:
        """The destination tokenizer's chat_template is set to the source's value."""
        src = _make_fast_tokenizer()
        src.chat_template = "my_template"
        dst = _make_fast_tokenizer()
        model = _make_tiny_model(len(dst))

        clone_chat_template(model, dst, src)

        assert dst.chat_template == "my_template"

    def test_new_special_tokens_added(self) -> None:
        """After cloning, destination tokenizer contains all source special tokens."""
        src = _make_fast_tokenizer(extra_specials=["<|im_start|>", "<|im_end|>"])
        src.chat_template = "template"
        dst = _make_fast_tokenizer()  # no im_start/im_end
        model = _make_tiny_model(len(dst))

        clone_chat_template(model, dst, src)

        dst_vocab = dst.get_vocab()
        assert "<|im_start|>" in dst_vocab
        assert "<|im_end|>" in dst_vocab

    def test_vocab_grew(self) -> None:
        """After cloning a source that adds two new special tokens, len(tokenizer) increases."""
        src = _make_fast_tokenizer(extra_specials=["<|im_start|>", "<|im_end|>"])
        src.chat_template = "template"
        dst = _make_fast_tokenizer()
        len_before = len(dst)
        model = _make_tiny_model(len_before)

        clone_chat_template(model, dst, src)

        assert len(dst) == len_before + 2

    def test_resize_invariant(self) -> None:
        """CRITICAL (risk α10): embedding matrix rows == len(tokenizer) after clone."""
        src = _make_fast_tokenizer(extra_specials=["<|im_start|>", "<|im_end|>"])
        src.chat_template = "template"
        dst = _make_fast_tokenizer()
        model = _make_tiny_model(len(dst))

        model, dst = clone_chat_template(model, dst, src)

        embed_rows = model.get_input_embeddings().weight.shape[0]
        assert embed_rows == len(dst), (
            f"Embedding rows ({embed_rows}) must equal tokenizer vocab size ({len(dst)})"
        )

    def test_noop_when_tokens_already_present(self) -> None:
        """When destination already has all source special tokens, vocab does not grow."""
        src = _make_fast_tokenizer(extra_specials=["<|im_start|>", "<|im_end|>"])
        src.chat_template = "template"
        dst = _make_fast_tokenizer(extra_specials=["<|im_start|>", "<|im_end|>"])
        len_before = len(dst)
        model = _make_tiny_model(len_before)

        model, dst = clone_chat_template(model, dst, src)

        assert len(dst) == len_before
        assert model.get_input_embeddings().weight.shape[0] == len_before

    def test_resize_invariant_noop_case(self) -> None:
        """Resize invariant holds even when no new tokens were added (no-op case)."""
        src = _make_fast_tokenizer(extra_specials=["<|im_start|>"])
        src.chat_template = "template"
        dst = _make_fast_tokenizer(extra_specials=["<|im_start|>"])
        model = _make_tiny_model(len(dst))

        model, dst = clone_chat_template(model, dst, src)

        assert model.get_input_embeddings().weight.shape[0] == len(dst)

    def test_returns_model_and_tokenizer(self) -> None:
        """Return value is a (model, tokenizer) tuple of the same objects passed in."""
        src = _make_fast_tokenizer()
        src.chat_template = "t"
        dst = _make_fast_tokenizer()
        model = _make_tiny_model(len(dst))

        ret_model, ret_tok = clone_chat_template(model, dst, src)

        assert ret_model is model
        assert ret_tok is dst

    def test_raises_on_source_without_chat_template(self) -> None:
        """ValueError is raised when the source tokenizer has no chat_template."""
        src = _make_fast_tokenizer()
        src.chat_template = None  # unset
        dst = _make_fast_tokenizer()
        model = _make_tiny_model(len(dst))

        with pytest.raises(ValueError, match="chat_template"):
            clone_chat_template(model, dst, src)

    def test_import_from_impl_path(self) -> None:
        """Functions are importable from the implementation module path."""
        from opaque.api.alignment.data._chat_template import (
            clone_chat_template as _clone,
            get_training_chat_template as _get,
        )

        assert callable(_clone)
        assert callable(_get)
