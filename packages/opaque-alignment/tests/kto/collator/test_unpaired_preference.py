# Copyright (c) 2025 Opaque Authors
# SPDX-License-Identifier: Apache-2.0
"""Tests for the KTO unpaired-preference collator (plan §7.6, §7.2).

Covers :func:`unpaired_preference_collator` imported from the concrete impl
path (the public façade is not wired yet — unit β.W).

Test plan
---------
1. ``label`` bool tensor is built correctly for mixed True/False batches,
   all-True, and all-False batches.
2. ``calculate_KL=True`` with KL fields → KL keys emitted;
   ``calculate_KL=False`` → KL keys absent even when inputs carry them.
3. ``calculate_KL=True`` but KL fields absent from inputs → no crash, KL
   keys omitted.
4. ``completion_labels`` pad positions are filled with ``-100`` (not the
   pad_token_id).
5. Determinism: calling the same collator twice on equal inputs returns
   identical tensors.
6. Key-set stability: mandatory keys always present; optional keys absent
   when not applicable.
7. ``reference_logps`` and ``reference_KL_logps`` follow the all-or-nothing
   rule (present in output iff present in *all* input examples).
"""

from __future__ import annotations

import torch

from opaque.api.alignment.kto.collator._unpaired_preference import (
    unpaired_preference_collator,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_PAD = 0  # pad_token_id used throughout
_MAX = 8  # max_length used throughout

MANDATORY_KEYS = frozenset(
    [
        "completion_input_ids",
        "completion_attention_mask",
        "completion_labels",
        "label",
    ]
)
KL_KEYS = frozenset(
    [
        "KL_completion_input_ids",
        "KL_completion_attention_mask",
        "KL_completion_labels",
    ]
)


def _make_example(
    *,
    label: bool,
    input_ids: list[int] | None = None,
    labels: list[int] | None = None,
    with_kl: bool = False,
    with_ref: bool = False,
    with_ref_kl: bool = False,
) -> dict:
    """Build a minimal per-example dict."""
    input_ids = input_ids if input_ids is not None else [10, 20, 30]
    # Default: first token is prompt (label=-100), rest are completion targets.
    labels = labels if labels is not None else [-100, 20, 30]
    ex: dict = {
        "completion_input_ids": input_ids,
        "completion_labels": labels,
        "label": label,
    }
    if with_kl:
        ex["KL_completion_input_ids"] = [11, 21, 31]
        ex["KL_completion_labels"] = [-100, 21, 31]
    if with_ref:
        ex["reference_logps"] = -1.5
    if with_ref_kl:
        ex["reference_KL_logps"] = -2.0
    return ex


# ---------------------------------------------------------------------------
# 1. label tensor correctness
# ---------------------------------------------------------------------------


def test_label_mixed_true_false() -> None:
    """Mixed labels: result is a bool tensor with exactly the right values."""
    collate = unpaired_preference_collator(_PAD, _MAX)
    batch = [
        _make_example(label=True),
        _make_example(label=False),
        _make_example(label=True),
    ]
    out = collate(batch)
    assert out["label"].dtype == torch.bool
    assert out["label"].shape == (3,)
    expected = torch.tensor([True, False, True])
    assert torch.equal(out["label"], expected)


def test_label_all_true() -> None:
    """All-True batch: label tensor is all True."""
    collate = unpaired_preference_collator(_PAD, _MAX)
    batch = [_make_example(label=True) for _ in range(4)]
    out = collate(batch)
    assert out["label"].all()
    assert out["label"].shape == (4,)


def test_label_all_false() -> None:
    """All-False batch: label tensor is all False."""
    collate = unpaired_preference_collator(_PAD, _MAX)
    batch = [_make_example(label=False) for _ in range(4)]
    out = collate(batch)
    assert not out["label"].any()
    assert out["label"].shape == (4,)


# ---------------------------------------------------------------------------
# 2. KL keys: calculate_KL=True with KL fields present
# ---------------------------------------------------------------------------


def test_kl_keys_emitted_when_calculate_kl_true_and_fields_present() -> None:
    """KL keys appear when calculate_KL=True and all examples have KL fields."""
    collate = unpaired_preference_collator(_PAD, _MAX, calculate_KL=True)
    batch = [
        _make_example(label=True, with_kl=True),
        _make_example(label=False, with_kl=True),
    ]
    out = collate(batch)
    for key in KL_KEYS:
        assert key in out, f"Expected key '{key}' in output"


def test_kl_keys_absent_when_calculate_kl_false() -> None:
    """KL keys absent when calculate_KL=False, even if inputs carry them."""
    collate = unpaired_preference_collator(_PAD, _MAX, calculate_KL=False)
    batch = [
        _make_example(label=True, with_kl=True),
        _make_example(label=False, with_kl=True),
    ]
    out = collate(batch)
    for key in KL_KEYS:
        assert key not in out, f"Key '{key}' should be absent when calculate_KL=False"


# ---------------------------------------------------------------------------
# 3. calculate_KL=True but KL fields absent: no crash, KL keys omitted
# ---------------------------------------------------------------------------


def test_kl_keys_absent_when_inputs_lack_kl_fields() -> None:
    """No crash and no KL keys when calculate_KL=True but inputs lack KL fields."""
    collate = unpaired_preference_collator(_PAD, _MAX, calculate_KL=True)
    batch = [
        _make_example(label=True),  # no KL fields
        _make_example(label=False),
    ]
    out = collate(batch)
    # Should not raise; KL keys must be absent.
    for key in KL_KEYS:
        assert key not in out, f"Key '{key}' must not appear when inputs lack KL fields"
    # Mandatory keys must still be present.
    assert MANDATORY_KEYS.issubset(out.keys())


def test_kl_keys_absent_when_only_some_inputs_have_kl() -> None:
    """Partial KL coverage: KL keys are suppressed (all-or-nothing rule)."""
    collate = unpaired_preference_collator(_PAD, _MAX, calculate_KL=True)
    batch = [
        _make_example(label=True, with_kl=True),
        _make_example(label=False),  # no KL
    ]
    out = collate(batch)
    for key in KL_KEYS:
        assert key not in out, (
            f"Key '{key}' must not appear when only partial KL coverage"
        )


# ---------------------------------------------------------------------------
# 4. completion_labels: pad positions filled with -100
# ---------------------------------------------------------------------------


def test_completion_labels_pad_is_minus_100() -> None:
    """Padding appended to completion_labels must be -100, not pad_token_id."""
    collate = unpaired_preference_collator(_PAD, max_length=6)
    # Sequence of length 3: positions [3,4,5] are padding.
    batch = [_make_example(label=True, input_ids=[10, 20, 30], labels=[-100, 20, 30])]
    out = collate(batch)
    labels = out["completion_labels"]
    # Positions 3..5 must be -100 (the ignore index), not 0 (pad_token_id).
    assert labels.shape == (1, 6)
    assert (labels[0, 3:] == -100).all(), "Pad positions in labels should be -100"
    # Non-pad positions must be preserved.
    assert labels[0, 0].item() == -100  # prompt token
    assert labels[0, 1].item() == 20
    assert labels[0, 2].item() == 30


def test_completion_input_ids_pad_is_pad_token_id() -> None:
    """Padding in completion_input_ids uses pad_token_id (0), not -100."""
    pad_id = 99
    collate = unpaired_preference_collator(pad_id, max_length=6)
    batch = [_make_example(label=True, input_ids=[10, 20], labels=[-100, 20])]
    out = collate(batch)
    ids = out["completion_input_ids"]
    assert ids.shape == (1, 6)
    assert (ids[0, 2:] == pad_id).all(), (
        "completion_input_ids pad must equal pad_token_id"
    )


def test_completion_attention_mask_zero_on_pad() -> None:
    """Attention mask is 0 on padded positions and 1 on real tokens."""
    pad_id = 5
    collate = unpaired_preference_collator(pad_id, max_length=6)
    batch = [_make_example(label=False, input_ids=[10, 20, 30], labels=[-100, 20, 30])]
    out = collate(batch)
    mask = out["completion_attention_mask"]
    assert mask.shape == (1, 6)
    assert (mask[0, :3] == 1).all()
    assert (mask[0, 3:] == 0).all()


# ---------------------------------------------------------------------------
# 5. Determinism
# ---------------------------------------------------------------------------


def test_determinism() -> None:
    """Calling the same collator twice on equal inputs yields identical tensors."""
    collate = unpaired_preference_collator(_PAD, _MAX)
    batch = [
        _make_example(label=True, with_kl=True),
        _make_example(label=False, with_kl=True),
    ]
    out1 = collate(batch)
    out2 = collate(batch)
    for key in out1:
        assert torch.equal(out1[key], out2[key]), (
            f"Non-deterministic output for key '{key}'"
        )


# ---------------------------------------------------------------------------
# 6. Key-set stability
# ---------------------------------------------------------------------------


def test_mandatory_keys_always_present() -> None:
    """All mandatory keys present regardless of optional-field presence."""
    collate = unpaired_preference_collator(_PAD, _MAX)
    batch = [_make_example(label=True)]
    out = collate(batch)
    assert MANDATORY_KEYS.issubset(out.keys())


def test_only_mandatory_keys_when_no_optional_fields() -> None:
    """When no optional fields are present, output has exactly the mandatory keys."""
    collate = unpaired_preference_collator(_PAD, _MAX, calculate_KL=True)
    batch = [_make_example(label=True), _make_example(label=False)]
    out = collate(batch)
    assert set(out.keys()) == MANDATORY_KEYS


def test_kl_shape_is_per_batch() -> None:
    """KL tensors have shape (B, Lk); Lk may differ from L."""
    collate = unpaired_preference_collator(_PAD, _MAX, calculate_KL=True)
    batch = [
        _make_example(
            label=True,
            input_ids=[1, 2],
            labels=[-100, 2],
            with_kl=True,
        ),
        _make_example(
            label=False,
            input_ids=[1, 2, 3],
            labels=[-100, 2, 3],
            with_kl=True,
        ),
    ]
    out = collate(batch)
    b = 2
    assert out["KL_completion_input_ids"].shape[0] == b
    assert out["KL_completion_attention_mask"].shape[0] == b
    assert out["KL_completion_labels"].shape[0] == b
    # All KL tensors share the same Lk.
    lk = out["KL_completion_input_ids"].shape[1]
    assert out["KL_completion_attention_mask"].shape[1] == lk
    assert out["KL_completion_labels"].shape[1] == lk


# ---------------------------------------------------------------------------
# 7. reference_logps / reference_KL_logps — all-or-nothing
# ---------------------------------------------------------------------------


def test_reference_logps_emitted_when_all_examples_have_them() -> None:
    """reference_logps present in output when every example carries it."""
    collate = unpaired_preference_collator(_PAD, _MAX)
    batch = [
        _make_example(label=True, with_ref=True),
        _make_example(label=False, with_ref=True),
    ]
    out = collate(batch)
    assert "reference_logps" in out
    assert out["reference_logps"].dtype == torch.float32
    assert out["reference_logps"].shape == (2,)


def test_reference_logps_absent_when_partial_coverage() -> None:
    """reference_logps absent if only some examples carry it."""
    collate = unpaired_preference_collator(_PAD, _MAX)
    batch = [
        _make_example(label=True, with_ref=True),
        _make_example(label=False),  # no reference_logps
    ]
    out = collate(batch)
    assert "reference_logps" not in out


def test_reference_kl_logps_emitted_when_all_examples_have_them() -> None:
    """reference_KL_logps present in output when every example carries it."""
    collate = unpaired_preference_collator(_PAD, _MAX)
    batch = [
        _make_example(label=True, with_ref_kl=True),
        _make_example(label=False, with_ref_kl=True),
    ]
    out = collate(batch)
    assert "reference_KL_logps" in out
    assert out["reference_KL_logps"].dtype == torch.float32
    assert out["reference_KL_logps"].shape == (2,)


def test_reference_kl_logps_absent_when_partial_coverage() -> None:
    """reference_KL_logps absent if only some examples carry it."""
    collate = unpaired_preference_collator(_PAD, _MAX)
    batch = [
        _make_example(label=True, with_ref_kl=True),
        _make_example(label=False),  # no reference_KL_logps
    ]
    out = collate(batch)
    assert "reference_KL_logps" not in out


def test_reference_logps_values_preserved() -> None:
    """Values in reference_logps and reference_KL_logps match the input."""
    collate = unpaired_preference_collator(_PAD, _MAX)
    batch = [
        _make_example(label=True, with_ref=True, with_ref_kl=True),
        _make_example(label=False, with_ref=True, with_ref_kl=True),
    ]
    out = collate(batch)
    expected_ref = torch.tensor([-1.5, -1.5], dtype=torch.float32)
    expected_kl = torch.tensor([-2.0, -2.0], dtype=torch.float32)
    assert torch.allclose(out["reference_logps"], expected_ref)
    assert torch.allclose(out["reference_KL_logps"], expected_kl)


def test_reference_logps_absent_when_none_have_them() -> None:
    """Both reference keys absent when no examples carry them."""
    collate = unpaired_preference_collator(_PAD, _MAX)
    batch = [_make_example(label=True), _make_example(label=False)]
    out = collate(batch)
    assert "reference_logps" not in out
    assert "reference_KL_logps" not in out


# ---------------------------------------------------------------------------
# 8. Truncation
# ---------------------------------------------------------------------------


def test_truncation_keep_start() -> None:
    """Sequences longer than max_length are truncated from the right (keep-start)."""
    max_len = 4
    collate = unpaired_preference_collator(_PAD, max_len)
    long_ids = [1, 2, 3, 4, 5, 6]
    long_labels = [-100, 2, 3, 4, 5, 6]
    batch = [_make_example(label=True, input_ids=long_ids, labels=long_labels)]
    out = collate(batch)
    assert out["completion_input_ids"].shape == (1, max_len)
    # First max_len tokens should be kept.
    assert out["completion_input_ids"][0, 0].item() == 1
    assert out["completion_input_ids"][0, 3].item() == 4
    # labels similarly truncated.
    assert out["completion_labels"][0, 0].item() == -100
    assert out["completion_labels"][0, 3].item() == 4


# ---------------------------------------------------------------------------
# 9. Output tensor dtypes
# ---------------------------------------------------------------------------


def test_output_tensor_dtypes() -> None:
    """completion_input_ids/labels/mask are long; label is bool."""
    collate = unpaired_preference_collator(_PAD, _MAX)
    batch = [_make_example(label=True), _make_example(label=False)]
    out = collate(batch)
    assert out["completion_input_ids"].dtype == torch.long
    assert out["completion_attention_mask"].dtype == torch.long
    assert out["completion_labels"].dtype == torch.long
    assert out["label"].dtype == torch.bool
