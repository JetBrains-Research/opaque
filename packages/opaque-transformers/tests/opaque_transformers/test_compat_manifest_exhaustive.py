"""Canary test: every upstream HF / TRL config field is bucketed exactly once.

This is the **drift catcher** for the opaque compat manifest. When upstream
HF or TRL ships a new ``TrainingArguments`` / ``SFTConfig`` / ``DPOConfig``
field on a release whose manifest opaque hasn't been updated for, the new
field will be unclassified by the bucket sets and this test fails.

The failure tells the maintainer:
1. Which new field appeared upstream.
2. To decide its bucket (DIRECT / RENAME / TRANSFORM / REJECT_IF_SET /
   DROP_WITH_WARN) and add it to the right constant in
   ``opaque.api.transformers.compat._hf`` (or ``_trl`` for the TRL surfaces
   once those land).

The test is the discipline that keeps the manifest in sync with upstream.
"""

from __future__ import annotations

import dataclasses

import pytest

# Required upstream — HF transformers is a runtime dep of opaque-transformers.
hf = pytest.importorskip("transformers")

from opaque.api.transformers.compat._hf import (  # noqa: E402
    HF_DIRECT_FIELDS,
    HF_DROP_FIELDS,
    HF_REJECTED_FIELDS,
    HF_RENAME_MAP,
    HF_TRANSFORM_MAP,
)


def _all_hf_field_names() -> set[str]:
    """Return every field name on HF's ``TrainingArguments`` dataclass."""
    return {f.name for f in dataclasses.fields(hf.TrainingArguments)}


def _opaque_buckets_for_hf() -> set[str]:
    """Union of every HF field name covered by opaque's manifest buckets."""
    return (
        set(HF_DIRECT_FIELDS)
        | set(HF_RENAME_MAP.keys())
        | set(HF_TRANSFORM_MAP.keys())
        | set(HF_REJECTED_FIELDS.keys())
        | set(HF_DROP_FIELDS.keys())
    )


def test_hf_manifest_buckets_are_disjoint():
    """No HF field appears in two different buckets — the manifest is a partition."""
    buckets = {
        "DIRECT": set(HF_DIRECT_FIELDS),
        "RENAME": set(HF_RENAME_MAP.keys()),
        "TRANSFORM": set(HF_TRANSFORM_MAP.keys()),
        "REJECT_IF_SET": set(HF_REJECTED_FIELDS.keys()),
        "DROP_WITH_WARN": set(HF_DROP_FIELDS.keys()),
    }
    for name_a, set_a in buckets.items():
        for name_b, set_b in buckets.items():
            if name_a >= name_b:
                continue
            overlap = set_a & set_b
            assert not overlap, (
                f"Manifest bucket overlap between {name_a} and {name_b}: "
                f"{sorted(overlap)}. Each HF field must be classified into "
                f"exactly one bucket."
            )


def test_hf_manifest_covers_every_upstream_field():
    """Every field on HF's ``TrainingArguments`` is bucketed.

    Drift case: HF adds a new field on a future release; this test
    surfaces it as the unclassified-field list. The fix is to add the
    new field to one of the constants in
    ``opaque.api.transformers.compat._hf``.
    """
    upstream = _all_hf_field_names()
    covered = _opaque_buckets_for_hf()
    missing = upstream - covered
    assert not missing, (
        f"HF ``TrainingArguments`` has fields not classified by the opaque "
        f"compat manifest:\n  {sorted(missing)}\n\n"
        f"This usually means upstream transformers added new fields on a "
        f"release the opaque manifest hasn't tracked yet. For each missing "
        f"field, decide its bucket and add it to the right constant in "
        f"``opaque.api.transformers.compat._hf``:\n"
        f"  - HF_DIRECT_FIELDS: same name + semantics as opaque\n"
        f"  - HF_RENAME_MAP: different name in opaque\n"
        f"  - HF_TRANSFORM_MAP: needs a derivation callable\n"
        f"  - HF_REJECTED_FIELDS: unsupported on the DP-SGD path\n"
        f"  - HF_DROP_FIELDS: irrelevant on opaque path (drop with warning)"
    )


def test_hf_manifest_does_not_have_phantom_fields():
    """Every field in the opaque manifest exists on HF's upstream class.

    Catches the reverse drift: HF removes a field, and opaque's manifest
    still references it. The fix is to delete the phantom entry from the
    manifest constant.
    """
    upstream = _all_hf_field_names()
    covered = _opaque_buckets_for_hf()
    phantoms = covered - upstream
    assert not phantoms, (
        f"opaque compat manifest references HF fields that no longer exist "
        f"upstream:\n  {sorted(phantoms)}\n\n"
        f"This means HF removed a field that opaque's manifest still "
        f"tracks. Remove the phantom entry from the corresponding constant "
        f"in ``opaque.api.transformers.compat._hf``."
    )
