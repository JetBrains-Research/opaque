"""Tests for :func:`compute_ref_logprobs_for_dataset` (plan §7.8, §11.9).

Covers the cache round-trip contract (§11.9): the first call runs the
reference forward (call counter advances) and attaches the requested columns;
a second call on the *same* dataset + ``cache_key`` + ``output_columns`` is a
cache HIT that skips ``ref`` entirely (a fresh counter stays at 0) and returns
identical column values. Also covers fingerprint sensitivity to ``cache_key``,
column presence / length / values, ``RefSpec`` construction for all four kinds,
and import from the implementation paths.
"""

from __future__ import annotations

import datasets
import pytest
import torch

from opaque.api.alignment.reference._precompute import (
    compute_ref_logprobs_for_dataset,
)
from opaque.api.alignment.reference.types import RefSpec

OUTPUT_COLUMNS = ("ref_chosen_logps", "ref_rejected_logps")


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------


def _make_dataset() -> datasets.Dataset:
    """A tiny preference-style dataset keyed by a per-row integer index."""
    return datasets.Dataset.from_dict(
        {
            "idx": [0, 1, 2, 3, 4],
            "prompt": ["p0", "p1", "p2", "p3", "p4"],
        }
    )


def _collator(rows: list[dict]) -> dict[str, torch.Tensor]:
    """Batch the raw rows into a dict carrying the per-row index tensor."""
    return {"idx": torch.tensor([r["idx"] for r in rows], dtype=torch.long)}


class _CountingRef:
    """A deterministic ``ref`` callable that records how often it is invoked.

    Per the precompute contract, ``ref(batch) -> dict[str, (B,) tensor]``. The
    returned logps are a fixed function of the row index so the test can assert
    exact values, and ``calls`` lets the cache-HIT test prove ``ref`` was never
    invoked on the second pass.
    """

    def __init__(self) -> None:
        self.calls = 0
        self.examples_seen = 0

    def __call__(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        self.calls += 1
        idx = batch["idx"].to(torch.float32)
        self.examples_seen += int(idx.numel())
        # Deterministic per-example values derived from the row index.
        return {
            "ref_chosen_logps": idx * 1.0,
            "ref_rejected_logps": idx * -2.0,
        }


def _expected(name: str, indices: list[int]) -> list[float]:
    """Mirror ``_CountingRef`` so tests assert against hand-computed values."""
    if name == "ref_chosen_logps":
        return [float(i) * 1.0 for i in indices]
    if name == "ref_rejected_logps":
        return [float(i) * -2.0 for i in indices]
    raise KeyError(name)


# ---------------------------------------------------------------------------
# Cache round-trip (§11.9)
# ---------------------------------------------------------------------------


def test_cache_round_trip_skips_ref_on_hit(tmp_path) -> None:
    """First call computes; second call (fresh ref) is a HIT and skips ref."""
    dataset = _make_dataset()
    cache_dir = str(tmp_path / "cache")
    cache_key = ("dpo", "model-v1")

    ref1 = _CountingRef()
    result1 = compute_ref_logprobs_for_dataset(
        dataset,
        ref1,
        _collator,
        OUTPUT_COLUMNS,
        batch_size=2,
        cache_key=cache_key,
        cache_dir=cache_dir,
    )
    # MISS: ref ran over every example.
    assert ref1.calls > 0, "first call should invoke ref (cache miss)"
    assert ref1.examples_seen == len(dataset)
    for name in OUTPUT_COLUMNS:
        assert name in result1.column_names

    # Second call: a FRESH ref whose counter starts at 0. Same dataset, key,
    # and output columns => cache HIT => ref must NOT be called.
    ref2 = _CountingRef()
    result2 = compute_ref_logprobs_for_dataset(
        dataset,
        ref2,
        _collator,
        OUTPUT_COLUMNS,
        batch_size=2,
        cache_key=cache_key,
        cache_dir=cache_dir,
    )
    assert ref2.calls == 0, "second call should be a cache HIT and skip ref"

    # Columns identical between the two calls.
    for name in OUTPUT_COLUMNS:
        assert result1[name] == result2[name]


def test_different_cache_key_recomputes(tmp_path) -> None:
    """A different cache_key is a MISS: ref runs again under a new fingerprint."""
    dataset = _make_dataset()
    cache_dir = str(tmp_path / "cache")

    ref_a = _CountingRef()
    compute_ref_logprobs_for_dataset(
        dataset,
        ref_a,
        _collator,
        OUTPUT_COLUMNS,
        batch_size=2,
        cache_key=("dpo", "model-A"),
        cache_dir=cache_dir,
    )
    assert ref_a.calls > 0

    # Different cache_key => distinct fingerprint => recompute.
    ref_b = _CountingRef()
    compute_ref_logprobs_for_dataset(
        dataset,
        ref_b,
        _collator,
        OUTPUT_COLUMNS,
        batch_size=2,
        cache_key=("dpo", "model-B"),
        cache_dir=cache_dir,
    )
    assert ref_b.calls > 0, "different cache_key should force recompute (MISS)"


def test_different_output_columns_recomputes(tmp_path) -> None:
    """output_columns participate in the fingerprint: a different set => MISS."""
    dataset = _make_dataset()
    cache_dir = str(tmp_path / "cache")
    cache_key = ("dpo", "model-v1")

    ref_full = _CountingRef()
    compute_ref_logprobs_for_dataset(
        dataset,
        ref_full,
        _collator,
        OUTPUT_COLUMNS,
        batch_size=2,
        cache_key=cache_key,
        cache_dir=cache_dir,
    )
    assert ref_full.calls > 0

    # Same dataset + key but only one requested column => different fingerprint.
    ref_single = _CountingRef()
    compute_ref_logprobs_for_dataset(
        dataset,
        ref_single,
        _collator,
        ("ref_chosen_logps",),
        batch_size=2,
        cache_key=cache_key,
        cache_dir=cache_dir,
    )
    assert ref_single.calls > 0, "different output_columns should force a MISS"


# ---------------------------------------------------------------------------
# Output columns: presence, length, values
# ---------------------------------------------------------------------------


def test_output_columns_present_with_correct_length_and_values(tmp_path) -> None:
    """Added columns exist, have len(dataset), and match the deterministic ref."""
    dataset = _make_dataset()
    cache_dir = str(tmp_path / "cache")
    indices = list(dataset["idx"])

    result = compute_ref_logprobs_for_dataset(
        dataset,
        _CountingRef(),
        _collator,
        OUTPUT_COLUMNS,
        batch_size=2,
        cache_key=("vals",),
        cache_dir=cache_dir,
    )

    for name in OUTPUT_COLUMNS:
        assert name in result.column_names
        assert len(result[name]) == len(dataset)
        got = [float(v) for v in result[name]]
        assert got == pytest.approx(_expected(name, indices))


def test_cached_values_match_computed_values(tmp_path) -> None:
    """Values restored from the .npz cache equal the freshly computed ones."""
    dataset = _make_dataset()
    cache_dir = str(tmp_path / "cache")
    indices = list(dataset["idx"])

    compute_ref_logprobs_for_dataset(
        dataset,
        _CountingRef(),
        _collator,
        OUTPUT_COLUMNS,
        batch_size=3,
        cache_key=("rt",),
        cache_dir=cache_dir,
    )
    # Second call: HIT, values come from the cache file.
    cached_result = compute_ref_logprobs_for_dataset(
        dataset,
        _CountingRef(),
        _collator,
        OUTPUT_COLUMNS,
        batch_size=3,
        cache_key=("rt",),
        cache_dir=cache_dir,
    )
    for name in OUTPUT_COLUMNS:
        got = [float(v) for v in cached_result[name]]
        assert got == pytest.approx(_expected(name, indices))


def test_missing_cache_dir_is_created(tmp_path) -> None:
    """A non-existent cache_dir is created on the first (miss) write."""
    dataset = _make_dataset()
    cache_dir = tmp_path / "nested" / "does_not_exist_yet"
    assert not cache_dir.exists()

    compute_ref_logprobs_for_dataset(
        dataset,
        _CountingRef(),
        _collator,
        OUTPUT_COLUMNS,
        batch_size=2,
        cache_key=("mkdir",),
        cache_dir=str(cache_dir),
    )
    assert cache_dir.exists(), "cache_dir should be created on first write"
    npz_files = list(cache_dir.glob("*.npz"))
    assert npz_files, "expected a .npz cache file to be written"


# ---------------------------------------------------------------------------
# RefSpec construction (§7.8)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "kind",
    [
        "separate_model",
        "lora_ref_adapter",
        "lora_disable_adapter",
        "callable",
    ],
)
def test_refspec_constructs_for_all_kinds(kind: str) -> None:
    """RefSpec constructs for each of the four §7.8 kinds."""
    spec = RefSpec(kind=kind)  # type: ignore[arg-type]
    assert spec.kind == kind
    assert spec.adapter_name is None


def test_refspec_carries_adapter_name() -> None:
    """The optional adapter_name field is recorded for the LoRA-ref kind."""
    spec = RefSpec(kind="lora_ref_adapter", adapter_name="ref")
    assert spec.adapter_name == "ref"


def test_refspec_is_frozen() -> None:
    """RefSpec is an immutable record (frozen dataclass)."""
    spec = RefSpec(kind="separate_model")
    with pytest.raises(Exception):
        spec.kind = "callable"  # type: ignore[misc]
