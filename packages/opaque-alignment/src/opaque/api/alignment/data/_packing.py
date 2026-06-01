"""Dataset packing transforms for sequence packing in language model training.

Ports the packing utilities from ``trl/data_utils.py`` (TRL ≥ 0.18,
lines 686-789) as pure-Python + ``datasets`` transforms with no PyTorch
dependency.  Three strategies are provided:

- :func:`pack_wrapped` — simple "greedy wrapped" packing: all token IDs are
  concatenated into one long stream, then sliced into fixed-length chunks.
  Classic and fast; may split a sequence mid-way.

- :func:`pack_bfd` — Best-Fit-Decreasing bin packing: sequences are sorted
  longest-first and placed into the bin with the least remaining capacity
  that can still fit the sequence (opens a new bin when none can).
  Produces fewer, better-filled bins than naive greedy packing.

- :func:`pack_bfd_split` — like :func:`pack_bfd` but sequences longer than
  ``max_length`` are first split into ``max_length``-length pieces so that
  no tokens are ever dropped.

All three functions accept a ``datasets.Dataset`` whose rows have an
``input_ids: list[int]`` column (and optionally ``attention_mask`` and
``labels``) and return a new ``Dataset`` with:

- ``input_ids`` — the packed token IDs (length ≤ max_length for BFD variants;
  exactly max_length for wrapped, except possibly the final chunk).
- ``seq_lengths`` — a ``list[int]`` recording the lengths of the original
  sequences that were concatenated into this row.  The sum of ``seq_lengths``
  equals ``len(input_ids)`` for every row.  This column is the contract for
  downstream attention masking.
- ``attention_mask`` — carried/rebuilt if present in the input (1 for real
  tokens, 0 for pad; no padding is added by packing itself).
- ``labels`` — carried/rebuilt if present in the input.

**Downstream attention decision (Phase θ open question):**
Packed examples require a block-diagonal attention mask so tokens only
attend within their original sequence.  The intended backend is
FlexAttention (a block-mask built from ``seq_lengths``); the fallback is
SDPA with an explicit 4D block-diagonal mask constructed from ``seq_lengths``
at collation time.  The packing transform itself is backend-agnostic — it
only emits ``seq_lengths``.  The collator that consumes these examples is
responsible for choosing and constructing the appropriate mask given the
available backend.  See: arXiv:2409.10524 (LD-DPO) and PyTorch FlexAttention
documentation for the block-mask API.

**TRL reference:** ``trl/data_utils.py`` lines 686-789 (circa TRL 0.18).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from datasets import Dataset

__all__ = ["pack_bfd", "pack_bfd_split", "pack_wrapped"]

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

_OPTIONAL_COLS = ("attention_mask", "labels")


def _has_col(dataset: "Dataset", col: str) -> bool:
    return col in dataset.column_names


def _build_rows(
    bins: list[list[list[int]]],
    *,
    has_attention_mask: bool,
    has_labels: bool,
) -> dict[str, list]:
    """Convert list-of-bins into the columnar dict expected by ``Dataset.from_dict``."""
    input_ids_col: list[list[int]] = []
    seq_lengths_col: list[list[int]] = []
    attention_mask_col: list[list[int]] = []
    labels_col: list[list[int]] = []

    for bin_seqs in bins:
        packed_ids: list[int] = []
        packed_am: list[int] = []
        packed_labels: list[int] = []
        lengths: list[int] = []

        for seq in bin_seqs:
            ids = seq[0]
            packed_ids.extend(ids)
            lengths.append(len(ids))
            if has_attention_mask:
                packed_am.extend(seq[1])
            if has_labels:
                packed_labels.extend(seq[-1] if has_attention_mask else seq[1])

        input_ids_col.append(packed_ids)
        seq_lengths_col.append(lengths)
        if has_attention_mask:
            attention_mask_col.append(packed_am)
        if has_labels:
            labels_col.append(packed_labels)

    result: dict[str, list] = {
        "input_ids": input_ids_col,
        "seq_lengths": seq_lengths_col,
    }
    if has_attention_mask:
        result["attention_mask"] = attention_mask_col
    if has_labels:
        result["labels"] = labels_col
    return result


def _extract_sequences(
    dataset: "Dataset",
    *,
    has_attention_mask: bool,
    has_labels: bool,
) -> list[list[list[int]]]:
    """Return a list of sequences; each sequence is a list of parallel lists
    (always starting with ``input_ids``, then optionally ``attention_mask``,
    then optionally ``labels``)."""
    seqs: list[list[list[int]]] = []
    for i in range(len(dataset)):
        row = dataset[i]
        seq: list[list[int]] = [list(row["input_ids"])]
        if has_attention_mask:
            seq.append(list(row["attention_mask"]))
        if has_labels:
            seq.append(list(row["labels"]))
        seqs.append(seq)
    return seqs


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def pack_wrapped(dataset: "Dataset", max_length: int) -> "Dataset":
    """Greedy 'wrapped' packing: concatenate all sequences end-to-end and split
    into contiguous chunks of exactly ``max_length`` (the simple, classic
    packing strategy).

    All tokens from all sequences are concatenated into a single stream and
    then sliced into fixed-length chunks.  The final incomplete chunk is
    **included** (it may be shorter than ``max_length``).  No tokens are
    dropped.

    The ``seq_lengths`` column records the lengths of the original sequences
    that contributed tokens to each chunk.  Because sequences may be split
    across chunk boundaries, ``seq_lengths`` within a chunk may sum to exactly
    ``max_length`` (for full chunks) or less (for the final chunk), and a
    single original sequence may appear split across two adjacent chunks.

    Args:
        dataset: A ``datasets.Dataset`` with an ``input_ids: list[int]``
            column.  May also contain ``attention_mask`` and ``labels``
            columns (carried through consistently).
        max_length: Target chunk length (number of tokens per output row).

    Returns:
        A new ``datasets.Dataset`` with columns ``input_ids``,
        ``seq_lengths``, and (if present in the input) ``attention_mask``
        and ``labels``.  Every output ``input_ids`` list has length exactly
        ``max_length`` except possibly the last.

    Examples:
        >>> from datasets import Dataset
        >>> ds = Dataset.from_dict({"input_ids": [[1, 2, 3], [4, 5], [6, 7, 8, 9]]})
        >>> out = pack_wrapped(ds, max_length=4)
        >>> out["input_ids"]
        [[1, 2, 3, 4], [5, 6, 7, 8], [9]]
        >>> out["seq_lengths"]
        [[3, 1], [1, 3], [1]]
    """
    from datasets import Dataset as HFDataset

    if len(dataset) == 0:
        return dataset

    has_am = _has_col(dataset, "attention_mask")
    has_labels = _has_col(dataset, "labels")

    # Build flat streams for each column.
    flat_ids: list[int] = []
    flat_am: list[int] = []
    flat_labels: list[int] = []
    # Track where each original sequence starts/ends in the flat stream.
    boundaries: list[tuple[int, int]] = []  # (start, end) inclusive-exclusive

    for i in range(len(dataset)):
        row = dataset[i]
        ids = list(row["input_ids"])
        start = len(flat_ids)
        flat_ids.extend(ids)
        end = len(flat_ids)
        boundaries.append((start, end))
        if has_am:
            flat_am.extend(list(row["attention_mask"]))
        if has_labels:
            flat_labels.extend(list(row["labels"]))

    total_tokens = len(flat_ids)
    if total_tokens == 0:
        return dataset

    # Slice flat stream into max_length chunks and rebuild seq_lengths.
    input_ids_col: list[list[int]] = []
    attention_mask_col: list[list[int]] = []
    labels_col: list[list[int]] = []
    seq_lengths_col: list[list[int]] = []

    # We walk through boundaries in sync with chunks.
    boundary_idx = 0
    # Position within the current boundary's original sequence that we have
    # already consumed into a chunk.
    boundary_consumed = 0

    for chunk_start in range(0, total_tokens, max_length):
        chunk_end = min(chunk_start + max_length, total_tokens)
        chunk_len = chunk_end - chunk_start

        input_ids_col.append(flat_ids[chunk_start:chunk_end])
        if has_am:
            attention_mask_col.append(flat_am[chunk_start:chunk_end])
        if has_labels:
            labels_col.append(flat_labels[chunk_start:chunk_end])

        # Compute seq_lengths: walk boundaries that overlap this chunk.
        lengths: list[int] = []
        remaining_in_chunk = chunk_len
        while remaining_in_chunk > 0 and boundary_idx < len(boundaries):
            seq_start, seq_end = boundaries[boundary_idx]
            seq_len = seq_end - seq_start
            available_in_seq = seq_len - boundary_consumed

            take = min(available_in_seq, remaining_in_chunk)
            lengths.append(take)
            remaining_in_chunk -= take
            boundary_consumed += take

            if boundary_consumed == seq_len:
                # Fully consumed this original sequence.
                boundary_idx += 1
                boundary_consumed = 0

        seq_lengths_col.append(lengths)

    result: dict[str, list] = {
        "input_ids": input_ids_col,
        "seq_lengths": seq_lengths_col,
    }
    if has_am:
        result["attention_mask"] = attention_mask_col
    if has_labels:
        result["labels"] = labels_col

    return HFDataset.from_dict(result)


def pack_bfd(dataset: "Dataset", max_length: int) -> "Dataset":
    """Best-Fit-Decreasing bin packing.

    Sequences longer than ``max_length`` are **dropped** (with a warning).
    If you need to preserve long sequences, use :func:`pack_bfd_split`.

    Algorithm:

    1. Extract all sequences that fit within ``max_length`` (others are
       skipped and a warning is emitted).
    2. Sort sequences by length descending (Decreasing step of BFD).
    3. For each sequence, find the open bin with the least remaining space
       that can still fit the sequence ("best fit").  If no bin can fit the
       sequence, open a new bin.
    4. Place the sequence into the chosen bin.

    The ``seq_lengths`` column records constituent original lengths in the
    order they were placed into each bin.

    Args:
        dataset: A ``datasets.Dataset`` with an ``input_ids: list[int]``
            column.  May also contain ``attention_mask`` and ``labels``.
        max_length: Maximum number of tokens per packed output row.  Sequences
            longer than this value are dropped.

    Returns:
        A new ``datasets.Dataset`` with columns ``input_ids``, ``seq_lengths``,
        and (if present in the input) ``attention_mask`` and ``labels``.
        Every output ``input_ids`` list has ``len <= max_length``; every
        ``seq_lengths`` list sums to ``len(input_ids)`` for that row.

    Examples:
        >>> from datasets import Dataset
        >>> ds = Dataset.from_dict({"input_ids": [[1, 2, 3], [4], [5, 6]]})
        >>> out = pack_bfd(ds, max_length=4)
        >>> # [1,2,3,4] in one bin, [5,6] in another
    """
    import warnings
    from datasets import Dataset as HFDataset

    if len(dataset) == 0:
        return dataset

    has_am = _has_col(dataset, "attention_mask")
    has_labels = _has_col(dataset, "labels")

    # Extract sequences with their lengths; filter oversized.
    sequences: list[tuple[int, list[list[int]]]] = []  # (length, [ids, am?, labels?])
    n_dropped = 0
    for i in range(len(dataset)):
        row = dataset[i]
        ids = list(row["input_ids"])
        length = len(ids)
        if length > max_length:
            n_dropped += 1
            continue
        seq: list[list[int]] = [ids]
        if has_am:
            seq.append(list(row["attention_mask"]))
        if has_labels:
            seq.append(list(row["labels"]))
        sequences.append((length, seq))

    if n_dropped:
        warnings.warn(
            f"pack_bfd: {n_dropped} sequence(s) with length > {max_length} were "
            "dropped. Use pack_bfd_split to preserve all tokens.",
            stacklevel=2,
        )

    if not sequences:
        return HFDataset.from_dict({"input_ids": [], "seq_lengths": []})

    # Sort descending by length (Decreasing step).
    sequences.sort(key=lambda x: x[0], reverse=True)

    # Best-fit using a min-heap keyed by (remaining_space, bin_index).
    # We want the bin with the LEAST remaining space that still fits.
    # Strategy: maintain bins as a sorted structure; for each sequence we want
    # the bin where remaining_space >= seq_len AND remaining_space is minimised.
    #
    # Implementation: use a list of (remaining_space, bin_index) sorted; for
    # each sequence do a linear scan from smallest remaining_space upward.
    # For the dataset sizes typical in ML (thousands of sequences), this is
    # fast enough.  A more efficient structure (e.g. a sorted container) can
    # be substituted without changing the public API.

    # bins[i] = list of sequences in bin i
    bins: list[list[list[list[int]]]] = []
    bin_remaining: list[int] = []

    for seq_len, seq in sequences:
        # Find best-fit bin: smallest remaining >= seq_len.
        best_bin = -1
        best_remaining = max_length + 1  # sentinel: larger than any valid value

        for b_idx, remaining in enumerate(bin_remaining):
            if remaining >= seq_len and remaining < best_remaining:
                best_remaining = remaining
                best_bin = b_idx

        if best_bin == -1:
            # Open a new bin.
            bins.append([[s for s in seq]])
            bin_remaining.append(max_length - seq_len)
        else:
            bins[best_bin].append(seq)
            bin_remaining[best_bin] -= seq_len

    # Build output columns.
    return HFDataset.from_dict(
        _build_rows(bins, has_attention_mask=has_am, has_labels=has_labels)
    )


def pack_bfd_split(dataset: "Dataset", max_length: int) -> "Dataset":
    """Best-Fit-Decreasing bin packing with splitting of oversized sequences.

    Like :func:`pack_bfd`, but sequences longer than ``max_length`` are first
    split into pieces of at most ``max_length`` tokens before packing.  **No
    tokens are dropped.**

    Split pieces are treated as independent sequences for packing purposes.
    Their ``seq_lengths`` entries reflect the piece lengths (which sum to the
    original sequence length).

    Args:
        dataset: A ``datasets.Dataset`` with an ``input_ids: list[int]``
            column.  May also contain ``attention_mask`` and ``labels``.
        max_length: Maximum number of tokens per packed output row and maximum
            piece length when splitting.

    Returns:
        A new ``datasets.Dataset`` with columns ``input_ids``, ``seq_lengths``,
        and (if present in the input) ``attention_mask`` and ``labels``.
        Every output ``input_ids`` list has ``len <= max_length``; every
        ``seq_lengths`` list sums to ``len(input_ids)`` for that row.  Every
        token from the input appears in exactly one output row.

    Examples:
        >>> from datasets import Dataset
        >>> ds = Dataset.from_dict({"input_ids": [[1, 2, 3, 4, 5, 6, 7]]})
        >>> out = pack_bfd_split(ds, max_length=4)
        >>> # [1,2,3,4] and [5,6,7] treated as separate sequences, then BFD-packed
    """
    from datasets import Dataset as HFDataset

    if len(dataset) == 0:
        return dataset

    has_am = _has_col(dataset, "attention_mask")
    has_labels = _has_col(dataset, "labels")

    # Extract and split oversized sequences.
    sequences: list[tuple[int, list[list[int]]]] = []  # (length, [ids, am?, labels?])
    for i in range(len(dataset)):
        row = dataset[i]
        ids = list(row["input_ids"])
        am = list(row["attention_mask"]) if has_am else None
        labels = list(row["labels"]) if has_labels else None

        # Split into max_length pieces.
        for start in range(0, max(len(ids), 1), max_length):
            piece_ids = ids[start : start + max_length]
            if not piece_ids:
                continue
            piece: list[list[int]] = [piece_ids]
            if has_am and am is not None:
                piece.append(am[start : start + max_length])
            if has_labels and labels is not None:
                piece.append(labels[start : start + max_length])
            sequences.append((len(piece_ids), piece))

    if not sequences:
        return HFDataset.from_dict({"input_ids": [], "seq_lengths": []})

    # Sort descending by length (Decreasing step).
    sequences.sort(key=lambda x: x[0], reverse=True)

    # Best-fit packing (same algorithm as pack_bfd).
    bins: list[list[list[list[int]]]] = []
    bin_remaining: list[int] = []

    for seq_len, seq in sequences:
        best_bin = -1
        best_remaining = max_length + 1

        for b_idx, remaining in enumerate(bin_remaining):
            if remaining >= seq_len and remaining < best_remaining:
                best_remaining = remaining
                best_bin = b_idx

        if best_bin == -1:
            bins.append([[s for s in seq]])
            bin_remaining.append(max_length - seq_len)
        else:
            bins[best_bin].append(seq)
            bin_remaining[best_bin] -= seq_len

    return HFDataset.from_dict(
        _build_rows(bins, has_attention_mask=has_am, has_labels=has_labels)
    )
