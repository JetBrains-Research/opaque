"""Cached, model-agnostic reference-logprob precomputation.

:func:`compute_ref_logprobs_for_dataset` runs a single one-shot forward pass
over a :class:`datasets.Dataset` to materialise per-example reference logprobs
(e.g. ``ref_chosen_logps`` / ``ref_rejected_logps`` for DPO) and attaches them
to the dataset as new columns. Results are persisted to a content-addressed
``.npz`` cache so the (expensive) reference forward runs at most once per
``(dataset, cache_key, output_columns)`` triple.

**Outside vmap only.** This helper deliberately breaks the vmap-safety contract
: it iterates a ``DataLoader``, runs a forward pass under
``torch.no_grad()``, gathers across ranks, and writes a file. It must be called
*before* the per-example gradient loop, never inside ``vmap``/``grad``.

**Mechanism-agnostic by construction.** ``ref`` is a plain callable
``dict[str, Tensor] -> dict[str, Tensor]``. A :class:`PreTrainedModel` is wrapped
into such a callable by the caller (the trainer / example), which keeps this
module independent of any particular model class and trivially unit-testable
with a synthetic ``ref``.

**Cache fingerprint.** The cache filename is the SHA-256 of
``(dataset._fingerprint or repr(dataset), repr(cache_key), tuple(output_columns))``.
Including ``cache_key`` and ``output_columns`` in the digest is what prevents
collisions across model checkpoints, preprocessing variants, and differing
requested column sets — callers vary ``cache_key`` (e.g. ``("dpo", model_name)``)
as the escape hatch.

**Cross-rank handling.** Per-column shards are concatenated across ranks
with :func:`gather_for_metrics`; only :func:`is_main_process` writes the cache;
:func:`wait_for_everyone` synchronises before the function returns so non-main
ranks observe the file on the next run. In a single-process context the gather
is the identity and ``is_main_process`` is ``True``.
"""

from __future__ import annotations

import hashlib
import os
import tempfile
from collections.abc import Callable, Sequence
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from opaque.distributed import (
    gather_for_metrics,
    is_main_process,
    wait_for_everyone,
)
from opaque.serialization import from_state_dict, state_dict

__all__ = ["compute_ref_logprobs_for_dataset"]


# Default cache root, lazily created on first miss. A subdirectory of the
# system temp dir keeps the cache off the dataset's own storage and lets the
# OS reclaim it.
_DEFAULT_CACHE_SUBDIR = "opaque_ref_cache"


def _cache_fingerprint(
    dataset: Any,
    cache_key: tuple,
    output_columns: Sequence[str],
) -> str:
    """SHA-256 over the dataset identity, ``cache_key`` and ``output_columns``.

    Uses ``dataset._fingerprint`` when present (the ``datasets`` content hash),
    falling back to ``repr(dataset)``. ``cache_key`` and ``output_columns`` are
    folded in so two runs that differ only in the requested columns or in the
    caller-supplied key get distinct cache files.
    """
    dataset_id = getattr(dataset, "_fingerprint", None)
    if dataset_id is None:
        dataset_id = repr(dataset)
    hasher = hashlib.sha256()
    hasher.update(repr(dataset_id).encode("utf-8"))
    hasher.update(repr(cache_key).encode("utf-8"))
    hasher.update(repr(tuple(output_columns)).encode("utf-8"))
    return hasher.hexdigest()


def _cache_path(
    cache_dir: str | None,
    fingerprint: str,
) -> str:
    """Resolve the ``.npz`` path, defaulting to ``<tmp>/opaque_ref_cache``."""
    if cache_dir is None:
        cache_dir = os.path.join(tempfile.gettempdir(), _DEFAULT_CACHE_SUBDIR)
    return os.path.join(cache_dir, f"{fingerprint}.npz")


def _load_cache(
    path: str,
    output_columns: Sequence[str],
) -> dict[str, np.ndarray] | None:
    """Load ``output_columns`` from a ``.npz`` cache, or ``None`` on miss.

    A miss is any of: file absent, a requested column absent from the archive,
    or a load error (treated as a stale/corrupt cache that should be recomputed
    rather than crash the run).
    """
    if not os.path.exists(path):
        return None
    try:
        with np.load(path) as archive:
            flat = {key: archive[key] for key in archive.files}
    except Exception:
        return None
    if any(name not in flat for name in output_columns):
        return None
    # Reconstruct the column dict via the serialization template so the load
    # path mirrors the save path (state_dict round-trip, §12.2). The template
    # shape is taken from the archived arrays themselves (the per-example count
    # is not known statically), so the round-trip validates dtype + structure
    # without an artificial length assumption.
    template = {name: np.empty_like(flat[name]) for name in output_columns}
    try:
        restored = from_state_dict(template, flat)
    except Exception:
        return None
    if any(name not in restored for name in output_columns):
        return None
    return {name: np.asarray(restored[name]) for name in output_columns}


def _save_cache(
    path: str,
    columns: dict[str, np.ndarray],
) -> None:
    """Flatten ``columns`` through ``state_dict`` and write the ``.npz`` cache.

    The parent directory is created if missing. ``state_dict`` flattens the
    column dict into a flat ``str -> array`` mapping; the values are
    coerced to NumPy arrays for ``numpy.savez``.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    flat = state_dict(columns)
    arrays: dict[str, np.ndarray] = {}
    for key, value in flat.items():
        if isinstance(value, torch.Tensor):
            arrays[key] = value.detach().cpu().numpy()
        else:
            arrays[key] = np.asarray(value)
    np.savez(path, **arrays)


def compute_ref_logprobs_for_dataset(
    dataset: Any,
    ref: Callable[[dict[str, torch.Tensor]], dict[str, torch.Tensor]],
    collator: Callable[[list[dict]], dict[str, torch.Tensor]],
    output_columns: Sequence[str],
    *,
    batch_size: int = 8,
    cache_key: tuple = (),
    cache_dir: str | None = None,
) -> Any:
    """Compute per-example reference logprobs once, with a ``.npz`` cache.

    Runs a single pass over ``dataset`` calling ``ref(batch)`` for each
    collated batch, collecting one ``(B,)`` tensor per name in
    ``output_columns``, and attaches the concatenated per-example results to
    ``dataset`` as new columns. The forward runs under ``torch.no_grad()``.
    Results are cached to a content-addressed ``.npz`` file keyed on
    ``(dataset._fingerprint, cache_key, output_columns)``.

    **Outside vmap only** — see the module docstring.

    Args:
        dataset: A :class:`datasets.Dataset`. Iterated via a ``DataLoader`` with
            ``collator``; the result is returned with ``output_columns`` added
            through :meth:`datasets.Dataset.add_column`.
        ref: A callable mapping a collated batch ``dict[str, Tensor]`` to a dict
            containing one ``(B,)`` tensor per name in ``output_columns``. A
            :class:`PreTrainedModel` is wrapped into such a callable by the
            caller, keeping this mechanism-agnostic.
        collator: ``Callable[[list[dict]], dict[str, Tensor]]`` producing a
            batched dict from a list of raw rows (the same collator later used
            for training).
        output_columns: The keys ``ref`` returns, e.g.
            ``("ref_chosen_logps", "ref_rejected_logps")``. Also folded into the
            cache fingerprint.
        batch_size: ``DataLoader`` batch size for the forward pass. Default 8.
        cache_key: Caller-supplied opaque tuple folded into the cache
            fingerprint — the escape hatch against collisions across model
            checkpoints / preprocessing variants. Default ``()``.
        cache_dir: Directory for the ``.npz`` cache. Defaults to
            ``<tempdir>/opaque_ref_cache`` (created on first miss).

    Returns:
        ``dataset`` with one new column per name in ``output_columns``, each of
        length ``len(dataset)``.
    """
    columns = tuple(output_columns)
    fingerprint = _cache_fingerprint(dataset, cache_key, columns)
    path = _cache_path(cache_dir, fingerprint)

    cached = _load_cache(path, columns)
    if cached is not None:
        # HIT: attach cached columns WITHOUT calling ``ref``.
        result = dataset
        for name in columns:
            result = result.add_column(name, cached[name].tolist())
        return result

    # MISS: run the reference forward once over the dataset.
    loader = DataLoader(dataset, batch_size=batch_size, collate_fn=collator)
    shards: dict[str, list[torch.Tensor]] = {name: [] for name in columns}
    with torch.no_grad():
        for batch in loader:
            out = ref(batch)
            for name in columns:
                value = out[name]
                if not isinstance(value, torch.Tensor):
                    value = torch.as_tensor(value)
                shards[name].append(value.detach())

    # Concatenate local shards, then gather across ranks (identity in 1-proc).
    gathered: dict[str, torch.Tensor] = {}
    for name in columns:
        local = (
            torch.cat(shards[name], dim=0)
            if shards[name]
            else torch.empty((0,), dtype=torch.float32)
        )
        gathered[name] = gather_for_metrics(local)

    # Only the main process writes the cache; all ranks sync before returning.
    if is_main_process():
        cpu_columns = {
            name: gathered[name].detach().cpu().contiguous() for name in columns
        }
        _save_cache(path, cpu_columns)
    wait_for_everyone()

    result = dataset
    for name in columns:
        values = gathered[name].detach().cpu().numpy().tolist()
        result = result.add_column(name, values)
    return result
