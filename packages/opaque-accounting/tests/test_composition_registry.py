"""Regression test for the Composed/Repeated DpProcess registry.

Before commit ``1696e7aa fix(accounting): register Composed/Repeated in
DpProcess registry`` the composition module's ``__init__`` imported
:class:`Composed` and :class:`Repeated` only lazily (inside the
``cached`` factory's late binding), so the dataclass
``__init_subclass__`` hook that fills ``_PROCESS_REGISTRY`` never
fired during a plain ``import opaque.accounting``.  Round-tripping a
tree that contained either node then failed deserialization with
``ValueError: Unknown nested DpProcess type 'Composed' in field 'inner'``
inside :func:`opaque.api.accounting.core._process_codec._load_dp_process`.

This test reconstructs that minimal failure surface using only
opaque-accounting-owned process types (EpsDelta, Composed, Repeated,
CachedProcess), so it does not depend on opaque-dpsgd or opaque-dpftrl
being installed.  Without A.1 the ``from_state_dict`` call raises;
with A.1 the tree round-trips clean.
"""

from __future__ import annotations

import json

import opaque.accounting as acc
from opaque.api.accounting.core._base import _PROCESS_REGISTRY
from opaque.serialization import from_state_dict, state_dict


def test_process_registry_contains_composition_nodes() -> None:
    """Plain ``import opaque.accounting`` must register composition nodes."""
    assert "Composed" in _PROCESS_REGISTRY, (
        "Composed missing from _PROCESS_REGISTRY — round-1 checkpoint-resume bug. "
        "Verify the import side-effect in "
        "opaque/api/accounting/core/composition/__init__.py."
    )
    assert "Repeated" in _PROCESS_REGISTRY, (
        "Repeated missing from _PROCESS_REGISTRY — round-1 checkpoint-resume bug."
    )
    assert "CachedProcess" in _PROCESS_REGISTRY


def test_cached_composed_repeated_roundtrips_via_json() -> None:
    """``cached(compose(repeat(EpsDelta, 3), EpsDelta))`` must round-trip.

    Mirrors the structural pattern DPTrainer writes into its accountant
    checkpoint: a cached opaque-boundary at the top, a Composed node
    splicing the prior phase's repeated step, and a Repeated wrapper
    for the homogeneous repetition.  This is the exact tree shape that
    raised on resume in round-1 before A.1.
    """
    leaf_a = acc.eps_delta(0.5, 1e-6)
    leaf_b = acc.eps_delta(0.1, 1e-7)

    # Composed(Repeated(leaf_a, 3), leaf_b)  =  (leaf_a * 3) | leaf_b
    tree = acc.cached(acc.compose(acc.repeat(leaf_a, 3), leaf_b))

    sd = dict(state_dict(tree))
    # JSON round-trip exercises the codec's full path; the codec emits
    # only Python primitives and plain containers, so JSON is lossless.
    blob = json.dumps(sd)
    restored_sd = json.loads(blob)

    # Template-driven restore: from_state_dict overwrites a freshly-built
    # tree of the same shape.
    template = acc.cached(acc.compose(acc.repeat(leaf_a, 3), leaf_b))
    restored = from_state_dict(template, restored_sd)

    assert restored == tree, (
        "Tree did not round-trip through state_dict / from_state_dict; "
        "structural equality should hold for the (cached, composed, "
        "repeated, EpsDelta) tree."
    )
