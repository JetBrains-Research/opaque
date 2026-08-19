"""The default collation helper preserves the container it was given."""

from __future__ import annotations

import collections

import pytest
import torch

from opaque.api.auditing.attacks._helpers import _default_collate


def test_stacks_native_arrays_along_a_new_leading_axis() -> None:
    out = _default_collate([torch.ones(2), torch.zeros(2)])
    assert tuple(out.shape) == (2, 2)


def test_namedtuple_fields_are_bound_positionally() -> None:
    """A namedtuple takes its fields positionally, not as one iterable.

    Building it the way a plain tuple is built binds the whole column list to
    the first field and raises on the rest, so a dataset yielding namedtuple
    examples could not be audited at all.
    """
    point = collections.namedtuple("point", "x y")
    out = _default_collate([point(torch.ones(2), torch.zeros(2)) for _ in range(3)])

    assert isinstance(out, point)
    assert out._fields == ("x", "y")
    assert tuple(out.x.shape) == (3, 2)


@pytest.mark.parametrize(
    "factory", [dict, collections.OrderedDict], ids=["dict", "ordered_dict"]
)
def test_mapping_type_is_preserved(factory) -> None:
    examples = [factory(a=torch.ones(2)), factory(a=torch.zeros(2))]
    out = _default_collate(examples)

    assert type(out) is factory
    assert tuple(out["a"].shape) == (2, 2)


@pytest.mark.parametrize("factory", [tuple, list], ids=["tuple", "list"])
def test_sequence_type_is_preserved(factory) -> None:
    examples = [factory([torch.ones(2), torch.zeros(2)]) for _ in range(3)]
    out = _default_collate(examples)

    assert type(out) is factory
    assert tuple(out[0].shape) == (3, 2)


def test_unknown_leaf_needs_an_explicit_collate_fn() -> None:
    with pytest.raises((TypeError, ValueError)):
        _default_collate(["not an array", "either"])
