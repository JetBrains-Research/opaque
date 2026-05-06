# Copyright (c) 2025 Opaque Authors
# SPDX-License-Identifier: Apache-2.0
"""Tests for the empty_collate wrapper."""

import pytest
import torch

from opaque.functional._collate import _empty_like, empty_collate


class TestEmptyLike:
    """Tests for the _empty_like helper."""

    def test_tensor(self):
        t = torch.tensor([[1, 2, 3], [4, 5, 6]])
        result = _empty_like(t)
        assert result.shape == (0, 3)
        assert result.dtype == t.dtype

    def test_dict_of_tensors(self):
        template = {
            "input_ids": torch.randint(0, 100, (4, 10)),
            "labels": torch.randint(0, 100, (4, 10)),
            "attention_mask": torch.ones(4, 10),
        }
        result = _empty_like(template)
        assert isinstance(result, dict)
        assert set(result.keys()) == set(template.keys())
        for k in template:
            assert result[k].shape[0] == 0
            assert result[k].shape[1:] == template[k].shape[1:]
            assert result[k].dtype == template[k].dtype

    def test_tuple_of_tensors(self):
        template = (torch.randn(3, 5), torch.randint(0, 10, (3, 5)))
        result = _empty_like(template)
        assert isinstance(result, tuple)
        assert len(result) == 2
        assert result[0].shape == (0, 5)
        assert result[1].shape == (0, 5)
        assert result[1].dtype == torch.int64

    def test_list_of_tensors(self):
        template = [torch.randn(2, 4), torch.randn(2, 4)]
        result = _empty_like(template)
        assert isinstance(result, list)
        assert len(result) == 2
        assert result[0].shape == (0, 4)

    def test_nested_dict_with_tuple(self):
        template = {"out": (torch.randn(2, 3),), "mask": torch.ones(2, 3)}
        result = _empty_like(template)
        assert result["out"][0].shape == (0, 3)
        assert result["mask"].shape == (0, 3)

    def test_preserves_dtype(self):
        for dtype in [torch.float16, torch.bfloat16, torch.int32, torch.bool]:
            t = torch.zeros(3, 4, dtype=dtype)
            assert _empty_like(t).dtype == dtype


class TestPoissonCollate:
    """Tests for the empty_collate wrapper."""

    def test_nonempty_passes_through(self):
        def collate(examples):
            return torch.stack(examples)

        wrapped = empty_collate(collate)
        tensors = [torch.tensor([1, 2]), torch.tensor([3, 4])]
        result = wrapped(tensors)
        assert torch.equal(result, torch.stack(tensors))

    def test_empty_before_nonempty_falls_through(self):
        """Before any template is learned, empty batch falls through to collate_fn."""
        wrapped = empty_collate(lambda ex: (_ for _ in ()).throw(IndexError))
        with pytest.raises(IndexError):
            wrapped([])

    def test_empty_after_nonempty_returns_structure(self):
        """After learning structure, empty batch returns zero-batch-dim output."""

        def collate(examples):
            return {"x": torch.stack(examples), "y": torch.ones(len(examples), 5)}

        wrapped = empty_collate(collate)

        # First call: non-empty, learns template
        wrapped([torch.tensor([1, 2]), torch.tensor([3, 4])])

        # Second call: empty, returns learned structure
        result = wrapped([])
        assert isinstance(result, dict)
        assert result["x"].shape == (0, 2)
        assert result["y"].shape == (0, 5)

    def test_empty_after_learning_preserves_dtype(self):
        def collate(examples):
            return (torch.tensor(examples, dtype=torch.float16),)

        wrapped = empty_collate(collate)
        wrapped([[1.0, 2.0]])
        result = wrapped([])
        assert result[0].dtype == torch.float16

    def test_preserves_function_name(self):
        def my_collate(examples):
            return examples

        wrapped = empty_collate(my_collate)
        assert wrapped.__name__ == "my_collate"

    def test_template_captured_once(self):
        """Template is captured from the first non-empty call only."""
        call_count = [0]

        def collate(examples):
            call_count[0] += 1
            return {"ids": torch.randn(len(examples), call_count[0])}

        wrapped = empty_collate(collate)
        wrapped([1])  # first: feature dim 1
        wrapped([1, 2])  # second: feature dim 2 (template unchanged)
        result = wrapped([])  # empty: uses template from first call
        assert result["ids"].shape == (0, 1)

    def test_tuple_output(self):
        """Handles collate functions that return tuples."""

        def collate(examples):
            return (torch.stack(examples),)

        wrapped = empty_collate(collate)
        wrapped([torch.randn(5)])
        result = wrapped([])
        assert isinstance(result, tuple)
        assert result[0].shape == (0, 5)

    def test_decorator_usage(self):
        """Can be used as a decorator."""

        @empty_collate
        def collate(examples):
            return (torch.stack(examples),)

        collate([torch.randn(3)])
        result = collate([])
        assert isinstance(result, tuple)
        assert result[0].shape == (0, 3)
