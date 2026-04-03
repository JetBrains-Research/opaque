# Copyright (c) 2025 Opaque Authors
# SPDX-License-Identifier: Apache-2.0
"""Tests for poisson_collate wrapper and DataCollator compat patch."""

import torch

from opaque.sampling.collate import poisson_collate


class TestPoissonCollate:
    """Tests for the poisson_collate wrapper."""

    def test_empty_returns_none(self):
        def collate(examples):
            return torch.stack(examples)

        wrapped = poisson_collate(collate)
        assert wrapped([]) is None

    def test_nonempty_passes_through(self):
        def collate(examples):
            return torch.stack(examples)

        wrapped = poisson_collate(collate)
        tensors = [torch.tensor([1, 2]), torch.tensor([3, 4])]
        result = wrapped(tensors)
        assert torch.equal(result, torch.stack(tensors))

    def test_preserves_function_name(self):
        def my_collate(examples):
            return examples

        wrapped = poisson_collate(my_collate)
        assert wrapped.__name__ == "my_collate"

    def test_none_is_falsy(self):
        """Training loops can use `if batch is None: continue`."""
        wrapped = poisson_collate(lambda ex: ex)
        result = wrapped([])
        assert result is None
        assert not result  # falsy


class TestDataCollatorPatch:
    """Tests for the DataCollatorForLanguageModeling compat patch."""

    def test_empty_examples_returns_empty_tensors(self):
        transformers = __import__("transformers", fromlist=["DataCollatorForLanguageModeling"])
        DataCollatorForLanguageModeling = transformers.DataCollatorForLanguageModeling
        AutoTokenizer = transformers.AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained("gpt2")
        tokenizer.pad_token = tokenizer.eos_token
        collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

        # This would crash without the patch (IndexError on examples[0])
        result = collator([])

        assert "input_ids" in result
        assert "labels" in result
        assert result["input_ids"].shape[0] == 0
        assert result["labels"].shape[0] == 0
        assert result["input_ids"].dtype == torch.long
        assert result["labels"].dtype == torch.long

    def test_nonempty_examples_unchanged(self):
        transformers = __import__("transformers", fromlist=["DataCollatorForLanguageModeling"])
        DataCollatorForLanguageModeling = transformers.DataCollatorForLanguageModeling
        AutoTokenizer = transformers.AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained("gpt2")
        tokenizer.pad_token = tokenizer.eos_token
        collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

        examples = [{"input_ids": [1, 2, 3]}, {"input_ids": [4, 5, 6]}]
        result = collator(examples)

        assert result["input_ids"].shape[0] == 2
        assert result["labels"].shape[0] == 2
