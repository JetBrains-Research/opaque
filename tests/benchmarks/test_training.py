from __future__ import annotations

import pytest
from benchmarks.cases.training import parse_training_summary
from benchmarks.core import BenchmarkError


def test_parse_training_summary_extracts_native_metrics() -> None:
    output = """
Training results:
  Total steps: 20
Performance:
  Throughput: 3.7 samples/s
  Steps/s: 0.93
  Peak memory: 52.21 GB
"""

    assert parse_training_summary(output) == {
        "total_steps": 20,
        "samples_per_second": 3.7,
        "steps_per_second": 0.93,
        "peak_memory_gb": 52.21,
    }


def test_parse_training_summary_rejects_incomplete_output() -> None:
    with pytest.raises(BenchmarkError, match="missing final metrics"):
        parse_training_summary("Training failed before the final summary")
