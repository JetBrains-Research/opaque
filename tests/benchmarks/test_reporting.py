from __future__ import annotations

import json
from typing import TYPE_CHECKING

from benchmarks.core import SCHEMA_VERSION, capture_sources, source_digest
from benchmarks.reporting import render_repository

if TYPE_CHECKING:
    from pathlib import Path


def test_render_repository_links_tables_to_result_data(tmp_path: Path) -> None:
    source = tmp_path / "implementation.py"
    source.write_text("VALUE = 1\n", encoding="utf-8")
    sources = capture_sources(tmp_path, ["implementation.py"])
    result = {
        "schema_version": SCHEMA_VERSION,
        "case": {
            "id": "optimizers.state",
            "description": "State size.",
            "config": {"size": 8},
            "sources": sources,
            "source_digest": source_digest(sources),
        },
        "provenance": {
            "timestamp_utc": "2026-08-06T05:00:00Z",
            "command": ["python", "-m", "benchmarks", "run"],
            "git": {"commit": "a" * 40, "dirty": False, "dirty_paths": []},
            "hardware": {
                "machine": "arm64",
                "cpu": "Example CPU",
                "memory_bytes": 1024,
                "accelerator": None,
            },
            "software": {
                "platform": "Example OS",
                "python": "3.12.0",
                "packages": {"opaque": "0.1.0"},
            },
        },
        "measurements": [
            {
                "name": "adamw",
                "parameters": {},
                "metrics": {
                    "state_bytes": {
                        "value": 2048,
                        "unit": "byte",
                        "statistic": "exact",
                        "method": "unique_tensor_storage",
                    }
                },
            }
        ],
        "comparisons": [],
        "notes": [],
    }
    results = tmp_path / "benchmarks" / "results"
    results.mkdir(parents=True)
    (results / "optimizer.json").write_text(json.dumps(result), encoding="utf-8")

    markdown = render_repository(tmp_path)

    assert (
        "[optimizer.json](https://github.com/JetBrains-Research/opaque/blob/main/"
        "benchmarks/results/optimizer.json)" in markdown
    )
    assert "optimizers.state" in markdown
    assert "2,048 byte" in markdown
    assert "`exact; unique_tensor_storage`" in markdown
    assert "Example CPU" in markdown


def test_render_repository_is_deterministic(tmp_path: Path) -> None:
    (tmp_path / "benchmarks" / "results").mkdir(parents=True)

    assert render_repository(tmp_path) == render_repository(tmp_path)
