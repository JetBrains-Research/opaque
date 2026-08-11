from __future__ import annotations

from pathlib import Path

import pytest
from benchmarks.claims import scan_file, scan_repository


@pytest.mark.parametrize(
    "text",
    [
        "The optimized path is 2.4x faster.",
        "This reduces peak memory by 2 GB.",
        "Deterministic execution is typically 10-30% slower.",
        "The measured crossover is ~16 experts and reaches ~2x at 128 experts.",
    ],
)
def test_scan_file_finds_precise_performance_claims(text: str) -> None:
    findings = scan_file(Path("docs/guide.md"), text)

    assert len(findings) == 1
    assert findings[0].line == 1


@pytest.mark.parametrize(
    "text",
    [
        "A smaller discretization such as 1e-5 is slower but more accurate.",
        "Direct convolution is O(n²).",
        "The benchmark used a batch size of 16.",
    ],
)
def test_scan_file_ignores_configuration_and_complexity_numbers(text: str) -> None:
    assert scan_file(Path("docs/guide.md"), text) == []


def test_scan_repository_checks_docs_and_public_source_but_not_tests(
    tmp_path: Path,
) -> None:
    docs = tmp_path / "docs"
    source = tmp_path / "packages" / "example" / "src"
    tests = tmp_path / "packages" / "example" / "tests"
    docs.mkdir()
    source.mkdir(parents=True)
    tests.mkdir(parents=True)
    (docs / "guide.md").write_text("Uses 3x less memory.\n", encoding="utf-8")
    (source / "api.py").write_text(
        '"""Runs 25% faster on the reference device."""\n', encoding="utf-8"
    )
    (tests / "test_perf.py").write_text(
        "# Assert at least 30% memory savings.\n", encoding="utf-8"
    )

    findings = scan_repository(tmp_path)

    assert {finding.path.relative_to(tmp_path).as_posix() for finding in findings} == {
        "docs/guide.md",
        "packages/example/src/api.py",
    }


def test_scan_repository_excludes_generated_benchmark_evidence(tmp_path: Path) -> None:
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "benchmarks.md").write_text(
        "Generated result: 2.1x faster.\n", encoding="utf-8"
    )

    assert scan_repository(tmp_path) == []


def test_scan_file_ignores_internal_rust_comments_but_checks_rustdoc() -> None:
    text = """
// Internal fixture is 75x larger for this test.
/// Public path is 2.4x faster on the reference workload.
"""

    findings = scan_file(Path("src/lib.rs"), text)

    assert len(findings) == 1
    assert findings[0].line == 3
