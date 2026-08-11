from __future__ import annotations

from typing import TYPE_CHECKING

from benchmarks.cases import list_cases
from benchmarks.core import (
    BenchmarkError,
    benchmark_source_paths,
    capture_sources,
    load_result,
    validate_result,
)
from benchmarks.inventory import load_claims, validate_claim_inventory
from benchmarks.reporting import render_repository

if TYPE_CHECKING:
    from pathlib import Path


def result_files(root: Path) -> list[Path]:
    directory = root / "benchmarks" / "results"
    return sorted(directory.glob("*.json")) if directory.is_dir() else []


def check_repository(root: Path, *, require_clean_results: bool = False) -> list[str]:
    errors: list[str] = []
    cases = {case.case_id: case for case in list_cases()}
    artifacts = result_files(root)
    if not artifacts:
        errors.append("no committed benchmark result JSON files found")
    result_case_ids: set[str] = set()
    for path in artifacts:
        try:
            result = load_result(path)
        except BenchmarkError as error:
            errors.append(str(error))
            continue
        relative = path.relative_to(root).as_posix()
        errors.extend(
            f"{relative}: {error}" for error in validate_result(result, root=root)
        )
        case_id = result.get("case", {}).get("id")
        if case_id not in cases:
            errors.append(f"{relative}: unknown benchmark case {case_id!r}")
        else:
            result_case_ids.add(case_id)
            expected_sources = set(
                capture_sources(root, benchmark_source_paths(cases[case_id]))
            )
            recorded_sources = set(result.get("case", {}).get("sources", {}))
            if recorded_sources != expected_sources:
                errors.append(f"{relative}: stale source set; rerun case {case_id!r}")
        if require_clean_results and result.get("provenance", {}).get("git", {}).get(
            "dirty"
        ):
            errors.append(f"{relative}: result was recorded from a dirty worktree")

    claims_path = root / "benchmarks" / "claims.toml"
    if not claims_path.is_file():
        errors.append("missing benchmarks/claims.toml")
    else:
        try:
            claims = load_claims(claims_path)
        except BenchmarkError as error:
            errors.append(str(error))
        else:
            errors.extend(
                validate_claim_inventory(
                    root,
                    claims,
                    case_ids=set(cases),
                    result_case_ids=result_case_ids,
                )
            )

    generated_path = root / "docs" / "benchmarks.md"
    if not generated_path.is_file():
        errors.append("missing generated docs/benchmarks.md")
    else:
        expected = render_repository(root)
        if generated_path.read_text(encoding="utf-8") != expected:
            errors.append(
                "docs/benchmarks.md is stale; run `uv run python -m benchmarks render`"
            )
    return errors


__all__ = ["check_repository", "result_files"]
