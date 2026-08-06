from __future__ import annotations

from typing import TYPE_CHECKING

from benchmarks.inventory import load_claims, validate_claim_inventory

if TYPE_CHECKING:
    from pathlib import Path


def _write_inventory(root: Path, *, status: str = "under_investigation") -> None:
    inventory = root / "benchmarks" / "claims.toml"
    inventory.parent.mkdir()
    inventory.write_text(
        f'''[[claim]]
id = "example-speed"
statement = "The optimized path is 2.4x faster."
classification = "benchmark"
status = "{status}"
case = "example.case"
decision = "Retain only when matching evidence is committed."
current_path = "docs/guide.md"
current_match = "2.4x faster"
''',
        encoding="utf-8",
    )


def test_inventory_classifies_current_performance_claim(tmp_path: Path) -> None:
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "guide.md").write_text(
        "The optimized path is 2.4x faster.\n", encoding="utf-8"
    )
    _write_inventory(tmp_path)

    errors = validate_claim_inventory(
        tmp_path,
        load_claims(tmp_path / "benchmarks" / "claims.toml"),
        case_ids={"example.case"},
        result_case_ids=set(),
    )

    assert errors == []


def test_inventory_rejects_unclassified_numeric_claim(tmp_path: Path) -> None:
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "guide.md").write_text(
        "The optimized path is 2.4x faster.\nAnother path uses 3x less memory.\n",
        encoding="utf-8",
    )
    _write_inventory(tmp_path)

    errors = validate_claim_inventory(
        tmp_path,
        load_claims(tmp_path / "benchmarks" / "claims.toml"),
        case_ids={"example.case"},
        result_case_ids=set(),
    )

    assert any("unclassified numeric performance claim" in error for error in errors)


def test_supported_claim_requires_committed_case_result(tmp_path: Path) -> None:
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "guide.md").write_text(
        "The optimized path is 2.4x faster.\n", encoding="utf-8"
    )
    _write_inventory(tmp_path, status="supported")
    claims = load_claims(tmp_path / "benchmarks" / "claims.toml")

    errors = validate_claim_inventory(
        tmp_path,
        claims,
        case_ids={"example.case"},
        result_case_ids=set(),
    )
    assert any("requires a committed result" in error for error in errors)

    assert (
        validate_claim_inventory(
            tmp_path,
            claims,
            case_ids={"example.case"},
            result_case_ids={"example.case"},
        )
        == []
    )
