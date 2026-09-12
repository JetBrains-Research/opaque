"""Integrity checks for the authoritative citation records."""

from __future__ import annotations

import re
import tomllib
from pathlib import Path

import pytest

_ROOT = Path(__file__).parents[1]
_REGISTRY = _ROOT / "citation_records.toml"
_SOURCE_ROOTS = (_ROOT / "packages", _ROOT / "docs")
_SOURCE_SUFFIXES = {".md", ".py", ".rs", ".toml"}
_CITATION_RECORD = re.compile(
    r"Citation:\s+arXiv:(?P<id>\d{4}\.\d{4,5});\s+"
    r"(?P<author>[^;\n]+);\s+(?P<title>[^\n]+)"
)
_UNRESOLVED_KEY = re.compile(r"\\?\[[A-Z][A-Za-z-]*\d{2}\\?\]")
_KNOWN_BAD_METADATA = (
    "Defazio, Yaida",
    "DiscoPOP: Discovering Preference Optimization Procedures",
    "Banded Inverse Square Root for DP Matrix Factorization",
)


def _records() -> dict[str, dict[str, object]]:
    data = tomllib.loads(_REGISTRY.read_text())
    records = data["records"]
    assert len(records) == len({record["id"] for record in records})
    return {record["id"]: record for record in records}


def _source_files() -> list[Path]:
    return sorted(
        path
        for root in _SOURCE_ROOTS
        for path in root.rglob("*")
        if path.is_file() and path.suffix in _SOURCE_SUFFIXES
    )


_RECORDS = _records()
_SOURCES = _source_files()
_CITATION_SITES = [
    (path, match.group("id"), match.group("author"), match.group("title"))
    for path in _SOURCES
    for match in _CITATION_RECORD.finditer(path.read_text())
]


@pytest.mark.parametrize(
    ("path", "citation_id", "author", "title"),
    _CITATION_SITES,
    ids=lambda value: (
        str(value.relative_to(_ROOT)) if isinstance(value, Path) else value
    ),
)
def test_audited_citation_site_resolves_to_registered_record(
    path: Path, citation_id: str, author: str, title: str
) -> None:
    """Generate the audited metadata checklist directly from source markers."""
    text = path.read_text()
    record = _RECORDS[citation_id]
    canonical_url = record["url"]
    authors = record["authors"]
    expected_author = authors[0] if len(authors) == 1 else f"{authors[0]} et al."

    assert canonical_url == f"https://arxiv.org/abs/{citation_id}"
    assert canonical_url in text or re.search(
        rf"arxiv:{re.escape(citation_id)}", text, re.IGNORECASE
    )
    assert author == expected_author
    assert title == record["title"]


@pytest.mark.parametrize("citation_id", sorted(_RECORDS))
def test_audited_record_has_a_source_site(citation_id: str) -> None:
    assert any(site_id == citation_id for _, site_id, _, _ in _CITATION_SITES)


def test_source_has_no_unresolved_or_known_bad_citation_metadata() -> None:
    failures: list[str] = []
    for path in _SOURCES:
        text = path.read_text()
        if match := _UNRESOLVED_KEY.search(text):
            failures.append(
                f"{path.relative_to(_ROOT)}: unresolved key {match.group()}"
            )
        failures.extend(
            f"{path.relative_to(_ROOT)}: known bad metadata {bad_metadata!r}"
            for bad_metadata in _KNOWN_BAD_METADATA
            if bad_metadata in text
        )
    assert not failures, "\n".join(failures)
