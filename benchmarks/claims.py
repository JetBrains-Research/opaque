from __future__ import annotations

import re
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path

_PERFORMANCE = re.compile(
    r"faster|slower|speedup|throughput|latency|overhead|memory|peak|runtime|"
    r"performance|saving|reduction|crossover|efficien|bandwidth",
    re.IGNORECASE,
)
_EMPIRICAL_RATIO = re.compile(
    r"(?<![\w.])(?:[~≈<>]=?\s*)?\d+(?:\.\d+)?\s*(?:[x×]|%)",
    re.IGNORECASE,
)
_RESOURCE_AMOUNT = re.compile(
    r"(?<![\w.])\d+(?:\.\d+)?\s*(?:[KMGT]i?B|ms|milliseconds?|seconds?|"
    r"minutes?|hours?)\b",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class ClaimFinding:
    path: Path
    line: int
    text: str


def scan_file(path: Path, text: str) -> list[ClaimFinding]:
    lines = text.splitlines()
    findings: list[ClaimFinding] = []
    for index, line in enumerate(lines):
        stripped = line.lstrip()
        if (
            path.suffix == ".rs"
            and stripped.startswith("//")
            and not stripped.startswith(("///", "//!"))
        ):
            continue
        if not (_EMPIRICAL_RATIO.search(line) or _RESOURCE_AMOUNT.search(line)):
            continue
        window = " ".join(lines[max(0, index - 1) : min(len(lines), index + 2)])
        if _PERFORMANCE.search(window):
            findings.append(ClaimFinding(path=path, line=index + 1, text=line.strip()))
    return findings


def scan_repository(root: Path) -> list[ClaimFinding]:
    candidates: set[Path] = set()
    readme = root / "README.md"
    if readme.is_file():
        candidates.add(readme)
    docs = root / "docs"
    if docs.is_dir():
        candidates.update(docs.rglob("*.md"))
    packages = root / "packages"
    if packages.is_dir():
        candidates.update(packages.glob("*/README.md"))
        for source_root in packages.glob("*/src"):
            for suffix in ("*.py", "*.rs", "*.md"):
                candidates.update(source_root.rglob(suffix))

    excluded = {(root / "docs" / "benchmarks.md").resolve()}
    findings: list[ClaimFinding] = []
    for path in sorted(candidates):
        if path.resolve() in excluded:
            continue
        try:
            findings.extend(scan_file(path, path.read_text(encoding="utf-8")))
        except OSError:
            continue
    return findings


__all__ = ["ClaimFinding", "scan_file", "scan_repository"]
