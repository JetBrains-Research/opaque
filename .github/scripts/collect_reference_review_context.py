#!/usr/bin/env python3
"""Collect broad reference-like context from changed files for gh-aw review."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
FILE_PATTERN = re.compile(r"^(README\.md|docs/.*|packages/.*\.(md|py|rs))$")
REFERENCE_RE = re.compile(
    r"(arxiv\.org/abs/|arXiv:|doi\.org/|doi:|et al\.|[A-Z][A-Za-z-]+ et al\. [0-9]{4}|[A-Z][A-Za-z-]+, [A-Z][A-Za-z-]+ [0-9]{4}|paper\b|privacy auditing|differential privacy|DP-[A-Za-z0-9-]+)",
    re.IGNORECASE,
)


def _changed_files(base: str) -> list[str]:
    result = subprocess.run(
        ["git", "--no-pager", "diff", "--name-only", f"{base}...HEAD"],
        check=True,
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )
    files = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    return [path for path in files if FILE_PATTERN.match(path)]


def _extract_reference_lines(path: Path) -> list[dict[str, object]]:
    entries: list[dict[str, object]] = []
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if REFERENCE_RE.search(line):
            entries.append(
                {
                    "file": str(path.relative_to(REPO_ROOT)),
                    "line": line_no,
                    "text": line.strip(),
                }
            )
    return entries


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", default="origin/main")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    files = _changed_files(args.base)
    entries: list[dict[str, object]] = []
    for rel_path in files:
        entries.extend(_extract_reference_lines(REPO_ROOT / rel_path))

    payload = {"base": args.base, "files": files, "entries": entries}
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(
        f"Wrote {len(entries)} reference-like lines from {len(files)} changed files "
        f"to {args.output}."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
