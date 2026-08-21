#!/usr/bin/env python3
"""Fail when a test leg executed fewer tests than it is required to.

A leg whose tests all skip still exits zero, so a marker typo, a missing
optional backend, or a deleted test file silently turns a required check into a
no-op. Exit codes cannot express this: pytest's exit 5 covers "nothing
collected", not "everything collected was skipped". Legs that must observe real
execution pass ``--min-executed``.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from xml.etree import ElementTree as ET


def _executed(report: Path) -> int:
    """Return the number of test cases in a JUnit report that actually ran."""
    root = ET.parse(report).getroot()
    return sum(
        int(suite.get("tests", 0)) - int(suite.get("skipped", 0))
        for suite in root.iter("testsuite")
    )


def main() -> int:
    """Compare executed test counts against the required minimum."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--report-dir", type=Path, required=True)
    parser.add_argument("--min-executed", type=int, default=1)
    args = parser.parse_args()

    reports = sorted(args.report_dir.glob("*.xml"))
    if not reports:
        print(f"::error::no pytest reports were written to {args.report_dir}.")
        return 1

    executed = sum(_executed(report) for report in reports)
    if executed < args.min_executed:
        print(
            f"::error::{args.report_dir} reports {executed} executed tests across "
            f"{len(reports)} report(s); at least {args.min_executed} required. "
            f"Every selected test skipped, or the selection is empty."
        )
        return 1

    print(f"{args.report_dir}: {executed} executed tests across {len(reports)} report(s).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
