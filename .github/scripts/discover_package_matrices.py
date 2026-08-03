#!/usr/bin/env python3
"""Discover package matrices for GitHub Actions from the packages/ tree."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import tomllib


REPO_ROOT = Path(__file__).resolve().parents[2]
PACKAGES_DIR = REPO_ROOT / "packages"


def _package_entries() -> list[dict[str, object]]:
    entries: list[dict[str, object]] = []
    for pyproject_path in sorted(PACKAGES_DIR.glob("*/pyproject.toml")):
        data = tomllib.loads(pyproject_path.read_text())
        package_dir = pyproject_path.parent
        dist_name = data["project"]["name"]
        build_backend = data["build-system"]["build-backend"]
        entries.append(
            {
                "dir": package_dir.name,
                "dist": dist_name,
                "name": dist_name.removeprefix("opaque-"),
                "path": package_dir.relative_to(REPO_ROOT).as_posix(),
                "native": build_backend == "maturin",
            }
        )
    return entries


def _outputs() -> dict[str, object]:
    packages = _package_entries()
    return {
        "test_shards": [
            *(
                {"name": package["name"], "paths": package["path"]}
                for package in packages
            ),
            {"name": "repo-tests", "paths": "tests"},
        ],
        "python_build_packages": [
            {
                "dir": package["dir"],
                "dist": package["dist"],
                "path": package["path"],
            }
            for package in packages
            if not package["native"]
        ],
        "native_build_packages": [
            {
                "dir": package["dir"],
                "dist": package["dist"],
                "path": package["path"],
            }
            for package in packages
            if package["native"]
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--github-output",
        type=Path,
        help="Write GitHub Actions step outputs to this file.",
    )
    args = parser.parse_args()

    outputs = _outputs()
    if args.github_output is not None:
        with args.github_output.open("a", encoding="utf-8") as fh:
            for key, value in outputs.items():
                fh.write(f"{key}={json.dumps(value, separators=(',', ':'))}\n")
    else:
        print(json.dumps(outputs, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
