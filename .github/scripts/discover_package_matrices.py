#!/usr/bin/env python3
"""Discover package matrices for GitHub Actions from the packages/ tree."""

from __future__ import annotations

import argparse
import json
import tomllib
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
PACKAGES_DIR = REPO_ROOT / "packages"

# Stable, timing-balanced groups keep the five CPU-oriented matrices compact.
# Discovery must stay reproducible, so it does not query historical GitHub runs.
_TEST_SHARD_GROUPS = (
    (
        "dp-mechanisms",
        "DP mechanisms",
        ("opaque-dpsgd", "opaque-dpftrl"),
    ),
    (
        "runtime-patches",
        "Runtime and patches",
        (
            "opaque-engine",
            "opaque-optimizers",
            "opaque-patches",
        ),
    ),
    (
        "transformers-auditing",
        "Transformers and auditing",
        ("opaque-transformers", "opaque-auditing"),
    ),
    (
        "foundation-alignment",
        "Foundation and alignment",
        ("opaque-base", "opaque-accounting", "opaque-alignment"),
    ),
)


def _has_pytest_marker(path: Path, marker: str) -> bool:
    needle = f"pytest.mark.{marker}"
    return any(needle in test_path.read_text() for test_path in path.rglob("*.py"))


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
                "cuda_tests": _has_pytest_marker(package_dir / "tests", "cuda"),
                "distributed_tests": _has_pytest_marker(
                    package_dir / "tests", "distributed"
                ),
            }
        )
    return entries


def _outputs() -> dict[str, object]:
    packages = _package_entries()
    packages_by_dist = {package["dist"]: package for package in packages}
    grouped_dists: set[str] = set()
    test_shards: list[dict[str, object]] = []
    for name, label, dists in _TEST_SHARD_GROUPS:
        grouped_packages = [
            packages_by_dist[dist] for dist in dists if dist in packages_by_dist
        ]
        if not grouped_packages:
            continue
        grouped_dists.update(package["dist"] for package in grouped_packages)
        test_shards.append(
            {
                "label": label,
                "name": name,
                "paths": [package["path"] for package in grouped_packages],
            }
        )

    test_shards.extend(
        {
            "label": package["dist"],
            "name": package["name"],
            "paths": [package["path"]],
        }
        for package in packages
        if package["dist"] not in grouped_dists
    )
    test_shards.append(
        {"label": "integration", "name": "integration", "paths": ["tests"]}
    )
    return {
        "test_shards": test_shards,
        "cuda_test_shards": [
            *(
                {
                    "label": package["dist"],
                    "name": package["name"],
                    "paths": [package["path"]],
                }
                for package in packages
                if package["cuda_tests"]
            ),
            *(
                [{"label": "integration", "name": "integration", "paths": ["tests"]}]
                if _has_pytest_marker(REPO_ROOT / "tests", "cuda")
                else []
            ),
        ],
        "distributed_test_shards": [
            *(
                {
                    "label": package["dist"],
                    "name": package["name"],
                    "paths": [package["path"]],
                }
                for package in packages
                if package["distributed_tests"]
            ),
            *(
                [{"label": "integration", "name": "integration", "paths": ["tests"]}]
                if _has_pytest_marker(REPO_ROOT / "tests", "distributed")
                else []
            ),
        ],
        "python_build_packages": [
            {"dir": ".", "dist": "opaque", "path": "."},
            *(
                {
                    "dir": package["dir"],
                    "dist": package["dist"],
                    "path": package["path"],
                }
                for package in packages
                if not package["native"]
            ),
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
    """Print or export the discovered package matrices."""
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
