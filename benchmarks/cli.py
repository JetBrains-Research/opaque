from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any

from benchmarks.cases import get_case, list_cases
from benchmarks.core import (
    BenchmarkError,
    RunOptions,
    build_result,
    load_result,
    validate_result,
    write_result,
)
from benchmarks.reporting import render_repository
from benchmarks.verify import check_repository, result_files

if TYPE_CHECKING:
    from collections.abc import Sequence

_ROOT = Path(__file__).resolve().parents[1]


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m benchmarks",
        description="Run and validate Opaque's repository-owned benchmarks.",
    )
    subparsers = parser.add_subparsers(dest="action", required=True)
    subparsers.add_parser("list", help="List benchmark cases and presets.")

    run = subparsers.add_parser("run", help="Run one benchmark case.")
    run.add_argument("case")
    run.add_argument("--preset", default="reference")
    run.add_argument("--device")
    run.add_argument("--warmup", type=int, default=3)
    run.add_argument("--repeats", type=int, default=10)
    run.add_argument("--set", dest="overrides", action="append", default=[])
    run.add_argument("--output", type=Path)

    validate = subparsers.add_parser("validate", help="Validate result artifacts.")
    validate.add_argument("paths", nargs="*", type=Path)
    validate.add_argument("--require-clean", action="store_true")

    render = subparsers.add_parser("render", help="Regenerate the evidence page.")
    render.add_argument("--output", type=Path, default=Path("docs/benchmarks.md"))

    check = subparsers.add_parser(
        "check", help="Check results, claim inventory, and generated docs."
    )
    check.add_argument("--require-clean-results", action="store_true")
    return parser


def _apply_overrides(config: dict[str, Any], overrides: Sequence[str]) -> None:
    for override in overrides:
        if "=" not in override:
            raise BenchmarkError(f"Override must use key=JSON syntax: {override!r}")
        dotted_key, raw_value = override.split("=", 1)
        parts = dotted_key.split(".")
        target: dict[str, Any] = config
        for part in parts[:-1]:
            child = target.get(part)
            if not isinstance(child, dict):
                raise BenchmarkError(f"Unknown configuration key {dotted_key!r}")
            target = child
        if parts[-1] not in target:
            raise BenchmarkError(f"Unknown configuration key {dotted_key!r}")
        try:
            target[parts[-1]] = json.loads(raw_value)
        except json.JSONDecodeError as error:
            raise BenchmarkError(
                f"Override value for {dotted_key!r} must be valid JSON"
            ) from error


def _run_case(args: argparse.Namespace, argv: Sequence[str]) -> int:
    case = get_case(args.case)
    if args.preset not in case.presets:
        choices = ", ".join(case.presets)
        raise BenchmarkError(
            f"Unknown preset {args.preset!r} for {case.case_id}; choose: {choices}"
        )
    config = copy.deepcopy(dict(case.presets[args.preset]))
    _apply_overrides(config, args.overrides)
    device = args.device or case.devices[0]
    options = RunOptions(device=device, warmup=args.warmup, repeats=args.repeats)
    run = case.run(config, options)
    command = ["uv", "run", "python", "-m", "benchmarks", *argv]
    result = build_result(
        case,
        config,
        run,
        root=_ROOT,
        command=command,
        device=device,
    )
    errors = validate_result(result, root=_ROOT)
    if errors:
        raise BenchmarkError("Invalid generated result:\n- " + "\n- ".join(errors))
    output = args.output or Path(
        "benchmarks/results/"
        f"{case.case_id.replace('.', '-')}-{args.preset}-{device}.json"
    )
    if not output.is_absolute():
        output = _ROOT / output
    write_result(result, output)
    print(output.relative_to(_ROOT))
    for comparison in run.comparisons:
        print(f"{comparison['name']}: {comparison['value']:.6g} {comparison['unit']}")
    return 0


def _validate(args: argparse.Namespace) -> int:
    paths = args.paths or result_files(_ROOT)
    if not paths:
        raise BenchmarkError("No benchmark result files found")
    errors: list[str] = []
    for path in paths:
        if not path.is_absolute():
            path = _ROOT / path
        result = load_result(path)
        errors.extend(
            f"{path.relative_to(_ROOT)}: {error}"
            for error in validate_result(result, root=_ROOT)
        )
        if args.require_clean and result.get("provenance", {}).get("git", {}).get(
            "dirty"
        ):
            errors.append(f"{path.relative_to(_ROOT)}: dirty worktree result")
    if errors:
        raise BenchmarkError("\n".join(errors))
    print(f"Validated {len(paths)} result(s).")
    return 0


def _render(args: argparse.Namespace) -> int:
    output = args.output if args.output.is_absolute() else _ROOT / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(render_repository(_ROOT), encoding="utf-8")
    print(output.relative_to(_ROOT))
    return 0


def _list() -> int:
    for case in list_cases():
        print(
            f"{case.case_id}\tdevices={','.join(case.devices)}\t"
            f"presets={','.join(case.presets)}"
        )
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    arguments = list(sys.argv[1:] if argv is None else argv)
    parser = _parser()
    args = parser.parse_args(arguments)
    try:
        if args.action == "list":
            return _list()
        if args.action == "run":
            return _run_case(args, arguments)
        if args.action == "validate":
            return _validate(args)
        if args.action == "render":
            return _render(args)
        if args.action == "check":
            errors = check_repository(
                _ROOT, require_clean_results=args.require_clean_results
            )
            if errors:
                raise BenchmarkError("\n".join(errors))
            print("Benchmark evidence is current.")
            return 0
    except (BenchmarkError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    raise AssertionError(f"Unhandled action: {args.action}")


__all__ = ["main"]
