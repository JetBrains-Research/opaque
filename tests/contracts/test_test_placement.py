"""Tests live in the wheel that depends on every package they import.

A test in package A that imports package B is only valid when A directly
or transitively depends on B. Otherwise the test would silently pull B
into A's resolved environment via the test runner's monorepo install,
even though the published wheel A cannot.

The optional-extras case (e.g. ``opaque-dpsgd[optimizers]``) is permitted
without further analysis: importing ``opaque.optimizers`` from a dpsgd
test is accepted because the extra is declared on the wheel, even if the
top-level module is not in the default cone. We trust file-head
``pytest.importorskip`` to handle the runtime case (not statically
checked here — the contract tests run with ``--extra all``).
"""

from __future__ import annotations

import ast
import pathlib
import sys
import tomllib

from .conftest import PACKAGES_DIR

# Top-level ``opaque.*`` import roots shipped by each wheel. Updated phase
# by phase as the wheel set evolves. This is the source-of-truth mapping
# that turns wheel names into import roots.
WHEEL_IMPORT_ROOTS: dict[str, tuple[str, ...]] = {
    "opaque-core": (
        "opaque.types",
        "opaque.pytree",
        "opaque._noise_allocation",
        "opaque._clipping",
        "opaque.random",
        "opaque.functional",
        "opaque.distributed",
        "opaque.scheduling",
        "opaque.serialization",
        "opaque.optimizers",
        "opaque.profiling",
    ),
    "opaque-base": (
        "opaque.api.base",
        "opaque.serialization",
    ),
    "opaque-engine": (
        "opaque.api.engine",
        "opaque.types",
        "opaque.pytree",
        "opaque.random",
        "opaque.distributed",
        "opaque.functional",
        "opaque.scheduling",
        "opaque.profiling",
    ),
    "opaque-optimizers": (
        "opaque.api.optimizers",
        "opaque.optimizers",
    ),
    "opaque-accounting": (
        "opaque.accounting",
        "opaque.api.accounting.core",
    ),
    "opaque-dpsgd": (
        "opaque.dpsgd",
        "opaque.api.dpsgd",
        "opaque.api.accounting.dpsgd",
    ),
    "opaque-dpftrl": (
        "opaque.dpftrl",
        "opaque.api.dpftrl",
        "opaque.api.accounting.dpftrl",
    ),
    "opaque-auditing": (
        "opaque.auditing",
        "opaque.api.auditing",
    ),
    "opaque-patches": (
        "opaque.patches",
        "opaque.api.patches",
    ),
    "opaque-transformers": (
        "opaque.transformers",
        "opaque.api.transformers",
    ),
}


def _read_pyproject(wheel: str) -> dict:
    path = PACKAGES_DIR / wheel / "pyproject.toml"
    if not path.exists():
        return {}
    with path.open("rb") as f:
        return tomllib.load(f)


def _direct_opaque_deps(wheel: str) -> set[str]:
    """Return the ``opaque-*`` wheel names this wheel declares as deps,
    including all optional-dependencies extras (we treat extras as
    permitted for tests that use ``pytest.importorskip``)."""
    pyproject = _read_pyproject(wheel)
    proj = pyproject.get("project", {})
    deps: set[str] = set()
    for dep in proj.get("dependencies", []):
        name = _dep_name(dep)
        if name.startswith("opaque-"):
            deps.add(name)
    for extras in proj.get("optional-dependencies", {}).values():
        for dep in extras:
            name = _dep_name(dep)
            if name.startswith("opaque-"):
                deps.add(name)
    return deps


def _dep_name(dep: str) -> str:
    # ``opaque-core>=0,<1`` → ``opaque-core``; strip extras and version.
    return (
        dep.split("[", 1)[0]
        .split(">=", 1)[0]
        .split("<=", 1)[0]
        .split("==", 1)[0]
        .split("<", 1)[0]
        .split(">", 1)[0]
        .strip()
    )


def _transitive_opaque_deps(wheel: str, _seen: set[str] | None = None) -> set[str]:
    seen = _seen if _seen is not None else set()
    if wheel in seen:
        return set()
    seen.add(wheel)
    deps = {wheel}
    for d in _direct_opaque_deps(wheel):
        deps |= _transitive_opaque_deps(d, seen)
    return deps


def _allowed_roots(wheel: str) -> set[str]:
    roots: set[str] = set()
    for d in _transitive_opaque_deps(wheel):
        roots.update(WHEEL_IMPORT_ROOTS.get(d, ()))
    return roots


def _opaque_imports(path: pathlib.Path) -> set[str]:
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"))
    except SyntaxError:
        return set()
    out: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.startswith("opaque"):
                    out.add(alias.name)
        elif isinstance(node, ast.ImportFrom):
            mod = node.module or ""
            if mod.startswith("opaque"):
                out.add(mod)
    return out


def _root_match(import_path: str, roots: set[str]) -> bool:
    # Plain ``import opaque`` is always allowed: it imports the namespace
    # package itself, which any wheel ships into.
    if import_path == "opaque":
        return True
    return any(import_path == r or import_path.startswith(r + ".") for r in roots)


# Baseline allowlist of pre-existing cross-cone test imports that the user
# explicitly wants flagged for future placement decisions rather than
# silently broken or outright failed today. Each entry is
# ``(test_path_relative_to_repo_root, imported_module)``.
#
# The contract test fails on ANY violation outside this set — so refactor
# work cannot accidentally introduce new cross-cone tests. As each phase
# moves tests into their proper home, the corresponding entries here are
# removed; the test stays green throughout.
KNOWN_CROSS_CONE_IMPORTS: frozenset[tuple[str, str]] = frozenset(
    {
        # opaque-core tests that use dpsgd primitives — phase 5 will move
        # them to opaque-dpsgd/tests/ during the dpsgd reshape.
        (
            "packages/opaque-core/tests/clipping/test_empty_batch.py",
            "opaque.dpsgd.clipping",
        ),
        (
            "packages/opaque-core/tests/clipping/test_empty_batch.py",
            "opaque.dpsgd.clipping._adaptive",
        ),
        (
            "packages/opaque-core/tests/clipping/test_empty_batch.py",
            "opaque.dpsgd.clipping._distributed",
        ),
        (
            "packages/opaque-core/tests/clipping/test_empty_batch.py",
            "opaque.dpsgd.noise",
        ),
        (
            "packages/opaque-core/tests/functional/test_compile.py",
            "opaque.dpsgd.noise",
        ),
        (
            "packages/opaque-core/tests/functional/test_precision.py",
            "opaque.dpsgd.noise",
        ),
        (
            "packages/opaque-core/tests/rng/test_rng_helpers.py",
            "opaque.dpsgd.noise",
        ),
        # Cross-stack DPSGD↔DPFTRL tests. Mutual non-dependency; no
        # natural home. Park here until the bidirectional placement
        # decision is taken (likely a top-level ``tests/integration/``
        # parking lot or moved into the most-downstream wheel — phase 5
        # opens this question).
        (
            "packages/opaque-dpsgd/tests/accounting/test_lambda_cgd.py",
            "opaque.dpftrl.accounting",
        ),
        (
            "packages/opaque-dpsgd/tests/accounting/test_lambda_cgd.py",
            "opaque.dpftrl.accounting.mechanisms.types",
        ),
        (
            "packages/opaque-dpsgd/tests/accounting/test_mf_mechanisms.py",
            "opaque.dpftrl.accounting",
        ),
        (
            "packages/opaque-dpsgd/tests/accounting/test_mf_mechanisms.py",
            "opaque.dpftrl.accounting.amplification.types",
        ),
        (
            "packages/opaque-dpsgd/tests/accounting/test_mf_mechanisms.py",
            "opaque.dpftrl.accounting.mechanisms.types",
        ),
        (
            "packages/opaque-dpsgd/tests/accounting/test_state_dict.py",
            "opaque.dpftrl.accounting",
        ),
        (
            "packages/opaque-dpsgd/tests/accounting/test_state_dict.py",
            "opaque.dpftrl.accounting.amplification._b_min_sep",
        ),
        (
            "packages/opaque-dpsgd/tests/accounting/test_state_dict.py",
            "opaque.dpftrl.accounting.amplification._poisson",
        ),
        (
            "packages/opaque-dpsgd/tests/accounting/test_state_dict.py",
            "opaque.dpftrl.accounting.mechanisms._band_mf",
        ),
        (
            "packages/opaque-dpsgd/tests/distributed/test_ddp_integration.py",
            "opaque.dpftrl.noise",
        ),
        (
            "packages/opaque-dpsgd/tests/distributed/test_noise_determinism.py",
            "opaque.dpftrl.noise",
        ),
        (
            "packages/opaque-dpftrl/tests/noise/test_auto_clipping_mf_noise.py",
            "opaque.dpsgd.clipping",
        ),
        (
            "packages/opaque-dpftrl/tests/noise/test_band_mf_noise.py",
            "opaque.dpsgd.accounting",
        ),
        (
            "packages/opaque-dpftrl/tests/noise/test_mf_noise.py",
            "opaque.dpsgd.accounting",
        ),
        # Patches tests use dpsgd's clipping to verify gradient flow
        # through the kernel patches survives. Mutual non-dependency
        # between patches and dpsgd; phase 6 revisits placement.
        (
            "packages/opaque-patches/tests/_helpers.py",
            "opaque.dpsgd.clipping",
        ),
        (
            "packages/opaque-patches/tests/kernels/test_autocast.py",
            "opaque.dpsgd.clipping",
        ),
        (
            "packages/opaque-patches/tests/kernels/test_compile.py",
            "opaque.dpsgd.clipping",
        ),
        (
            "packages/opaque-patches/tests/peft/test_fused_lora_qkv.py",
            "opaque.dpsgd.clipping",
        ),
        (
            "packages/opaque-patches/tests/peft/test_qwen2_lora.py",
            "opaque.dpsgd.clipping",
        ),
        (
            "packages/opaque-patches/tests/torch/test_checkpoint.py",
            "opaque.dpsgd.clipping",
        ),
        (
            "packages/opaque-patches/tests/torch/test_cpu_offload.py",
            "opaque.dpsgd.clipping",
        ),
        (
            "packages/opaque-patches/tests/transformers/components/test_attention.py",
            "opaque.dpsgd.clipping",
        ),
        (
            "packages/opaque-patches/tests/transformers/models/_test_utils.py",
            "opaque.dpsgd.clipping",
        ),
    }
)


def test_each_wheels_tests_respect_dep_cone() -> None:
    new_violations: list[str] = []
    repo_root = PACKAGES_DIR.parent
    for wheel_dir in sorted(PACKAGES_DIR.iterdir()):
        if not wheel_dir.is_dir():
            continue
        wheel = wheel_dir.name
        tests = wheel_dir / "tests"
        if not tests.exists():
            continue
        roots = _allowed_roots(wheel)
        if not roots:
            roots = set(WHEEL_IMPORT_ROOTS.get(wheel, ()))
        for path in tests.rglob("*.py"):
            for imp in _opaque_imports(path):
                if _root_match(imp, roots):
                    continue
                rel = str(path.relative_to(repo_root))
                if (rel, imp) in KNOWN_CROSS_CONE_IMPORTS:
                    continue
                new_violations.append(
                    f"{rel}: imports {imp} which is outside "
                    f"{wheel}'s transitive dep cone"
                )

    assert not new_violations, (
        "New test placement violations (test imports outside the wheel's "
        "transitive dep cone, not in KNOWN_CROSS_CONE_IMPORTS):\n"
        + "\n".join(f"  - {v}" for v in new_violations)
        + "\n\nIf this is a legitimate cross-cutting test, add it to "
        "KNOWN_CROSS_CONE_IMPORTS in tests/contracts/test_test_placement.py "
        "with a comment explaining the placement decision; otherwise move "
        "the test to a wheel that depends on every package it imports."
    )


def test_python_version_supports_tomllib() -> None:
    # Sanity check: stdlib tomllib lands in 3.11; this repo requires-python
    # >=3.11,<3.13 so the test runner always has it.
    assert sys.version_info >= (3, 11)
