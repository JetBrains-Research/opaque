"""End-to-end stack guides import from the context-correct stack only.

``docs/user-guide/dp-sgd.md`` is the DP-SGD walkthrough; every
``opaque.<stack>.*`` import inside its code blocks must be
``opaque.dpsgd.*``. Same for ``docs/user-guide/dp-ftrl.md`` with
``opaque.dpftrl.*``. ``opaque.<concern>`` imports for cross-cutting
surfaces (``opaque.types``, ``opaque.serialization``,
``opaque.optimizers``, ``opaque.functional``, ``opaque.random``,
``opaque.distributed``, ``opaque.scheduling``, ``opaque.profiling``,
``opaque.accounting``, ``opaque.auditing``, ``opaque.patches``,
``opaque.transformers``) are allowed in either guide.

The check is intentionally narrow: it only inspects the two
end-to-end guides. Topic pages (clipping.md, noise.md, sampling.md,
accounting.md) cover both stacks side by side and use the
context-correct path per section; they don't fit a "single stack per
file" rule and are not scanned here.
"""

from __future__ import annotations

import pathlib
import re

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]


def _opaque_stack_imports(text: str) -> set[str]:
    """Return the set of stack identifiers (``"dpsgd"``, ``"dpftrl"``,
    or ``"OTHER:<full path>"``) referenced by an
    ``opaque.<stack>.*`` import in the given text. Only stack roots
    (dpsgd, dpftrl) trigger discipline; everything else is skipped.
    """
    pattern = re.compile(r"\bopaque\.(dpsgd|dpftrl)\b")
    return {m.group(1) for m in pattern.finditer(text)}


def test_dp_sgd_guide_uses_dpsgd_only() -> None:
    path = REPO_ROOT / "docs" / "user-guide" / "dp-sgd.md"
    assert path.exists(), path
    found = _opaque_stack_imports(path.read_text(encoding="utf-8"))
    leaks = found - {"dpsgd"}
    assert not leaks, (
        f"docs/user-guide/dp-sgd.md references foreign stack(s) "
        f"{sorted(leaks)} — DP-SGD content must use opaque.dpsgd.* paths "
        f"only. (Cross-stack mentions in prose with no import statement "
        f"are still flagged here; rephrase to avoid the literal "
        f"``opaque.dpftrl`` token, or move the discussion to a topic page.)"
    )


def test_dp_ftrl_guide_uses_dpftrl_only() -> None:
    path = REPO_ROOT / "docs" / "user-guide" / "dp-ftrl.md"
    assert path.exists(), path
    found = _opaque_stack_imports(path.read_text(encoding="utf-8"))
    leaks = found - {"dpftrl"}
    assert not leaks, (
        f"docs/user-guide/dp-ftrl.md references foreign stack(s) "
        f"{sorted(leaks)} — DP-FTRL content must use opaque.dpftrl.* paths "
        f"only."
    )


def test_user_facing_docs_do_not_leak_internal_namespace() -> None:
    """``opaque.api.*`` paths only appear under ``docs/extending/``.

    Mentions of the literal token ``opaque.api.`` in prose — like
    "the ``opaque.api.*`` contributor surface" — are forbidden
    everywhere except ``docs/extending/`` (where they're documented
    as the contributor entry point).
    """
    docs_dir = REPO_ROOT / "docs"
    forbidden_subdirs = (
        "user-guide",
        "reference",
        "getting-started",
        "mechanisms",
        "tutorials",
    )
    violations: list[str] = []
    for sub in forbidden_subdirs:
        for path in (docs_dir / sub).rglob("*.md"):
            text = path.read_text(encoding="utf-8")
            # Allow the *literal string* ``opaque.api.*`` (a glob)
            # used as illustrative prose ("the ``opaque.api.*``
            # plug-in surface"); flag concrete paths like
            # ``opaque.api.engine.X``.
            if re.search(r"opaque\.api\.[a-z_][a-z_0-9]*", text):
                rel = path.relative_to(REPO_ROOT)
                violations.append(str(rel))

    assert not violations, (
        "User-facing docs must not reference concrete ``opaque.api.*`` "
        "paths — the contributor namespace is documented only under "
        "``docs/extending/``. Move references in:\n"
        + "\n".join(f"  - {v}" for v in sorted(violations))
    )
