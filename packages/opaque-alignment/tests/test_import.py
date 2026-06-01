"""Smoke test: the alignment façade and implementation namespaces import.

Both ``opaque.alignment`` (public façade) and ``opaque.api.alignment``
(implementation namespace) must import cleanly, and importing one must not
shadow the shared ``opaque`` / ``opaque.api`` PEP 420 namespaces (sibling
wheels such as ``opaque.engine`` must remain importable alongside).
"""

from __future__ import annotations


def test_facade_imports() -> None:
    import opaque.alignment  # noqa: F401

    assert hasattr(opaque.alignment, "__all__")


def test_impl_namespace_imports() -> None:
    import opaque.api.alignment  # noqa: F401

    assert hasattr(opaque.api.alignment, "__all__")


def test_namespace_coexists_with_engine() -> None:
    # Importing alignment must not break sibling wheels sharing ``opaque.*``.
    import opaque.alignment  # noqa: F401
    import opaque.distributed  # from opaque-engine

    assert hasattr(opaque.distributed, "gather_for_metrics")
